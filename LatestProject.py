
import os
from typing import TypedDict, Literal, List, Dict, Any, Annotated
from langgraph.graph import Graph, StateGraph, END, add_messages
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document, BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import json
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

TAVILY_API_KEY= os.getenv("TAVILY_API_KEY")
GEMINI_API_KEY= os.getenv("GEMINI_API_KEY")

# Initialize models and tools
model = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    google_api_key=GEMINI_API_KEY,
    temperature=0.1
)

tavily_search = TavilySearchResults(
    max_results=5,
    tavily_api_key=TAVILY_API_KEY
)

# Initialize embeddings for RAG
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# State definition with messages for conversation history
class NewsAnalysisState(TypedDict):
    messages: Annotated[list, add_messages]  # Conversation history
    query: str
    llm_response: str
    rag_response: str
    web_crawler_response: str
    validation_result: Dict[str, Any]
    final_result: str  
    supervisor_decision: str
    iteration_count: int
    max_iterations: int
    is_validated: bool


# Node 1: Updated Supervisor Node with IPL Focus
def supervisor_node(state: NewsAnalysisState) -> NewsAnalysisState:
    """
    Supervisor node that decides which specialized node to call next
    """
    logger.info("Supervisor Node: Making routing decision")
    
    query = state["query"]
    iteration_count = state.get("iteration_count", 0)
    
    # Supervisor decision logic
    supervisor_prompt = ChatPromptTemplate.from_template("""
    You are a supervisor that routes queries to the most appropriate processing node.
    
    Query: {query}
    Current iteration: {iteration_count}
    
    Based on the query, decide which node should handle this:
    - "llm_call": For general questions, analysis, or creative tasks that do not require external data or specific historical IPL information.
    - "rag": For questions related to IPL history up to and including the 2024 season. This is your primary knowledge base for IPL data.
    - "web_crawler": For current events, real-time information, or recent news specifically about the IPL **beyond the 2024 season (e.g., 2025 or future events)**. This should be used only if the `rag` node cannot fulfill the request.
    
    Respond with just the node name (llm_call, rag, or web_crawler).
    
    Consider:
    - If the query asks for IPL information up to and including 2024 (e.g., "IPL 2023 winner", "Mumbai Indians history") -> **rag**
    - If the query explicitly asks for "latest IPL information", "IPL 2025", "current IPL news", or future IPL events -> **web_crawler**
    - If it's a general analytical question about cricket that isn't tied to specific IPL historical data or current events -> **llm_call**
    """)
    
    decision_chain = supervisor_prompt | model
    
    try:
        response = decision_chain.invoke({
            "query": query,
            "iteration_count": iteration_count + 1
        })
        
        decision = response.content.strip().lower()
        
        # Validate decision
        valid_decisions = ["llm_call", "rag", "web_crawler"]
        if decision in valid_decisions:
            decision = "llm_call"  # Default fallback
        
        state["supervisor_decision"] = decision
        state["iteration_count"] = iteration_count
        
        logger.info(f"Supervisor decision: {decision}")
        
    except Exception as e:
        logger.error(f"Error in supervisor node: {e}")
        state["supervisor_decision"] = "llm_call"  # Default fallback
    
    return state

# Node 2: Router Function
def router_function(state: NewsAnalysisState) -> Literal["llm_call", "rag", "web_crawler"]:
    """
    Router function that determines the next node based on supervisor decision
    """
    decision = state.get("supervisor_decision", "llm_call")
    logger.info(f"Router: Directing to {decision}")
    return decision

# Node 3.1: LLM Call Node
def llm_call_node(state: NewsAnalysisState) -> NewsAnalysisState:
    """
    LLM node for general analysis and responses
    """
    logger.info("LLM Call Node: Processing query")
    
    query = state["query"]
    messages = state.get("messages", [])
    
    # Add processing message
    processing_message = SystemMessage(content="LLM Node: Processing query with general AI analysis")
    
    llm_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert analyst. Provide a comprehensive and accurate response to the following query.
        Consider the conversation history for context if relevant."""),
        ("human", "Query: {query}\n\nProvide a detailed, factual, and well-structured response. Include relevant context and explanations.")
    ])
    
    try:
        llm_chain = llm_prompt | model
        response = llm_chain.invoke({"query": query})
        
        state["llm_response"] = response.content
        
        # Add response to conversation history
        response_message = AIMessage(content=f"LLM Response: {response.content}")
        state["messages"] = messages + [processing_message, response_message]
        
        logger.info("LLM Call Node: Response generated successfully")
        
    except Exception as e:
        logger.error(f"Error in LLM call node: {e}")
        error_response = f"Error generating LLM response: {str(e)}"
        state["llm_response"] = error_response
        
        # Add error to conversation history
        error_message = AIMessage(content=f"LLM Error: {error_response}")
        state["messages"] = messages + [processing_message, error_message]
    
    return state

# Node 3.2: RAG Node
def rag_node(state: NewsAnalysisState) -> NewsAnalysisState:
    """
    RAG node for IPL knowledge base retrieval and response
    """
    logger.info("RAG Node: Processing query with IPL knowledge retrieval")

    query = state["query"]
    messages = state.get("messages", [])

    # Add processing message
    processing_message = SystemMessage(content="RAG Node: Retrieving IPL information from knowledge base")
    
    try:
        # --- IPL Data Loading and Processing (moved inside the function) ---
        logger.info("RAG Node: Loading and processing IPL data...")
        loader = DirectoryLoader("./ipl_data", glob="./*.txt", loader_cls=TextLoader)
        docs = loader.load()
        textsplitter = RecursiveCharacterTextSplitter(
            chunk_size=200,
            chunk_overlap=50
        )
        new_docs = textsplitter.split_documents(documents=docs)
        vectorstore = FAISS.from_documents(new_docs, embeddings)
        rag_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
        logger.info("RAG Node: IPL data loaded and retriever initialized.")
        # --- End of IPL Data Loading and Processing ---

        # Retrieve relevant documents
        retrieved_docs = rag_retriever.invoke(query)

        # Combine retrieved content
        context = "\n\n".join([doc.page_content for doc in retrieved_docs])

        rag_prompt = ChatPromptTemplate.from_messages([
            ("system", """Based on the following IPL context from the knowledge base, answer the query accurately.
            This knowledge base contains IPL information up to and including the 2024 season.
            Consider the conversation history for additional context if relevant."""),
            ("human", """IPL Context:
{context}

Query: {query}

Provide a response based on the IPL context. If the context doesn't contain relevant information, 
mention that the information is not available in the current IPL knowledge base (up to 2024 season).""")
        ])

        rag_chain = rag_prompt | model
        response = rag_chain.invoke({
            "context": context,
            "query": query
        })

        state["rag_response"] = response.content
        logger.info(f"RAG Node: IPL response generated successfully: {response.content}")
        # Add response to conversation history
        retrieval_info = f"Retrieved {len(retrieved_docs)} IPL documents from knowledge base"
        retrieval_message = SystemMessage(content=retrieval_info)
        response_message = AIMessage(content=f"RAG Response: {response.content}")
        state["messages"] = messages + [processing_message, retrieval_message, response_message]

        logger.info("RAG Node: IPL response generated successfully")

    except Exception as e:
        logger.error(f"Error in RAG node: {e}")
        error_response = f"Error in RAG processing: {str(e)}"
        state["rag_response"] = error_response

        # Add error to conversation history
        error_message = AIMessage(content=f"RAG Error: {error_response}")
        state["messages"] = messages + [processing_message, error_message]

    return state
# Node 3.3: Web Crawler Node
def web_crawler_node(state: NewsAnalysisState) -> NewsAnalysisState:
    """
    Web crawler node using Tavily AI for real-time IPL information
    """
    logger.info("Web Crawler Node: Fetching real-time IPL information")
    
    query = state["query"]
    messages = state.get("messages", [])
    
    # Add processing message
    processing_message = SystemMessage(content="Web Crawler Node: Searching for real-time IPL information using Tavily AI")
    
    try:
        # Enhance query for IPL-specific search
        enhanced_query = f"IPL {query}" if "IPL" not in query.upper() else query
        
        # Search for real-time information
        search_results = tavily_search.run(enhanced_query)
        
        # Process search results
        if isinstance(search_results, list) and search_results:
            # Extract and format search results
            formatted_results = []
            for result in search_results[:3]:  # Top 3 results
                if isinstance(result, dict):
                    title = result.get("title", "")
                    content = result.get("content", "")
                    url = result.get("url", "")
                    formatted_results.append(f"Title: {title}\nContent: {content}\nSource: {url}\n")
            
            search_context = "\n---\n".join(formatted_results)
        else:
            search_context = "No recent IPL information found."
        
        # Generate response based on search results
        web_prompt = ChatPromptTemplate.from_messages([
            ("system", """Based on the following real-time IPL search results, provide a comprehensive answer to the query.
            Focus on the most recent and relevant IPL information.
            Consider the conversation history for additional context if relevant."""),
            ("human", """IPL Search Results:
{search_context}

Query: {query}

Synthesize the IPL information from the search results to provide an accurate, up-to-date response.
Include source references where appropriate.""")
        ])
        
        web_chain = web_prompt | model
        response = web_chain.invoke({
            "search_context": search_context,
            "query": query
        })
        
        state["web_crawler_response"] = response.content
        logger.info(f"web_crawler Node: IPL response generated successfully: {response.content}")
        # Add response to conversation history
        search_info = f"Found {len(search_results) if isinstance(search_results, list) else 0} IPL search results"
        search_message = SystemMessage(content=search_info)
        response_message = AIMessage(content=f"Web Crawler Response: {response.content}")
        state["messages"] = messages + [processing_message, search_message, response_message]
        
        logger.info("Web Crawler Node: IPL response generated successfully")
        
    except Exception as e:
        logger.error(f"Error in web crawler node: {e}")
        error_response = f"Error in web crawling: {str(e)}"
        state["web_crawler_response"] = error_response
        
        # Add error to conversation history
        error_message = AIMessage(content=f"Web Crawler Error: {error_response}")
        state["messages"] = messages + [processing_message, error_message]
    
    return state

# Node 4: Validation Node
def validation_node(state: NewsAnalysisState) -> NewsAnalysisState:
    """
    Validation node to check the quality and accuracy of generated outputs
    """
    logger.info("Validation Node: Validating generated output")
    
    messages = state.get("messages", [])
    
    # Add validation processing message
    validation_message = SystemMessage(content="Validation Node: Analyzing response quality and accuracy")
    
    # Determine which response to validate based on supervisor decision
    supervisor_decision = state.get("supervisor_decision", "llm_call")
    
    if supervisor_decision == "llm_call":
        response_to_validate = state.get("llm_response", "")
        source = "LLM"
    elif supervisor_decision == "rag":
        response_to_validate = state.get("rag_response", "")
        source = "RAG"
    else:  # web_crawler
        response_to_validate = state.get("web_crawler_response", "")
        source = "Web Crawler"
    
    if not response_to_validate:
        validation_result = {
            "is_valid": False,
            "reason": "No response to validate",
            "confidence": 0.0
        }
        state["validation_result"] = validation_result
        state["is_validated"] = False
        
        # Add to conversation history
        error_message = AIMessage(content="Validation Failed: No response to validate")
        state["messages"] = messages + [validation_message, error_message]
        return state
    
    # Validation logic with IPL context
    validation_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a quality validator specializing in IPL cricket information. Evaluate the following response for:
1. Accuracy and factual correctness (especially for IPL data)
2. Completeness of the answer
3. Relevance to the original query
4. Overall quality and coherence

Consider the conversation history for additional context."""),
        ("human", """Original Query: {query}
Response to Validate ({source}): {response}

Provide your evaluation in the following JSON format:
{{
    "is_valid": true/false,
    "reason": "Brief explanation of validation decision",
    "confidence": 0.0-1.0,
    "suggestions": "Suggestions for improvement if not valid"
}}

Consider the response valid if it:
- Addresses the query appropriately
- Contains accurate IPL information (when applicable)
- Is well-structured and coherent
- Provides sufficient detail""")
    ])
    
    try:
        validation_chain = validation_prompt | model
        validation_response = validation_chain.invoke({
            "query": state["query"],
            "response": response_to_validate,
            "source": source
        })
        
        # Parse validation result
        try:
            validation_result = json.loads(validation_response.content)
            logger.info(f"validation Node: IPL response generated successfully: {validation_response.content}")
        except:
            # Fallback parsing if JSON parsing fails
            content = validation_response.content.lower()
            is_valid = "true" in content and "is_valid" in content
            validation_result = {
                "is_valid": is_valid,
                "reason": "Validation completed",
                "confidence": 0.7,
                "suggestions": ""
            }
        
        state["validation_result"] = validation_result
        state["is_validated"] = validation_result.get("is_valid", False)
        
        # Add validation result to conversation history
        validation_status = "PASSED" if state["is_validated"] else "FAILED"
        result_message = AIMessage(content=f"Validation {validation_status}: {validation_result.get('reason', 'No reason provided')}")
        state["messages"] = messages + [validation_message, result_message]
        
        logger.info(f"Validation result: {validation_result}")
        
    except Exception as e:
        logger.error(f"Error in validation node: {e}")
        validation_result = {
            "is_valid": False,
            "reason": f"Validation error: {str(e)}",
            "confidence": 0.0
        }
        state["validation_result"] = validation_result
        state["is_validated"] = False
        
        # Add error to conversation history
        error_message = AIMessage(content=f"Validation Error: {str(e)}")
        state["messages"] = messages + [validation_message, error_message]
    
    return state

# Final output node
def generate_final_output(state: NewsAnalysisState) -> NewsAnalysisState:
    """
    Generate the final validated output
    """
    logger.info("Final Output Node: Generating final response")
    
    messages = state.get("messages", [])
    supervisor_decision = state.get("supervisor_decision", "llm_call")
    
    # Add final processing message
    final_message = SystemMessage(content="Final Output Node: Generating final validated response")
    
    if supervisor_decision == "llm_call":
        final_response = state.get("llm_response", "")
    elif supervisor_decision == "rag":
        final_response = state.get("rag_response", "")
    else:  # web_crawler
        final_response = state.get("web_crawler_response", "")
    
    validation_info = state.get("validation_result", {})
    
    # Add validation metadata to final output
    final_output = f"""
=== IPL ANALYSIS SYSTEM RESPONSE ===

Query: {state['query']}
Processing Method: {supervisor_decision.upper()}
Validation Status: {'PASSED' if state.get('is_validated') else 'FAILED'}
Total Iterations: {state.get('iteration_count', 0)}

Response:
{final_response}

Validation Details:
- Confidence: {validation_info.get('confidence', 'N/A')}
- Validation Reason: {validation_info.get('reason', 'N/A')}

=== CONVERSATION HISTORY ===
Total Messages: {len(messages)}
"""
    
    # Add conversation history summary
    if messages:
        final_output += "\nKey Steps:\n"
        for i, msg in enumerate(messages[-5:], 1):  # Show last 5 messages
            msg_type = type(msg).__name__
            content_preview = msg.content[:100] + "..." if len(msg.content) > 100 else msg.content
            final_output += f"{i}. [{msg_type}] {content_preview}\n"
    
    state["final_result"] = final_output
    
    # Add final output to conversation history
    completion_message = AIMessage(content="Final output generated successfully")
    state["messages"] = messages + [final_message, completion_message]
    
    return state

# Decision functions for routing
def should_continue_validation(state: NewsAnalysisState) -> Literal["supervisor", "generate_final"]:
    """
    Decide whether to continue validation loop or generate final output
    """
    is_validated = state.get("is_validated", False)
    iteration_count = state.get("iteration_count", 0)
    max_iterations = state.get("max_iterations", 3)
    
    if is_validated or iteration_count >= max_iterations:
        return "generate_final"
    else:
        logger.info(f"Validation failed, returning to supervisor (iteration {iteration_count})")
        return "supervisor"

# Build the graph
def create_news_analysis_graph():
    """
    Create and configure the LangGraph workflow
    """
    # Create the graph
    workflow = StateGraph(NewsAnalysisState)
    
    # Add nodes
    workflow.add_node("supervisor", supervisor_node)
    workflow.add_node("llm_call", llm_call_node)
    workflow.add_node("rag", rag_node)
    workflow.add_node("web_crawler", web_crawler_node)
    workflow.add_node("validation", validation_node)
    workflow.add_node("generate_final", generate_final_output)
    
    # Set entry point
    workflow.set_entry_point("supervisor")
    
    # Add conditional edges from supervisor to specialized nodes
    workflow.add_conditional_edges(
        "supervisor",
        router_function,
        {
            "llm_call": "llm_call",
            "rag": "rag",
            "web_crawler": "web_crawler"
        }
    )
    
    # All specialized nodes go to validation
    workflow.add_edge("llm_call", "validation")
    workflow.add_edge("rag", "validation")
    workflow.add_edge("web_crawler", "validation")
    
    # Conditional edge from validation
    workflow.add_conditional_edges(
        "validation",
        should_continue_validation,
        {
            "supervisor": "supervisor",
            "generate_final": "generate_final"
        }
    )
    
    # Final output ends the workflow
    workflow.add_edge("generate_final", END)
    
    # Compile the graph
    app = workflow.compile()
    from IPython.display import Image, display
    Display = display(Image(app.get_graph().draw_mermaid_png()))
    return app

# Main execution function
def run_news_analysis(query: str) -> str:
    """
    Run the IPL-focused news analysis system with a given query
    """
    # Create the graph
    app = create_news_analysis_graph()
    
    # Initialize state
    initial_state = NewsAnalysisState(
        query=query,
        llm_response="",
        rag_response="",
        web_crawler_response="",
        validation_result={},
        final_result="",
        supervisor_decision="",
        iteration_count=0,
        max_iterations=3,
        is_validated=False,
        messages=[]  # Initialize messages list
    )
    
    logger.info(f"Starting IPL analysis for query: {query}")
    
    try:
        # Run the workflow
        result = app.invoke(initial_state)
        
        return result["final_result"]
        
    except Exception as e:
        logger.error(f"Error running IPL analysis: {e}")
        return f"Error processing query: {str(e)}"

# Example usage
if __name__ == "__main__":
    # Example queries to test different routing paths with IPL focus
    test_queries = [
        "Who won IPL 2024?",  # Should route to RAG
        "What are the latest IPL 2025 auction news?",  # Should route to web crawler
        "Analyze the batting strategies in T20 cricket",  # Should route to LLM
        "Mumbai Indians IPL history",  # Should route to RAG
        "IPL 2025, player transfers and news",  # Should route to web crawler
    ]
    
    print("=== IPL-FOCUSED NEWS ANALYSIS AND FACT-CHECKING SYSTEM ===\n")
    
    for i, query in enumerate(test_queries, 1):
        print(f"Test Query {i}: {query}")
        print("-" * 50)
        
        # Run the analysis
        result = run_news_analysis(query)
        print(result)
        print("\n" + "="*80 + "\n")



