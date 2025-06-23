import os
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage, SystemMessage, ToolMessage, HumanMessage
from operator import add as add_messages
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0, max_tokens=1000)

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

pdf_path = os.path.join(os.path.dirname(__file__), "data", "Stock_Market_Performance_2024.pdf")
pages = PyPDFLoader(pdf_path).load_and_split()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

pages_split = text_splitter.split_documents(pages)

chroma_persist_directory = os.path.join(os.path.dirname(__file__), "chroma_db")
collection_name = "stock_market_performance"

if not os.path.exists(chroma_persist_directory):
    os.makedirs(chroma_persist_directory)

vector_store = Chroma.from_documents(
    pages_split,
    embeddings,
    persist_directory=chroma_persist_directory,
    collection_name=collection_name
)

retriever = vector_store.as_retriever(
    search_type="similarity", # Type of search to perform
    search_kwargs={"k": 4} # Number of documents to retrieve
)

@tool
def retriever_tool(query: str) -> str:
    """Retrieve relevant documents based on the query."""
    docs = retriever.invoke(query)
    if not docs:
        return "No relevant documents found."
    
    results = []
    for doc in docs:
        results.append(f"Document: {doc.page_content}")
    
    return "\n\n".join(results)

tools = [retriever_tool]
tools_dict = {tool.name: tool for tool in tools} # Create a dictionary of tools for easy access and to check if the LLM is using the right tools

llm = llm.bind_tools(tools)

class AgentState(TypedDict):
    """State of the RAG agent."""
    messages: Annotated[Sequence[BaseMessage], add_messages]

def should_continue(state: AgentState) -> bool:
    """Determine if the agent should continue."""
    result = state["messages"][-1]
    return hasattr(result, "tool_calls") and len(result.tool_calls) > 0

system_prompt = """
You are an intelligent AI assistant who answers questions about Stock Market Performance in 2024 based on the PDF document loaded into your knowledge base.
Use the retriever tool available to answer questions about the stock market performance data. You can make multiple calls if needed.
If you need to look up some information before asking a follow up question, you are allowed to do that!
Please always cite the specific parts of the documents you use in your answers.
"""

def llm_agent(state: AgentState) -> AgentState:
    """LLM agent that processes the messages and generates a response."""
    messages = state["messages"]

    response = llm.invoke(
        [SystemMessage(content=system_prompt)] + messages
    )
    
    return {"messages": [response]}

def retriever_agent(state: AgentState) -> AgentState:
    """Execute the retriever tool based on the LLM's response."""
    # We know the last message is a ToolMessage because this node is only called after the LLM has made a tool call
    tool_calls = state["messages"][-1].tool_calls
    results = []

    for t in tool_calls:
        if t["name"] in tools_dict:
            tool = tools_dict[t["name"]]
            result = tool.invoke(t["args"].get("query", ""))
        else:
            print(f"Tool {t['name']} does not exist.")
            result = f" Incorrect Tool! {t['name']} does not exist. Please retry and select a valid tool."
        
        results.append(ToolMessage(tool_call_id=t['id'], name=t['name'], content=str(result)))
    
    print("Tool execution completed. Returning results to the LLM.")
    # Append the results to the messages
    return {"messages": results}

graph = StateGraph(AgentState)

graph.add_node("llm_agent", llm_agent)
graph.add_node("retriever_agent", retriever_agent)

graph.add_conditional_edges(
    "llm_agent",
    should_continue,
    {
        True: "retriever_agent",
        False: END
    }
)
graph.add_edge(START, "llm_agent")
graph.add_edge("retriever_agent", "llm_agent")

rag_agent = graph.compile()

def run_rag_agent():
    while True:
        user_input = input("Enter your question (type 'exit' or 'quit' to stop): ")
        if user_input.strip().lower() in ["exit", "quit"]:
            print("Exiting RAG agent.")
            break
        state = {"messages": [HumanMessage(content=user_input)]}
        result = rag_agent.invoke(state)
        print("----------------- RAG Agent Response -----------------")
        print(result["messages"][-1].content if result["messages"] else "No response generated.")
        print("------------------------------------------------------")


if __name__ == "__main__":
    run_rag_agent()