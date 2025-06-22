from typing import TypedDict, Sequence, Annotated
# What is Annotated? - # https://docs.python.org/3/library/typing.html#typing.Annotated
# What is Sequence? - # https://docs.python.org/3/library/typing.html#typing.Sequence
from langchain_core.messages import BaseMessage, ToolMessage, SystemMessage, HumanMessage, AIMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, InjectedState
from langgraph.types import Command
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool, InjectedToolCallId

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]  # Annotated sequence of BaseMessage since BaseMessage is the parent class of all message types in LangChain
    document_content: str

@tool
def update(content: str, tool_call_id: Annotated[str, InjectedToolCallId], config: RunnableConfig) -> Command:
    """Updates the document with the provided content.
    
    Args:
        content (str): The content to add to the document."""
    # Using Command helps to update the state of the agent with the new content.
    # We use the tool_call_id to track which tool call is being processed and return a ToolMessage with the updated content.
    # config is not used here, but it can be used to access the state variables if needed like:
        # state = config["configurable"].get("state_variable")
    return Command(
        update={
            "document_content": content,
            "messages": [
                ToolMessage(
                    content=f"Document updated! The current content is: \n{content}",
                    tool_call_id=tool_call_id,
                )
            ]
        }
    )

@tool
def save(filename: str, state: Annotated[AgentState, InjectedState]) -> str:
    """Saves the document content to a text file.
    
    Args:
        filename (str): The name of the file to save the document content."""
    print(f"Saving document to {filename}...")
    print(f"Current document content: {state['document_content']}")
    if not filename.endswith('.txt'):
        filename += '.txt'
    try:
        with open(filename, 'w') as f:
            file_content = state["document_content"]
            file_content = file_content.strip()
            print("Document content to save:", file_content)
            f.write(file_content)
    except Exception as e:
        return f"Error saving file: {str(e)}"
    return f"Document saved to {filename}."

tools = [update, save]

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3).bind_tools(tools)

def setup(state: AgentState) -> AgentState:
    """Initialize the agent state with an empty document."""
    state["document_content"] = ""
    return state

def agent(state: AgentState) -> AgentState:
    """Process the state by invoking the LLM with the messages."""
    system_message = SystemMessage(
        content=f"""
        You are a document drafting assistant. You can help users draft documents by updating content and saving it to a file.

        - The user will provide content to update the document, use the 'update' tool to add content to the document.
        - When the user wants to save the document, use the 'save' tool with a filename
        - Make sure to always show the current state of the document after each update.

        The current document content is:
        {state["document_content"]}
        """
    )

    if state["messages"]:
        user_input = input("Query: ")
        user_msg = HumanMessage(content=user_input)
    else:
        user_msg = HumanMessage(content="Hello! I'm ready to draft a document with you.")

    response = llm.invoke([system_message] + state["messages"] + [user_msg])

    print("------------- AI Response -------------", response.content)
    if hasattr(response, 'tool_calls') and response.tool_calls:
        print(f"Using tools: {[tool_["name"] for tool_ in response.tool_calls]}")
    
    return {"messages": [response]}

def should_continue(state: AgentState) -> str:
    """Determine if we should continue or end the conversation."""
    last_message = state["messages"][-1]
    if isinstance(last_message, ToolMessage) and "save" in last_message.content.lower() and "document" in last_message.content.lower():
        return "end"
    return "continue"

graph = StateGraph(AgentState)
graph.add_node("setup", setup)
graph.add_node("agent", agent)
graph.add_node("should_continue", lambda state: state)
graph.add_node("tool_node", ToolNode(tools=tools))

graph.add_edge(START, "setup")
graph.add_edge("setup", "agent")
graph.add_conditional_edges(
    "tool_node",
    should_continue,
    {
        "continue": "agent",
        "end": END,
    },
)
graph.add_edge("agent", "tool_node")

app = graph.compile()

def print_message(messages):
    """If the last 3 messages contain a tool call, print the tool call result."""
    if not messages:
        return
    if isinstance(messages[-1], ToolMessage):
        print("--------------------------------------------------")
        print(f"🛠️ Tool Result: \n{messages[-1].content}")
        print("--------------------------------------------------")

def run_agent():
    """Run the agent and print the stream of messages with a friendly interface."""
    initial_state = AgentState(messages=[])
    print("👋 Hello! I'm your Document Drafter AI Assistant.")
    print("📝 Let's start drafting your document together!")
    print("--------------------------------------------------")
    for message in app.stream(initial_state, stream_mode="values"):
        if "messages" in message:
            print_message(message["messages"])
    print("✅ Session ended. Your document work is complete! 🚀")

if __name__ == "__main__":
    run_agent()