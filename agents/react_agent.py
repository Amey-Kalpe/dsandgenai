from typing import TypedDict, Sequence, Annotated
# What is Annotated? - # https://docs.python.org/3/library/typing.html#typing.Annotated
# What is Sequence? - # https://docs.python.org/3/library/typing.html#typing.Sequence
from langchain_core.messages import BaseMessage, ToolMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages] # Annotated sequence of BaseMessage since BaseMessage is the parent class of all message types in LangChain

@tool
def addition(a: int, b: int) -> int:
    """Adds two integer numbers."""
    return a + b

@tool
def subtraction(a: int, b: int) -> int:
    """Subtracts two integer numbers."""
    return a - b

@tool
def multiplication(a: int, b: int) -> int:
    """Multiplies two integer numbers."""
    return a * b

tools = [addition, subtraction, multiplication]

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3).bind_tools(tools)

def process(state: AgentState) -> AgentState:
    """Process the state by invoking the LLM with the messages."""
    system_message = SystemMessage(
        content="You are my AI assistant. Answer my questions to the best of your ability. If you don't know the answer, say 'I don't know'."
    )
    response = llm.invoke([system_message] + state["messages"])
    return {"messages": [response]}


def should_continue(state: AgentState) -> AgentState:
    """Determine if the conversation should continue by checking if any more tool calls are needed."""
    last_message = state["messages"][-1]
    if not last_message.tool_calls:
        return "end"
    return "continue"

graph = StateGraph(AgentState)
graph.add_node("agent", process)
graph.add_node("should_continue", lambda state: state)
graph.add_node("tool_node", ToolNode(tools=tools))

graph.add_edge(START, "agent")
graph.add_conditional_edges(
    "agent",
    should_continue,
    {
        "continue": "tool_node",
        "end": END,
    },
)
graph.add_edge("tool_node", "agent")

app = graph.compile()

def print_stream(stream):
    """Print the stream of messages."""
    for message in stream:
        last_message = message["messages"][-1]
        if isinstance(last_message, tuple):
            print(last_message)
        else:
            last_message.pretty_print()

inputs = {"messages": [("user", "Add 40 and 12, subtract 2 from the result and multiply the result by 2.")]}  # Initial user message
print_stream(app.stream(inputs, stream_mode="values"))