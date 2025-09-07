"""LangGraph single-node graph template.

Returns a predefined response. Replace logic and configuration as needed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, TypedDict, Sequence, Annotated
from langgraph.graph.message import add_messages

# What is Annotated? - # https://docs.python.org/3/library/typing.html#typing.Annotated
# What is Sequence? - # https://docs.python.org/3/library/typing.html#typing.Sequence
from langchain_core.messages import BaseMessage, ToolMessage, SystemMessage
from langgraph.graph import StateGraph
from langgraph.runtime import Runtime
from langgraph.prebuilt import ToolNode
from langgraph.types import interrupt
from langgraph.store.memory import InMemoryStore  # Long-term Memory
from langgraph.checkpoint.memory import MemorySaver  # Short-term Memory
from langchain_openai import ChatOpenAI
from agent.tools import tools

# Initialize long-term memory store for persistent data between conversations
in_memory_store = InMemoryStore()

# Initialize checkpointer for short-term memory within a single thread/conversation
checkpointer = MemorySaver()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3).bind_tools(tools)


class Context(TypedDict):
    """Context parameters for the agent.

    Set these when creating assistants OR when invoking the graph.
    See: https://langchain-ai.github.io/langgraph/cloud/how-tos/configuration_cloud/
    """

    my_configurable_param: str


class State(TypedDict):
    """Input state for the agent.

    Defines the initial structure of incoming data.
    See: https://langchain-ai.github.io/langgraph/concepts/low_level/#state
    """

    messages: Annotated[Sequence[BaseMessage], add_messages]


def call_model(state: State, runtime: Runtime[Context]) -> Dict[str, Any]:
    """Process input and returns output.

    Can use runtime context to alter behavior.
    """
    system_message = SystemMessage(
        content="You are my AI assistant. Answer my questions to the best of your ability. If you don't know the answer, say 'I don't know'."
    )
    response = llm.invoke([system_message] + state["messages"])
    if response.type == "ai" and response.additional_kwargs.get("tool_calls", None):
        print(f"{response.type=}")
        print(response.additional_kwargs.get("tool_calls", None))
        for tc in response.tool_calls:
            # Pause execution until human approves
            _ = interrupt(
                {
                    "awaiting_user_approval": True,
                    "tool_name": tc["name"],
                    "tool_args": tc.get("args", {}),
                }
            )
        # _ = interrupt(
        #     {
        #         "awaiting": response.tool_calls[0]["name"],
        #         "args": response.tool_calls[0].get("args", {}),
        #     }
        # )
    return {"messages": [response]}
    # return {
    #     "changeme": "output from call_model. "
    #     f"Configured with {runtime.context.get('my_configurable_param')}"
    # }


def should_continue(state: State) -> State:
    """Determine if the conversation should continue by checking if any more tool calls are needed."""
    last_message = state["messages"][-1]
    if not last_message.tool_calls:
        return "end"
    return "continue"


# Define the graph
graph = (
    StateGraph(State, context_schema=Context)
    .add_node(call_model)
    .add_node("tool_node", ToolNode(tools=tools))
    .add_edge("__start__", "call_model")
    .add_conditional_edges(
        "call_model",
        should_continue,
        {
            "continue": "tool_node",
            "end": "__end__",
        },
    )
    .add_edge("tool_node", "call_model")
    .compile(name="React Agent", checkpointer=checkpointer, store=in_memory_store)
)
