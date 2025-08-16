import os
from langgraph.graph import StateGraph, START, END
from agents.memory_agent.node import create_memory
from agents.memory_agent.node import load_memory
from agents.supervisor.agent import supervisor_prebuilt
from agents.verification.node.verify_info import verify_info, should_interrupt
from agents.verification.node.human_input import human_input
from memory import in_memory_store
from memory import checkpointer
from agent_state import State

os.environ["LANGSMITH_TRACING"] = "true"  # Enables LangSmith tracing
os.environ["LANGSMITH_PROJECT"] = (
    "langgraph-multi-agent"  # Project name for organizing LangSmith traces
)


multi_agent_verify = StateGraph(State)

multi_agent_verify.add_node("verify_info", verify_info)
multi_agent_verify.add_node("human_input", human_input)
multi_agent_verify.add_node("load_memory", load_memory)
multi_agent_verify.add_node("supervisor", supervisor_prebuilt)
multi_agent_verify.add_node("create_memory", create_memory)

multi_agent_verify.add_edge(START, "verify_info")
multi_agent_verify.add_conditional_edges(
    "verify_info",
    should_interrupt,
    {"continue": "load_memory", "interrupt": "human_input"},
)

multi_agent_verify.add_edge("human_input", "verify_info")
# After loading memory, pass control to the supervisor
multi_agent_verify.add_edge("load_memory", "supervisor")
# After supervisor completes, save any new memory
multi_agent_verify.add_edge("supervisor", "create_memory")
# After creating/updating memory, the workflow ends
multi_agent_verify.add_edge("create_memory", END)


multi_agent_verify_graph = multi_agent_verify.compile(
    name="multi_agent_verify", checkpointer=checkpointer, store=in_memory_store
)
