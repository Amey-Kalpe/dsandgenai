import os
from langgraph.checkpoint.memory import MemorySaver  # Short-term Memory
from langgraph.store.memory import InMemoryStore  # Long-term Memory

os.environ["LANGSMITH_TRACING"] = "true"  # Enables LangSmith tracing
os.environ["LANGSMITH_PROJECT"] = "langgraph-multi-agent"  # Project name for organizing LangSmith traces

# Initialize long-term memory store for persistent data between conversations
in_memory_store = InMemoryStore()

# Initialize checkpointer for short-term memory within a single thread/conversation
checkpointer = MemorySaver()

