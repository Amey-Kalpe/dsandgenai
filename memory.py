from langgraph.store.memory import InMemoryStore  # Long-term Memory
from langgraph.checkpoint.memory import MemorySaver  # Short-term Memory

# Initialize long-term memory store for persistent data between conversations
in_memory_store = InMemoryStore()

# Initialize checkpointer for short-term memory within a single thread/conversation
checkpointer = MemorySaver()
