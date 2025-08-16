"""
This module defines and configures the invoice information subagent for a customer support bot using LangGraph's pre-built ReAct agent architecture.

The subagent specializes in handling customer queries related to invoices and billing information. It leverages a language model (OpenAI's GPT-4o-mini),
a set of invoice-specific tools, and custom prompts to reason about and respond to user requests.
The agent maintains conversation context and persistent data using a checkpointer and an in-memory store.

Key components:
- Language model initialization for controlled, context-aware responses.
- Integration of invoice tools for database queries and information retrieval.
- Custom prompt for guiding the agent's behavior in invoice-related scenarios.
- State schema for structured data flow between agent nodes.
- Checkpointer and memory store for managing conversation state and persistence.
- Utility function to visualize the compiled agent graph structure.

Example usage (commented out):
- Demonstrates how to invoke the agent with a user question and display the resulting conversation.
"""

from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
from agent_state import State
from .tools import invoice_tools
from .prompts import invoice_subagent_prompt
from memory import in_memory_store
from memory import checkpointer

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)


# Create the invoice information subagent using LangGraph's pre-built ReAct agent
# This agent specializes in handling customer invoice queries and billing information
invoice_information_subagent = create_react_agent(
    llm,  # Language model for reasoning and responses
    tools=invoice_tools,  # Invoice-specific tools for database queries
    name="invoice_information_subagent",  # Unique identifier for the agent
    prompt=invoice_subagent_prompt,  # System instructions for invoice handling
    state_schema=State,  # State schema for data flow between nodes
    checkpointer=checkpointer,  # Short-term memory for conversation context
    store=in_memory_store,  # Long-term memory store for persistent data
)


# ---- Test Agent ----
# import uuid
# from langchain_core.messages import HumanMessage

# # Generate a unique thread ID for this conversation session
# thread_id = uuid.uuid4()

# # Define the user's question about their recent invoice and employee assistance
# question = "My customer id is 1. What was my most recent invoice, and who was the employee that helped me with it?"

# # Set up configuration with the thread ID for maintaining conversation context
# config = {"configurable": {"thread_id": thread_id}}

# # Invoke the invoice information subagent with the user's question
# # The agent will use its tools to search for invoice information and employee details
# result = invoice_information_subagent.invoke({"messages": [HumanMessage(content=question)]}, config=config)

# # Display all messages from the conversation in a formatted way
# for message in result["messages"]:
#     message.pretty_print()
