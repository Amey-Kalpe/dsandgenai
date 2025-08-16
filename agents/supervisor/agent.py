from langchain_openai import ChatOpenAI
from langgraph_supervisor import create_supervisor
from agents.music_assistant.agent import music_catalog_subagent
from agents.invoice_assistant.agent import invoice_information_subagent
from agents.supervisor.prompts import supervisor_prompt
from memory import in_memory_store
from memory import checkpointer
from agent_state import State

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)

# Create supervisor workflow using LangGraph's pre-built supervisor
# The supervisor coordinates between multiple subagents based on the incoming queries
supervisor_prebuilt_workflow = create_supervisor(
    agents=[
        invoice_information_subagent,
        music_catalog_subagent,
    ],  # List of subagents to supervise
    output_mode="last_message",  # Return only the final response (alternative: "full_history")
    model=llm,  # Language model for supervisor reasoning and routing decisions
    prompt=(supervisor_prompt),  # System instructions for the supervisor agent
    state_schema=State,  # State schema defining data flow structure
)

# Compile the supervisor workflow with memory components
# - checkpointer: Enables short-term memory within conversation threads
# - store: Provides long-term memory storage across conversations
supervisor_prebuilt = supervisor_prebuilt_workflow.compile(
    name="music_catalog_subagent", checkpointer=checkpointer, store=in_memory_store
)

# ---- Test Agent ----
# import uuid
# from langchain_core.messages import HumanMessage

# # Generate a unique thread ID for this conversation session
# thread_id = uuid.uuid4()

# # Define a question that tests both invoice and music catalog capabilities
# question = "My customer ID is 1. How much was my most recent purchase? What albums do you have by U2?"

# # Set up configuration with the thread ID for maintaining conversation context
# config = {"configurable": {"thread_id": thread_id}}

# # Invoke the supervisor workflow with the multi-part question
# # The supervisor will route to appropriate subagents for invoice and music queries
# result = supervisor_prebuilt.invoke(
#     {"messages": [HumanMessage(content=question)]}, config=config
# )

# # Display all messages from the conversation in a formatted way
# for message in result["messages"]:
#     message.pretty_print()
