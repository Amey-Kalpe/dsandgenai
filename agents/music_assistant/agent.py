"""
This module defines the music assistant agent workflow for handling music-related queries within a customer support bot.
It leverages LangGraph's StateGraph to orchestrate reasoning and tool execution nodes, enabling the agent to decide when to
invoke music tools or respond directly to user queries. The workflow supports short-term and long-term memory via a checkpointer
and in-memory store, and is designed for extensibility and integration with other subagents in the system.

Key components:
- ToolNode for music-related tool execution.
- Reasoning node (`music_assistant`) for decision-making.
- Conditional routing based on whether tool invocation is required.
- Support for conversational context via thread IDs.
- Example test code (commented) for agent invocation and message display.
"""

from langgraph.prebuilt import ToolNode
from langgraph.graph import StateGraph, START, END
from .tools import music_tools
from agent_state import State
from .node import music_assistant, should_continue as ma_should_continue
from langsmith import utils
from graph import checkpointer, in_memory_store


music_tool_node = ToolNode(music_tools)

# Create a new StateGraph instance for the music workflow
music_workflow = StateGraph(State)

# Add nodes to the graph
# music_assistant: The reasoning node that decides which tools to invoke or responds directly
music_workflow.add_node("music_assistant", music_assistant)
# music_tool_node: The execution node that handles all music-related tool calls
music_workflow.add_node("music_tool_node", music_tool_node)

# Add edges to define the flow of the graph
# Set the entry point - all queries start with the music assistant
music_workflow.add_edge(START, "music_assistant")

# Add conditional edge from music_assistant based on whether tools need to be called
music_workflow.add_conditional_edges(
    "music_assistant",
    # Conditional function that determines the next step
    ma_should_continue,
    {
        # If tools need to be executed, route to tool node
        "continue": "music_tool_node",
        # If no tools needed, end the workflow
        "end": END,
    },
)

# After tool execution, always return to the music assistant for further processing
music_workflow.add_edge("music_tool_node", "music_assistant")

# Compile the graph with checkpointer for short-term memory and store for long-term memory
music_catalog_subagent = music_workflow.compile(
    name="music_catalog_subagent", 
    checkpointer=checkpointer, 
    store=in_memory_store
)


# ---- Test agent ----
# import uuid
# from langchain_core.messages import HumanMessage

# # Generate a unique thread ID for this conversation session
# thread_id = uuid.uuid4()

# # Define the user's question about music recommendations
# question = "I like the Rolling Stones. What songs do you recommend by them or by other artists that I might like?"

# # Set up configuration with the thread ID for maintaining conversation context
# config = {"configurable": {"thread_id": thread_id}}

# # Invoke the music catalog subagent with the user's question
# # The agent will use its tools to search for Rolling Stones music and provide recommendations
# result = music_catalog_subagent.invoke({"messages": [HumanMessage(content=question)]}, config=config)

# # Display all messages from the conversation in a formatted way
# for message in result["messages"]:
#     message.pretty_print()
