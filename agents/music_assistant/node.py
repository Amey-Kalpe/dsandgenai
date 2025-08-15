from langchain_core.runnables import RunnableConfig
from langchain_core.messages import SystemMessage
from agents.music_assistant.prompts import generate_music_assistant_prompt
from agent_state import State
from agents.music_assistant.tools import music_tools
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
llm_with_music_tools = llm.bind_tools(music_tools)


def music_assistant(state: State, config: RunnableConfig):
    """
    Music assistant node that handles music catalog queries and recommendations.

    This node processes customer requests related to music discovery, album searches,
    artist information, and personalized recommendations based on stored preferences.

    Args:
        state (State): Current state containing customer_id, messages, loaded_memory, etc.
        config (RunnableConfig): Configuration for the runnable execution

    Returns:
        dict: Updated state with the assistant's response message
    """
    # Retrieve long-term memory preferences if available
    memory = "None"
    if "loaded_memory" in state:
        memory = state["loaded_memory"]

    # Generate instructions for the music assistant agent
    music_assistant_prompt = generate_music_assistant_prompt(memory)

    # Invoke the language model with tools and system prompt
    # The model can decide whether to use tools or respond directly
    response = llm_with_music_tools.invoke(
        [SystemMessage(music_assistant_prompt)] + state["messages"]
    )

    # Return updated state with the assistant's response
    return {"messages": [response]}


def should_continue(state: State, config: RunnableConfig):
    """
    Conditional edge function that determines the next step in the ReAct agent workflow.

    This function examines the last message in the conversation to decide whether the agent
    should continue with tool execution or end the conversation.

    Args:
        state (State): Current state containing messages and other workflow data
        config (RunnableConfig): Configuration for the runnable execution

    Returns:
        str: Either "continue" to execute tools or "end" to finish the workflow
    """
    # Get all messages from the current state
    messages = state["messages"]

    # Examine the most recent message to check for tool calls
    last_message = messages[-1]

    # If the last message doesn't contain any tool calls, the agent is done
    if not last_message.tool_calls:
        return "end"
    # If there are tool calls present, continue to execute them
    else:
        return "continue"
