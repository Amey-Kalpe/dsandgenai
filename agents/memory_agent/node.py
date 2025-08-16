from pydantic import BaseModel, Field
from typing import List
from langchain_core.messages import SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.store.base import BaseStore
from agent_state import State
from .prompts import create_memory_prompt
from langchain_openai.chat_models import ChatOpenAI
from utils.format_user_memory import format_user_memory
from agent_state import State
from langchain_core.runnables import RunnableConfig
from langgraph.store.base import BaseStore

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)


# Pydantic model to define the structure of the user profile for memory storage
class UserProfile(BaseModel):
    customer_id: str = Field(description="The customer ID of the customer")
    music_preferences: List[str] = Field(
        description="The music preferences of the customer"
    )


# Node: create_memory
def create_memory(state: State, config: RunnableConfig, store: BaseStore):
    """
    Analyzes conversation history and updates the user's long-term memory profile.

    This node extracts new music preferences shared by the customer during the
    conversation and persists them in the InMemoryStore for future interactions.
    """
    # Get the user_id from the configurable part of the config or from the state
    user_id = str(config["configurable"].get("user_id", state["customer_id"]))

    # Define the namespace and key for the memory profile
    namespace = ("memory_profile", user_id)
    key = "user_memory"

    # Retrieve the existing memory profile for the user
    existing_memory = store.get(namespace, key)

    # Format the existing memory for the LLM prompt
    formatted_memory = ""
    if existing_memory and existing_memory.value:
        existing_memory_dict = existing_memory.value
        # Ensure 'music_preferences' is treated as a list, even if it might be missing or None
        music_prefs = existing_memory_dict.get("music_preferences", [])
        if music_prefs:
            formatted_memory = f"Music Preferences: {', '.join(music_prefs)}"

    # Prepare the system message for the LLM to update memory
    formatted_system_message = SystemMessage(
        content=create_memory_prompt.format(
            conversation=state["messages"], memory_profile=formatted_memory
        )
    )

    # Invoke the LLM with the UserProfile schema to get structured updated memory
    updated_memory = llm.with_structured_output(UserProfile).invoke(
        [formatted_system_message]
    )

    # Store the updated memory profile
    store.put(namespace, key, {"memory": updated_memory})


# Node: load_memory
def load_memory(state: State, config: RunnableConfig, store: BaseStore):
    """
    Loads music preferences from the long-term memory store for a given user.

    This node fetches previously saved user preferences to provide context
    for the current conversation, enabling personalized responses.
    """
    # Get the user_id from the configurable part of the config
    # In our evaluation setup, we might pass user_id via config
    user_id = config["configurable"].get(
        "user_id", state["customer_id"]
    )  # Use customer_id if user_id not in config

    # Define the namespace and key for accessing memory in the store
    namespace = ("memory_profile", user_id)
    key = "user_memory"

    # Retrieve existing memory for the user
    existing_memory = store.get(namespace, key)
    formatted_memory = ""

    # Format the retrieved memory if it exists and has content
    if existing_memory and existing_memory.value:
        formatted_memory = format_user_memory(existing_memory.value)

    # Update the state with the loaded and formatted memory
    return {"loaded_memory": formatted_memory}
