import uuid
from langchain_core.messages import HumanMessage
from langgraph.types import Command
from graph import multi_agent_verify_graph
from memory import in_memory_store


def main():
    # HITL
    # Without Customer ID
    # thread_id = uuid.uuid4()
    # question = "How much was my most recent purchase?"
    # config = {"configurable": {"thread_id": thread_id}}

    # result = multi_agent_verify_graph.invoke(
    #     {"messages": [HumanMessage(content=question)]}, config=config
    # )
    # for message in result["messages"]:
    #     message.pretty_print()

    # # Resume from the interrupt, providing the phone number for verification
    # question = "My phone number is +55 (12) 3923-555."
    # result = multi_agent_verify_graph.invoke(Command(resume=question), config=config)
    # for message in result["messages"]:
    #     message.pretty_print()

    # No HITL
    # With customer ID
    # thread_id = uuid.uuid4()
    # question = "How much was my most recent purchase? My customer ID is 1"
    # config = {"configurable": {"thread_id": thread_id}}

    # result = multi_agent_verify_graph.invoke(
    #     {"messages": [HumanMessage(content=question)]}, config=config
    # )
    # for message in result["messages"]:
    #     message.pretty_print()

    # No HITL
    # With customer info
    thread_id = uuid.uuid4()

    question = "My phone number is +55 (12) 3923-5555. How much was my most recent purchase? What albums do you have by the Rolling Stones?"
    config = {"configurable": {"thread_id": thread_id}}

    result = multi_agent_verify_graph.invoke(
        {"messages": [HumanMessage(content=question)]}, config=config
    )
    for message in result["messages"]:
        message.pretty_print()

    user_id = "1"  # Assuming customer ID 1 was used in the previous interaction
    namespace = ("memory_profile", user_id)
    memory = in_memory_store.get(namespace, "user_memory")

    # Access the UserProfile object stored under the "memory" key
    saved_music_preferences = memory.value.get("memory").music_preferences

    print("\nSaved music preferences for user ID 1")
    print(saved_music_preferences)

    ### OUTPUT ###
    # ["Rolling Stones"]


if __name__ == "__main__":
    main()
