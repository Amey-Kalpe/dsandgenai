import uuid
from langchain_core.messages import HumanMessage
from langgraph.types import Command
from graph import multi_agent_verify_graph


def main():
    # Without Customer ID
    thread_id = uuid.uuid4()
    question = "How much was my most recent purchase?"
    config = {"configurable": {"thread_id": thread_id}}

    result = multi_agent_verify_graph.invoke(
        {"messages": [HumanMessage(content=question)]}, config=config
    )
    for message in result["messages"]:
        message.pretty_print()

    # Resume from the interrupt, providing the phone number for verification
    question = "My phone number is +55 (12) 3923-555."
    result = multi_agent_verify_graph.invoke(Command(resume=question), config=config)
    for message in result["messages"]:
        message.pretty_print()

    print("*" * 50)
    print("*" * 50)

    # With customer ID
    thread_id = uuid.uuid4()
    question = "How much was my most recent purchase? My customer ID is 1"
    config = {"configurable": {"thread_id": thread_id}}

    result = multi_agent_verify_graph.invoke(
        {"messages": [HumanMessage(content=question)]}, config=config
    )
    for message in result["messages"]:
        message.pretty_print()


if __name__ == "__main__":
    main()
