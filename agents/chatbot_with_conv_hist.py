from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from typing import TypedDict, List, Union
from langgraph.checkpoint.memory import MemorySaver

memory = MemorySaver()

class AgentState(TypedDict):
    messages: List[Union[HumanMessage, AIMessage]]

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)

def process(state: AgentState) -> AgentState:
    """Process the state by invoking the LLM with the messages."""
    response = llm.invoke(state["messages"])
    print(f"Response: {response.content}")
    
    # Append the AI response to the messages
    state["messages"].append(AIMessage(content=response.content))
    
    return state

graph = StateGraph(AgentState)
graph.add_node("process", process)
graph.add_edge(START, "process")
graph.add_edge("process", END)
# Add memory checkpointer when compiling the graph
app = graph.compile(checkpointer=memory)

conversation_history = []

# Create a config with user thread
config = {"configurable": {"thread_id": "1"}}

while True:
    user_input = input("You: ")
    if user_input.lower() in ["exit", "quit"]:
        break
    
    # Create a new state with the user's message
    state = AgentState(messages=conversation_history + [HumanMessage(content=user_input)])
    
    # Run the graph with the current state, and add the config for the user
    state = app.invoke(state, config=config)
    
    # Update conversation history
    conversation_history = state["messages"]
    if len(conversation_history) > 10:
        # Keep the last 10 messages to avoid memory overflow
        conversation_history = conversation_history[-10:]

# Once the chatbpt exits, the conversation history and state gets wiped
# So, save the conversation history to a file
with open("conversation_history.txt", "w") as f:
    for message in conversation_history:
        if isinstance(message, HumanMessage):
            f.write(f"You: {message.content}\n")
        elif isinstance(message, AIMessage):
            f.write(f"AI: {message.content}\n")
    f.write("\nEnd of conversation.\n")

print("Conversation history saved to conversation_history.txt")