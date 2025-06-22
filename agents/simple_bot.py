from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from typing import TypedDict, List

class AgentState(TypedDict):
    messages: List[HumanMessage]

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)

def process(state: AgentState) -> AgentState:
    """Process the state by invoking the LLM with the messages."""
    response = llm.invoke(state["messages"])
    print(f"Response: {response.content}")
    
    return state

graph = StateGraph(AgentState)
graph.add_node("process", process)
graph.add_edge(START, "process")
graph.add_edge("process", END)

app = graph.compile()

while True:
    user_input = input("You: ")
    if user_input.lower() in ["exit", "quit"]:
        break
    
    # Create a new state with the user's message
    state = AgentState(messages=[HumanMessage(content=user_input)])
    
    # Run the graph with the current state
    app.invoke(state)