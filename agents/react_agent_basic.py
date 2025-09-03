from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain.agents import initialize_agent
from langchain_community.tools import TavilySearchResults
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)

search_tool = TavilySearchResults()

agent = initialize_agent(
    tools=[search_tool], llm=llm, agent="zero-shot-react-description", verbose=True
)

print(agent.invoke(HumanMessage("Give me a funny tweet today about Pune?")))
