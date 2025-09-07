import json
from fastmcp import FastMCP, Context
import uvicorn
from agent.graph import graph
from langchain_core.messages import HumanMessage
from langgraph.types import Command

mcp = FastMCP("Langgraph Supervisor MCP")


@mcp.tool(name="calculator")
def get_calculator_response(query: str, ctx: Context) -> str:
    """
    This tool helps perform mathematical operations.
    """
    thread_id = ctx.session_id
    # question = "How much was my most recent purchase?"
    config = {"configurable": {"thread_id": thread_id}}

    result = graph.invoke({"messages": [HumanMessage(content=query)]}, config=config)
    if result.get("__interrupt__", None):
        return json.dumps(result["__interrupt__"][0].value)
    return result["messages"][-1].content


@mcp.tool(name="resume_execution")
def resume_execution(query: str, ctx: Context) -> str:
    "This tool will resume execution after user approval."
    thread_id = ctx.session_id
    # question = "How much was my most recent purchase?"
    config = {"configurable": {"thread_id": thread_id}}
    result = graph.invoke(Command(resume=query), config=config)
    return result["messages"][-1].content


app = mcp.http_app("/mcp")

if __name__ == "__main__":
    uvicorn.run(app=app, port=8000)
