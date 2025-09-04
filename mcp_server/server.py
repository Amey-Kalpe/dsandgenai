from fastmcp import FastMCP, Context
import uvicorn
from graph import multi_agent_verify_graph
from langchain_core.messages import HumanMessage

mcp = FastMCP("Langgraph Supervisor MCP")


@mcp.tool(name="Customer Assistant")
def get_customer_assistant_response(query: str, ctx: Context) -> str:
    """
    This tool helps the user with querying the database with their
    purchase details or invoice info. It can also answer
    """
    thread_id = ctx.session_id
    # question = "How much was my most recent purchase?"
    config = {"configurable": {"thread_id": thread_id}}

    result = multi_agent_verify_graph.invoke(
        {"messages": [HumanMessage(content=query)]}, config=config
    )
    return result["messages"][-1].content


app = mcp.streamable_http_app("/mcp")

if __name__ == "__main__":
    uvicorn.run(app=app, port=8000)
