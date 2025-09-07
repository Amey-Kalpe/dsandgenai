---
applyTo: '**'
---
When a user asks to perform some mathematical operation, use the tools provided by the `calulcator` mcp server.
The `calulcator` mcp server runs langgraph behind the scenes and supports human-in-the-loop.
If you receive a message that indicates getting user approval or information from the user, you must get approval from the user mentioning the mcp server wants to perform the tool call.
Then use the `resume_execution` tool to resume execution, until the final response for the query is not received.