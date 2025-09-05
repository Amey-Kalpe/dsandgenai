from langchain_core.tools import tool


@tool
def addition(a: int, b: int) -> int:
    """Adds two integer numbers."""
    return a + b


@tool
def subtraction(a: int, b: int) -> int:
    """Subtracts two integer numbers."""
    return a - b


@tool
def multiplication(a: int, b: int) -> int:
    """Multiplies two integer numbers."""
    return a * b


tools = [addition, subtraction, multiplication]
