import json
import random
from fastmcp import FastMCP

# Create the FastMCP server instance
mcp = FastMCP(name = "Simple Calculator Server")

@mcp.tool
def add(a: float, b: float) -> float:
    """
    Add two numbers.
    Args:
        a (float): The first number.
        b (float): The second number.
    Returns:
        float: The sum of the two numbers.
    """
    return a + b

# tool: generate a random number
@mcp.tool
def generate_random_number(min_value: int = 1, max_value: int = 100) -> int:
    """
    Generate a random integer between min_value and max_value.
    Args:
        min_value (int): The minimum value (inclusive).
        max_value (int): The maximum value (inclusive).
    Returns:
        int: A random integer between min_value and max_value.
    """
    return random.randint(min_value, max_value)

# Resource: Server information
@mcp.resource("info://server")
def server_info() -> str:
    """
    Get information about this server.
    """
    info = {
        "name": "Simple Calculator Server",
        "version": "1.0",
        "description": "A simple server that provides basic calculator functions.",
        "tools": {
            "add": {
                "description": "Add two numbers.",
                "parameters": {
                    "a": {"type": "float", "description": "The first number."},
                    "b": {"type": "float", "description": "The second number."}
                },
                "returns": {"type": "float", "description": "The sum of the two numbers."}
            },
            "generate_random_number": {
                "description": "Generate a random integer between min_value and max_value.",
                "parameters": {
                    "min_value": {"type": "int", "description": "The minimum value (inclusive)."},
                    "max_value": {"type": "int", "description": "The maximum value (inclusive)."}
                },
                "returns": {"type": "int", "description": "A random integer between min_value and max_value."}
            }
        }
    }
    return json.dumps(info, indent=4)

if __name__ == "__main__":
    mcp.run(transport="http", host="0.0.0.0", port=8000)
