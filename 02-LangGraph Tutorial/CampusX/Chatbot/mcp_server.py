from __future__ import annotations
from fastmcp import FastMCP


mcp = FastMCP("arithmetic_calculator")

def _as_number(x):
    # Accept ints/floats or numeric strings; raise clean errors otherwise
    if isinstance(x, (int, float)):
        return x
    try:
        return float(x)
    except (ValueError, TypeError):
        raise ValueError(f"Expected a number (int/float or numeric string), got {x!r}")

@mcp.tool()
async def add(a: float, b: float) -> float:
    """Add two numbers"""
    return _as_number(a) + _as_number(b)

@mcp.tool()
async def sub(a: float, b: float) -> float:
    """Subtract two numbers"""
    return _as_number(a) - _as_number(b)

@mcp.tool()
async def mul(a: float, b: float) -> float:
    """Multiply two numbers"""
    return _as_number(a) * _as_number(b)

@mcp.tool()
async def div(a: float, b: float) -> float:
    """Divide two numbers"""
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return _as_number(a) / _as_number(b)

@mcp.tool()
async def power(base: float, exponent: float) -> float:
    """Raise a base to an exponent"""
    return _as_number(base) ** _as_number(exponent)

@mcp.tool()
async def modulus(a: float, b: float) -> float:
    """Return the remainder of a divided by b"""
    return _as_number(a) % _as_number(b)

if __name__ == "__main__":
    mcp.run()
