import os
import asyncio
import sqlite3
import requests
import threading
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.tools import tool, BaseTool
from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver
# from langgraph.checkpoint.memory import InMemorySaver
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

load_dotenv()

# Dedicated async loop for backend tasks
_ASYNC_LOOP   = asyncio.new_event_loop()
_ASYNC_THREAD = threading.Thread(target = _ASYNC_LOOP.run_forever, daemon = True)
_ASYNC_THREAD.start()

def _submit_async(coro):
    return asyncio.run_coroutine_threadsafe(coro, _ASYNC_LOOP)

def run_async(coro):
    return _submit_async(coro).result()

def submit_async_task(coro):
    """
    Submit an async task to the dedicated event loop.
    """
    return _submit_async(coro)

# Define the LLM
grow_api_key = os.getenv("GROQ_API_KEY", None)
llm = ChatGroq(api_key = grow_api_key, model = "openai/gpt-oss-20b")

################################################################################
# Add tools to our LangGraph
################################################################################

# Define the tools
search_tool = DuckDuckGoSearchRun(region = "us-en")

# @tool
# def calculator(first_num: float, second_num: float, operation: str) -> dict:
#     """
#     Perform a basic arithmetic operation on two numbers.

#     Args:
#         first_num (float): The first number.
#         second_num (float): The second number.
#         operation (str): The operation to perform. Must be one of 'add', 'sub', 'mul', and 'div'.

#     Returns:
#         dict: A dictionary containing the result of the operation.
#     """
#     try:
#         if operation == "add":
#             result = first_num + second_num
#         elif operation == "sub":
#             result = first_num - second_num
#         elif operation == "mul":
#             result = first_num * second_num
#         elif operation == "div":
#             if second_num == 0:
#                 return {"error": "Division by zero is not allowed"}
#             result = first_num / second_num
#         else:
#             return {"error": f"Unsupported operation `{operation}`"}
        
#         return {"first_num": first_num, "second_num": second_num, "operation": operation, "result": result}
#     except Exception as e:
#         return {"error": str(e)}


@tool
def get_stock_price(symbol: str) -> dict:
    """
    Fetch latest stock price for a given symbol (e.g., 'AAPL', 'TSLA')
    using Alpha Vantage with API key in the URL.
    """
    url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey={os.getenv('ALPHAVANTAGE_API_KEY')}"
    response = requests.get(url)
    return response.json()

# @tool
# def list_github_prs(owner: str, repo: str, state: str = "open", per_page: int = 3) -> dict:
#     """
#     List GitHub pull requests for a given repository.

#     Args:
#         owner (str): The owner of the repository.
#         repo (str): The name of the repository.
#         state (str, optional): The state of the pull requests to list. Defaults to "open".
#         per_page (int, optional): The number of pull requests to list per page. Defaults to 3.

#     Returns:
#         dict: A dictionary containing the list of pull requests.
#     """
#     url = f"https://api.github.com/repos/{owner}/{repo}/pulls?state={state}&per_page={per_page}"
#     response = requests.get(url)
#     return response.json()

# make tool list
# tools = [get_stock_price, search_tool, calculator, list_github_prs]
# tools = [get_stock_price, search_tool, calculator]

# Make the LLM tool-aware
# llm_with_tools = llm.bind_tools(tools)

# MCP client for local FastMCP server
client = MultiServerMCPClient({
    "arithmetic_calculator": {
        "transport": "stdio",
        "command": "uv",
        "args": ["run", "mcp_server.py"]
    },
    # You can add multiple MCP servers
    "expense": {
        "transport": "streamable_http",
        "url": "https://splendid-gold-dingo.fastmcp.app/mcp"
    }
})

def load_mcp_tools() -> list[BaseTool]:
    try:
        return run_async(client.get_tools())
    except Exception as e:
        print(f"Error loading MCP tools: {e}")
        return []

# bind all the tools to our LLM
mcp_tools = load_mcp_tools()
tools = [search_tool, get_stock_price, *mcp_tools]
llm_with_tools = llm.bind_tools(tools) if tools else llm

################################################################################
# Define the State for the LangGraph
################################################################################

# Define the state of the LangGraph Chatbot
class ChatState(TypedDict): 
    messages: Annotated[list[BaseMessage], add_messages]


################################################################################
# Define the graph using LangGraph
################################################################################

conn         = sqlite3.connect("chatbot.db", check_same_thread=False)
checkpointer = SqliteSaver(conn = conn, serde = JsonPlusSerializer())

# Define the chat node
async def chat_node(state: ChatState):
    messages = state['messages']
    response = await llm_with_tools.ainvoke(messages)
    return {"messages": [response]}

# tools node
tool_node = ToolNode(tools)

# Initialize the graph
graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_edge(START, "chat_node")

if tool_node:
    graph.add_node("tools", tool_node)
    graph.add_conditional_edges("chat_node", tools_condition)
    graph.add_edge("tools", "chat_node")
else:
    graph.add_edge("chat_node", END)

chatbot = graph.compile(checkpointer=checkpointer)

# helpers

async def main():
    question = "Find the modulus of 132354 and 23 and give answer like a cricket commentator."
    result = await chatbot.ainvoke({"messages": [HumanMessage(content=question)]}, config = {"configurable": {"thread_id": "thread-01"}})
    print(result["messages"][-1].content)

# checkpointer = InMemorySaver()
# conn = sqlite3.connect("chatbot.db", check_same_thread=False)
# checkpointer = SqliteSaver(conn = conn, serde=JsonPlusSerializer())

# graph = StateGraph(ChatState)

# # define the nodes of the graph
# graph.add_node("chat_node", chat_node)
# graph.add_node("tools", tool_node)

# # define the edges of the graph
# graph.add_edge(START, "chat_node")
# graph.add_conditional_edges("chat_node", tools_condition)
# graph.add_edge("tools", "chat_node")

# chatbot = graph.compile(checkpointer=checkpointer)


#### ? Enable the Streaming Mode Example Below ! ####
# CONFIG = {"configurable": {"thread_id": "thread-01"}}
# for message_chunk, metadata in chatbot.stream(
#     {"messages": [HumanMessage(content="What is the recipe to make pasta")]},
#     config = CONFIG,
#     stream_mode =  "messages"
# ):
#     if message_chunk.content:
#         print(message_chunk.content, end=" ", flush=True)

# extract all the threads
def retrieve_all_threads():
    all_threads = set()
    for checkpoint in checkpointer.list(None):
        all_threads.add(checkpoint.config["configurable"]["thread_id"])
    return list(all_threads)


# running the graph
if __name__ == "__main__":
    asyncio.run(main())