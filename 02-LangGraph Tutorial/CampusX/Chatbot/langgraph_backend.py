import os
import asyncio
import sqlite3
import requests
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver
# from langgraph.checkpoint.memory import InMemorySaver
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

load_dotenv()

# Define the LLM
grow_api_key = os.getenv("GROQ_API_KEY", None)
llm = ChatGroq(
    api_key = grow_api_key,
    model = "openai/gpt-oss-20b"
)

# Define the tools
search_tool = DuckDuckGoSearchRun()

@tool
def calculator(first_num: float, second_num: float, operation: str) -> dict:
    """
    Perform a basic arithmetic operation on two numbers.

    Args:
        first_num (float): The first number.
        second_num (float): The second number.
        operation (str): The operation to perform. Must be one of 'add', 'sub', 'mul', and 'div'.

    Returns:
        dict: A dictionary containing the result of the operation.
    """
    try:
        if operation == "add":
            result = first_num + second_num
        elif operation == "sub":
            result = first_num - second_num
        elif operation == "mul":
            result = first_num * second_num
        elif operation == "div":
            if second_num == 0:
                return {"error": "Division by zero is not allowed"}
            result = first_num / second_num
        else:
            return {"error": f"Unsupported operation `{operation}`"}
        
        return {"first_num": first_num, "second_num": second_num, "operation": operation, "result": result}
    except Exception as e:
        return {"error": str(e)}
    

@tool
def get_stock_price(symbol: str) -> dict:
    """
    Fetch latest stock price for a given symbol (e.g., 'AAPL', 'TSLA')
    using Alpha Vantage with API key in the URL.
    """
    url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey={os.getenv('ALPHAVANTAGE_API_KEY')}"
    response = requests.get(url)
    return response.json()


SERVERS = {
    "github": {
        "transport": "stdio",
        "command": "/usr/bin/python3",
        "args": [
            "/path/to/github_mcp_server.py"
        ]
    }
}

# make tool list
tools = [get_stock_price, search_tool, calculator]
# Make the LLM tool-aware
llm_with_tools = llm.bind_tools(tools)

# Define the state of the LangGraph Chatbot
class ChatState(TypedDict): 
    messages: Annotated[list[BaseMessage], add_messages]

conn = sqlite3.connect("chatbot.db", check_same_thread=False)
checkpointer = SqliteSaver(conn = conn, serde=JsonPlusSerializer())

async def build_graph(checkpointer):
    async def chat_node(state: ChatState):
        messages = state['messages']
        response = await llm.ainvoke(messages)
        return {"messages": [response]}

    tool_node = ToolNode(tools)

    chatbot = StateGraph(ChatState)
    chatbot.add_node("chat_node", chat_node)
    chatbot.add_node("tools", tool_node)

    chatbot.add_edge(START, "chat_node")
    chatbot.add_conditional_edges("chat_node", tools_condition)
    chatbot.add_edge("tools", "chat_node")
    chatbot.add_edge("chat_node", END)

    chatbot.compile(checkpointer=checkpointer)
    return chatbot

async def main():
    chatbot = await build_graph(checkpointer)
    question = "Find the modulus of 132354 and 23 and give answer like a cricket commentator."
    result = chatbot.ainvoke({"messages": [HumanMessage(content=question)]}, config = {"configurable": {"thread_id": "thread-01"}})
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


#### ! Enable the Streaming Mode Example Below ! ####
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