import uuid
import queue
import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langgraph_backend2 import chatbot, retrieve_all_threads, submit_async_task

#################################################################################
######################## ! Utility Functions ! ##################################
#################################################################################

def generate_thread_id() -> str:
    return str(uuid.uuid4())

def reset_chat():
    thread_id = generate_thread_id()

    st.session_state["thread_id"] = thread_id
    add_thread(thread_id)

    st.session_state["message_history"] = []

def add_thread(thread_id: str):
    if thread_id not in st.session_state["chat_threads"]:
        st.session_state["chat_threads"].append(thread_id)

def load_conversation(thread_id: str):
    return chatbot.get_state(config = {"configurable": {"thread_id": thread_id}}).values.get("messages", [])

##################################################################################
########################## ! Session Setup ! #####################################
##################################################################################
# Initialize message history
if "message_history" not in st.session_state:
    st.session_state["message_history"] = []

# Initialize thread ID for conversations
if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = generate_thread_id()

if "chat_threads" not in st.session_state:
    st.session_state["chat_threads"] = list(retrieve_all_threads())

add_thread(st.session_state["thread_id"])

##################################################################################
############################### ! Sidebar UI ! ###################################
##################################################################################

st.sidebar.title("LangGraph Chatbot")

if st.sidebar.button("New Chat"):
    reset_chat()

st.sidebar.header("My Conversations")
for thread_id in st.session_state["chat_threads"][::-1]:
    if st.sidebar.button(str(thread_id)):
        st.session_state["thread_id"] = thread_id
        messages = load_conversation(thread_id)

        temp_messages = []
        for msg in messages:
            role = "user" if isinstance(msg, HumanMessage) else "assistant"
            temp_messages.append({"role": role, "content": msg.content})
        st.session_state["message_history"] = temp_messages


#####################################################################################
################################## ! Main UI Code ! #################################
#####################################################################################

st.title("LangGraph Chatbot with Groq LLM")

# Display all the previous messages
for message in st.session_state["message_history"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_input = st.chat_input("Type your message here...")

# Handle user input and generate a dummy response
if user_input:
    st.session_state["message_history"].append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.markdown(user_input)

    # ! Without Streaming Mode
    # ai_message = chatbot.invoke({"messages": [HumanMessage(content=user_input)]}, config = CONFIG)["messages"][-1].content
    # st.session_state["message_history"].append({"role": "assistant", "content": ai_message})
    # with st.chat_message("assistant"):
    #     st.markdown(ai_message)

    # ! With Streaming Mode
    CONFIG = {
        "configurable": {"thread_id": st.session_state["thread_id"]},
        "metadata": {
            "thread_id": st.session_state["thread_id"]
        },
        "run_name": "chat_turn"
    }

    # ! Below commented code will not work properly because we have added the tools to the graph
    # with st.chat_message("assistant"):
    #     ai_message = st.write_stream(
    #         message_chunk.content for message_chunk, _ in chatbot.stream(
    #             {"messages": [HumanMessage(content=user_input)]},
    #             config = CONFIG,
    #             stream_mode =  "messages"
    #         )
    #     )
    # st.session_state["message_history"].append({"role": "assistant", "content": ai_message})

    # ! Below commented code will use the only AI Streaming using the tools graph but the status of the tool will not be displayed to the UI
    # with st.chat_message("assistant"):
    #     def ai_only_stream():
    #         for message_chunk, metadata in chatbot.stream(
    #             {"messages": [HumanMessage(content=user_input)]},
    #             config=CONFIG,
    #             stream_mode="messages"
    #         ):
    #             if isinstance(message_chunk, AIMessage):
    #                 # yield only assistant tokens
    #                 yield message_chunk.content

    #     ai_message = st.write_stream(ai_only_stream())

    # st.session_state["message_history"].append({"role": "assistant", "content": ai_message})

    # ! Below code will use the streaming of the tool calling graph as well as will display which tool is used to the UI
    # Assistant streaming block
    with st.chat_message("assistant"):
        # Use a mutable holder so the generator can set/modify it
        status_holder = {"box": None}

        def ai_only_stream():
            # for message_chunk, metadata in chatbot.astream(
            #     {"messages": [HumanMessage(content=user_input)]},
            #     config=CONFIG,
            #     stream_mode="messages",
            # ):
            #     # Lazily create & update the SAME status container when any tool runs
            #     if isinstance(message_chunk, ToolMessage):
            #         tool_name = getattr(message_chunk, "name", "tool")
            #         if status_holder["box"] is None:
            #             status_holder["box"] = st.status(
            #                 f"🔧 Using `{tool_name}` …", expanded=True
            #             )
            #         else:
            #             status_holder["box"].update(
            #                 label=f"🔧 Using `{tool_name}` …",
            #                 state="running",
            #                 expanded=True,
            #             )

            #     # Stream ONLY assistant tokens
            #     if isinstance(message_chunk, AIMessage):
            #         yield message_chunk.content

            event_queue: queue.Queue = queue.Queue()

            async def run_stream():
                try:
                    async for message_chunk, metadata in chatbot.astream(
                        {"messages": [HumanMessage(content=user_input)]},
                        config=CONFIG,
                        stream_mode="messages",
                    ):
                        event_queue.put((message_chunk, metadata))
                except Exception as e:
                    event_queue.put(f"error: {e}")
                finally:
                    event_queue.put(None)

            submit_async_task(run_stream())

            while True:
                item = event_queue.get()
                if item is None:
                    break

                if isinstance(item, str) and item.startswith("error:"):
                    raise Exception(item)
                
                message_chunk, metadata = item

                # Lazily create & update the SAME status container when any tool runs
                if isinstance(message_chunk, ToolMessage):
                    tool_name = getattr(message_chunk, "name", "tool")
                    if status_holder["box"] is None:
                        status_holder["box"] = st.status(
                            f"🔧 Using `{tool_name}` …", expanded=True
                        )
                    else:
                        status_holder["box"].update(
                            label=f"🔧 Using `{tool_name}` …",
                            state="running",
                            expanded=True,
                        )

                # Stream ONLY assistant tokens
                if isinstance(message_chunk, AIMessage) and message_chunk.content:
                    yield message_chunk.content

        ai_message = st.write_stream(ai_only_stream())

        # Finalize only if a tool was actually used
        if status_holder["box"] is not None:
            status_holder["box"].update(
                label="✅ Tool finished", state="complete", expanded=False
            )

    # Save assistant message
    st.session_state["message_history"].append(
        {"role": "assistant", "content": ai_message}
    )