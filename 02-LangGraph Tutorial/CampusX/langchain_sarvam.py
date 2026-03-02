import os
from typing import Any, List, Optional, Dict
from sarvamai import SarvamAI
from pydantic import Field, PrivateAttr, BaseModel
from langchain_core.outputs import ChatResult, ChatGeneration
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.utils.function_calling import convert_to_openai_tool

class ChatSarvam(BaseChatModel):
    api_key: str = Field(...)
    temperature: float = 0.5
    top_p: float = 1
    max_tokens: int = 1000
    model_name: str = "sarvam-2b" # Or your specific Sarvam model
    _client: SarvamAI = PrivateAttr()

    def model_post_init(self, __context: Any) -> None:
        self._client = SarvamAI(api_subscription_key=self.api_key)

    @property
    def _llm_type(self) -> str:
        return "sarvam-chat-model"

    def bind_tools(
        self,
        tools: List[Any],
        tool_choice: Optional[str] = None,
        **kwargs: Any,
    ) -> "ChatSarvam":
        """This allows .with_structured_output to function."""
        formatted_tools = [convert_to_openai_tool(tool) for tool in tools]
        return self.bind(tools=formatted_tools, tool_choice=tool_choice, **kwargs)

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> ChatResult:
        # 1. Convert LangChain messages to Sarvam format
        sarvam_messages = []
        for msg in messages:
            if isinstance(msg, HumanMessage):
                role = "user"
            elif isinstance(msg, AIMessage):
                role = "assistant"
            elif isinstance(msg, SystemMessage):
                role = "system"
            else:
                role = "user"
            sarvam_messages.append({"role": role, "content": msg.content})

        # 2. Extract tools and tool_choice from kwargs (injected by bind_tools)
        tools = kwargs.get("tools")
        tool_choice = kwargs.get("tool_choice")

        # 3. Call the Sarvam API
        # Note: We pass through tool-related arguments if they exist
        response = self._client.chat.completions(
            model=self.model_name,
            messages=sarvam_messages,
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_tokens,
            tools=tools,
            tool_choice=tool_choice,
        )

        content = response.choices[0].message.content
        
        # 4. Handle Tool Calls in the response (Necessary for Structured Output)
        # LangChain's structured output parser looks at the tool_calls field
        tool_calls = []
        if hasattr(response.choices[0].message, "tool_calls") and response.choices[0].message.tool_calls:
            for tc in response.choices[0].message.tool_calls:
                tool_calls.append({
                    "name": tc.function.name,
                    "args": tc.function.arguments, # This should be a JSON string or dict
                    "id": tc.id
                })

        message = AIMessage(content=content if content else "", tool_calls=tool_calls)

        return ChatResult(
            generations=[ChatGeneration(message=message)]
        )