from sarvamai import SarvamAI
from typing import Any, List, Optional
from langchain_core.outputs import ChatResult, ChatGeneration
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from pydantic import Field, PrivateAttr

class ChatSarvam(BaseChatModel):
    api_key: str = Field(...)
    temperature: float = 0.5
    top_p: float = 1
    max_tokens: int = 1000

    _client: SarvamAI = PrivateAttr()   # ✅ Private attribute

    def model_post_init(self, __context: Any) -> None:
        self._client = SarvamAI(
            api_subscription_key=self.api_key
        )

    @property
    def _llm_type(self) -> str:
        return "sarvam-chat-model"

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> ChatResult:

        sarvam_messages = []
        for msg in messages:
            if isinstance(msg, HumanMessage):
                role = "user"
            elif isinstance(msg, AIMessage):
                role = "assistant"
            else:
                role = "system"

            sarvam_messages.append({
                "role": role,
                "content": msg.content
            })

        response = self._client.chat.completions(
            messages=sarvam_messages,
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_tokens,
        )

        content = response.choices[0].message.content

        return ChatResult(
            generations=[
                ChatGeneration(
                    message=AIMessage(content=content)
                )
            ]
        )