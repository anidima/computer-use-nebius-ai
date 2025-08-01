# computer_use_demo/adapters.py
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, List, Dict, Literal, Union

# --- 1. Внутренние (нейтральные) структуры данных ---
# Эти классы не зависят ни от Anthropic, ни от OpenAI

@dataclass
class InternalToolCall:
    id: str
    name: str
    input: Dict[str, Any]

@dataclass
class InternalToolResult:
    tool_call_id: str
    content: str
    # Nebius/OpenAI не поддерживает картинки в tool results,
    # так что это поле будет использоваться только для Anthropic.
    base64_image: str | None = None 
    is_error: bool = False

@dataclass
class InternalMessage:
    role: Literal["user", "assistant", "tool"]
    content: Union[str, List[Union[InternalToolCall, InternalToolResult, Dict[str, Any]]]]

# --- 2. Абстрактный класс Адаптера ---

class BaseAdapter(ABC):
    @abstractmethod
    def to_api_messages(self, messages: List[InternalMessage]) -> Any:
        """Преобразует внутренний формат сообщений в формат API."""
        pass

    @abstractmethod
    def from_api_response(self, response: Any) -> InternalMessage:
        """Преобразует ответ от API во внутренний формат."""
        pass
    
    @abstractmethod
    def to_api_tools(self, tool_collection: Any) -> Any:
        """Преобразует описание инструментов в формат API."""
        pass

# --- 3. Реализация для Nebius/OpenAI ---

class NebiusOpenAIAdapter(BaseAdapter):
    def to_api_messages(self, messages: List[InternalMessage]) -> List[Dict[str, Any]]:
        api_messages = []
        for msg in messages:
            if msg.role == "assistant" and isinstance(msg.content, list):
                # Это ответ модели с вызовом инструментов
                tool_calls = [
                    {
                        "id": tool.id,
                        "type": "function",
                        "function": {"name": tool.name, "arguments": json.dumps(tool.input)}
                    }
                    for tool in msg.content if isinstance(tool, InternalToolCall)
                ]
                api_messages.append({"role": "assistant", "tool_calls": tool_calls})
            elif msg.role == "tool" and isinstance(msg.content, list):
                 # Это результат работы инструмента
                 for tool_result in msg.content:
                     if isinstance(tool_result, InternalToolResult):
                        api_messages.append({
                            "role": "tool",
                            "tool_call_id": tool_result.tool_call_id,
                            "name": "имя_инструмента_здесь", # OpenAI требует имя, его нужно будет где-то хранить
                            "content": tool_result.content
                        })
            else:
                # Обычное текстовое сообщение
                api_messages.append({"role": msg.role, "content": msg.content})
        return api_messages

    def from_api_response(self, response: Any) -> InternalMessage:
        choice = response.choices[0].message
        if choice.tool_calls:
            tool_calls = [
                InternalToolCall(
                    id=call.id,
                    name=call.function.name,
                    input=json.loads(call.function.arguments)
                ) for call in choice.tool_calls
            ]
            return InternalMessage(role="assistant", content=tool_calls)
        else:
            return InternalMessage(role="assistant", content=choice.content)

    def to_api_tools(self, tool_collection: Any) -> Any:
        # tool_collection - это ваш текущий ToolCollection
        # Нужно преобразовать его в формат OpenAI
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters # Это уже JSON Schema, что идеально подходит
                }
            }
            for tool in tool_collection.to_params() # to_params() возвращает список словарей
        ]
