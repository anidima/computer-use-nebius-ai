import json
import os

from pydantic import BaseModel, Field
from typing import Literal
from openai import OpenAI

client = OpenAI(
    base_url="https://api.studio.nebius.com/v1/",
    api_key=os.getenv('NEBIUS_API_KEY')
)

model = "mistralai/Mistral-Small-3.1-24B-Instruct-2503"

class GetCurrentWeatherParams(BaseModel):
    city: str = Field(
        ...,
        description="The city to find the weather for, e.g. 'San Francisco'"
    )
    state: str = Field(
        ...,
        description=(
            "The two-letter abbreviation for the state that the city is in, "
            "e.g. 'CA' for California"
        )
    )
    unit: Literal['celsius', 'fahrenheit'] = Field(
        ...,
        description="The unit to fetch the temperature in"
    )

tools = [{
    "type": "function",
    "function": {
        "name": "get_current_weather",
        "description": "Get the current weather in a given location",
        "parameters": GetCurrentWeatherParams.model_json_schema()
    }
}]

messages = [
    {
        "role": "user",
        "content": "Hi! How are you doing today?"
    },
    {
        "role": "assistant",
        "content": "I'm doing well! How can I help you?"
    },
    {
        "role": "user",
        "content": (
            "Can you tell me what the temperature will be in Dallas, "
            "in Fahrenheit?"
        )
    }
]

chat_completion = client.chat.completions.create(
    messages=messages,
    model=model,
    tools=tools,
    tool_choice={
        "type": "function",
        "function": {
            "name": "get_current_weather"
        }
    }
)

messages.append({
    "role": "assistant",
    "tool_calls": chat_completion.choices[0].message.tool_calls
})

# Simulate a tool call
def get_current_weather(city: str, state: str, unit: 'str'):
    return (
      "The weather in Dallas, Texas is 85 degrees Fahrenheit. "
      "It is partly cloudy, with highs in the 90's."
    )


available_tools = {"get_current_weather": get_current_weather}

completion_tool_calls = chat_completion.choices[0].message.tool_calls
for call in completion_tool_calls:
    tool_to_call = available_tools[call.function.name]
    args = json.loads(call.function.arguments)
    result = tool_to_call(**args)
    print(result)
    messages.append({
        "role": "tool",
        "content": result,
        "tool_call_id": call.id,
        "name": call.function.name
    })


#Messages array with response:
[
  {
    "role": "user",
    "content": "Hi! How are you doing today?"
  },
  {
    "role": "assistant",
    "content": "I'm doing well! How can I help you?"
  },
  {
    "role": "user",
    "content": "Can you tell me what the temperature will be in Dallas, in Fahrenheit?"
  },
  {
    "role": "assistant",
    "tool_calls": [
      {
        "id": "chatcmpl-tool-99a7259c139e4aa986549d07cde8df8f",
        "type": "function",
        "function": {
          "name": "get_current_weather",
          "arguments": "{ \"city\" : \"Dallas\" , \"state\": \"Texas\", \"unit\": \"fahrenheit\" }"
        }
      }
    ]
  },
  {
    "role": "tool",
    "content": "The weather in Dallas, Texas is 85 degrees fahrenheit. It is partly cloudy, with highs in the 90's.",
    "tool_call_id": "chatcmpl-tool-99a7259c139e4aa986549d07cde8df8f",
    "name": "get_current_weather"
  }
]
