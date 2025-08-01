#!/usr/bin/env python3

"""
Additional test to verify tool_choice="none" support in Qwen models
"""

import sys
import os
import asyncio
from openai import OpenAI

def setup_client():
    """Setup Nebius OpenAI client"""
    api_key = os.getenv("NEBIUS_API_KEY")
    if not api_key:
        print("⚠️  NEBIUS_API_KEY not set")
        return None
    
    return OpenAI(
        base_url="https://api.studio.nebius.com/v1/",
        api_key=api_key
    )

async def test_tool_choice_none_with_tools():
    """Test tool_choice='none' with tools provided"""
    print("🔍 Testing tool_choice='none' with tools provided")
    
    client = setup_client()
    if not client:
        return
    
    # Simple calculator tool
    tools = [{
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Perform basic arithmetic calculations",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Mathematical expression to evaluate"
                    }
                },
                "required": ["expression"]
            }
        }
    }]
    
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is 2 + 3? Please answer without using tools."}
    ]
    
    models_to_test = [
        "Qwen/Qwen2-VL-72B-Instruct",
        "Qwen/Qwen2.5-VL-72B-Instruct"
    ]
    
    for model in models_to_test:
        print(f"\n--- Testing {model} ---")
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                tools=tools,
                tool_choice="none",  # Explicitly don't use tools
                max_tokens=512
            )
            
            choice = response.choices[0].message
            print(f"✅ Response with tool_choice='none':")
            print(f"   Content: {choice.content}")
            if choice.tool_calls:
                print(f"   ⚠️  Unexpected tool calls: {len(choice.tool_calls)}")
            else:
                print(f"   ✅ No tool calls (as expected)")
            
        except Exception as e:
            print(f"❌ Error with tool_choice='none': {e}")

async def test_simple_auto():
    """Test simple auto with basic tools"""
    print("\n🔍 Testing simple tool_choice='auto' scenario")
    
    client = setup_client()
    if not client:
        return
    
    # Very simple tool
    tools = [{
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "City name"}
                },
                "required": ["location"]
            }
        }
    }]
    
    messages = [
        {"role": "user", "content": "What's the weather like in Paris?"}
    ]
    
    models_to_test = [
        "Qwen/Qwen2-VL-72B-Instruct",
        "Qwen/Qwen2.5-VL-72B-Instruct"
    ]
    
    for model in models_to_test:
        print(f"\n--- Testing {model} with simple auto ---")
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                tools=tools,
                tool_choice="auto",
                max_tokens=512
            )
            
            choice = response.choices[0].message
            print(f"✅ Response with tool_choice='auto':")
            if choice.content:
                print(f"   Content: {choice.content[:200]}...")
            if choice.tool_calls:
                print(f"   Tool calls: {len(choice.tool_calls)}")
                for tc in choice.tool_calls:
                    print(f"   - {tc.function.name}: {tc.function.arguments}")
            
        except Exception as e:
            print(f"❌ Error with tool_choice='auto': {e}")

if __name__ == "__main__":
    asyncio.run(test_tool_choice_none_with_tools())
    asyncio.run(test_simple_auto())
