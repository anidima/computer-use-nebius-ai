#!/usr/bin/env python3

"""
Test script to check how Nebius models respond when tools are provided 
but tool_choice is set to None.
"""

import asyncio
import json
import os
from openai import OpenAI

# Simple calculator tool definition for testing
CALCULATOR_TOOL = {
    "type": "function",
    "function": {
        "name": "calculator",
        "description": "Perform basic mathematical calculations",
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "Mathematical expression to evaluate (e.g., '2 + 2', '10 * 5')"
                }
            },
            "required": ["expression"]
        }
    }
}

def test_nebius_tool_choice_none():
    """Test Nebius models with tool_choice=None"""
    
    # Get API key from environment or prompt user
    api_key = os.getenv("NEBIUS_API_KEY")
    if not api_key:
        api_key = input("Enter your Nebius API key: ").strip()
    
    # Initialize OpenAI client for Nebius
    client = OpenAI(
        base_url="https://api.studio.nebius.com/v1/",
        api_key=api_key
    )
    
    # Test models
    models_to_test = [
        "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "meta-llama/Meta-Llama-3.1-70B-Instruct", 
        "Qwen/Qwen2.5-72B-Instruct",
        "mistralai/Mistral-7B-Instruct-v0.3"
    ]
    
    # Test message that could trigger tool usage
    test_message = "Can you calculate 15 * 23 + 47 for me?"
    
    print("🧪 Testing Nebius models with tool_choice=None")
    print("=" * 60)
    
    for model in models_to_test:
        print(f"\n📋 Testing model: {model}")
        print("-" * 40)
        
        try:
            # Make API call with tools but tool_choice=None
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "user", "content": test_message}
                ],
                max_tokens=1000,
                tools=[CALCULATOR_TOOL],
                tool_choice=None  # This is the key parameter we're testing
            )
            
            # Analyze the response
            message = response.choices[0].message
            
            print(f"✅ Model responded successfully")
            print(f"📝 Response content: {message.content}")
            
            if message.tool_calls:
                print(f"🔧 Tool calls made: {len(message.tool_calls)}")
                for i, tool_call in enumerate(message.tool_calls):
                    print(f"   Tool {i+1}: {tool_call.function.name}")
                    print(f"   Arguments: {tool_call.function.arguments}")
            else:
                print("🚫 No tool calls made")
            
            # Check if model provided direct calculation
            if any(word in message.content.lower() for word in ['calculator', 'calculate', '392']):
                print("💡 Model seems aware of calculation context")
            
        except Exception as e:
            print(f"❌ Error with {model}: {str(e)}")
    
    print("\n" + "=" * 60)
    print("🏁 Test completed!")


if __name__ == "__main__":
    test_nebius_tool_choice_none()
