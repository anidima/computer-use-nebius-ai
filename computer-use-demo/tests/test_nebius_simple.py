#!/usr/bin/env python3

"""
Simple test to understand Nebius Mistral model tool behavior
"""

import sys
import os
from openai import OpenAI

# Add path to import computer_use_demo modules
sys.path.append(os.path.join(os.path.dirname(__file__), 'computer_use_demo'))

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

def test_without_tools():
    """Test simple conversation without tools"""
    print("🔍 Testing without tools")
    
    client = setup_client()
    if not client:
        return
    
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello! Can you help me understand how to use tools?"}
    ]
    
    try:
        response = client.chat.completions.create(
            model="mistralai/Mistral-Small-3.1-24B-Instruct-2503",
            messages=messages,
            max_tokens=512
        )
        
        choice = response.choices[0].message
        print(f"✅ Response without tools:")
        print(f"   Content: {choice.content}")
        
    except Exception as e:
        print(f"❌ Error without tools: {e}")

def test_with_tools_no_choice():
    """Test with tools but no tool_choice parameter"""
    print("\n🔍 Testing with tools, no tool_choice parameter")
    
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
        {"role": "system", "content": "You are a calculator assistant."},
        {"role": "user", "content": "What is 2 + 3?"}
    ]
    
    try:
        response = client.chat.completions.create(
            model="mistralai/Mistral-Small-3.1-24B-Instruct-2503",
            messages=messages,
            tools=tools,
            max_tokens=512
        )
        
        choice = response.choices[0].message
        print(f"✅ Response with tools (no tool_choice):")
        if choice.content:
            print(f"   Content: {choice.content}")
        if choice.tool_calls:
            print(f"   Tool calls: {len(choice.tool_calls)}")
            for tc in choice.tool_calls:
                print(f"   - {tc.function.name}: {tc.function.arguments}")
        
    except Exception as e:
        print(f"❌ Error with tools (no tool_choice): {e}")

def test_with_specific_tool():
    """Test with specific tool choice"""
    print("\n🔍 Testing with specific tool choice")
    
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
        {"role": "system", "content": "You are a calculator assistant."},
        {"role": "user", "content": "What is 2 + 3?"}
    ]
    
    try:
        response = client.chat.completions.create(
            model="mistralai/Mistral-Small-3.1-24B-Instruct-2503",
            messages=messages,
            tools=tools,
            tool_choice={
                "type": "function",
                "function": {"name": "calculate"}
            },
            max_tokens=512
        )
        
        choice = response.choices[0].message
        print(f"✅ Response with specific tool choice:")
        if choice.content:
            print(f"   Content: {choice.content}")
        if choice.tool_calls:
            print(f"   Tool calls: {len(choice.tool_calls)}")
            for tc in choice.tool_calls:
                print(f"   - {tc.function.name}: {tc.function.arguments}")
        
    except Exception as e:
        print(f"❌ Error with specific tool choice: {e}")

if __name__ == "__main__":
    print("🚀 Nebius Mistral Tool Behavior Analysis")
    print("=" * 50)
    
    test_without_tools()
    test_with_tools_no_choice()
    test_with_specific_tool()
    
    print("\n✨ Analysis completed!")
