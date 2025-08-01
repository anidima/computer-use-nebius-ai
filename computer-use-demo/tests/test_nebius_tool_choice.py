#!/usr/bin/env python3

"""
Test script to verify how Nebius provider handles different tool_choice values
with multiple tools available.
"""

import sys
import os
import json
import asyncio
from typing import Any, Dict, List

# Add path to import computer_use_demo modules
sys.path.append(os.path.join(os.path.dirname(__file__), 'computer_use_demo'))

from openai import OpenAI
from computer_use_demo.loop import (
    _convert_anthropic_tools_to_openai,
    _convert_openai_response_to_anthropic,
    APIProvider
)
from computer_use_demo.tools import ToolCollection, TOOL_GROUPS_BY_VERSION

def setup_nebius_client():
    """Setup Nebius OpenAI client"""
    # You need to set your API key here or in environment
    api_key = os.getenv("NEBIUS_API_KEY")
    if not api_key:
        print("⚠️  Warning: NEBIUS_API_KEY not set. Using placeholder.")
        api_key = "your-api-key-here"
    
    return OpenAI(
        base_url="https://api.studio.nebius.com/v1/",
        api_key=api_key
    )

# Models to test
MODELS_TO_TEST = [
    "mistralai/Mistral-Small-3.1-24B-Instruct-2503",
    "google/gemma-3-27b-it", 
    "Qwen/Qwen2-VL-72B-Instruct",
    "Qwen/Qwen2.5-VL-72B-Instruct"
]

def get_test_tools():
    """Get available tools for testing"""
    # Set required environment variables for computer tool
    os.environ.setdefault("WIDTH", "1920")
    os.environ.setdefault("HEIGHT", "1080")
    
    tool_version = "computer_use_20250124"
    tool_group = TOOL_GROUPS_BY_VERSION[tool_version]
    tool_collection = ToolCollection(*(ToolCls() for ToolCls in tool_group.tools))
    
    # Convert to OpenAI format
    anthropic_tools = tool_collection.to_params()
    openai_tools = _convert_anthropic_tools_to_openai(anthropic_tools)
    
    return openai_tools, [tool["function"]["name"] for tool in openai_tools]

def create_test_messages():
    """Create test messages that could use multiple tools"""
    return [
        {
            "role": "system",
            "content": "You are a helpful assistant with access to computer tools. You can take screenshots, edit files, and run bash commands."
        },
        {
            "role": "user", 
            "content": "I need to take a screenshot of my desktop and then create a simple text file with the current date. Can you help me with this?"
        }
    ]

async def test_tool_choice_none(client: OpenAI, tools: List[Dict], tool_names: List[str], model: str):
    """Test with tool_choice=None"""
    print(f"🔍 Testing tool_choice=None with {model}")
    
    messages = create_test_messages()
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools,
            tool_choice=None,  # Let model decide whether to use tools or not
            max_tokens=1024
        )
        
        choice = response.choices[0].message
        
        print(f"✅ Response with tool_choice=None:")
        if choice.content:
            print(f"   Content: {choice.content[:200]}...")
        
        if choice.tool_calls:
            print(f"   Tool calls made: {len(choice.tool_calls)}")
            for tool_call in choice.tool_calls:
                print(f"   - {tool_call.function.name}: {tool_call.function.arguments[:100]}...")
        else:
            print("   No tool calls made")
            
        return response
        
    except Exception as e:
        print(f"❌ Error with tool_choice=None: {e}")
        return None

async def test_no_tool_choice(client: OpenAI, tools: List[Dict], tool_names: List[str], model: str):
    """Test without specifying tool_choice at all"""
    print(f"\n🔍 Testing without tool_choice parameter with {model}")
    
    messages = create_test_messages()
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools,
            # No tool_choice parameter at all
            max_tokens=1024
        )
        
        choice = response.choices[0].message
        
        print(f"✅ Response without tool_choice:")
        if choice.content:
            print(f"   Content: {choice.content[:200]}...")
        
        if choice.tool_calls:
            print(f"   Tool calls made: {len(choice.tool_calls)}")
            for tool_call in choice.tool_calls:
                print(f"   - {tool_call.function.name}: {tool_call.function.arguments[:100]}...")
        else:
            print("   No tool calls made")
            
        return response
        
    except Exception as e:
        print(f"❌ Error without tool_choice: {e}")
        return None

async def test_tool_choice_auto(client: OpenAI, tools: List[Dict], tool_names: List[str], model: str):
    """Test with tool_choice='auto'"""
    print(f"\n🔍 Testing tool_choice='auto' with {model}")
    
    messages = create_test_messages()
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools,
            tool_choice="auto",  # Let model automatically decide
            max_tokens=1024
        )
        
        choice = response.choices[0].message
        
        print(f"✅ Response with tool_choice='auto':")
        if choice.content:
            print(f"   Content: {choice.content[:200]}...")
        
        if choice.tool_calls:
            print(f"   Tool calls made: {len(choice.tool_calls)}")
            for tool_call in choice.tool_calls:
                print(f"   - {tool_call.function.name}: {tool_call.function.arguments[:100]}...")
        else:
            print("   No tool calls made")
            
        return response
        
    except Exception as e:
        print(f"❌ Error with tool_choice='auto': {e}")
        return None

async def test_tool_choice_required(client: OpenAI, tools: List[Dict], tool_names: List[str], model: str):
    """Test with tool_choice='required'"""
    print(f"\n🔍 Testing tool_choice='required' with {model}")
    
    messages = create_test_messages()
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools,
            tool_choice="required",  # Force model to use tools
            max_tokens=1024
        )
        
        choice = response.choices[0].message
        
        print(f"✅ Response with tool_choice='required':")
        if choice.content:
            print(f"   Content: {choice.content[:200]}...")
        
        if choice.tool_calls:
            print(f"   Tool calls made: {len(choice.tool_calls)}")
            for tool_call in choice.tool_calls:
                print(f"   - {tool_call.function.name}: {tool_call.function.arguments[:100]}...")
        else:
            print("   ⚠️  No tool calls made (unexpected with 'required')")
            
        return response
        
    except Exception as e:
        print(f"❌ Error with tool_choice='required': {e}")
        return None

async def test_tool_choice_specific(client: OpenAI, tools: List[Dict], tool_names: List[str], model: str):
    """Test with tool_choice specifying a specific tool"""
    print(f"\n🔍 Testing tool_choice with specific tool using {model}")
    
    # Choose the first available tool
    if not tool_names:
        print("   ⚠️  No tools available for specific choice test")
        return None
        
    specific_tool = tool_names[0]  # Use first tool (likely 'computer')
    messages = create_test_messages()
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools,
            tool_choice={
                "type": "function",
                "function": {"name": specific_tool}
            },
            max_tokens=1024
        )
        
        choice = response.choices[0].message
        
        print(f"✅ Response with tool_choice='{specific_tool}':")
        if choice.content:
            print(f"   Content: {choice.content[:200]}...")
        
        if choice.tool_calls:
            print(f"   Tool calls made: {len(choice.tool_calls)}")
            for tool_call in choice.tool_calls:
                print(f"   - {tool_call.function.name}: {tool_call.function.arguments[:100]}...")
                if tool_call.function.name != specific_tool:
                    print(f"   ⚠️  Expected {specific_tool}, got {tool_call.function.name}")
        else:
            print("   ⚠️  No tool calls made (unexpected with specific tool choice)")
            
        return response
        
    except Exception as e:
        print(f"❌ Error with tool_choice='{specific_tool}': {e}")
        return None

async def test_model(client: OpenAI, tools: List[Dict], tool_names: List[str], model: str):
    """Test a specific model with all tool_choice variations"""
    print(f"\n{'='*60}")
    print(f"🔬 Testing Model: {model}")
    print(f"{'='*60}")
    
    responses = {}
    
    responses["None"] = await test_tool_choice_none(client, tools, tool_names, model)
    responses["No choice"] = await test_no_tool_choice(client, tools, tool_names, model)
    responses["Auto"] = await test_tool_choice_auto(client, tools, tool_names, model)
    responses["Required"] = await test_tool_choice_required(client, tools, tool_names, model)
    responses["Specific"] = await test_tool_choice_specific(client, tools, tool_names, model)
    
    return responses

def compare_responses(responses: Dict[str, Any], model: str):
    """Compare responses from different tool_choice values"""
    print(f"\n📊 Comparison Summary for {model}:")
    print("=" * 50)
    
    for choice_type, response in responses.items():
        if response is None:
            print(f"{choice_type:15}: Failed")
            continue
            
        choice = response.choices[0].message
        has_content = bool(choice.content)
        has_tools = bool(choice.tool_calls)
        tool_count = len(choice.tool_calls) if choice.tool_calls else 0
        
        print(f"{choice_type:15}: Content={has_content}, Tools={has_tools}, Count={tool_count}")
        
        if choice.tool_calls:
            tools_used = [tc.function.name for tc in choice.tool_calls]
            print(f"{'':15}  Tools used: {', '.join(tools_used)}")

async def main():
    """Main test function"""
    print("🚀 Testing Nebius Tool Choice Behavior Across Multiple Models")
    print("=" * 70)
    
    # Setup
    client = setup_nebius_client()
    tools, tool_names = get_test_tools()
    
    print(f"Available tools: {tool_names}")
    print(f"Total tools: {len(tools)}")
    print(f"Models to test: {len(MODELS_TO_TEST)}")
    
    all_results = {}
    
    # Test each model
    for model in MODELS_TO_TEST:
        try:
            model_responses = await test_model(client, tools, tool_names, model)
            all_results[model] = model_responses
            compare_responses(model_responses, model)
        except Exception as e:
            print(f"❌ Failed to test model {model}: {e}")
            all_results[model] = {"error": str(e)}
    
    # Final summary
    print(f"\n{'='*70}")
    print("🎯 FINAL SUMMARY ACROSS ALL MODELS")
    print(f"{'='*70}")
    
    for model, results in all_results.items():
        if "error" in results:
            print(f"{model:<40}: ❌ Failed - {results['error']}")
            continue
            
        successful_tests = sum(1 for r in results.values() if r is not None)
        total_tests = len(results)
        print(f"{model:<40}: ✅ {successful_tests}/{total_tests} tests passed")
        
        # Show which tool_choice values worked
        working = [choice for choice, resp in results.items() if resp is not None]
        if working:
            print(f"{'':42}Working: {', '.join(working)}")
    
    print("\n✨ Multi-model test completed!")

if __name__ == "__main__":
    asyncio.run(main())
