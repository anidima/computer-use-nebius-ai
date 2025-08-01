#!/usr/bin/env python3

"""
Advanced test using project infrastructure to test tool_choice=None behavior.
"""

import asyncio
import json
import os
import sys
from typing import Any, Dict, List

# Add the project path to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from computer_use_demo.tools.nebius_collection import NebiusToolCollection
from computer_use_demo.tools.nebius_adapter import convert_tool_classes_to_nebius
from computer_use_demo.tools.groups import TOOL_GROUPS_BY_VERSION
from computer_use_demo.tools.bash import BashTool20250124
from computer_use_demo.tools.edit import EditTool20250124
try:
    from openai import OpenAI
except ImportError:
    print("⚠️ OpenAI package not found. Please install it with: pip install openai")
    sys.exit(1)


async def test_with_project_tools():
    """Test using the project's tool infrastructure"""
    
    print("🔬 Advanced test with project tools")
    print("=" * 50)
    
    # Get API key
    api_key = os.getenv("NEBIUS_API_KEY")
    if not api_key:
        api_key = input("Enter your Nebius API key: ").strip()
    
    # Create tool collection using project infrastructure
    tool_group = TOOL_GROUPS_BY_VERSION["computer_use_20250124"]
    nebius_tools = convert_tool_classes_to_nebius(tool_group.tools)
    tool_collection = NebiusToolCollection(*nebius_tools)
    
    # Initialize client
    client = OpenAI(
        base_url="https://api.studio.nebius.com/v1/",
        api_key=api_key
    )
    
    # Test scenarios
    test_cases = [
        {
            "name": "Bash command request",
            "message": "Please list the files in the current directory",
            "expected_behavior": "Should use bash tool"
        },
        {
            "name": "File editing request", 
            "message": "Create a simple text file with the content 'Hello World'",
            "expected_behavior": "Should use edit tool"
        },
        {
            "name": "General conversation",
            "message": "Hello, how are you today?",
            "expected_behavior": "Should respond without tools"
        },
        {
            "name": "Ambiguous request",
            "message": "Can you help me with my work?",
            "expected_behavior": "Unclear if tools will be used"
        }
    ]
    
    models = [
        "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "Qwen/Qwen2.5-72B-Instruct"
    ]
    
    for model in models:
        print(f"\n🤖 Testing model: {model}")
        print("-" * 40)
        
        for test_case in test_cases:
            print(f"\n📝 Test: {test_case['name']}")
            print(f"💬 Message: {test_case['message']}")
            print(f"🎯 Expected: {test_case['expected_behavior']}")
            
            try:
                tools = tool_collection.to_params()
                
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "user", "content": test_case["message"]}
                    ],
                    max_tokens=1000,
                    tools=tools,
                    tool_choice=None  # Key parameter being tested
                )
                
                message = response.choices[0].message
                
                print(f"✅ Response: {message.content}")
                
                if message.tool_calls:
                    print(f"🔧 Used {len(message.tool_calls)} tool(s):")
                    for tool_call in message.tool_calls:
                        print(f"   - {tool_call.function.name}: {tool_call.function.arguments}")
                else:
                    print("🚫 No tools used")
                    
            except Exception as e:
                print(f"❌ Error: {str(e)}")
            
            print()


if __name__ == "__main__":
    asyncio.run(test_with_project_tools())
