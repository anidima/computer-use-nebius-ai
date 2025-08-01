#!/usr/bin/env python3

"""
Test script to check the format of tools that are passed to OpenAI API conversion
"""

import sys
import os

# Set required environment variables for tools
os.environ['WIDTH'] = '1920'
os.environ['HEIGHT'] = '1080'
os.environ['DISPLAY_NUM'] = '1'

sys.path.append(os.path.join(os.path.dirname(__file__), 'computer_use_demo'))

from computer_use_demo.tools import TOOL_GROUPS_BY_VERSION, ToolCollection
import json

def test_tools_format():
    # Get the latest tool group
    tool_group = TOOL_GROUPS_BY_VERSION["computer_use_20250124"]
    tool_collection = ToolCollection(*(ToolCls() for ToolCls in tool_group.tools))
    
    # Get Anthropic format
    anthropic_tools = tool_collection.to_params()
    
    print("=== ANTHROPIC TOOLS FORMAT ===")
    for i, tool in enumerate(anthropic_tools):
        print(f"\nTool {i+1}:")
        print(json.dumps(tool, indent=2, default=str))
    
    # Test our conversion function
    from computer_use_demo.loop import _convert_anthropic_tools_to_openai
    
    openai_tools = _convert_anthropic_tools_to_openai(anthropic_tools)
    
    print("\n\n=== OPENAI TOOLS FORMAT ===")
    for i, tool in enumerate(openai_tools):
        print(f"\nTool {i+1}:")
        print(json.dumps(tool, indent=2, default=str))

if __name__ == "__main__":
    test_tools_format()
