#!/usr/bin/env python3

"""
Test script to verify that Nebius provider correctly sets Mistral model
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'computer_use_demo'))

# Mock streamlit session_state
class MockSessionState:
    def __init__(self):
        self.provider = None
        self.model = None
        self.tool_version = None
        self.has_thinking = None
        self.output_tokens = None
        self.max_output_tokens = None
        self.thinking_budget = None

# Set up mock
import streamlit as st
mock_session_state = MockSessionState()
st.session_state = mock_session_state

from computer_use_demo.loop import APIProvider

# Import after mocking
from computer_use_demo.streamlit import PROVIDER_TO_DEFAULT_MODEL_NAME, _reset_model, _reset_model_conf

def test_nebius_model():
    print("=== Testing Nebius Model Configuration ===")
    
    # Test 1: Check default model mapping
    nebius_default = PROVIDER_TO_DEFAULT_MODEL_NAME[APIProvider.NEBIUS]
    print(f"✓ Nebius default model: {nebius_default}")
    assert nebius_default == "mistralai/Mistral-Small-3.1-24B-Instruct-2503"
    
    # Test 2: Test model reset for Nebius
    st.session_state.provider = APIProvider.NEBIUS
    _reset_model()
    print(f"✓ Model after reset: {st.session_state.model}")
    assert st.session_state.model == "mistralai/Mistral-Small-3.1-24B-Instruct-2503"
    
    # Test 3: Test model configuration
    _reset_model_conf()
    print(f"✓ Tool version: {st.session_state.tool_version}")
    print(f"✓ Max output tokens: {st.session_state.max_output_tokens}")
    print(f"✓ Default output tokens: {st.session_state.output_tokens}")
    print(f"✓ Has thinking: {st.session_state.has_thinking}")
    
    # Verify configuration
    assert st.session_state.tool_version == "computer_use_20250124"
    assert st.session_state.max_output_tokens == 32_000
    assert st.session_state.output_tokens == 1024 * 8
    assert st.session_state.has_thinking == False
    
    print("\n🎉 All tests passed! Nebius correctly uses Mistral model.")

if __name__ == "__main__":
    test_nebius_model()
