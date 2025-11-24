"""Test script to verify ToolResponse fix"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.adapters.alfworld_adapter import AlfworldAdapter
from agentscope.tool import ToolResponse

def test_tool_response():
    """Verify that step() returns ToolResponse"""
    adapter = AlfworldAdapter()
    
    # Reset environment
    task = adapter.reset()
    print(f"Task: {task}")
    
    # Execute a step
    result = adapter.step("go to fridge 1")
    
    # Verify it's a ToolResponse
    assert isinstance(result, ToolResponse), f"Expected ToolResponse, got {type(result)}"
    
    # Verify it has content
    assert hasattr(result, 'content'), "ToolResponse should have content"
    assert isinstance(result.content, str), "ToolResponse content should be string"
    
    # Verify metadata
    assert hasattr(result, 'metadata'), "ToolResponse should have metadata"
    assert 'done' in result.metadata, "metadata should contain 'done'"
    assert 'won' in result.metadata, "metadata should contain 'won'"
    assert 'reward' in result.metadata, "metadata should contain 'reward'"
    
    print("✓ ToolResponse structure is correct")
    print(f"Content: {result.content[:100]}...")
    print(f"Metadata: {result.metadata}")

if __name__ == "__main__":
    test_tool_response()
    print("\n✓ All tests passed!")
