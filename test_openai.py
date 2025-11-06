#!/usr/bin/env python3
"""Test script for OpenAI client"""

from src.llm_client import create_llm_client

# Test OpenAI client with auto-detection
print("Testing OpenAI client with gpt-3.5-turbo...")
client = create_llm_client("gpt-3.5-turbo", backend="auto")

# Test simple generation
prompt = "Say 'Hello, I am GPT-3.5-turbo' and nothing else."
print(f"\nPrompt: {prompt}")
response = client.generate(prompt, temperature=0.0, max_new_tokens=50)
print(f"Response: {response}")

print("\n✓ OpenAI client test successful!")
