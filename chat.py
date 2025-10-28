#!/usr/bin/env python3
"""CLI Chat Interface for LLM Models

Interactive terminal-based conversation interface supporting multiple LLM models.

Usage:
    python chat.py
"""

import os
import re
import sys
import logging
import contextlib
from typing import List, Dict, Optional

# Suppress all vLLM progress bars and logs BEFORE importing
os.environ["VLLM_LOGGING_LEVEL"] = "ERROR"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

from src.llm_client import create_llm_client, LLMClient

# ANSI color codes for terminal output
class Colors:
    BLUE = '\033[94m'      # User messages
    GREEN = '\033[92m'     # Model responses
    RED = '\033[91m'       # Config/system messages
    RESET = '\033[0m'      # Reset to default

# Suppress Python logging for all libraries
logging.getLogger("vllm").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("torch").setLevel(logging.ERROR)
logging.getLogger("torch.distributed").setLevel(logging.ERROR)
logging.getLogger("networkx").setLevel(logging.ERROR)

# Suppress warnings from PyTorch distributed
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch.distributed")
warnings.filterwarnings("ignore", message=".*destroy_process_group.*")


class ChatSession:
    """Manages a multi-turn chat session with an LLM."""

    # Model configuration registry
    MODEL_CONFIG = {
        "qwen": {
            "name": "Qwen/Qwen2.5-7B-Instruct",
            "backend": "vllm",
            "context_window": 4096,
            "display_name": "Qwen 2.5 7B",
        },
        "saul": {
            "name": "Equall/Saul-7B-Instruct-v1",
            "backend": "vllm",
            "context_window": 4096,
            "display_name": "Saul 7B",
        },
        "haiku": {
            "name": "haiku",  # Alias, resolved by create_llm_client
            "backend": "claude",
            "context_window": 200000,
            "display_name": "Claude Haiku",
        },
    }

    SAFETY_MARGIN = 512  # Reserve tokens for generation (local models)
    CLAUDE_SAFETY_MARGIN = 2048  # Larger buffer for Claude due to token estimation uncertainty

    # Pre-compiled command patterns for efficiency
    _CMD_EXIT = re.compile(r'^/exit\s*$')
    _CMD_MODEL = re.compile(r'^/model(?:\s+(\w+))?\s*$')
    _CMD_PROMPT = re.compile(r'^/prompt(?:\s+(.*))?$', re.IGNORECASE)
    _CMD_CLEAN = re.compile(r'^/clean\s*$')
    _CMD_HELP = re.compile(r'^/help\s*$')

    def __init__(self, initial_model: str = "qwen", system_prompt: str = ""):
        """
        Initialize ChatSession with a model.

        Args:
            initial_model: Model key from MODEL_CONFIG (default: "qwen")
            system_prompt: Initial system prompt (default: empty string)
        """
        self.llm_client: Optional[LLMClient] = None
        self.model_key = initial_model.lower()
        self.system_prompt = system_prompt
        self.conversation_history: List[Dict[str, str]] = []

        # Validate model and initialize context_window from config
        if self.model_key not in self.MODEL_CONFIG:
            available = ', '.join(self.MODEL_CONFIG.keys())
            raise ValueError(f"Unknown model: {initial_model}\nAvailable models: {available}")

        self.context_window = self.MODEL_CONFIG[self.model_key]["context_window"]

        # Load initial model
        if not self.switch_model(initial_model):
            # Exit if initial model fails to load (no fallback available)
            print(f"FATAL: Cannot initialize chat without a working model.")
            sys.exit(1)

    def switch_model(self, model_key: str) -> bool:
        """
        Switch to a different model.

        Args:
            model_key: Model identifier from MODEL_CONFIG

        Returns:
            True if successful, False if failed (should exit program)
        """
        model_key = model_key.lower()

        if model_key not in self.MODEL_CONFIG:
            print(f"Error: Unknown model '{model_key}'.")
            print(f"Available models: {', '.join(self.MODEL_CONFIG.keys())}")
            return False

        try:
            config = self.MODEL_CONFIG[model_key]
            print(f"\nLoading {config['display_name']}...")

            # Validate API key exists in environment (should be loaded from .env via dotenv)
            if config["backend"] == "claude":
                if not os.getenv("ANTHROPIC_API_KEY"):
                    print(f"\nError: ANTHROPIC_API_KEY not found in environment.")
                    print("Please ensure your .env file is loaded or set the environment variable.")
                    print("See README_CHAT.md for setup instructions.")
                    return False

            # Create new client (suppress loading messages and progress bars)
            with open(os.devnull, 'w') as devnull:
                with contextlib.redirect_stderr(devnull), contextlib.redirect_stdout(devnull):
                    self.llm_client = create_llm_client(
                        model_name=config["name"],
                        backend=config["backend"],
                        max_new_tokens=512,
                        temperature=0.7,
                    )

            self.model_key = model_key
            self.context_window = config["context_window"]

            print(f"{Colors.RED}Model loaded successfully!{Colors.RESET}")
            self.show_startup_info()
            return True

        except Exception as e:
            print(f"\nFATAL ERROR: Failed to load model: {e}")
            print("Program exiting.")
            return False

    def count_tokens(self, text: str) -> int:
        """
        Count tokens in text using tokenizer or fallback estimation.

        Args:
            text: Text to count tokens for

        Returns:
            Approximate token count
        """
        if not text:
            return 0

        try:
            # Try to use tokenizer (available for TransformersClient, VLLMClient)
            if hasattr(self.llm_client, 'tokenizer'):
                return len(self.llm_client.tokenizer.encode(text))
        except Exception as e:
            # Log tokenizer failure for debugging
            if not hasattr(self, '_tokenizer_warning_shown'):
                print(f"Note: Token counting using estimation ({type(e).__name__})")
                self._tokenizer_warning_shown = True

        # Fallback: rough estimate (3-5 chars ≈ 1 token, varies by model/language)
        # May be inaccurate - actual tokenizer preferred when available
        return max(len(text) // 4, 1)

    def get_display_name(self) -> str:
        """Get friendly display name for current model."""
        return self.MODEL_CONFIG[self.model_key]["display_name"]

    def get_system_prompt_tokens(self) -> int:
        """Get token count for system prompt."""
        return self.count_tokens(self.system_prompt) if self.system_prompt else 0

    def get_remaining_context(self) -> int:
        """
        Calculate remaining tokens after safety margin.

        Returns:
            Number of tokens remaining in context window
        """
        full_prompt = self.build_prompt()
        used_tokens = self.count_tokens(full_prompt)

        # Use larger safety margin for Claude models due to estimation uncertainty
        safety_margin = self.CLAUDE_SAFETY_MARGIN if self.MODEL_CONFIG[self.model_key]["backend"] == "claude" else self.SAFETY_MARGIN

        remaining = self.context_window - used_tokens - safety_margin

        # Warn if context is exceeded
        if remaining < 0:
            print(f"\nWARNING: Context window exceeded by {abs(remaining)} tokens!")
            print("Consider using /clean to clear conversation history.")
            return 0

        return remaining

    def build_prompt(self) -> str:
        """
        Build full prompt from system prompt + conversation history.

        Returns:
            Formatted prompt string
        """
        parts = []

        if self.system_prompt:
            parts.append(f"System: {self.system_prompt}\n")

        for msg in self.conversation_history:
            role = msg["role"].capitalize()
            parts.append(f"{role}: {msg['content']}\n")

        return "".join(parts)

    def add_to_history(self, role: str, content: str) -> None:
        """
        Add message to conversation history.

        Args:
            role: "user" or "assistant"
            content: Message content
        """
        self.conversation_history.append({"role": role, "content": content})

    def chat(self, user_input: str) -> str:
        """
        Send message and get response.

        Args:
            user_input: User's message

        Returns:
            Assistant's response
        """
        # Add user message to history
        self.add_to_history("user", user_input)

        # Build full prompt with history
        full_prompt = self.build_prompt() + "Assistant: "

        # Generate response
        try:
            # Temperature already set at client initialization
            # Suppress progress bars by redirecting stderr
            with open(os.devnull, 'w') as devnull:
                with contextlib.redirect_stderr(devnull):
                    response = self.llm_client.generate(full_prompt)

            # Clean up response (remove any leading/trailing whitespace)
            response = response.strip()

            # Add assistant response to history
            self.add_to_history("assistant", response)

            return response

        except (ConnectionError, TimeoutError) as e:
            print(f"\nNetwork error: {e}")
            print("Please check your internet connection and try again.")
            self.conversation_history.pop()
            return ""

        except MemoryError as e:
            print(f"\nOut of memory: {e}")
            print("Try using /clean to clear history or switch to a smaller model with /model")
            self.conversation_history.pop()
            return ""

        except KeyboardInterrupt:
            print("\nGeneration interrupted by user")
            self.conversation_history.pop()
            raise  # Re-raise to let outer handler deal with it

        except Exception as e:
            import traceback
            print(f"\nError generating response: {type(e).__name__}: {e}")
            print("\nFull traceback:")
            traceback.print_exc()
            print("Your message was removed from history. Please try again or rephrase.")
            # Remove user message from history since generation failed
            self.conversation_history.pop()
            return ""

    def execute_command(self, user_input: str) -> Optional[str]:
        """
        Parse and execute command.

        Args:
            user_input: Raw user input

        Returns:
            Command result message, "COMMAND_EXIT" to exit, or None if not a command
        """
        # Normalize to lowercase for case-insensitive matching
        cmd_lower = user_input.strip().lower()

        # Exit command
        if self._CMD_EXIT.match(cmd_lower):
            return "COMMAND_EXIT"

        # Model command
        model_match = self._CMD_MODEL.match(cmd_lower)
        if model_match:
            if model_match.group(1):
                # Switch model
                new_model = model_match.group(1).lower()
                if not self.switch_model(new_model):
                    # Model switch failed, continue with current model
                    return f"Failed to switch to '{new_model}'. Continuing with {self.get_display_name()}."
                return f"Switched to {self.get_display_name()}"
            else:
                # List models
                return self._list_models()

        # Prompt command - extract original case for the prompt text
        prompt_match = self._CMD_PROMPT.match(user_input.strip())
        if prompt_match:
            if prompt_match.group(1):
                # Set new prompt (preserve original case)
                new_prompt = prompt_match.group(1).strip().strip('"\'')
                old_prompt = self.system_prompt
                self.system_prompt = new_prompt

                # Clear conversation when prompt changes
                if old_prompt != new_prompt:
                    self.conversation_history = []

                token_count = self.get_system_prompt_tokens()
                return f"System prompt updated ({token_count} tokens)\nConversation history cleared."
            else:
                # Show current prompt
                if self.system_prompt:
                    token_count = self.get_system_prompt_tokens()
                    return f'Current system prompt: "{self.system_prompt}" ({token_count} tokens)'
                else:
                    return "System prompt is empty"

        # Clean command
        if self._CMD_CLEAN.match(cmd_lower):
            self.conversation_history = []
            return "Conversation history cleared. System prompt preserved."

        # Help command
        if self._CMD_HELP.match(cmd_lower):
            return self._show_help()

        # Not a command
        return None

    def show_startup_info(self) -> None:
        """Display model information at startup."""
        token_count = self.get_system_prompt_tokens()

        print(f"{Colors.RED}\n" + "="*80)
        print(f"MODEL: {self.get_display_name()}")
        print(f"BACKEND: {self.MODEL_CONFIG[self.model_key]['backend']}")
        print(f"CONTEXT WINDOW: {self.context_window:,} tokens")
        print("="*80)

        if self.system_prompt:
            print(f"\nSYSTEM PROMPT ({token_count} tokens):")
            print(f'  "{self.system_prompt}"')
        else:
            print("\nSYSTEM PROMPT: (empty)")

        print(f"\nType /help for commands or start chatting.{Colors.RESET}")
        print()

    def show_context_summary(self) -> None:
        """Display context usage after each message."""
        remaining = self.get_remaining_context()
        safety_margin = self.CLAUDE_SAFETY_MARGIN if self.MODEL_CONFIG[self.model_key]["backend"] == "claude" else self.SAFETY_MARGIN
        used = self.context_window - remaining - safety_margin
        percent = (used / self.context_window) * 100 if self.context_window > 0 else 0

        print(f"{Colors.RED}\n[Context: {remaining:,} / {self.context_window:,} tokens remaining ({percent:.1f}% used)]{Colors.RESET}")

    def _show_help(self) -> str:
        """Return formatted help text."""
        return """
Available Commands (case-insensitive):
  /exit                    - Exit the chat
  /model [name]            - List models or switch to specified model
  /prompt [text]           - Show current prompt or set new system prompt
  /clean                   - Clear conversation history
  /help                    - Show this help message

Available Models:
  qwen                     - Qwen 2.5 7B Instruct (4,096 tokens, vLLM)
  saul                     - Saul 7B Legal Model (4,096 tokens, vLLM)
  haiku                    - Claude Haiku 4.5 (200,000 tokens, API)

Examples:
  /model haiku
  /prompt You are a helpful math tutor.
  /clean

Note: Setting a new system prompt will clear the conversation history.
"""

    def _list_models(self) -> str:
        """Return formatted model list."""
        models = []
        for key, config in self.MODEL_CONFIG.items():
            marker = " (current)" if key == self.model_key else ""
            models.append(f"  {key:10} - {config['display_name']} ({config['context_window']:,} tokens){marker}")

        return "Available models:\n" + "\n".join(models)


def main():
    """Main CLI loop."""
    print(f"{Colors.RED}{'='*80}")
    print("CLI Chat Interface")
    print("="*80)
    print(f"\nInitializing with default model (Qwen 2.5 7B)...{Colors.RESET}")

    # Initialize with default model
    try:
        session = ChatSession(initial_model="qwen", system_prompt="")
    except Exception as e:
        print(f"{Colors.RED}Failed to initialize: {e}{Colors.RESET}")
        sys.exit(1)

    # Main loop
    while True:
        try:
            # Display prompt in blue
            prompt_prefix = f"{Colors.BLUE}[{session.get_display_name()}] You: {Colors.RESET}"
            user_input = input(prompt_prefix).strip()

            # Skip empty input
            if not user_input:
                continue

            # Try to execute as command
            cmd_result = session.execute_command(user_input)

            if cmd_result == "COMMAND_EXIT":
                print(f"\n{Colors.RED}Goodbye!{Colors.RESET}")
                break
            elif cmd_result is not None:
                # Command was executed (print in red for config messages)
                print(f"{Colors.RED}\n{cmd_result}\n{Colors.RESET}")
            else:
                # Not a command, treat as chat message
                print()  # Blank line before response
                response = session.chat(user_input)

                if response:
                    # Print model response in green
                    print(f"{Colors.GREEN}{session.get_display_name()}: {response}{Colors.RESET}")
                    session.show_context_summary()
                print()  # Blank line after response

        except KeyboardInterrupt:
            print("\n\nInterrupted. Type /exit to quit.")
            print()
        except EOFError:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"\nError: {e}")
            print()


if __name__ == "__main__":
    main()
