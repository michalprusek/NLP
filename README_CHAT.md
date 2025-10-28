# CLI Chat Interface

Interactive terminal-based conversation interface for LLM models.

## Quick Start

```bash
# Run with default model (Qwen 2.5 7B)
uv run python chat.py

# Or use standard python (if dependencies installed)
python chat.py
```

## Features

- **Multi-turn conversations** with conversation history
- **Multiple model support**: Qwen 2.5 7B, Saul 7B (legal), Claude Haiku
- **System prompts** for customizing model behavior
- **Context tracking** showing remaining tokens after each message
- **Easy model switching** during conversation

## Available Commands

All commands are case-insensitive:

| Command | Usage | Description |
|---------|-------|-------------|
| `/exit` | `/exit` | Exit the chat |
| `/model` | `/model` or `/model haiku` | List models or switch to specified model |
| `/prompt` | `/prompt` or `/prompt Your text here` | Show or set system prompt |
| `/clean` | `/clean` | Clear conversation history |
| `/help` | `/help` | Show help message |

## Supported Models

| Model | Backend | Context Window | Description |
|-------|---------|----------------|-------------|
| `qwen` | vLLM | 4,096 tokens | Qwen 2.5 7B Instruct (default) |
| `saul` | vLLM | 4,096 tokens | Saul 7B Legal Model |
| `haiku` | Claude API | 200,000 tokens | Claude Haiku 4.5 |

## Example Session

```
[Qwen 2.5 7B] You: What is 2+2?

Qwen 2.5 7B: 2+2 equals 4.

[Context: 3,890 / 4,096 tokens remaining (5.0% used)]

[Qwen 2.5 7B] You: /prompt You are a helpful math tutor who explains step by step.

System prompt updated (15 tokens)
Conversation history cleared.

[Qwen 2.5 7B] You: What is 15 * 23?

Qwen 2.5 7B: Let me solve 15 × 23 step by step:
15 × 23 = 15 × (20 + 3)
= (15 × 20) + (15 × 3)
= 300 + 45
= 345

[Context: 3,802 / 4,096 tokens remaining (7.2% used)]

[Qwen 2.5 7B] You: /model haiku

Loading Claude Haiku...
Model loaded successfully!

[Claude Haiku] You: Hello!
...
```

## Configuration

### For Claude Models (Haiku)

You need an Anthropic API key. Add it to your `.env` file:

```bash
# Copy example file
cp .env.example .env

# Edit .env and add your key
ANTHROPIC_API_KEY=your_api_key_here
```

### For Local Models (Qwen, Saul)

Local models use vLLM backend and require:
- CUDA GPU (for vLLM)
- Sufficient GPU memory (~8GB for 7B models)

## Behavior Notes

1. **System Prompt Changes**: When you set a new system prompt with `/prompt`, the conversation history is automatically cleared.

2. **Model Switching**: You can switch models mid-conversation. The conversation history is preserved.

3. **Context Limits**: When you approach the context window limit, consider using `/clean` to clear history.

4. **Model Loading Failures**: If a model fails to load (e.g., out of memory), the program will exit. This is intentional to prevent an inconsistent state.

5. **Token Counting**:
   - For local models (Qwen, Saul): Uses actual tokenizer (accurate)
   - For Claude models: Uses approximation (~4 characters per token)

## Troubleshooting

### "ModuleNotFoundError: No module named 'torch'"

Use `uv run python chat.py` instead of `python chat.py` to ensure dependencies are loaded.

### "ANTHROPIC_API_KEY not found"

Make sure you have a `.env` file with your Anthropic API key for Claude models.

### "CUDA out of memory"

Try using a smaller model or use CPU backend (though this is not configured by default in chat.py).

### Model loading is slow

First load of vLLM models can take 30-60 seconds. Subsequent generations are fast.

## Implementation Details

- **File**: `chat.py` (~350 lines)
- **Architecture**: Single ChatSession class managing all state
- **Dependencies**: Uses existing `src/llm_client.py` infrastructure
- **No modifications**: Existing codebase remains unchanged
