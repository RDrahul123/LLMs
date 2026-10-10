# Module 4: Working with LLM APIs

> **Level:** Intermediate | **Time:** ~2 hours | **Prerequisites:** Module 3 | **Code:** chatbot.py, openai_examples.py, claude_examples.py, streaming_demo.py

---

## Why Use LLM APIs?

Instead of running models yourself, LLM APIs let you access state-of-the-art models with a simple HTTP call. Benefits:

- No GPU required — runs on the provider's servers
- Pay only for what you use
- Always the latest model version
- Simple authentication and billing

---

## API Overview

| Provider | Library / interface | Example model family | Cost model |
|----------|----------------------|----------------------|------------|
| OpenAI | `openai` | GPT family | Provider pricing varies by model and may change |
| Anthropic | `anthropic` | Claude family | Provider pricing varies by model and may change |
| Google | `google-genai` | Gemini family | Provider pricing varies by model and may change |
| Local | Ollama API | Open-weight models | No per-token API fee; compute, power, and hosting still have costs |

Use each provider's current pricing and model documentation when estimating costs. The API snippets below are learning examples; verify model identifiers and feature availability before deploying an application.

![How an API request moves between your application and a model provider](../assets/diagrams/llm-lifecycle.svg)

---

## OpenAI API

### Basic Request

```python
from openai import OpenAI

client = OpenAI()  # Uses OPENAI_API_KEY env var

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "user", "content": "What is the capital of France?"}
    ]
)
print(response.choices[0].message.content)
# → "The capital of France is Paris."
```

### System Prompt

```python
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "You are a helpful math tutor. Explain step by step."},
        {"role": "user", "content": "What is 15% of 200?"}
    ]
)
```

### Multi-Turn Conversation

```python
messages = [
    {"role": "system", "content": "You are a friendly cooking assistant."},
    {"role": "user", "content": "What's a good pasta recipe?"},
]

# First turn
response = client.chat.completions.create(
    model="gpt-4o-mini", messages=messages
)
messages.append({"role": "assistant", "content": response.choices[0].message.content})

# Second turn — model remembers context
messages.append({"role": "user", "content": "Can I make it vegetarian?"})
response = client.chat.completions.create(
    model="gpt-4o-mini", messages=messages
)
```

### JSON Mode

```python
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "user", "content": "List 3 planets with their moons. Return as JSON."}
    ],
    response_format={"type": "json_object"}
)
```

---

## Anthropic Claude API

```python
import anthropic

client = anthropic.Anthropic()  # Uses ANTHROPIC_API_KEY env var

message = client.messages.create(
    model="claude-3-5-sonnet-20241022",
    max_tokens=1024,
    system="You are a helpful coding assistant.",
    messages=[
        {"role": "user", "content": "Explain the difference between async and threading in Python."}
    ]
)
print(message.content[0].text)
```

Key differences from OpenAI:
- Uses `messages.create()` instead of `chat.completions.create()`
- System prompt is a separate parameter (not in messages)
- Response is `message.content[0].text`

---

## Google Gemini API

```python
from google import genai

client = genai.Client()  # Uses GOOGLE_API_KEY env var

response = client.models.generate_content(
    model="gemini-2.0-flash",
    contents="What are the key differences between Python and JavaScript?",
)
print(response.text)
```

---

## Ollama (Local, Free)

Run open-source models locally with no API key.

```bash
# Install Ollama: https://ollama.com
# Pull a model
ollama pull llama3.1:8b

# Run
ollama run llama3.1:8b
```

### Python with Ollama

```python
import requests

response = requests.post("http://localhost:11434/api/chat", json={
    "model": "llama3.1:8b",
    "messages": [{"role": "user", "content": "Hello!"}],
    "stream": False
})
print(response.json()["message"]["content"])
```

---

## Streaming Responses

Streaming returns tokens as they're generated, providing a better user experience.

### OpenAI Streaming

```python
stream = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Tell me a story"}],
    stream=True
)

for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
print()
```

### Anthropic Streaming

```python
with client.messages.stream(
    model="claude-3-5-sonnet-20241022",
    max_tokens=1024,
    messages=[{"role": "user", "content": "Tell me a story"}]
) as stream:
    for text in stream.text_stream:
        print(text, end="", flush=True)
print()
```

---

## Cost Comparison

Model prices, limits, and names change over time. Compare current provider pricing for both input and output tokens, caching, batch processing, and any tool or multimodal usage. For development, begin with an inexpensive model that meets the quality bar; do not select production models by price alone—run the evaluation set from Module 10.

---

## Try It Yourself

```bash
# Run the chatbot
python 03-llm-apis/chatbot.py --model gpt-4o-mini

# Run with a custom system prompt
python 03-llm-apis/chatbot.py --model gpt-4o-mini --system "You are a pirate. Respond in pirate dialect."

# Run with local Ollama
python 03-llm-apis/chatbot.py --model llama3.1:8b
```

---

## What's Next?

You can now build with LLM APIs. But what if your model needs access to your own data?

**→ [Module 5: Retrieval-Augmented Generation (RAG)](../04-rag/README.md)**

Learn how to give LLMs access to documents, databases, and your own knowledge base.

---

## Further Reading

- [OpenAI API Reference](https://platform.openai.com/docs/api-reference)
- [Anthropic API Reference](https://docs.anthropic.com/en/api/messages)
- [Ollama Documentation](https://ollama.com/library)
- [LiteLLM — Unified API for 100+ LLMs](https://github.com/BerriAI/litellm)

## Visual Deep Dive

Continue to **[Module 4 Deep Dive](deep-dive.md)** for request lifecycles, stateless conversation history, streaming, tool-call round trips, retries, and cost estimation.
