# Module 3 — LLM APIs 🟠

> **Level:** Beginner → Intermediate | **Time:** ~2 hours | **Code:** Python

---

## Why Use LLM APIs?

Instead of running models locally (requires expensive GPUs), you can call state-of-the-art models via API — pay only for what you use.

```
Your App  →  HTTP Request  →  LLM API Server  →  AI Response  →  Your App
```

---

## APIs Covered in This Module

| Provider | Model | Strengths |
|---|---|---|
| **OpenAI** | GPT-4o, GPT-4o-mini | Best overall, huge ecosystem |
| **Anthropic** | Claude 3.5 Sonnet | Long context, safety, reasoning |
| **Google** | Gemini 1.5 Pro | Multimodal, massive context |
| **Ollama** | LLaMA 3, Mistral | Free, runs locally, private |

---

## Setup

### 1. Install Dependencies
```bash
pip install openai anthropic google-generativeai python-dotenv
```

### 2. Get API Keys
- OpenAI: https://platform.openai.com/api-keys
- Anthropic: https://console.anthropic.com/
- Google: https://aistudio.google.com/

### 3. Create `.env` file
```bash
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GOOGLE_API_KEY=AI...
```

---

## OpenAI API

### Basic Completion
```python
from openai import OpenAI

client = OpenAI()  # reads OPENAI_API_KEY from env

response = client.chat.completions.create(
    model="gpt-4o-mini",          # cheapest smart model
    messages=[
        {"role": "user", "content": "Explain recursion in one paragraph."}
    ],
    temperature=0.7,
    max_tokens=500
)

print(response.choices[0].message.content)
```

### With System Prompt
```python
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "You are a friendly Python tutor."},
        {"role": "user", "content": "What's a list comprehension?"}
    ]
)
```

### Multi-Turn Conversation
```python
messages = [{"role": "system", "content": "You are a helpful assistant."}]

while True:
    user_input = input("You: ")
    if user_input.lower() == "quit":
        break

    messages.append({"role": "user", "content": user_input})

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages
    )

    reply = response.choices[0].message.content
    messages.append({"role": "assistant", "content": reply})
    print(f"AI: {reply}\n")
```

### Structured Output (JSON Mode)
```python
import json

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {
            "role": "user",
            "content": "Extract: name, email from: 'Hi, I am John Doe, john@example.com'"
        }
    ],
    response_format={"type": "json_object"}
)

data = json.loads(response.choices[0].message.content)
print(data["name"])   # John Doe
print(data["email"])  # john@example.com
```

---

## Anthropic Claude API

```python
import anthropic

client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env

message = client.messages.create(
    model="claude-3-5-sonnet-20241022",
    max_tokens=1024,
    system="You are a data science expert.",
    messages=[
        {"role": "user", "content": "What's the difference between L1 and L2 regularization?"}
    ]
)

print(message.content[0].text)
```

### Claude with Long Context
Claude shines with large documents (up to 200k tokens):

```python
with open("long_document.txt", "r") as f:
    document = f.read()

message = client.messages.create(
    model="claude-3-5-sonnet-20241022",
    max_tokens=2048,
    messages=[
        {
            "role": "user",
            "content": f"Here is a document:\n\n{document}\n\nSummarize the top 5 key points."
        }
    ]
)
```

---

## Google Gemini API

```python
import google.generativeai as genai
import os

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

model = genai.GenerativeModel("gemini-1.5-flash")

response = model.generate_content("Write a haiku about machine learning.")
print(response.text)
```

---

## Ollama (Free, Local, Private)

Run models 100% on your machine — no API key needed.

### Install Ollama
```bash
# MacOS/Linux
curl -fsSL https://ollama.ai/install.sh | sh

# Pull a model
ollama pull llama3
ollama pull mistral
```

### Use with Python
```python
from openai import OpenAI  # Ollama uses OpenAI-compatible API

client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama"  # required but unused
)

response = client.chat.completions.create(
    model="llama3",
    messages=[
        {"role": "user", "content": "What is the capital of France?"}
    ]
)

print(response.choices[0].message.content)
```

---

## Streaming Responses

Instead of waiting for the full response, stream it token-by-token:

```python
stream = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Write a short story about a robot."}],
    stream=True
)

for chunk in stream:
    if chunk.choices[0].delta.content is not None:
        print(chunk.choices[0].delta.content, end="", flush=True)
```

---

## Cost Comparison (Approximate, as of 2024)

| Model | Input (per 1M tokens) | Output (per 1M tokens) |
|---|---|---|
| GPT-4o | $5.00 | $15.00 |
| GPT-4o-mini | $0.15 | $0.60 |
| Claude 3.5 Sonnet | $3.00 | $15.00 |
| Gemini 1.5 Flash | $0.075 | $0.30 |
| Ollama (local) | **Free** | **Free** |

> 💡 For development/learning, use `gpt-4o-mini` or `gemini-flash` — they're nearly as good and 20x cheaper.

---

## 📁 Files in This Module

| File | Description |
|---|---|
| [`openai_examples.py`](./openai_examples.py) | Full OpenAI API examples |
| [`claude_examples.py`](./claude_examples.py) | Anthropic Claude examples |
| [`chatbot.py`](./chatbot.py) | Complete terminal chatbot |
| [`streaming_demo.py`](./streaming_demo.py) | Streaming response demo |

---

## ✅ What's Next?

👉 **[Module 4 — RAG](../04-rag/README.md)**
