# 🧠 Learn LLMs — A Complete Practical Course

> From zero to building real AI applications with Large Language Models.

![Course Banner](https://img.shields.io/badge/Level-All%20Levels-blue?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)
![PRs Welcome](https://img.shields.io/badge/PRs-Welcome-orange?style=for-the-badge)
![Maintained](https://img.shields.io/badge/Maintained-Yes-brightgreen?style=for-the-badge)

---

## 📚 What Is This?

This is a **free, open-source course** on Large Language Models (LLMs) — designed for everyone from complete beginners to developers who want to build production-grade AI apps.

No fluff. Just clean explanations, working code, and real examples.

---

## 🗺️ Course Roadmap

```
📦 llm-course
 ┣ 📂 01-introduction         → What are LLMs? How do they work?
 ┣ 📂 02-prompt-engineering   → Craft prompts like a pro
 ┣ 📂 03-llm-apis             → Use OpenAI, Claude, Gemini APIs
 ┣ 📂 04-rag                  → Build retrieval-augmented generation apps
 ┗ 📂 05-fine-tuning          → Fine-tune models on your own data
```

---

## 🧭 Who Is This For?

| 👤 Profile | ✅ You'll Learn |
|---|---|
| **Beginners** | What LLMs are, how to prompt them, use APIs |
| **Developers** | Build RAG pipelines, integrate LLM APIs into apps |
| **ML Engineers** | Fine-tune models, evaluate outputs, go to production |

---

## 📖 Modules

### 🟢 Module 1 — Introduction to LLMs
> *Perfect for beginners. No code required.*

- What is an LLM?
- How does training work? (Tokens, transformers, attention)
- Famous models: GPT-4, Claude, Gemini, Mistral, LLaMA
- Key concepts: Temperature, context window, hallucinations

📁 [`01-introduction/`](./01-introduction/README.md)

---

### 🟡 Module 2 — Prompt Engineering
> *The skill that 10x's your LLM output quality.*

- Zero-shot, few-shot, chain-of-thought prompting
- Role prompting & system prompts
- Prompt templates & reusability
- Common mistakes and how to fix them

📁 [`02-prompt-engineering/`](./02-prompt-engineering/README.md)

---

### 🟠 Module 3 — LLM APIs
> *Connect your apps to powerful AI models.*

- OpenAI API (GPT-4o, GPT-3.5)
- Anthropic Claude API
- Google Gemini API
- Open-source models via Ollama & HuggingFace
- Building a simple chatbot

📁 [`03-llm-apis/`](./03-llm-apis/README.md)

---

### 🔵 Module 4 — RAG (Retrieval Augmented Generation)
> *Give your LLM access to your own data.*

- Why RAG? The problem with LLM memory
- Embeddings & vector databases (FAISS, Pinecone, ChromaDB)
- Build a RAG pipeline from scratch
- Chat with your PDF

📁 [`04-rag/`](./04-rag/README.md)

---

### 🔴 Module 5 — Fine-Tuning LLMs
> *Train a model on your own data.*

- When to fine-tune vs. prompt engineer vs. RAG
- Preparing your dataset (JSONL format)
- Fine-tuning with OpenAI & HuggingFace
- Evaluating your fine-tuned model

📁 [`05-fine-tuning/`](./05-fine-tuning/README.md)

---

## 🚀 Quick Start

```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/llm-course.git
cd llm-course

# Install Python dependencies
pip install -r requirements.txt

# Set your API keys
cp .env.example .env
# → Edit .env and add your keys
```

---

## 🔧 Prerequisites

- Python 3.9+
- Basic Python knowledge (for code modules)
- API keys: [OpenAI](https://platform.openai.com/) | [Anthropic](https://console.anthropic.com/) *(free tiers available)*

---

## 🤝 Contributing

Contributions are welcome! Found a bug, have a better example, or want to add a new module?

1. Fork the repo
2. Create a branch: `git checkout -b feature/your-idea`
3. Commit and push
4. Open a Pull Request

See [`CONTRIBUTING.md`](./CONTRIBUTING.md) for details.

---

## 📄 License

MIT License — free to use, share, and build on.

---

## ⭐ Star This Repo

If this helped you, give it a star! It helps others find it.

---

*Made with ❤️ to make AI education accessible to everyone.*
