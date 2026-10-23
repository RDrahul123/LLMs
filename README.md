<div align="center">

# Learn LLMs

### A practical, visual course—from first principles to production

Understand how large language models work, build applications with them, and learn how to evaluate and operate those systems responsibly.

[![Open the course](https://img.shields.io/badge/Explore_the_live_course-Visit_site-2563eb?style=for-the-badge)](https://rdrahul123.github.io/LLMs/)
[![Read the study guide](https://img.shields.io/badge/Start_with_the_study_guide-Read_now-7c3aed?style=for-the-badge)](./LLM_STUDY_GUIDE.md)

<br>

![All levels](https://img.shields.io/badge/Level-All%20levels-2563eb)
![12 modules](https://img.shields.io/badge/Course-12%20modules-0f766e)
![5 projects](https://img.shields.io/badge/Practice-5%20projects-ea580c)
![MIT license](https://img.shields.io/badge/License-MIT-16a34a)
![Contributions welcome](https://img.shields.io/badge/Contributions-Welcome-f59e0b)

</div>

<br>

![A visual map of how trained models are used inside AI applications](./assets/diagrams/llm-lifecycle.svg)

## Start learning

New to LLMs? The **[Complete LLM Study Guide](./LLM_STUDY_GUIDE.md)** brings the key concepts together in one guided path—from tokens, Transformers, and model training to RAG, agents, evaluation, deployment, and safety. Continue with focused lessons, diagrams, code examples, and hands-on projects.

| Choose your next step | |
|---|---|
| **Read the guide** | [Study all the core concepts in one place](./LLM_STUDY_GUIDE.md) |
| **Browse the course** | [Open the live learning website](https://rdrahul123.github.io/LLMs/) |
| **Start with the basics** | [Introduction to LLMs](./01-introduction/README.md) |
| **Build something real** | [Explore the five course projects](./projects/README.md) |

## What you’ll learn

- **Understand the model:** tokens, embeddings, attention, Transformers, training, and generation.
- **Build with models:** prompts, APIs, retrieval-augmented generation, and fine-tuning.
- **Create capable systems:** tool use, agents, advanced retrieval, and multimodal applications.
- **Ship responsibly:** evaluation, deployment, security, safety, and responsible AI.

Every module includes an overview and a linked visual deep dive. The course website adds guided navigation, search, rendered diagrams, and browser-local reading progress. You can read the lessons without API keys; keys are only needed for examples that use hosted models.

## The 12-module roadmap

| Module | Learn about | Lesson |
|---:|---|---|
| 1 | LLM foundations, training, tokens, and model behavior | [Introduction](./01-introduction/README.md) |
| 2 | Tokenization, embeddings, attention, and Transformer blocks | [Transformer architecture](./02-transformer-architecture/README.md) |
| 3 | Clear instructions, examples, and reliable output formats | [Prompt engineering](./02-prompt-engineering/README.md) |
| 4 | Hosted and local model APIs, chat, and streaming | [LLM APIs](./03-llm-apis/README.md) |
| 5 | Embeddings, vector search, and retrieval-augmented generation | [RAG](./04-rag/README.md) |
| 6 | Dataset preparation, fine-tuning, and model adaptation | [Fine-tuning](./05-fine-tuning/README.md) |
| 7 | Prompt chaining and advanced reasoning strategies | [Advanced prompting](./07-advanced-prompting/README.md) |
| 8 | Tool calling, agent loops, and memory | [AI agents](./08-ai-agents/README.md) |
| 9 | Hybrid and graph search, reranking, and multimodal RAG | [Advanced RAG](./09-advanced-rag/README.md) |
| 10 | Benchmarks, custom evaluation, and human review | [Evaluation](./10-evaluation/README.md) |
| 11 | APIs, streaming, monitoring, cost, and reliability | [Deployment](./11-deployment/README.md) |
| 12 | Alignment, prompt injection, bias, and responsible AI | [Safety & ethics](./12-safety-ethics/README.md) |

> **Note:** The first six lessons retain their original directory names; the table shows the intended course sequence.

## Build your skills with projects

Apply the lessons in five end-to-end projects:

| Project | What you’ll build | Skills |
|---|---|---|
| [RAG Chatbot](./projects/01-rag-chatbot/) | Chat with a collection of PDFs | RAG, ChromaDB, FastAPI |
| [Code Review Agent](./projects/02-code-review-agent/) | Review code with an AI-assisted workflow | Agents, tools, ReAct |
| [ContextIQ — Document Q&A](./projects/03-document-qa/) | A citation-first document question-answering app | RAG, BM25, Streamlit, evaluation |
| [Customer Support Bot](./projects/04-customer-support-bot/) | Automate support responses using intent and knowledge | Fine-tuning, RAG, classification |
| [Research Assistant](./projects/05-research-assistant/) | Analyze and summarize research papers | Agents, extraction, summarization |

See the [project overview](./projects/README.md) for setup and prerequisites.

## Pick a learning path

| Your goal | Recommended path |
|---|---|
| **I’m new to LLMs** | Modules 1–4: learn the foundations, then make your first API-backed application. |
| **I build applications** | Modules 3–7: practice prompting, APIs, RAG, and adaptation. |
| **I work with ML** | Modules 1–2, 5–6, and 9–10: connect model internals to adaptation and measurement. |
| **I’m shipping to production** | Modules 8–12: design controlled agents, evaluate behavior, deploy, and manage risk. |
| **I want the full course** | Follow all 12 modules and finish with a project. |

Browse the [visual diagram gallery](./assets/diagrams/README.md) or look up terminology in the [LLM systems glossary](./GLOSSARY.md).

## Run the course locally

**You need:** Python 3.9 or newer. API keys are optional and only needed for examples that call hosted models. Ollama is optional for running compatible models locally.

```bash
git clone https://github.com/RDrahul123/LLMs.git
cd LLMs

# Build and preview the course website
python -m pip install -r course_site/requirements.txt
python course_site/prepare_docs.py
python -m mkdocs serve
```

Open the local URL printed by MkDocs. To run an individual code example, follow that module or project's README and install only its required dependencies.

### Check the course files

Run the lightweight repository checks without installing the optional ML dependencies:

```bash
python scripts/check_course_docs.py
```

The checker validates local Markdown links, Python syntax, and SVG files. See [`course_site/README.md`](./course_site/README.md) for deployment instructions and [`CONTRIBUTING.md`](./CONTRIBUTING.md) to help improve the course.

## Contributing

Found an issue, have a clearer explanation, or want to add an example? Contributions are welcome. Start with the [contributor guide](./CONTRIBUTING.md), or open an issue to discuss a larger change.

## License

This project is available under the [MIT License](./LICENSE).

<div align="center">

**Learn something useful?** [Explore the course](https://rdrahul123.github.io/LLMs/) · [Star the repository](https://github.com/RDrahul123/LLMs)

</div>
