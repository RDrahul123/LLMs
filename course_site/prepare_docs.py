"""Assemble the existing course content into a clean MkDocs documentation tree."""

from __future__ import annotations

import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BUILD_ROOT = ROOT / ".course-site-build"
DOCS_DIR = BUILD_ROOT / "docs"
MARKER = BUILD_ROOT / ".generated-by-course-site"

MODULE_DIRS = (
    "01-introduction",
    "02-transformer-architecture",
    "02-prompt-engineering",
    "03-llm-apis",
    "04-rag",
    "05-fine-tuning",
    "07-advanced-prompting",
    "08-ai-agents",
    "09-advanced-rag",
    "10-evaluation",
    "11-deployment",
    "12-safety-ethics",
)


def _ignore_generated_and_private(_: str, names: list[str]) -> set[str]:
    """Keep virtual environments, secrets, caches, and model artifacts out of the site."""
    ignored = {
        ".git",
        ".venv",
        "venv",
        "__pycache__",
        ".streamlit",
        "secrets.toml",
        ".env",
    }
    ignored.update(name for name in names if name.endswith((".pyc", ".pyo", ".pt", ".pth", ".bin", ".safetensors")))
    return set(names).intersection(ignored)


def _copy_tree(source_name: str) -> None:
    source = ROOT / source_name
    if not source.is_dir():
        raise FileNotFoundError(f"Expected course content directory: {source}")
    shutil.copytree(
        source,
        DOCS_DIR / source_name,
        dirs_exist_ok=True,
        ignore=_ignore_generated_and_private,
    )


def _write_course_pages() -> None:
    (DOCS_DIR / "index.md").write_text(
        """# Learn LLMs, from first principles to production

<div class="course-hero">
<p class="course-eyebrow">A free, practical, visual course</p>
<h1>Understand the model.<br>Build the system.</h1>
<p class="course-lead">Work through 12 guided modules, learn from interactive diagrams and runnable examples, then apply the ideas in five end-to-end projects.</p>
<p><a class="md-button md-button--primary" href="01-introduction/">Start Module 1</a> <a class="md-button" href="learning-paths/">Choose a learning path</a></p>
<div class="course-hero-stats"><span><strong>12</strong> modules</span><span><strong>5</strong> projects</span><span><strong>All levels</strong> beginner → production</span></div>
</div>

## Learn by building

The course moves from how language models work to how to build reliable applications around them. Each module overview links to a deeper lesson with diagrams, worked examples, common failure modes, and questions to check your understanding.

<div class="grid cards" markdown>

-   :material-brain: **Foundations**

    Understand tokens, training, model behavior, and Transformer architecture.

    **Modules 1–2** · [Start learning](01-introduction/README.md)

-   :material-message-text: **Build with models**

    Design prompts, use model APIs, connect private knowledge, and adapt models.

    **Modules 3–6** · [Explore the build path](02-prompt-engineering/README.md)

-   :material-robot: **Advanced systems**

    Compose reasoning strategies, tool-using agents, and production retrieval.

    **Modules 7–9** · [Explore advanced systems](07-advanced-prompting/README.md)

-   :material-shield-check: **Ship responsibly**

    Evaluate quality, deploy with operational controls, and reason about safety.

    **Modules 10–12** · [Explore production topics](10-evaluation/README.md)

</div>

## A practical learning sequence

```mermaid
flowchart LR
    A[Learn the model<br/>1–2] --> B[Build the interface<br/>3–4]
    B --> C[Connect knowledge<br/>5–6]
    C --> D[Compose systems<br/>7–9]
    D --> E[Evaluate and operate<br/>10–11]
    E --> F[Protect users and data<br/>12]
    F --> P[Build a project]
```

For one connected reference covering the concepts in sequence, read the [Complete LLM Study Guide](LLM_STUDY_GUIDE.md).

Choose a [learning path](learning-paths.md) that fits your background. You can read the lessons without an API key; provider keys are only needed for examples that call hosted models.

## Put the ideas into practice

Build a [PDF RAG chatbot](projects/01-rag-chatbot/README.md), a [code review agent](projects/02-code-review-agent/README.md), [ContextIQ document Q&A](projects/03-document-qa/README.md), a [customer-support bot](projects/04-customer-support-bot/README.md), or a [research assistant](projects/05-research-assistant/README.md).

## Ready to begin?

Start with **[Module 1: Introduction to LLMs](01-introduction/README.md)**. For the source code, contribution guide, and local examples, visit the [GitHub repository](https://github.com/RDrahul123/LLMs).
""",
        encoding="utf-8",
    )
    (DOCS_DIR / "learning-paths.md").write_text(
        """# Choose your learning path

All paths are self-paced. A lesson can be read without installing dependencies; run the examples when you are ready to experiment.

## New to LLMs

**Goal:** Understand what language models do and make your first API-backed application.

1. [Introduction](01-introduction/README.md) — tokens, training, limitations
2. [Transformer architecture](02-transformer-architecture/README.md) — attention, blocks, generation
3. [Prompt engineering](02-prompt-engineering/README.md) — task design and output contracts
4. [LLM APIs](03-llm-apis/README.md) — requests, streaming, tools, and failures

**Build next:** [RAG Chatbot](projects/01-rag-chatbot/README.md)

## Application developer

**Goal:** Connect an LLM to real data and produce a testable application.

1. Review [prompting](02-prompt-engineering/deep-dive.md) and [API behavior](03-llm-apis/deep-dive.md).
2. Build [basic RAG](04-rag/deep-dive.md) and learn [advanced retrieval](09-advanced-rag/deep-dive.md).
3. Study [evaluation](10-evaluation/deep-dive.md) before comparing models or prompts.
4. Learn [deployment](11-deployment/deep-dive.md) and [safety](12-safety-ethics/deep-dive.md).

**Build next:** [ContextIQ Document Q&A](projects/03-document-qa/README.md)

## ML engineer

**Goal:** Understand model internals, adaptation, and measurement.

1. Study [Transformers](02-transformer-architecture/deep-dive.md) and the [fine-tuning decision guide](05-fine-tuning/deep-dive.md).
2. Compare prompting, retrieval, and adaptation on a fixed evaluation set.
3. Explore [advanced RAG](09-advanced-rag/README.md) and [LLM evaluation](10-evaluation/README.md).
4. Measure latency, cost, quality, and safety together before deployment.

**Build next:** [Research Assistant](projects/05-research-assistant/README.md)

## Production / technical lead

**Goal:** Design controlled, observable systems that can be operated responsibly.

1. Read [agents and tool permissions](08-ai-agents/deep-dive.md).
2. Establish a representative [evaluation suite](10-evaluation/deep-dive.md).
3. Design [service reliability and observability](11-deployment/deep-dive.md).
4. Threat-model [prompt injection, access control, and incident response](12-safety-ethics/deep-dive.md).

**Build next:** [Code Review Agent](projects/02-code-review-agent/README.md)—keep its mock analyzers clearly labeled and human-verify findings.

## Optional setup for code examples

```bash
python -m venv .venv-course
# Windows PowerShell: .venv-course\\Scripts\\Activate.ps1
# macOS/Linux:       source .venv-course/bin/activate
python -m pip install -r requirements.txt
```

The full repository requirements include optional ML and evaluation packages. For the static course site alone, install only `course_site/requirements.txt`. Review each module's requirements before installing GPU-heavy libraries.
""",
        encoding="utf-8",
    )


def prepare_docs() -> Path:
    """Assemble the documentation tree in a marked build directory."""
    if BUILD_ROOT.exists() and not MARKER.exists():
        raise RuntimeError(
            f"Refusing to replace {BUILD_ROOT}: its course-site ownership marker is missing."
        )
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    MARKER.write_text("Generated by course_site/prepare_docs.py\n", encoding="utf-8")

    for module in MODULE_DIRS:
        _copy_tree(module)
    _copy_tree("projects")
    _copy_tree("assets/diagrams")
    shutil.copy2(ROOT / "GLOSSARY.md", DOCS_DIR / "GLOSSARY.md")
    shutil.copy2(ROOT / "LLM_STUDY_GUIDE.md", DOCS_DIR / "LLM_STUDY_GUIDE.md")
    shutil.copy2(ROOT / "course_site" / "brand.svg", DOCS_DIR / "assets" / "course-mark.svg")
    shutil.copytree(ROOT / "course_site" / "static", DOCS_DIR, dirs_exist_ok=True)
    _write_course_pages()
    return DOCS_DIR


if __name__ == "__main__":
    print(f"Prepared course documentation at {prepare_docs()}")
