#!/usr/bin/env python3
"""Research Assistant — Analyze research papers from arXiv or PDF."""

import argparse
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

import fitz  # PyMuPDF
import requests
from openai import OpenAI
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

console = Console()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")

client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

# ---------------------------------------------------------------------------
# Content extraction
# ---------------------------------------------------------------------------

def fetch_arxiv_html(arxiv_id: str) -> str:
    """Fetch the arXiv abstract page as text."""
    url = f"https://arxiv.org/abs/{arxiv_id}"
    console.print(f"[dim]Fetching {url}...[/dim]")
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    return resp.text


def parse_arxiv_id(url_or_id: str) -> str:
    """Extract arXiv ID from a URL or raw ID."""
    url_or_id = url_or_id.strip()
    # Match URLs like https://arxiv.org/abs/2301.07041 or http://arxiv.org/abs/2301.07041v2
    m = re.search(r"arxiv\.org/abs/(\d+\.\d+(?:v\d+)?)", url_or_id)
    if m:
        return m.group(1)
    # Match raw IDs like 2301.07041
    if re.match(r"^\d+\.\d+(?:v\d+)?$", url_or_id):
        return url_or_id
    return url_or_id


def extract_from_arxiv(arxiv_id: str) -> dict:
    """Extract paper metadata and abstract from arXiv."""
    html = fetch_arxiv_html(arxiv_id)

    # Parse title
    title_match = re.search(r'<meta\s+name="citation_title"\s+content="([^"]+)"', html)
    title = title_match.group(1) if title_match else "Unknown Title"

    # Parse authors
    authors = re.findall(r'<meta\s+name="citation_author"\s+content="([^"]+)"', html)

    # Parse abstract
    abstract_match = re.search(
        r'<blockquote\s+class="abstract[^"]*"[^>]*>\s*(?:<span[^>]*>)?(.*?)(?:</span>)?\s*</blockquote>',
        html, re.DOTALL
    )
    abstract = abstract_match.group(1).strip() if abstract_match else ""
    abstract = re.sub(r"<[^>]+>", "", abstract).strip()  # strip HTML tags
    abstract = re.sub(r"^\s*Abstract:\s*", "", abstract)

    # Parse date
    date_match = re.search(r'<meta\s+name="citation_date"\s+content="([^"]+)"', html)
    date = date_match.group(1) if date_match else "Unknown"

    # Parse categories
    categories = re.findall(r'<meta\s+name="citation_keywords"\s+content="([^"]+)"', html)

    return {
        "arxiv_id": arxiv_id,
        "title": title,
        "authors": authors,
        "abstract": abstract,
        "date": date,
        "categories": categories,
    }


def extract_from_pdf(pdf_path: str) -> dict:
    """Extract text from a PDF file."""
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    doc = fitz.open(str(path))
    pages = []
    for page in doc:
        pages.append(page.get_text())
    doc.close()
    full_text = "\n".join(pages)

    # Try to extract title (first line usually)
    lines = [l.strip() for l in full_text.split("\n") if l.strip()]
    title = lines[0] if lines else path.stem

    return {
        "arxiv_id": None,
        "title": title,
        "authors": [],
        "abstract": "",  # Will be extracted by LLM
        "date": "Unknown",
        "categories": [],
        "full_text": full_text[:15000],  # limit for token budget
    }

# ---------------------------------------------------------------------------
# LLM Analysis
# ---------------------------------------------------------------------------

ANALYSIS_PROMPT = """\
You are a research paper analyst. Analyze the following paper and produce a structured report.

Paper information:
- Title: {title}
- Authors: {authors}
- Date: {date}
- Categories: {categories}
- arXiv ID: {arxiv_id}

{content_section}

Produce a Markdown report with these exact sections:

## Summary
A 2-3 paragraph summary of the paper's main ideas and contributions.

## Key Contributions
A numbered list of the paper's main contributions (3-5 items).

## Methodology
A description of the methods, models, or approaches used.

## Key Findings
Bullet points of the main results and findings.

## Limitations
Bullet points of limitations acknowledged by the authors or identified by you.

## Related Work Suggestions
Suggest 3-5 related papers or research directions the reader should explore.
"""


def analyze_paper(paper: dict) -> str:
    """Use LLM to generate a structured analysis."""
    if not client:
        return _fallback_analysis(paper)

    title = paper.get("title", "Unknown")
    authors = ", ".join(paper.get("authors", [])) or "Unknown"
    date = paper.get("date", "Unknown")
    categories = ", ".join(paper.get("categories", [])) or "N/A"
    arxiv_id = paper.get("arxiv_id") or "N/A"

    if paper.get("full_text"):
        content_section = f"Full text (truncated):\n\n{paper['full_text']}"
    elif paper.get("abstract"):
        content_section = f"Abstract:\n\n{paper['abstract']}"
    else:
        content_section = "No content available."

    prompt = ANALYSIS_PROMPT.format(
        title=title,
        authors=authors,
        date=date,
        categories=categories,
        arxiv_id=arxiv_id,
        content_section=content_section,
    )

    resp = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": "You are a meticulous research paper analyst."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.3,
        max_tokens=2048,
    )
    return resp.choices[0].message.content


def _fallback_analysis(paper: dict) -> str:
    """Generate a basic analysis without LLM."""
    parts = [f"# {paper.get('title', 'Unknown')}\n"]
    if paper.get("authors"):
        parts.append(f"**Authors**: {', '.join(paper['authors'])}\n")
    if paper.get("abstract"):
        parts.append(f"## Abstract\n\n{paper['abstract']}\n")
    if paper.get("full_text"):
        parts.append(f"## Raw Text (first 2000 chars)\n\n{paper['full_text'][:2000]}\n")
    parts.append("\n[Note: Set OPENAI_API_KEY for full LLM-powered analysis]")
    return "\n".join(parts)


def ask_question(paper: dict, question: str, analysis: str, history: list[dict]) -> str:
    """Answer a question about the paper."""
    if not client:
        return "Set OPENAI_API_KEY to enable Q&A."

    paper_context = (
        f"Title: {paper.get('title', 'Unknown')}\n"
        f"Authors: {', '.join(paper.get('authors', []))}\n"
        f"Abstract: {paper.get('abstract', 'N/A')}\n"
        f"Analysis:\n{analysis}"
    )
    if paper.get("full_text"):
        paper_context += f"\n\nFull text (truncated):\n{paper['full_text'][:8000]}"

    messages = [
        {
            "role": "system",
            "content": (
                f"You are a research assistant discussing a paper. Answer questions based on the paper.\n\n"
                f"## Paper Context\n\n{paper_context}"
            ),
        },
        *history[-6:],
        {"role": "user", "content": question},
    ]
    resp = client.chat.completions.create(model=LLM_MODEL, messages=messages, temperature=0.3)
    return resp.choices[0].message.content


def compare_with_paper(paper: dict, other_text: str) -> str:
    """Compare the analyzed paper with another concept or paper."""
    if not client:
        return "Set OPENAI_API_KEY to enable comparison."

    paper_summary = (
        f"Title: {paper.get('title', 'Unknown')}\n"
        f"Abstract: {paper.get('abstract', 'N/A')}"
    )
    resp = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a research analyst. Compare two papers/concepts and identify "
                    "similarities, differences, strengths, and weaknesses.\n\n"
                    f"## Paper A\n\n{paper_summary}"
                ),
            },
            {"role": "user", "content": f"Compare with: {other_text}"},
        ],
        temperature=0.3,
    )
    return resp.choices[0].message.content


def format_report(paper: dict, analysis: str) -> str:
    """Format the full report as Markdown."""
    parts = []
    parts.append(f"# Paper Analysis: {paper.get('title', 'Unknown')}\n")
    parts.append("## Metadata\n")
    parts.append(f"- **Authors**: {', '.join(paper.get('authors', [])) or 'Unknown'}")
    parts.append(f"- **Date**: {paper.get('date', 'Unknown')}")
    if paper.get("arxiv_id"):
        parts.append(f"- **arXiv**: {paper['arxiv_id']}")
        parts.append(f"- **URL**: https://arxiv.org/abs/{paper['arxiv_id']}")
    if paper.get("categories"):
        parts.append(f"- **Categories**: {', '.join(paper['categories'])}")
    parts.append("")
    parts.append(analysis)
    return "\n".join(parts)

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Research Paper Analysis Assistant")
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--arxiv", type=str, help="arXiv URL or ID (e.g., 2301.07041)")
    input_group.add_argument("--pdf", type=str, help="Path to a PDF file")
    parser.add_argument("--interactive", action="store_true", help="Enter interactive Q&A mode")
    parser.add_argument("--output", type=str, default=None, help="Save analysis to Markdown file")

    args = parser.parse_args()

    console.print(Panel("[bold]Research Paper Assistant[/bold]", border_style="blue"))

    # Extract content
    if args.arxiv:
        arxiv_id = parse_arxiv_id(args.arxiv)
        paper = extract_from_arxiv(arxiv_id)
    else:
        paper = extract_from_pdf(args.pdf)

    console.print(f"[bold cyan]Title:[/bold cyan] {paper.get('title', 'Unknown')}")
    if paper.get("authors"):
        console.print(f"[dim]Authors: {', '.join(paper['authors'][:5])}{'...' if len(paper['authors']) > 5 else ''}[/dim]")

    # Analyze
    console.print("\n[bold]Analyzing paper...[/bold]")
    analysis = analyze_paper(paper)
    console.print()
    console.print(Markdown(analysis))

    # Save if requested
    if args.output:
        report = format_report(paper, analysis)
        Path(args.output).write_text(report, encoding="utf-8")
        console.print(f"\n[green]Report saved to {args.output}[/green]")

    # Interactive Q&A
    if args.interactive:
        console.print("\n[dim]Interactive mode. Type 'ask <question>', 'compare <text>', 'save', or 'quit'.[/dim]\n")
        history: list[dict] = []

        while True:
            try:
                user_input = console.input("[bold cyan]❓ >[/bold cyan] ").strip()
            except (EOFError, KeyboardInterrupt):
                console.print("\n[dim]Goodbye![/dim]")
                break

            if not user_input:
                continue

            if user_input.lower() == "quit":
                break
            if user_input.lower() == "save":
                report = format_report(paper, analysis)
                out_path = Path(args.output or f"{paper.get('arxiv_id', 'paper')}_analysis.md")
                out_path.write_text(report, encoding="utf-8")
                console.print(f"[green]Saved to {out_path}[/green]")
                continue
            if user_input.lower() == "summary":
                console.print(Markdown(analysis))
                continue

            if user_input.lower().startswith("compare "):
                other = user_input[8:].strip()
                result = compare_with_paper(paper, other)
                console.print()
                console.print(Markdown(result))
                continue

            question = user_input
            if user_input.lower().startswith("ask "):
                question = user_input[4:].strip()

            history.append({"role": "user", "content": question})
            answer = ask_question(paper, question, analysis, history)
            history.append({"role": "assistant", "content": answer})
            console.print()
            console.print(Markdown(answer))
            console.print()


if __name__ == "__main__":
    main()
