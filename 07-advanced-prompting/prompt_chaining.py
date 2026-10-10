"""
Module 07: Prompt Chaining Framework
Multi-step prompt pipelines with error handling,
parallel execution, and conditional branching.

Usage:
    python prompt_chaining.py
"""

import os
import json
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def call_llm(
    messages: list[dict],
    model: str = "gpt-4o-mini",
    temperature: float = 0.7,
    max_tokens: int = 1024,
) -> str:
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return resp.choices[0].message.content.strip()


def call_llm_json(
    messages: list[dict],
    model: str = "gpt-4o-mini",
    temperature: float = 0.3,
) -> dict:
    raw = call_llm(messages, model=model, temperature=temperature, max_tokens=2048)
    cleaned = re.sub(r"```json\s*|\s*```", "", raw)
    return json.loads(cleaned)


# ---------------------------------------------------------------------------
# Chain Step
# ---------------------------------------------------------------------------

class StepStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class ChainStep:
    """
    A single step in a prompt chain.

    Args:
        name: Unique identifier for this step.
        prompt_template: Function that receives context dict and returns
            the prompt messages to send to the LLM.
        parser: Optional function that extracts structured data from the
            raw LLM response. If None, raw text is used.
        validator: Optional function that checks if the step output is
            valid. Returns (is_valid, error_message).
        condition: Optional function that receives context dict and
            returns True if this step should run.
        model: LLM model to use for this step.
        temperature: Sampling temperature.
    """
    name: str
    prompt_template: Callable[[dict], list[dict]]
    parser: Optional[Callable[[str], Any]] = None
    validator: Optional[Callable[[Any], tuple[bool, str]]] = None
    condition: Optional[Callable[[dict], bool]] = None
    model: str = "gpt-4o-mini"
    temperature: float = 0.7
    status: StepStatus = field(default=StepStatus.PENDING, repr=False)
    output: Any = field(default=None, repr=False)
    error: Optional[str] = field(default=None, repr=False)
    raw_response: str = field(default="", repr=False)


# ---------------------------------------------------------------------------
# Prompt Chain
# ---------------------------------------------------------------------------

@dataclass
class ChainResult:
    """Result from running a prompt chain."""
    outputs: dict[str, Any] = field(default_factory=dict)
    raw_responses: dict[str, str] = field(default_factory=dict)
    steps_completed: list[str] = field(default_factory=list)
    steps_failed: list[str] = field(default_factory=list)
    steps_skipped: list[str] = field(default_factory=list)
    success: bool = True


class PromptChain:
    """
    A sequential prompt chain that runs steps in order, passing
    outputs as context to subsequent steps.

    Supports:
    - Sequential execution with context passing
    - Error handling between steps
    - Conditional steps
    - Parallel execution for independent steps
    - Step validation and retry
    """

    def __init__(
        self,
        steps: list[ChainStep],
        model: str = "gpt-4o-mini",
        fail_fast: bool = True,
    ):
        """
        Args:
            steps: Ordered list of ChainStep objects.
            model: Default model for steps.
            fail_fast: If True, stop chain on first failure.
        """
        self.steps = steps
        self.default_model = model
        self.fail_fast = fail_fast

    def run(self, initial_context: dict | None = None) -> ChainResult:
        """
        Execute the chain sequentially.

        Args:
            initial_context: Starting context dict. Each step's output
                is added to this dict under its step name.

        Returns:
            ChainResult with all outputs and status information.
        """
        context = initial_context or {}
        result = ChainResult()

        for step in self.steps:
            # Check condition
            if step.condition and not step.condition(context):
                step.status = StepStatus.SKIPPED
                result.steps_skipped.append(step.name)
                continue

            # Run step
            step.status = StepStatus.RUNNING
            print(f"  Running step: {step.name}")

            try:
                messages = step.prompt_template(context)
                model = step.model or self.default_model
                raw = call_llm(messages, model=model, temperature=step.temperature)
                step.raw_response = raw

                # Parse
                if step.parser:
                    output = step.parser(raw)
                else:
                    output = raw

                # Validate
                if step.validator:
                    is_valid, error_msg = step.validator(output)
                    if not is_valid:
                        step.status = StepStatus.FAILED
                        step.error = error_msg
                        result.steps_failed.append(step.name)
                        result.success = False
                        print(f"    Validation failed: {error_msg}")
                        if self.fail_fast:
                            break
                        continue

                step.status = StepStatus.COMPLETED
                step.output = output
                context[step.name] = output
                result.outputs[step.name] = output
                result.raw_responses[step.name] = raw
                result.steps_completed.append(step.name)
                print(f"    Completed: {step.name}")

            except Exception as e:
                step.status = StepStatus.FAILED
                step.error = str(e)
                result.steps_failed.append(step.name)
                result.success = False
                print(f"    Failed: {step.name} - {e}")
                if self.fail_fast:
                    break

        return result

    def run_parallel(self, initial_context: dict | None = None) -> ChainResult:
        """
        Run all independent steps (those with no dependencies) in parallel,
        then run remaining sequential steps.

        Steps are considered independent if their condition function
        returns True (or None) and they don't depend on other step outputs.
        """
        context = initial_context or {}
        result = ChainResult()

        def run_single_step(step):
            if step.condition and not step.condition(context):
                return step, StepStatus.SKIPPED, None, None, None

            messages = step.prompt_template(context)
            model = step.model or self.default_model
            raw = call_llm(messages, model=model, temperature=step.temperature)
            output = step.parser(raw) if step.parser else raw

            if step.validator:
                is_valid, error_msg = step.validator(output)
                if not is_valid:
                    return step, StepStatus.FAILED, output, raw, error_msg

            return step, StepStatus.COMPLETED, output, raw, None

        with ThreadPoolExecutor(max_workers=len(self.steps)) as pool:
            futures = {pool.submit(run_single_step, s): s for s in self.steps}
            for f in as_completed(futures):
                step, status, output, raw, error = f.result()
                step.status = status
                if status == StepStatus.COMPLETED:
                    step.output = output
                    step.raw_response = raw or ""
                    context[step.name] = output
                    result.outputs[step.name] = output
                    result.steps_completed.append(step.name)
                elif status == StepStatus.SKIPPED:
                    result.steps_skipped.append(step.name)
                else:
                    step.error = error
                    result.steps_failed.append(step.name)
                    result.success = False

        return result

    def summary(self) -> str:
        """Print a summary of the chain structure."""
        lines = ["Chain Summary:"]
        for i, step in enumerate(self.steps):
            deps = "conditional" if step.condition else "always"
            lines.append(
                f"  {i+1}. {step.name} [{deps}] - status: {step.status.value}"
            )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Helper: Default text parsers
# ---------------------------------------------------------------------------

def parse_json_array(text: str) -> list:
    """Extract a JSON array from LLM output."""
    cleaned = re.sub(r"```json\s*|\s*```", "", text)
    match = re.search(r"\[.*\]", cleaned, re.DOTALL)
    if match:
        return json.loads(match.group())
    return [text]


def parse_json_object(text: str) -> dict:
    """Extract a JSON object from LLM output."""
    cleaned = re.sub(r"```json\s*|\s*```", "", text)
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if match:
        return json.loads(match.group())
    return {"raw": text}


def parse_numbered_list(text: str) -> list[str]:
    """Parse a numbered list from text."""
    items = re.findall(r"\d+\.\s*(.+)", text)
    return items if items else [line.strip() for line in text.splitlines() if line.strip()]


# ---------------------------------------------------------------------------
# Example: Research Query Chain
# ---------------------------------------------------------------------------

def build_research_chain() -> PromptChain:
    """
    Build a prompt chain that takes a research query and produces
    a structured research summary.

    Steps:
    1. extract_topics: Identify main topics/sub-questions
    2. generate_queries: Create search queries for each topic
    3. synthesize: Combine findings into a summary
    4. format: Produce final formatted output
    """

    def extract_topics_prompt(context):
        query = context.get("query", "")
        return [{"role": "user", "content": (
            f"Break this research query into 3-5 distinct topics or sub-questions.\n"
            f"Return JSON array of objects with 'topic' and 'sub_question'.\n\n"
            f"Query: {query}"
        )}]

    def generate_queries_prompt(context):
        topics = context.get("extract_topics", [])
        topics_text = json.dumps(topics, indent=2) if isinstance(topics, list) else str(topics)
        return [{"role": "user", "content": (
            f"For each of these topics, generate a concise search query.\n\n"
            f"Topics:\n{topics_text}\n\n"
            f"Return JSON array of objects with 'topic', 'search_query', and 'key_points' (array)."
        )}]

    def synthesize_prompt(context):
        topics = context.get("generate_queries", [])
        topics_text = json.dumps(topics, indent=2) if isinstance(topics, list) else str(topics)
        return [{"role": "user", "content": (
            f"Based on these research topics and key points, write a comprehensive synthesis.\n\n"
            f"Topics:\n{topics_text}\n\n"
            f"Include: overview, key findings per topic, and connections between topics."
        )}]

    def format_prompt(context):
        synthesis = context.get("synthesize", "")
        return [{"role": "user", "content": (
            f"Format this research synthesis into a well-structured document with:\n"
            f"- Title\n- Executive Summary (2-3 sentences)\n"
            f"- Key Findings (bullet points)\n- Detailed Sections\n- Conclusion\n\n"
            f"Content:\n{synthesis}"
        )}]

    def validate_topics(output):
        if isinstance(output, list) and len(output) >= 2:
            return True, ""
        return False, "Expected at least 2 topics"

    def validate_queries(output):
        if isinstance(output, list) and len(output) >= 2:
            return True, ""
        return False, "Expected at least 2 search queries"

    return PromptChain(
        steps=[
            ChainStep(
                name="extract_topics",
                prompt_template=extract_topics_prompt,
                parser=parse_json_array,
                validator=validate_topics,
            ),
            ChainStep(
                name="generate_queries",
                prompt_template=generate_queries_prompt,
                parser=parse_json_array,
                validator=validate_queries,
            ),
            ChainStep(
                name="synthesize",
                prompt_template=synthesize_prompt,
            ),
            ChainStep(
                name="format",
                prompt_template=format_prompt,
            ),
        ],
        fail_fast=True,
    )


# ---------------------------------------------------------------------------
# Example: Content Generation Chain with Conditional Branching
# ---------------------------------------------------------------------------

def build_content_chain() -> PromptChain:
    """
    Chain that generates content with conditional formatting
    based on the intended audience.
    """

    def plan_prompt(context):
        topic = context.get("topic", "")
        audience = context.get("audience", "general")
        return [{"role": "user", "content": (
            f"Plan a short article about: {topic}\n"
            f"Target audience: {audience}\n\n"
            f"Return JSON with: 'title', 'outline' (array of section headings), "
            f"'tone' (formal/informal/technical), 'word_count_target' (number)"
        )}]

    def draft_prompt(context):
        plan = context.get("plan", {})
        outline = plan.get("outline", [])
        tone = plan.get("tone", "formal")
        return [{"role": "user", "content": (
            f"Write a detailed draft.\n"
            f"Title: {plan.get('title', '')}\n"
            f"Outline: {json.dumps(outline)}\n"
            f"Tone: {tone}\n"
            f"Word count: ~{plan.get('word_count_target', 300)} words\n\n"
            f"Write the full article content."
        )}]

    def technical_review_prompt(context):
        draft = context.get("draft", "")
        return [{"role": "user", "content": (
            f"Review this draft for technical accuracy and depth.\n\n"
            f"Draft:\n{draft}\n\n"
            f"Suggest 3 specific improvements. Return JSON array of suggestions."
        )}]

    def general_review_prompt(context):
        draft = context.get("draft", "")
        return [{"role": "user", "content": (
            f"Review this draft for clarity, engagement, and readability.\n\n"
            f"Draft:\n{draft}\n\n"
            f"Suggest 3 specific improvements. Return JSON array of suggestions."
        )}]

    def revise_prompt(context):
        draft = context.get("draft", "")
        plan = context.get("plan", {})
        # Pick the appropriate review based on tone
        tone = plan.get("tone", "general")
        review_key = "technical_review" if tone == "technical" else "general_review"
        review = context.get(review_key, [])
        review_text = json.dumps(review, indent=2) if isinstance(review, list) else str(review)
        return [{"role": "user", "content": (
            f"Revise this draft based on the review feedback.\n\n"
            f"Original draft:\n{draft}\n\n"
            f"Review feedback:\n{review_text}\n\n"
            f"Output the revised article."
        )}]

    def is_technical(context):
        plan = context.get("plan", {})
        return plan.get("tone") == "technical"

    def is_not_technical(context):
        plan = context.get("plan", {})
        return plan.get("tone") != "technical"

    return PromptChain(
        steps=[
            ChainStep(
                name="plan",
                prompt_template=plan_prompt,
                parser=parse_json_object,
            ),
            ChainStep(
                name="draft",
                prompt_template=draft_prompt,
            ),
            ChainStep(
                name="technical_review",
                prompt_template=technical_review_prompt,
                parser=parse_json_array,
                condition=is_technical,
            ),
            ChainStep(
                name="general_review",
                prompt_template=general_review_prompt,
                parser=parse_json_array,
                condition=is_not_technical,
            ),
            ChainStep(
                name="revise",
                prompt_template=revise_prompt,
            ),
        ],
        fail_fast=False,
    )


# ---------------------------------------------------------------------------
# Example: Data Extraction Pipeline
# ---------------------------------------------------------------------------

def build_extraction_chain() -> PromptChain:
    """
    Chain that extracts structured data from unstructured text.
    """

    def identify_fields_prompt(context):
        text = context.get("text", "")
        return [{"role": "user", "content": (
            f"Analyze this text and identify the key data fields present.\n\n"
            f"Text: {text[:1000]}\n\n"
            f"Return JSON array of field names (e.g., 'name', 'email', 'date', 'amount')."
        )}]

    def extract_fields_prompt(context):
        text = context.get("text", "")
        fields = context.get("identify_fields", [])
        fields_text = json.dumps(fields) if isinstance(fields, list) else str(fields)
        return [{"role": "user", "content": (
            f"Extract the following fields from the text.\n"
            f"Fields: {fields_text}\n\n"
            f"Text: {text[:1000]}\n\n"
            f"Return JSON object mapping each field to its extracted value. "
            f"Use null for fields not found."
        )}]

    def validate_extraction(output):
        if isinstance(output, dict) and len(output) > 0:
            return True, ""
        return False, "No fields were extracted"

    return PromptChain(
        steps=[
            ChainStep(
                name="identify_fields",
                prompt_template=identify_fields_prompt,
                parser=parse_json_array,
            ),
            ChainStep(
                name="extract_fields",
                prompt_template=extract_fields_prompt,
                parser=parse_json_object,
                validator=validate_extraction,
            ),
        ],
        fail_fast=True,
    )


# ---------------------------------------------------------------------------
# Main: Run all demos
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 70)
    print("  PROMPT CHAINING DEMO")
    print("=" * 70)

    # ---- 1. Research Query Chain ----
    print("\n\n--- 1. Research Query Chain ---")
    research_chain = build_research_chain()
    print(research_chain.summary())
    print()

    research_result = research_chain.run({"query": "Impact of AI on healthcare diagnostics"})
    print(f"\nSteps completed: {research_result.steps_completed}")
    print(f"Steps failed: {research_result.steps_failed}")
    print(f"Success: {research_result.success}")
    if research_result.outputs.get("format"):
        print(f"\nFinal output (first 500 chars):\n{research_result.outputs['format'][:500]}")

    # ---- 2. Content Generation Chain ----
    print("\n\n--- 2. Content Generation Chain (Technical) ---")
    content_chain = build_content_chain()
    print(content_chain.summary())
    print()

    content_result = content_chain.run({
        "topic": "Understanding transformer architecture",
        "audience": "ML engineers",
    })
    print(f"\nSteps completed: {content_result.steps_completed}")
    print(f"Steps skipped: {content_result.steps_skipped}")
    if content_result.outputs.get("revise"):
        print(f"\nFinal article (first 500 chars):\n{content_result.outputs['revise'][:500]}")

    # ---- 3. Data Extraction Chain ----
    print("\n\n--- 3. Data Extraction Chain ---")
    extraction_chain = build_extraction_chain()
    print(extraction_chain.summary())
    print()

    sample_text = (
        "Invoice #12345 dated March 15, 2024 from Acme Corp. "
        "Total amount: $4,567.89. Contact: billing@acme.com. "
        "Payment terms: Net 30. Reference: PO-2024-789."
    )
    extraction_result = extraction_chain.run({"text": sample_text})
    print(f"\nSteps completed: {extraction_result.steps_completed}")
    if extraction_result.outputs.get("extract_fields"):
        print(f"Extracted: {json.dumps(extraction_result.outputs['extract_fields'], indent=2)}")

    # ---- Summary ----
    print("\n" + "=" * 70)
    print("  All prompt chaining demos complete!")
    print("=" * 70)
