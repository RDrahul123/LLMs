"""
Cost Optimizer — Token usage optimization and budget management.

Implements:
  - Prompt compression (remove redundancy)
  - Model routing (cheap model for simple tasks, expensive for complex)
  - Token counting with tiktoken
  - Cost estimation per request
  - Budget management
  - Usage analytics and recommendations

Run: python cost_optimizer.py
"""

import json
import logging
import re
import time
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Token Counter (tiktoken)
# ---------------------------------------------------------------------------

try:
    import tiktoken

    def count_tokens(text: str, model: str = "gpt-4o") -> int:
        try:
            encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text))

    def count_message_tokens(messages: list[dict], model: str = "gpt-4o") -> int:
        """Count tokens in a messages list (includes overhead per message)."""
        try:
            encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            encoding = tiktoken.get_encoding("cl100k_base")

        total = 0
        for msg in messages:
            total += 4  # Message overhead
            total += len(encoding.encode(msg.get("role", "")))
            total += len(encoding.encode(msg.get("content", "")))
        total += 2  # Reply priming
        return total

except ImportError:
    def count_tokens(text: str, model: str = "gpt-4o") -> int:
        return len(text.split()) * 4 // 3  # Rough estimate

    def count_message_tokens(messages: list[dict], model: str = "gpt-4o") -> int:
        total = 0
        for msg in messages:
            total += 4 + len(msg.get("content", "").split()) * 4 // 3
        return total + 2


# ---------------------------------------------------------------------------
# Cost Estimator
# ---------------------------------------------------------------------------

MODEL_PRICING = {
    "gpt-4o": {"input": 2.50, "output": 10.00, "context_window": 128000},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60, "context_window": 128000},
    "gpt-4-turbo": {"input": 10.00, "output": 30.00, "context_window": 128000},
    "gpt-3.5-turbo": {"input": 0.50, "output": 1.50, "context_window": 16385},
    "text-embedding-3-small": {"input": 0.02, "output": 0.0, "context_window": 8191},
}


class CostEstimator:
    """Estimate costs for LLM requests."""

    @staticmethod
    def estimate(
        model: str,
        input_tokens: int,
        output_tokens: int,
    ) -> dict:
        pricing = MODEL_PRICING.get(model, {"input": 0.0, "output": 0.0})
        input_cost = (input_tokens / 1_000_000) * pricing["input"]
        output_cost = (output_tokens / 1_000_000) * pricing["output"]

        return {
            "model": model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "input_cost_usd": round(input_cost, 8),
            "output_cost_usd": round(output_cost, 8),
            "total_cost_usd": round(input_cost + output_cost, 8),
        }

    @staticmethod
    def compare_models(
        input_tokens: int,
        output_tokens: int,
        models: list[str] | None = None,
    ) -> list[dict]:
        if models is None:
            models = list(MODEL_PRICING.keys())
        return [
            CostEstimator.estimate(m, input_tokens, output_tokens)
            for m in models
            if "embedding" not in m
        ]


# ---------------------------------------------------------------------------
# Prompt Compressor
# ---------------------------------------------------------------------------


class PromptCompressor:
    """Reduce token count by removing redundancy from prompts."""

    @staticmethod
    def remove_redundant_whitespace(text: str) -> str:
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r" {2,}", " ", text)
        text = re.sub(r"\t+", "\t", text)
        return text.strip()

    @staticmethod
    def remove_filler_phrases(text: str) -> str:
        fillers = [
            r"\bplease\b",
            r"\bkindly\b",
            r"\bthank you\b",
            r"\bthanks\b",
            r"\bI would like you to\b",
            r"\bcan you\b",
            r"\bcould you\b",
            r"\bwould you\b",
            r"\bI need you to\b",
            r"\bhelp me\b",
        ]
        for pattern in fillers:
            text = re.sub(pattern, "", text, flags=re.IGNORECASE)
        return re.sub(r"\s{2,}", " ", text).strip()

    @staticmethod
    def summarize_history(
        messages: list[dict],
        max_history_tokens: int = 500,
        keep_recent: int = 4,
    ) -> list[dict]:
        """
        Compress conversation history by summarizing older messages.
        Keeps system prompt + recent messages, summarizes the rest.
        """
        system_msgs = [m for m in messages if m.get("role") == "system"]
        other_msgs = [m for m in messages if m.get("role") != "system"]

        if len(other_msgs) <= keep_recent:
            return messages

        recent = other_msgs[-keep_recent:]
        older = other_msgs[:-keep_recent]

        older_text = " ".join(m.get("content", "") for m in older)
        if count_tokens([{"role": "user", "content": older_text}]) <= max_history_tokens:
            return messages

        summary = (
            f"Previous conversation summary: The user discussed "
            f"{len(older)} messages covering topics including: "
            + older_text[:200] + "..."
        )

        return system_msgs + [{"role": "user", "content": summary}] + recent

    @staticmethod
    def compress_prompt(
        system_prompt: str,
        user_message: str,
        max_tokens: int = 2000,
    ) -> tuple[str, str]:
        """Compress system prompt and user message to fit token budget."""
        system = PromptCompressor.remove_redundant_whitespace(system_prompt)
        system = PromptCompressor.remove_filler_phrases(system)

        user = PromptCompressor.remove_redundant_whitespace(user_message)
        user = PromptCompressor.remove_filler_phrases(user)

        combined = count_tokens([
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ])

        if combined <= max_tokens:
            return system, user

        # Truncate user message to fit
        ratio = max_tokens / combined
        max_user_tokens = int(max_tokens * ratio * 0.8)
        words = user.split()
        truncated_words = []
        token_count = 0
        for word in words:
            if token_count + len(word) // 3 > max_user_tokens:
                break
            truncated_words.append(word)
            token_count += len(word) // 3

        return system, " ".join(truncated_words) + "..."


# ---------------------------------------------------------------------------
# Model Router
# ---------------------------------------------------------------------------


class TaskComplexity(Enum):
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"


class ModelRouter:
    """Route requests to appropriate models based on task complexity."""

    ROUTING_TABLE = {
        TaskComplexity.SIMPLE: "gpt-4o-mini",
        TaskComplexity.MODERATE: "gpt-4o",
        TaskComplexity.COMPLEX: "gpt-4o",
    }

    SIMPLE_PATTERNS = [
        r"\b(yes|no|true|false)\b",
        r"\b(thank|thanks|ok|okay)\b",
        r"\b(hi|hello|hey)\b",
        r"\b(what is|who is|when)\b.{0,20}\?$",
    ]

    COMPLEX_PATTERNS = [
        r"\b(analyze|compare|evaluate|explain why)\b",
        r"\b(write|create|generate).{0,30}(code|program|function)\b",
        r"\b(step by step|chain of thought|reason)\b",
        r"\b(trade.?offs?|pros and cons|alternatives)\b",
    ]

    @classmethod
    def classify(cls, prompt: str) -> TaskComplexity:
        lower = prompt.lower()

        for pattern in cls.COMPLEX_PATTERNS:
            if re.search(pattern, lower):
                return TaskComplexity.COMPLEX

        for pattern in cls.SIMPLE_PATTERNS:
            if re.search(pattern, lower):
                return TaskComplexity.SIMPLE

        if len(prompt.split()) > 100:
            return TaskComplexity.COMPLEX
        if len(prompt.split()) < 20:
            return TaskComplexity.SIMPLE

        return TaskComplexity.MODERATE

    @classmethod
    def route(cls, prompt: str) -> str:
        complexity = cls.classify(prompt)
        return cls.ROUTING_TABLE[complexity]

    @classmethod
    def route_with_reason(cls, prompt: str) -> dict:
        complexity = cls.classify(prompt)
        model = cls.ROUTING_TABLE[complexity]
        return {
            "model": model,
            "complexity": complexity.value,
            "prompt_length": len(prompt.split()),
            "reason": f"Task classified as {complexity.value} -> {model}",
        }


# ---------------------------------------------------------------------------
# Budget Manager
# ---------------------------------------------------------------------------


@dataclass
class BudgetConfig:
    daily_token_limit: int = 1_000_000
    monthly_token_limit: int = 10_000_000
    daily_cost_limit: float = 10.0
    monthly_cost_limit: float = 100.0
    alert_threshold: float = 0.8  # Alert at 80% usage


class BudgetManager:
    """Manage token and cost budgets with alerts."""

    def __init__(self, config: BudgetConfig | None = None):
        self.config = config or BudgetConfig()
        self.usage: list[dict] = []
        self.alerts: list[dict] = []

    def _get_period_usage(self, period: str = "day") -> dict:
        now = time.time()
        if period == "day":
            cutoff = now - 86400
        elif period == "month":
            cutoff = now - 30 * 86400
        else:
            cutoff = 0

        recent = [u for u in self.usage if u["timestamp"] > cutoff]
        total_tokens = sum(u["tokens"] for u in recent)
        total_cost = sum(u["cost_usd"] for u in recent)
        return {"tokens": total_tokens, "cost_usd": total_cost, "count": len(recent)}

    def can_use(self, estimated_tokens: int, model: str) -> dict:
        daily = self._get_period_usage("day")
        monthly = self._get_period_usage("month")
        cost = CostEstimator.estimate(model, estimated_tokens, 0)

        checks = {
            "daily_tokens": daily["tokens"] + estimated_tokens <= self.config.daily_token_limit,
            "monthly_tokens": monthly["tokens"] + estimated_tokens <= self.config.monthly_token_limit,
            "daily_cost": daily["cost_usd"] + cost["total_cost_usd"] <= self.config.daily_cost_limit,
            "monthly_cost": monthly["cost_usd"] + cost["total_cost_usd"] <= self.config.monthly_cost_limit,
        }

        return {
            "allowed": all(checks.values()),
            "checks": checks,
            "daily_usage": daily,
            "monthly_usage": monthly,
        }

    def record_usage(self, tokens: int, model: str, input_tokens: int = 0, output_tokens: int = 0):
        cost = CostEstimator.estimate(model, input_tokens or tokens // 2, output_tokens or tokens // 2)
        entry = {
            "tokens": tokens,
            "model": model,
            "cost_usd": cost["total_cost_usd"],
            "timestamp": time.time(),
        }
        self.usage.append(entry)

        # Check alerts
        daily = self._get_period_usage("day")
        daily_pct = daily["tokens"] / self.config.daily_token_limit
        if daily_pct >= self.config.alert_threshold:
            self.alerts.append({
                "type": "daily_token_threshold",
                "usage_pct": daily_pct,
                "message": f"Daily token usage at {daily_pct:.0%}",
                "timestamp": time.time(),
            })

        monthly = self._get_period_usage("month")
        monthly_cost_pct = monthly["cost_usd"] / self.config.monthly_cost_limit
        if monthly_cost_pct >= self.config.alert_threshold:
            self.alerts.append({
                "type": "monthly_cost_threshold",
                "usage_pct": monthly_cost_pct,
                "message": f"Monthly cost usage at {monthly_cost_pct:.0%}",
                "timestamp": time.time(),
            })

    def get_usage_report(self) -> dict:
        daily = self._get_period_usage("day")
        monthly = self._get_period_usage("month")
        return {
            "daily": {
                **daily,
                "token_limit": self.config.daily_token_limit,
                "cost_limit": self.config.daily_cost_limit,
                "token_pct": daily["tokens"] / self.config.daily_token_limit,
                "cost_pct": daily["cost_usd"] / self.config.daily_cost_limit,
            },
            "monthly": {
                **monthly,
                "token_limit": self.config.monthly_token_limit,
                "cost_limit": self.config.monthly_cost_limit,
                "token_pct": monthly["tokens"] / self.config.monthly_token_limit,
                "cost_pct": monthly["cost_usd"] / self.config.monthly_cost_limit,
            },
            "alerts": self.alerts[-10:],
        }


# ---------------------------------------------------------------------------
# Usage Analytics
# ---------------------------------------------------------------------------


class UsageAnalytics:
    """Analyze usage patterns and generate recommendations."""

    def __init__(self):
        self.records: list[dict] = []

    def record(self, prompt: str, model: str, input_tokens: int, output_tokens: int, latency_ms: float):
        complexity = ModelRouter.classify(prompt).value
        cost = CostEstimator.estimate(model, input_tokens, output_tokens)
        self.records.append({
            "prompt_length": len(prompt.split()),
            "model": model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "latency_ms": latency_ms,
            "cost_usd": cost["total_cost_usd"],
            "complexity": complexity,
            "timestamp": time.time(),
        })

    def analyze(self) -> dict:
        if not self.records:
            return {"message": "No data available"}

        by_model = defaultdict(lambda: {"count": 0, "total_cost": 0, "total_tokens": 0})
        by_complexity = defaultdict(lambda: {"count": 0, "total_cost": 0})
        total_cost = 0

        for r in self.records:
            by_model[r["model"]]["count"] += 1
            by_model[r["model"]]["total_cost"] += r["cost_usd"]
            by_model[r["model"]]["total_tokens"] += r["input_tokens"] + r["output_tokens"]
            by_complexity[r["complexity"]]["count"] += 1
            by_complexity[r["complexity"]]["total_cost"] += r["cost_usd"]
            total_cost += r["cost_usd"]

        # Find optimization opportunities
        recommendations = []
        for model, data in by_model.items():
            if model == "gpt-4o" and data["count"] > 10:
                simple_count = sum(1 for r in self.records if r["model"] == model and r["complexity"] == "simple")
                if simple_count > 5:
                    savings = simple_count * 0.002
                    recommendations.append(
                        f"Route {simple_count} simple requests from gpt-4o to gpt-4o-mini "
                        f"(estimated savings: ${savings:.4f})"
                    )

        return {
            "total_records": len(self.records),
            "total_cost_usd": round(total_cost, 6),
            "by_model": dict(by_model),
            "by_complexity": dict(by_complexity),
            "recommendations": recommendations,
        }


# ---------------------------------------------------------------------------
# Cost-Optimized Wrapper
# ---------------------------------------------------------------------------


class CostOptimizedLLM:
    """
    Wrapper that applies cost optimization to LLM calls:
      1. Route to cheapest appropriate model
      2. Compress prompts
      3. Enforce budgets
      4. Track usage
    """

    def __init__(self, budget_config: BudgetConfig | None = None):
        self.router = ModelRouter()
        self.compressor = PromptCompressor()
        self.budget = BudgetManager(budget_config)
        self.analytics = UsageAnalytics()
        self.estimator = CostEstimator()

    def prepare_request(
        self,
        messages: list[dict],
        model: str | None = None,
        compress: bool = True,
    ) -> dict:
        """Prepare an optimized request."""
        user_content = " ".join(
            m["content"] for m in messages if m.get("role") == "user"
        )

        # Route model if not specified
        if model is None:
            model = self.router.route(user_content)
            routing = self.router.route_with_reason(user_content)
        else:
            routing = {"model": model, "reason": "User-specified model"}

        # Compress if requested
        compression_savings = 0
        if compress:
            original_tokens = count_message_tokens(messages)
            compressed_messages = self.compressor.summarize_history(messages)
            compressed_tokens = count_message_tokens(compressed_messages)
            compression_savings = original_tokens - compressed_tokens
            messages = compressed_messages

        # Estimate cost
        input_tokens = count_message_tokens(messages, model)
        cost_estimate = self.estimator.estimate(model, input_tokens, 0)

        # Check budget
        budget_check = self.budget.can_use(input_tokens, model)

        return {
            "messages": messages,
            "model": model,
            "routing": routing,
            "input_tokens": input_tokens,
            "cost_estimate": cost_estimate,
            "compression_savings": compression_savings,
            "budget": budget_check,
        }

    def record_result(self, model: str, input_tokens: int, output_tokens: int, latency_ms: float):
        """Record actual usage after request completes."""
        tokens = input_tokens + output_tokens
        cost = self.estimator.estimate(model, input_tokens, output_tokens)
        self.budget.record_usage(tokens, model, input_tokens, output_tokens)
        self.analytics.record("", model, input_tokens, output_tokens, latency_ms)


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------


def demo_token_counting():
    print("\n=== Token Counting ===")
    texts = [
        "Hello, how are you?",
        "Write a comprehensive analysis of the economic implications of artificial intelligence on the global workforce.",
        "def fibonacci(n): return n if n <= 1 else fibonacci(n-1) + fibonacci(n-2)",
    ]
    for text in texts:
        tokens = count_tokens(text)
        print(f"  [{tokens:4d} tokens] {text[:60]}...")


def demo_cost_estimation():
    print("\n=== Cost Estimation ===")
    comparison = CostEstimator.compare_models(input_tokens=1000, output_tokens=500)
    for est in comparison:
        print(f"  {est['model']:20s}: ${est['total_cost_usd']:.6f} "
              f"({est['input_tokens']} in + {est['output_tokens']} out)")


def demo_model_routing():
    print("\n=== Model Routing ===")
    prompts = [
        "Hello",
        "What is the capital of France?",
        "Explain the trade-offs between microservices and monolithic architecture in detail, considering scalability, maintainability, and deployment complexity.",
        "Convert this list to JSON: [1, 2, 3]",
        "Write a step-by-step analysis of why my application is slow, including database queries, network latency, and CPU usage.",
    ]
    for prompt in prompts:
        result = ModelRouter.route_with_reason(prompt)
        print(f"  [{result['complexity']:8s}] -> {result['model']:15s} | {prompt[:50]}...")


def demo_prompt_compression():
    print("\n=== Prompt Compression ===")
    system = "You are a helpful assistant. Please be kind and polite. Thank you for using our service. You are an expert in Python programming."
    user = "Can you please help me write a function? I would like you to create a Python function that sorts a list. Thank you kindly."

    comp = PromptCompressor()
    compressed_system, compressed_user = comp.compress_prompt(system, user, max_tokens=200)

    orig_tokens = count_tokens(system + " " + user)
    comp_tokens = count_tokens(compressed_system + " " + compressed_user)
    print(f"  Original:  {orig_tokens} tokens")
    print(f"  Compressed: {comp_tokens} tokens")
    print(f"  Saved: {orig_tokens - comp_tokens} tokens ({(orig_tokens - comp_tokens) / orig_tokens:.0%})")


def demo_budget():
    print("\n=== Budget Management ===")
    bm = BudgetManager(BudgetConfig(daily_token_limit=5000, daily_cost_limit=0.05))

    for i in range(5):
        check = bm.can_use(1200, "gpt-4o")
        if check["allowed"]:
            bm.record_usage(1200, "gpt-4o")
            print(f"  Request {i + 1}: Allowed (daily: {check['daily_usage']['tokens']} tokens)")
        else:
            print(f"  Request {i + 1}: Denied")
            print(f"    Checks: {check['checks']}")

    report = bm.get_usage_report()
    print(f"  Daily usage: {report['daily']['token_pct']:.0%} of limit")
    if report["alerts"]:
        print(f"  Alerts: {[a['message'] for a in report['alerts']]}")


def demo_analytics():
    print("\n=== Usage Analytics ===")
    analytics = UsageAnalytics()

    import random
    for _ in range(30):
        model = random.choice(["gpt-4o", "gpt-4o-mini"])
        tokens = random.randint(100, 1000)
        analytics.record("test prompt", model, tokens, tokens // 2, random.uniform(200, 1000))

    report = analytics.analyze()
    print(f"  Total records: {report['total_records']}")
    print(f"  Total cost: ${report['total_cost_usd']:.6f}")
    for model, data in report["by_model"].items():
        print(f"  {model}: {data['count']} calls, ${data['total_cost']:.6f}")
    for rec in report["recommendations"]:
        print(f"  Recommendation: {rec}")


def demo_cost_optimized_llm():
    print("\n=== Cost-Optimized LLM Wrapper ===")
    llm = CostOptimizedLLM(BudgetConfig(daily_token_limit=100000))

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello, what is 2 + 2?"},
    ]

    result = llm.prepare_request(messages)
    print(f"  Model: {result['model']}")
    print(f"  Complexity: {result['routing']['complexity']}")
    print(f"  Input tokens: {result['input_tokens']}")
    print(f"  Cost estimate: ${result['cost_estimate']['total_cost_usd']:.6f}")
    print(f"  Budget allowed: {result['budget']['allowed']}")


if __name__ == "__main__":
    demo_token_counting()
    demo_cost_estimation()
    demo_model_routing()
    demo_prompt_compression()
    demo_budget()
    demo_analytics()
    demo_cost_optimized_llm()
    print("\nAll cost optimizer demos complete.")
