"""Custom Evaluation Suite Builder.

Build, run, and manage custom evaluation suites with flexible metric definitions.
Supports loading test cases from JSON/CSV and multiple scoring strategies.
"""

import csv
import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Callable, Any
from abc import ABC, abstractmethod


@dataclass
class EvalCase:
    """A single evaluation case."""
    id: str
    input_text: str
    expected_output: str
    actual_output: str = ""
    metadata: dict = field(default_factory=dict)
    category: str = "default"


class Metric(ABC):
    """Base class for evaluation metrics."""

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def score(self, expected: str, actual: str, case: EvalCase) -> dict:
        """Score a single case. Returns dict with 'score' and optional 'details'."""
        pass


class ExactMatchMetric(Metric):
    """Exact string match after normalization."""

    def __init__(self, case_sensitive: bool = False, strip: bool = True):
        self.case_sensitive = case_sensitive
        self.strip = strip

    @property
    def name(self) -> str:
        return "exact_match"

    def _normalize(self, text: str) -> str:
        result = text
        if self.strip:
            result = result.strip()
        if not self.case_sensitive:
            result = result.lower()
        result = re.sub(r"\s+", " ", result)
        return result

    def score(self, expected: str, actual: str, case: EvalCase) -> dict:
        match = self._normalize(expected) == self._normalize(actual)
        return {"score": 1.0 if match else 0.0, "details": {"match": match}}


class ContainsMetric(Metric):
    """Check if actual output contains the expected substring."""

    def __init__(self, case_sensitive: bool = False):
        self.case_sensitive = case_sensitive

    @property
    def name(self) -> str:
        return "contains"

    def score(self, expected: str, actual: str, case: EvalCase) -> dict:
        exp = expected if self.case_sensitive else expected.lower()
        act = actual if self.case_sensitive else actual.lower()
        contains = exp in act
        return {"score": 1.0 if contains else 0.0, "details": {"contains": contains}}


class RegexMatchMetric(Metric):
    """Check if actual output matches a regex pattern."""

    @property
    def name(self) -> str:
        return "regex_match"

    def score(self, expected: str, actual: str, case: EvalCase) -> dict:
        try:
            match = bool(re.search(expected, actual))
        except re.error:
            match = False
        return {"score": 1.0 if match else 0.0, "details": {"match": match}}


class LLMMetric(Metric):
    """Use an LLM to judge the output quality."""

    def __init__(self, judge_fn: Optional[Callable] = None, rubric: str = ""):
        self.judge_fn = judge_fn
        self.rubric = rubric or "Rate the quality of the response on a scale of 0 to 1."

    @property
    def name(self) -> str:
        return "llm_judge"

    def score(self, expected: str, actual: str, case: EvalCase) -> dict:
        if self.judge_fn is None:
            return {"score": 0.0, "details": {"error": "No judge function provided"}}

        prompt = (
            f"Reference answer: {expected}\n\n"
            f"Generated answer: {actual}\n\n"
            f"{self.rubric}\n"
            f"Respond with JSON: {{\"score\": <float 0-1>, \"reasoning\": \"<str>\"}}"
        )

        try:
            result = self.judge_fn(prompt)
            if isinstance(result, dict):
                return {"score": result.get("score", 0.0), "details": result}
            return {"score": float(result), "details": {}}
        except Exception as e:
            return {"score": 0.0, "details": {"error": str(e)}}


class SemanticSimilarityMetric(Metric):
    """Compute semantic similarity using embeddings."""

    def __init__(self, embed_fn: Optional[Callable] = None):
        self.embed_fn = embed_fn

    @property
    def name(self) -> str:
        return "semantic_similarity"

    def _cosine_similarity(self, a: list[float], b: list[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    def score(self, expected: str, actual: str, case: EvalCase) -> dict:
        if self.embed_fn is None:
            return {"score": 0.0, "details": {"error": "No embedding function provided"}}

        try:
            import math

            emb_expected = self.embed_fn(expected)
            emb_actual = self.embed_fn(actual)
            similarity = self._cosine_similarity(emb_expected, emb_actual)
            similarity = max(0.0, min(1.0, similarity))
            return {"score": similarity, "details": {"similarity": similarity}}
        except Exception as e:
            return {"score": 0.0, "details": {"error": str(e)}}


class CustomMetric(Metric):
    """Wrap any user-defined function as a metric."""

    def __init__(self, name: str, fn: Callable[[str, str, EvalCase], dict]):
        self._name = name
        self.fn = fn

    @property
    def name(self) -> str:
        return self._name

    def score(self, expected: str, actual: str, case: EvalCase) -> dict:
        return self.fn(expected, actual, case)


@dataclass
class EvalCaseResult:
    """Result of evaluating a single case across all metrics."""
    case_id: str
    category: str
    scores: dict[str, float] = field(default_factory=dict)
    details: dict[str, dict] = field(default_factory=dict)
    latency_ms: float = 0.0


class EvalDataset:
    """Load and manage evaluation datasets."""

    @staticmethod
    def from_json(path: str) -> list[EvalCase]:
        """Load eval cases from JSON file."""
        data = json.loads(Path(path).read_text())
        cases = []
        for item in data:
            cases.append(EvalCase(
                id=item.get("id", f"case-{len(cases)}"),
                input_text=item.get("input", item.get("input_text", "")),
                expected_output=item.get("expected", item.get("expected_output", "")),
                actual_output=item.get("actual", item.get("actual_output", "")),
                metadata=item.get("metadata", {}),
                category=item.get("category", "default"),
            ))
        return cases

    @staticmethod
    def from_csv(path: str) -> list[EvalCase]:
        """Load eval cases from CSV file."""
        cases = []
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                cases.append(EvalCase(
                    id=row.get("id", f"case-{i}"),
                    input_text=row.get("input", row.get("input_text", "")),
                    expected_output=row.get("expected", row.get("expected_output", "")),
                    actual_output=row.get("actual", row.get("actual_output", "")),
                    metadata=json.loads(row.get("metadata", "{}")),
                    category=row.get("category", "default"),
                ))
        return cases

    @staticmethod
    def from_dicts(items: list[dict]) -> list[EvalCase]:
        """Load eval cases from list of dicts."""
        cases = []
        for item in items:
            cases.append(EvalCase(
                id=item.get("id", f"case-{len(cases)}"),
                input_text=item.get("input", item.get("input_text", "")),
                expected_output=item.get("expected", item.get("expected_output", "")),
                actual_output=item.get("actual", item.get("actual_output", "")),
                metadata=item.get("metadata", {}),
                category=item.get("category", "default"),
            ))
        return cases

    @staticmethod
    def save_json(cases: list[EvalCase], path: str):
        """Save eval cases to JSON."""
        data = [
            {
                "id": c.id,
                "input": c.input_text,
                "expected": c.expected_output,
                "actual": c.actual_output,
                "metadata": c.metadata,
                "category": c.category,
            }
            for c in cases
        ]
        Path(path).write_text(json.dumps(data, indent=2))


class EvalSuite:
    """Run and manage a complete evaluation suite."""

    def __init__(self, name: str = "default"):
        self.name = name
        self.metrics: list[Metric] = []
        self.cases: list[EvalCase] = []
        self.results: list[EvalCaseResult] = []

    def add_metric(self, metric: Metric):
        self.metrics.append(metric)
        return self

    def add_cases(self, cases: list[EvalCase]):
        self.cases.extend(cases)
        return self

    def load_json(self, path: str):
        self.cases.extend(EvalDataset.from_json(path))
        return self

    def load_csv(self, path: str):
        self.cases.extend(EvalDataset.from_csv(path))
        return self

    def run_all(self) -> list[EvalCaseResult]:
        """Run all metrics on all cases."""
        self.results = []

        for case in self.cases:
            start = time.time()
            case_result = EvalCaseResult(
                case_id=case.id,
                category=case.category,
            )

            for metric in self.metrics:
                metric_result = metric.score(
                    case.expected_output, case.actual_output, case
                )
                case_result.scores[metric.name] = metric_result.get("score", 0.0)
                case_result.details[metric.name] = metric_result.get("details", {})

            case_result.latency_ms = (time.time() - start) * 1000
            self.results.append(case_result)

        return self.results

    def summary(self) -> dict:
        """Compute summary statistics across all results."""
        if not self.results:
            return {}

        summary = {"total_cases": len(self.results), "metrics": {}}

        for metric in self.metrics:
            scores = [r.scores.get(metric.name, 0.0) for r in self.results]
            if scores:
                summary["metrics"][metric.name] = {
                    "mean": sum(scores) / len(scores),
                    "min": min(scores),
                    "max": max(scores),
                    "pass_rate": sum(1 for s in scores if s >= 0.5) / len(scores),
                }

        categories = {}
        for r in self.results:
            cat = r.category
            if cat not in categories:
                categories[cat] = {"count": 0, "scores": {}}
            categories[cat]["count"] += 1
            for metric_name, score in r.scores.items():
                if metric_name not in categories[cat]["scores"]:
                    categories[cat]["scores"][metric_name] = []
                categories[cat]["scores"][metric_name].append(score)

        for cat_data in categories.values():
            cat_data["avg_scores"] = {
                m: sum(s) / len(s) if s else 0
                for m, s in cat_data["scores"].items()
            }
            del cat_data["scores"]

        summary["categories"] = categories
        summary["total_latency_ms"] = sum(r.latency_ms for r in self.results)

        return summary

    def summary_text(self) -> str:
        """Generate human-readable summary."""
        s = self.summary()
        if not s:
            return "No results available."

        lines = [
            f"Evaluation Suite: {self.name}",
            "=" * 50,
            f"Total Cases: {s['total_cases']}",
            f"Total Latency: {s.get('total_latency_ms', 0):.0f}ms",
            "",
            "Metric Scores:",
        ]

        for metric_name, data in s.get("metrics", {}).items():
            lines.append(
                f"  {metric_name}: "
                f"mean={data['mean']:.3f}, "
                f"min={data['min']:.3f}, "
                f"max={data['max']:.3f}, "
                f"pass_rate={data['pass_rate']:.1%}"
            )

        if s.get("categories"):
            lines.extend(["", "Per-Category Breakdown:"])
            for cat, data in sorted(s["categories"].items()):
                lines.append(f"  {cat} (n={data['count']}):")
                for m, avg in data.get("avg_scores", {}).items():
                    lines.append(f"    {m}: {avg:.3f}")

        return "\n".join(lines)

    def export(self, path: str):
        """Export results to JSON."""
        data = {
            "suite_name": self.name,
            "summary": self.summary(),
            "results": [
                {
                    "case_id": r.case_id,
                    "category": r.category,
                    "scores": r.scores,
                    "latency_ms": r.latency_ms,
                }
                for r in self.results
            ],
        }
        Path(path).write_text(json.dumps(data, indent=2))

    def failure_cases(self, metric_name: str, threshold: float = 0.5) -> list[EvalCaseResult]:
        """Find cases that scored below threshold on a metric."""
        return [
            r for r in self.results
            if r.scores.get(metric_name, 1.0) < threshold
        ]


def demo():
    """Demo custom evaluation suite for a summarization task."""
    print("Custom Evaluation Suite Demo")
    print("=" * 50)
    print("Task: Evaluating text summarization quality\n")

    # Define metrics
    exact = ExactMatchMetric(case_sensitive=False)
    contains = ContainsMetric()
    regex = RegexMatchMetric()
    custom_len = CustomMetric(
        "length_ratio",
        lambda exp, act, case: {
            "score": min(len(act) / max(len(exp), 1), 2.0) / 2.0,
            "details": {"expected_len": len(exp), "actual_len": len(act)},
        },
    )

    # Build test cases
    cases = [
        EvalCase(
            id="sum-001",
            input_text="The quick brown fox jumps over the lazy dog.",
            expected_output="A fox jumps over a dog.",
            actual_output="The fox jumped over the lazy dog.",
            category="simple",
        ),
        EvalCase(
            id="sum-002",
            input_text="Python is a high-level programming language created by Guido van Rossum. It emphasizes code readability and supports multiple programming paradigms.",
            expected_output="Python is a readable, multi-paradigm language by van Rossum.",
            actual_output="Python is a programming language known for its readability.",
            category="technical",
        ),
        EvalCase(
            id="sum-003",
            input_text="Machine learning is a subset of artificial intelligence that enables systems to learn from data.",
            expected_output="ML is an AI subset for learning from data.",
            actual_output="Machine learning is a branch of AI that learns from data.",
            category="technical",
        ),
        EvalCase(
            id="sum-004",
            input_text="The Earth orbits the Sun at approximately 107,000 kilometers per hour.",
            expected_output="Earth orbits the Sun at 107000 km/h.",
            actual_output="The planet travels around the sun at high speed.",
            category="factual",
        ),
        EvalCase(
            id="sum-005",
            input_text="Photosynthesis converts sunlight into chemical energy in plants.",
            expected_output="Photosynthesis turns sunlight into energy in plants.",
            actual_output="Plants use photosynthesis to convert light into chemical energy.",
            category="factual",
        ),
    ]

    # Build and run eval suite
    suite = EvalSuite(name="Summarization Eval")
    suite.add_metric(exact)
    suite.add_metric(contains)
    suite.add_metric(regex)
    suite.add_metric(custom_len)
    suite.add_cases(cases)
    suite.run_all()

    print(suite.summary_text())

    print("\n\nFailure Cases (exact_match < 0.5):")
    failures = suite.failure_cases("exact_match", 0.5)
    for f in failures:
        print(f"  {f.case_id}: scores={f.scores}")

    # Export results
    export_path = Path("eval_results.json")
    suite.export(str(export_path))
    print(f"\nResults exported to {export_path}")

    # Save/load dataset
    dataset_path = Path("eval_dataset.json")
    EvalDataset.save_json(cases, str(dataset_path))
    loaded = EvalDataset.from_json(str(dataset_path))
    print(f"Saved and reloaded {len(loaded)} cases from {dataset_path}")


if __name__ == "__main__":
    demo()
