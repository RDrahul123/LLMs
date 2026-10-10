"""Standard Benchmark Evaluation.

Run evaluations against established benchmarks: MMLU, HumanEval, GSM8K.
Includes accuracy calculation with confidence intervals.
"""

import json
import math
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Callable


@dataclass
class BenchmarkQuestion:
    """A single benchmark question."""
    id: str
    question: str
    choices: list[str] = field(default_factory=list)
    answer: str = ""
    metadata: dict = field(default_factory=dict)


@dataclass
class BenchmarkResult:
    """Result for a single question."""
    question_id: str
    predicted: str
    expected: str
    correct: bool
    latency_ms: float = 0.0
    tokens_used: int = 0


@dataclass
class BenchmarkReport:
    """Aggregated benchmark results."""
    benchmark_name: str
    total_questions: int
    correct: int
    accuracy: float
    confidence_interval_95: tuple[float, float]
    results: list[BenchmarkResult]
    per_category: dict = field(default_factory=dict)
    avg_latency_ms: float = 0.0
    total_tokens: int = 0

    def summary(self) -> str:
        lines = [
            f"Benchmark: {self.benchmark_name}",
            f"Total:     {self.total_questions}",
            f"Correct:   {self.correct}",
            f"Accuracy:  {self.accuracy:.1%}",
            f"95% CI:    [{self.confidence_interval_95[0]:.1%}, {self.confidence_interval_95[1]:.1%}]",
            f"Avg Latency: {self.avg_latency_ms:.0f}ms",
            f"Total Tokens: {self.total_tokens:,}",
        ]
        if self.per_category:
            lines.append("\nPer-Category Breakdown:")
            for cat, data in sorted(self.per_category.items()):
                lines.append(f"  {cat}: {data['correct']}/{data['total']} ({data['accuracy']:.1%})")
        return "\n".join(lines)


def wilson_score_interval(
    correct: int, total: int, z: float = 1.96
) -> tuple[float, float]:
    """Compute Wilson score confidence interval for a proportion."""
    if total == 0:
        return (0.0, 1.0)

    p = correct / total
    denominator = 1 + z**2 / total
    centre = (p + z**2 / (2 * total)) / denominator
    spread = z * math.sqrt((p * (1 - p) + z**2 / (4 * total)) / total) / denominator

    lower = max(0.0, centre - spread)
    upper = min(1.0, centre + spread)
    return (lower, upper)


class MMLUEvaluator:
    """Evaluate on MMLU-style multiple choice questions."""

    SUBJECTS = [
        "abstract_algebra", "anatomy", "astronomy", "business_ethics",
        "college_biology", "college_chemistry", "college_computer_science",
        "college_mathematics", "college_medicine", "college_physics",
        "computer_security", "conceptual_physics", "econometrics",
        "electrical_engineering", "elementary_mathematics", "formal_logic",
        "global_facts", "high_school_biology", "high_school_chemistry",
        "high_school_computer_science", "high_school_european_history",
        "high_school_geography", "high_school_government_and_politics",
        "high_school_macroeconomics", "high_school_mathematics",
        "high_school_microeconomics", "high_school_physics",
        "high_school_psychology", "high_school_statistics",
        "high_school_us_history", "high_school_world_history",
        "human_aging", "human_law", "international_law", "jurisprudence",
        "logical_fallacies", "machine_learning", "medical_genetics",
        "miscellaneous", "moral_disputes", "moral_scenarios",
        "nutrition", "philosophy", "prehistory", "professional_accounting",
        "professional_law", "professional_medicine", "professional_psychology",
        "public_relations", "security_studies", "sociology",
        "us_foreign_policy", "virology", "world_religions",
    ]

    CHOICES = ["A", "B", "C", "D"]

    def __init__(self, model_fn: Optional[Callable] = None):
        self.model_fn = model_fn or self._default_model

    def _default_model(self, question: str, choices: list[str]) -> str:
        """Default model: random choice (replace with actual model)."""
        return random.choice(self.CHOICES[: len(choices)])

    def format_question(self, question: BenchmarkQuestion) -> str:
        """Format a question for the model."""
        choices_text = "\n".join(
            f"{letter}. {choice}"
            for letter, choice in zip(self.CHOICES, question.choices)
        )
        return f"{question.question}\n\n{choices_text}\n\nAnswer:"

    def evaluate(
        self, questions: list[BenchmarkQuestion]
    ) -> BenchmarkReport:
        """Run evaluation on a list of questions."""
        results = []
        category_stats = {}

        for q in questions:
            start = time.time()
            predicted = self.model_fn(self.format_question(q), q.choices)
            latency = (time.time() - start) * 1000

            correct = predicted.strip().upper() == q.answer.strip().upper()
            results.append(BenchmarkResult(
                question_id=q.id,
                predicted=predicted,
                expected=q.answer,
                correct=correct,
                latency_ms=latency,
            ))

            category = q.metadata.get("subject", "unknown")
            if category not in category_stats:
                category_stats[category] = {"correct": 0, "total": 0}
            category_stats[category]["total"] += 1
            if correct:
                category_stats[category]["correct"] += 1

        total = len(results)
        correct_count = sum(1 for r in results if r.correct)
        accuracy = correct_count / total if total > 0 else 0.0
        ci = wilson_score_interval(correct_count, total)
        avg_latency = sum(r.latency_ms for r in results) / total if total > 0 else 0

        per_cat = {}
        for cat, data in category_stats.items():
            per_cat[cat] = {
                "correct": data["correct"],
                "total": data["total"],
                "accuracy": data["correct"] / data["total"] if data["total"] > 0 else 0,
            }

        return BenchmarkReport(
            benchmark_name="MMLU",
            total_questions=total,
            correct=correct_count,
            accuracy=accuracy,
            confidence_interval_95=ci,
            results=results,
            per_category=per_cat,
            avg_latency_ms=avg_latency,
        )


class HumanEvalEvaluator:
    """Evaluate on HumanEval-style code generation tasks."""

    def __init__(self, model_fn: Optional[Callable] = None):
        self.model_fn = model_fn or self._default_model

    def _default_model(self, prompt: str) -> str:
        """Default model: returns a stub (replace with actual model)."""
        return "def solution():\n    pass"

    def extract_code(self, response: str) -> str:
        """Extract Python code from model response."""
        if "```python" in response:
            parts = response.split("```python")
            if len(parts) > 1:
                code = parts[1].split("```")[0]
                return code.strip()
        if "```" in response:
            parts = response.split("```")
            if len(parts) >= 3:
                return parts[1].strip()
        return response.strip()

    def run_test(self, code: str, test_code: str) -> bool:
        """Run generated code against test cases."""
        namespace = {}
        try:
            exec(code, namespace)
            exec(test_code, namespace)
            return True
        except Exception:
            return False

    def evaluate(
        self,
        problems: list[dict],
        num_samples: int = 1,
    ) -> BenchmarkReport:
        """Evaluate code generation problems.

        Each problem should have:
        - task_id: str
        - prompt: str (function signature + docstring)
        - test: str (assertions to run)
        - entry_point: str (function name)
        """
        results = []

        for problem in problems:
            correct_count = 0
            for _ in range(num_samples):
                start = time.time()
                response = self.model_fn(problem["prompt"])
                latency = (time.time() - start) * 1000

                code = self.extract_code(response)
                full_code = code + "\n" + problem.get("test", "")
                passed = self.run_test(full_code, problem.get("test", ""))

                if passed:
                    correct_count += 1

                results.append(BenchmarkResult(
                    question_id=problem["task_id"],
                    predicted=code[:200],
                    expected=problem.get("entry_point", ""),
                    correct=passed,
                    latency_ms=latency,
                ))

        total = len(results)
        correct_count = sum(1 for r in results if r.correct)
        accuracy = correct_count / total if total > 0 else 0.0
        ci = wilson_score_interval(correct_count, total)
        avg_latency = sum(r.latency_ms for r in results) / total if total > 0 else 0

        return BenchmarkReport(
            benchmark_name="HumanEval",
            total_questions=total,
            correct=correct_count,
            accuracy=accuracy,
            confidence_interval_95=ci,
            results=results,
            avg_latency_ms=avg_latency,
        )

    def pass_at_k(self, n: int, c: int, k: int) -> float:
        """Compute pass@k metric.

        Args:
            n: total attempts
            c: correct attempts
            k: k in pass@k
        """
        if n - c < k:
            return 1.0
        return 1.0 - math.prod(1.0 - k / (n - i) for i in range(c))


class GSM8KEvaluator:
    """Evaluate on GSM8K-style math word problems."""

    def __init__(self, model_fn: Optional[Callable] = None):
        self.model_fn = model_fn or self._default_model

    def _default_model(self, question: str) -> str:
        """Default model: returns random number (replace with actual)."""
        return str(random.randint(0, 1000))

    def extract_answer(self, response: str) -> str:
        """Extract the final numerical answer from model response."""
        # Look for #### pattern (GSM8K format)
        if "####" in response:
            return response.split("####")[-1].strip()

        # Look for the last number in the response
        numbers = []
        for word in response.replace(",", "").split():
            try:
                float(word)
                numbers.append(word)
            except ValueError:
                continue

        return numbers[-1] if numbers else response.strip()

    def normalize_answer(self, answer: str) -> str:
        """Normalize a numerical answer for comparison."""
        answer = answer.strip().replace(",", "").replace("$", "")
        try:
            num = float(answer)
            if num == int(num):
                return str(int(num))
            return f"{num:.2f}"
        except ValueError:
            return answer.strip()

    def evaluate(self, problems: list[dict]) -> BenchmarkReport:
        """Evaluate math problems.

        Each problem should have:
        - question: str
        - answer: str (e.g., "#### 42")
        """
        results = []

        for i, problem in enumerate(problems):
            start = time.time()
            response = self.model_fn(problem["question"])
            latency = (time.time() - start) * 1000

            predicted = self.extract_answer(response)
            expected = self.extract_answer(problem.get("answer", ""))

            correct = self.normalize_answer(predicted) == self.normalize_answer(expected)

            results.append(BenchmarkResult(
                question_id=f"gsm8k-{i}",
                predicted=predicted,
                expected=expected,
                correct=correct,
                latency_ms=latency,
            ))

        total = len(results)
        correct_count = sum(1 for r in results if r.correct)
        accuracy = correct_count / total if total > 0 else 0.0
        ci = wilson_score_interval(correct_count, total)
        avg_latency = sum(r.latency_ms for r in results) / total if total > 0 else 0

        return BenchmarkReport(
            benchmark_name="GSM8K",
            total_questions=total,
            correct=correct_count,
            accuracy=accuracy,
            confidence_interval_95=ci,
            results=results,
            avg_latency_ms=avg_latency,
        )


class BenchmarkSuite:
    """Run and compare results across multiple benchmarks."""

    def __init__(self):
        self.reports: list[BenchmarkReport] = []

    def add_report(self, report: BenchmarkReport):
        self.reports.append(report)

    def comparison_table(self) -> str:
        """Generate a comparison table of all benchmark results."""
        if not self.reports:
            return "No results available."

        lines = [
            f"{'Benchmark':<15} {'Accuracy':>10} {'95% CI':>20} {'Avg Latency':>15} {'N':>8}",
            "-" * 70,
        ]

        for r in self.reports:
            ci_str = f"[{r.confidence_interval_95[0]:.1%}, {r.confidence_interval_95[1]:.1%}]"
            lines.append(
                f"{r.benchmark_name:<15} {r.accuracy:>10.1%} {ci_str:>20} {r.avg_latency_ms:>12.0f}ms {r.total_questions:>8}"
            )

        return "\n".join(lines)

    def export(self, path: str):
        """Export all results to JSON."""
        data = []
        for r in self.reports:
            data.append({
                "benchmark": r.benchmark_name,
                "total": r.total_questions,
                "correct": r.correct,
                "accuracy": r.accuracy,
                "ci_lower": r.confidence_interval_95[0],
                "ci_upper": r.confidence_interval_95[1],
                "avg_latency_ms": r.avg_latency_ms,
                "per_category": r.per_category,
            })
        Path(path).write_text(json.dumps(data, indent=2))


def demo():
    """Demo benchmark evaluation with synthetic data."""
    print("Benchmark Evaluation Suite")
    print("=" * 50)

    # MMLU Demo
    print("\n--- MMLU Evaluation ---")
    mmlu = MMLUEvaluator()
    mmlu_questions = [
        BenchmarkQuestion(
            id="mmlu-001",
            question="What is the capital of France?",
            choices=["London", "Berlin", "Paris", "Madrid"],
            answer="C",
            metadata={"subject": "geography"},
        ),
        BenchmarkQuestion(
            id="mmlu-002",
            question="Which planet is closest to the Sun?",
            choices=["Venus", "Mercury", "Earth", "Mars"],
            answer="B",
            metadata={"subject": "astronomy"},
        ),
        BenchmarkQuestion(
            id="mmlu-003",
            question="What is 2 + 2?",
            choices=["3", "4", "5", "6"],
            answer="B",
            metadata={"subject": "mathematics"},
        ),
    ]
    mmlu_report = mmlu.evaluate(mmlu_questions)
    print(mmlu_report.summary())

    # HumanEval Demo
    print("\n--- HumanEval Evaluation ---")
    humaneval = HumanEvalEvaluator()
    code_problems = [
        {
            "task_id": "HumanEval/0",
            "prompt": "def has_close_elements(numbers, threshold):\n    '''Check if any two numbers are closer than threshold.'''",
            "test": "assert has_close_elements([1.0, 2.0, 3.0], 0.5) == False",
            "entry_point": "has_close_elements",
        },
    ]
    humaneval_report = humaneval.evaluate(code_problems)
    print(humaneval_report.summary())

    # GSM8K Demo
    print("\n--- GSM8K Evaluation ---")
    gsm8k = GSM8KEvaluator()
    math_problems = [
        {
            "question": "Janet's ducks lay 16 eggs per day. She eats 3 for breakfast and bakes 4 into muffins. She sells the rest at $2 per egg. How much does she make per day?",
            "answer": "#### 18",
        },
        {
            "question": "A store has 50 apples. If 20% are red, how many apples are red?",
            "answer": "#### 10",
        },
    ]
    gsm8k_report = gsm8k.evaluate(math_problems)
    print(gsm8k_report.summary())

    # Comparison
    print("\n--- Benchmark Comparison ---")
    suite = BenchmarkSuite()
    suite.add_report(mmlu_report)
    suite.add_report(humaneval_report)
    suite.add_report(gsm8k_report)
    print(suite.comparison_table())

    # Pass@k example
    print("\n--- Pass@k Calculation ---")
    he = HumanEvalEvaluator()
    print(f"pass@1 (10 total, 7 correct): {he.pass_at_k(10, 7, 1):.4f}")
    print(f"pass@5 (10 total, 7 correct): {he.pass_at_k(10, 7, 5):.4f}")
    print(f"pass@10 (10 total, 7 correct): {he.pass_at_k(10, 7, 10):.4f}")


if __name__ == "__main__":
    demo()
