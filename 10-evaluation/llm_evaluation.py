"""LLM-as-Judge Evaluation System.

Uses a strong LLM (GPT-4) to evaluate outputs from other models
on multiple criteria with structured scoring and reasoning.
"""

import json
import statistics
import time
from dataclasses import dataclass, field, asdict
from typing import Optional
from pathlib import Path

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


@dataclass
class EvalCriteria:
    """Defines a single evaluation criterion."""
    name: str
    description: str
    scale: tuple[int, int] = (1, 5)
    weight: float = 1.0

    def format_scale(self) -> str:
        return f"{self.scale[0]}-{self.scale[1]}"


@dataclass
class EvalResult:
    """Result of evaluating a single criterion."""
    criteria_name: str
    score: int
    reasoning: str
    latency_ms: float = 0.0


@dataclass
class TestCase:
    """A single test case for evaluation."""
    id: str
    input_text: str
    expected_output: str
    actual_output: str
    metadata: dict = field(default_factory=dict)


@dataclass
class EvaluationReport:
    """Complete evaluation report for a test case."""
    test_case: TestCase
    results: list[EvalResult]
    total_score: float = 0.0
    weighted_score: float = 0.0
    evaluation_time_ms: float = 0.0

    def __post_init__(self):
        if self.results:
            self.total_score = sum(r.score for r in self.results) / len(self.results)
            total_weight = sum(1.0 for _ in self.results)
            self.weighted_score = (
                sum(r.score * 1.0 for r in self.results) / total_weight
                if total_weight > 0 else 0.0
            )


class LLMJudge:
    """Evaluates LLM outputs using another LLM as judge."""

    JUDGE_PROMPT = """You are an expert evaluator assessing the quality of an AI assistant's response.

## Input Query
{input_text}

## Expected Answer (Reference)
{expected_output}

## Actual Answer (To Evaluate)
{actual_output}

## Evaluation Criterion
**{criteria_name}**: {criteria_description}

Rate the actual answer on a scale of {scale_min}-{scale_max}.

Provide your evaluation as JSON:
{{
    "score": <integer between {scale_min} and {scale_max}>,
    "reasoning": "<brief explanation of your rating>"
}}

Be strict and precise. Only give high scores for truly excellent responses."""

    def __init__(
        self,
        model: str = "gpt-4",
        temperature: float = 0.0,
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ):
        if OpenAI is None:
            raise ImportError("Install openai: pip install openai")
        self.client = OpenAI()
        self.model = model
        self.temperature = temperature
        self.max_retries = max_retries
        self.retry_delay = retry_delay

    def _call_judge(self, prompt: str) -> dict:
        """Call the judge model with retry logic."""
        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.temperature,
                    response_format={"type": "json_object"},
                )
                content = response.choices[0].message.content
                return json.loads(content)
            except Exception as e:
                if attempt == self.max_retries - 1:
                    raise
                time.sleep(self.retry_delay * (attempt + 1))
        return {}

    def evaluate_single(
        self, test_case: TestCase, criteria: EvalCriteria
    ) -> EvalResult:
        """Evaluate a test case on a single criterion."""
        prompt = self.JUDGE_PROMPT.format(
            input_text=test_case.input_text,
            expected_output=test_case.expected_output,
            actual_output=test_case.actual_output,
            criteria_name=criteria.name,
            criteria_description=criteria.description,
            scale_min=criteria.scale[0],
            scale_max=criteria.scale[1],
        )

        start = time.time()
        result = self._call_judge(prompt)
        latency_ms = (time.time() - start) * 1000

        score = result.get("score", criteria.scale[0])
        score = max(criteria.scale[0], min(criteria.scale[1], score))

        return EvalResult(
            criteria_name=criteria.name,
            score=score,
            reasoning=result.get("reasoning", "No reasoning provided"),
            latency_ms=latency_ms,
        )

    def evaluate(
        self, test_case: TestCase, criteria_list: list[EvalCriteria]
    ) -> EvaluationReport:
        """Evaluate a test case on all criteria."""
        results = []
        start = time.time()

        for criteria in criteria_list:
            result = self.evaluate_single(test_case, criteria)
            results.append(result)

        total_time = (time.time() - start) * 1000

        report = EvaluationReport(
            test_case=test_case,
            results=results,
            evaluation_time_ms=total_time,
        )
        return report


class BatchEvaluator:
    """Runs evaluation across multiple test cases."""

    def __init__(self, judge: LLMJudge, criteria_list: list[EvalCriteria]):
        self.judge = judge
        self.criteria_list = criteria_list
        self.reports: list[EvaluationReport] = []

    def run(
        self,
        test_cases: list[TestCase],
        progress_callback=None,
    ) -> list[EvaluationReport]:
        """Run evaluation on all test cases."""
        self.reports = []
        total = len(test_cases)

        for i, tc in enumerate(test_cases):
            report = self.judge.evaluate(tc, self.criteria_list)
            self.reports.append(report)

            if progress_callback:
                progress_callback(i + 1, total, report)

        return self.reports

    def aggregate(self) -> dict:
        """Compute aggregate statistics across all evaluations."""
        if not self.reports:
            return {}

        stats = {}
        for criteria in self.criteria_list:
            scores = []
            for report in self.reports:
                for result in report.results:
                    if result.criteria_name == criteria.name:
                        scores.append(result.score)

            if scores:
                stats[criteria.name] = {
                    "mean": statistics.mean(scores),
                    "median": statistics.median(scores),
                    "stdev": statistics.stdev(scores) if len(scores) > 1 else 0.0,
                    "min": min(scores),
                    "max": max(scores),
                    "n": len(scores),
                    "distribution": {
                        str(i): scores.count(i)
                        for i in range(criteria.scale[0], criteria.scale[1] + 1)
                    },
                }

        all_scores = [
            r.score for report in self.reports for r in report.results
        ]
        stats["overall"] = {
            "mean": statistics.mean(all_scores) if all_scores else 0,
            "median": statistics.median(all_scores) if all_scores else 0,
            "stdev": statistics.stdev(all_scores) if len(all_scores) > 1 else 0,
            "total_evaluations": len(all_scores),
            "total_test_cases": len(self.reports),
        }

        return stats

    def generate_text_report(self) -> str:
        """Generate a human-readable text report."""
        stats = self.aggregate()
        lines = ["=" * 60, "LLM Evaluation Report", "=" * 60, ""]

        for criteria_name, s in stats.items():
            if criteria_name == "overall":
                continue
            lines.append(f"Criterion: {criteria_name}")
            lines.append(f"  Mean:   {s['mean']:.2f}")
            lines.append(f"  Median: {s['median']:.2f}")
            lines.append(f"  Std:    {s['stdev']:.2f}")
            lines.append(f"  Range:  [{s['min']}, {s['max']}]")
            lines.append(f"  N:      {s['n']}")
            lines.append("")

        if "overall" in stats:
            o = stats["overall"]
            lines.append("-" * 40)
            lines.append(f"Overall Mean:       {o['mean']:.2f}")
            lines.append(f"Overall Median:     {o['median']:.2f}")
            lines.append(f"Total Evaluations:  {o['total_evaluations']}")
            lines.append(f"Total Test Cases:   {o['total_test_cases']}")

        return "\n".join(lines)

    def generate_json_report(self) -> str:
        """Generate a JSON report."""
        stats = self.aggregate()
        detailed = []

        for report in self.reports:
            detailed.append({
                "test_case_id": report.test_case.id,
                "input": report.test_case.input_text[:200],
                "scores": {
                    r.criteria_name: {
                        "score": r.score,
                        "reasoning": r.reasoning,
                    }
                    for r in report.results
                },
                "total_score": report.total_score,
                "evaluation_time_ms": report.evaluation_time_ms,
            })

        output = {
            "summary": stats,
            "detailed_results": detailed,
        }
        return json.dumps(output, indent=2)

    def find_worst_cases(self, n: int = 5) -> list[EvaluationReport]:
        """Find the N lowest-scoring test cases."""
        sorted_reports = sorted(self.reports, key=lambda r: r.total_score)
        return sorted_reports[:n]

    def find_best_cases(self, n: int = 5) -> list[EvaluationReport]:
        """Find the N highest-scoring test cases."""
        sorted_reports = sorted(self.reports, key=lambda r: r.total_score, reverse=True)
        return sorted_reports[:n]


def demo():
    """Demo the LLM-as-judge evaluation system."""
    print("LLM-as-Judge Evaluation System")
    print("=" * 50)

    criteria = [
        EvalCriteria(
            name="accuracy",
            description="Is the response factually correct and complete?",
            scale=(1, 5),
        ),
        EvalCriteria(
            name="helpfulness",
            description="Does the response directly address the user's question?",
            scale=(1, 5),
        ),
        EvalCriteria(
            name="clarity",
            description="Is the response well-organized and easy to understand?",
            scale=(1, 5),
        ),
        EvalCriteria(
            name="safety",
            description="Does the response avoid harmful, biased, or inappropriate content?",
            scale=(1, 5),
        ),
    ]

    test_cases = [
        TestCase(
            id="tc-001",
            input_text="What is the capital of France?",
            expected_output="The capital of France is Paris.",
            actual_output="Paris is the capital and largest city of France, located in the north-central part of the country along the Seine River.",
        ),
        TestCase(
            id="tc-002",
            input_text="Explain what a neural network is in simple terms.",
            expected_output="A neural network is a computing system inspired by the human brain that learns patterns from data.",
            actual_output="Neural networks are computational architectures consisting of interconnected nodes organized in layers that apply nonlinear transformations to input data through weighted connections optimized via backpropagation.",
        ),
        TestCase(
            id="tc-003",
            input_text="Write a Python function to check if a number is prime.",
            expected_output="def is_prime(n): if n < 2: return False for i in range(2, int(n**0.5)+1): if n % i == 0: return False return True",
            actual_output="def is_prime(n):\n    if n < 2:\n        return False\n    for i in range(2, int(n**0.5) + 1):\n        if n % i == 0:\n            return False\n    return True",
        ),
    ]

    print("\nConfiguration:")
    print(f"  Criteria: {[c.name for c in criteria]}")
    print(f"  Test Cases: {len(test_cases)}")
    print()

    print("Running evaluation (requires OPENAI_API_KEY)...")
    print("(Using simulated results for demo - replace with real API calls)")
    print()

    # Simulate results for demo without requiring API key
    simulated_reports = _simulate_evaluation(test_cases, criteria)
    evaluator = BatchEvaluator(None, criteria)
    evaluator.reports = simulated_reports

    print(evaluator.generate_text_report())
    print("\n\nJSON Report:")
    print(evaluator.generate_json_report()[:500] + "...")

    print("\n\nWorst Cases:")
    for report in evaluator.find_worst_cases(2):
        print(f"  {report.test_case.id}: score={report.total_score:.2f}")

    print("\nBest Cases:")
    for report in evaluator.find_best_cases(2):
        print(f"  {report.test_case.id}: score={report.total_score:.2f}")


def _simulate_evaluation(
    test_cases: list[TestCase], criteria: list[EvalCriteria]
) -> list[EvaluationReport]:
    """Generate simulated evaluation results for demo."""
    import random
    random.seed(42)

    reports = []
    simulated_scores = [
        {"accuracy": 5, "helpfulness": 5, "clarity": 4, "safety": 5},
        {"accuracy": 4, "helpfulness": 3, "clarity": 2, "safety": 5},
        {"accuracy": 5, "helpfulness": 5, "clarity": 5, "safety": 5},
    ]

    for i, tc in enumerate(test_cases):
        results = []
        scores = simulated_scores[i % len(simulated_scores)]
        for c in criteria:
            results.append(EvalResult(
                criteria_name=c.name,
                score=scores.get(c.name, 3),
                reasoning=f"Simulated reasoning for {c.name} on {tc.id}",
                latency_ms=random.uniform(500, 2000),
            ))
        reports.append(EvaluationReport(
            test_case=tc,
            results=results,
            evaluation_time_ms=sum(r.latency_ms for r in results),
        ))

    return reports


if __name__ == "__main__":
    demo()
