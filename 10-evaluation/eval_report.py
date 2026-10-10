"""Evaluation Report Generator.

Load evaluation results, compute statistics, compare models,
identify failures, and generate markdown reports.
"""

import json
import math
import statistics
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional


@dataclass
class EvalResult:
    """A single evaluation result."""
    case_id: str
    model: str
    scores: dict[str, float]
    input_text: str = ""
    output_text: str = ""
    expected_text: str = ""
    metadata: dict = field(default_factory=dict)
    category: str = "default"
    latency_ms: float = 0.0


@dataclass
class ModelSummary:
    """Summary statistics for a model."""
    model_name: str
    total_cases: int
    metric_means: dict[str, float]
    metric_stds: dict[str, float]
    metric_mins: dict[str, float]
    metric_maxs: dict[str, float]
    avg_latency_ms: float
    category_scores: dict[str, dict[str, float]]


class EvalResultLoader:
    """Load evaluation results from various sources."""

    @staticmethod
    def from_json(path: str) -> list[EvalResult]:
        """Load results from JSON file."""
        data = json.loads(Path(path).read_text())
        results = []
        for item in data:
            results.append(EvalResult(
                case_id=item.get("case_id", ""),
                model=item.get("model", "unknown"),
                scores=item.get("scores", {}),
                input_text=item.get("input_text", ""),
                output_text=item.get("output_text", ""),
                expected_text=item.get("expected_text", ""),
                metadata=item.get("metadata", {}),
                category=item.get("category", "default"),
                latency_ms=item.get("latency_ms", 0.0),
            ))
        return results

    @staticmethod
    def from_dict_list(data: list[dict]) -> list[EvalResult]:
        """Load results from list of dicts."""
        results = []
        for item in data:
            results.append(EvalResult(
                case_id=item.get("case_id", ""),
                model=item.get("model", "unknown"),
                scores=item.get("scores", {}),
                input_text=item.get("input_text", ""),
                output_text=item.get("output_text", ""),
                expected_text=item.get("expected_text", ""),
                metadata=item.get("metadata", {}),
                category=item.get("category", "default"),
                latency_ms=item.get("latency_ms", 0.0),
            ))
        return results


class StatisticsCalculator:
    """Compute statistics from evaluation results."""

    @staticmethod
    def compute_summary(results: list[EvalResult]) -> ModelSummary:
        """Compute summary for a single model's results."""
        if not results:
            return ModelSummary(
                model_name="unknown",
                total_cases=0,
                metric_means={},
                metric_stds={},
                metric_mins={},
                metric_maxs={},
                avg_latency_ms=0,
                category_scores={},
            )

        model_name = results[0].model
        metrics = defaultdict(list)
        category_metrics = defaultdict(lambda: defaultdict(list))
        latencies = []

        for r in results:
            for metric, score in r.scores.items():
                metrics[metric].append(score)
                category_metrics[r.category][metric].append(score)
            latencies.append(r.latency_ms)

        metric_means = {m: statistics.mean(s) for m, s in metrics.items()}
        metric_stds = {
            m: statistics.stdev(s) if len(s) > 1 else 0.0
            for m, s in metrics.items()
        }
        metric_mins = {m: min(s) for m, s in metrics.items()}
        metric_maxs = {m: max(s) for m, s in metrics.items()}

        category_scores = {}
        for cat, cat_metrics in category_metrics.items():
            category_scores[cat] = {
                m: statistics.mean(s) for m, s in cat_metrics.items()
            }

        return ModelSummary(
            model_name=model_name,
            total_cases=len(results),
            metric_means=metric_means,
            metric_stds=metric_stds,
            metric_mins=metric_mins,
            metric_maxs=metric_maxs,
            avg_latency_ms=statistics.mean(latencies) if latencies else 0,
            category_scores=category_scores,
        )

    @staticmethod
    def compare_models(summaries: list[ModelSummary]) -> str:
        """Generate a comparison table between models."""
        if not summaries:
            return "No models to compare."

        all_metrics = set()
        for s in summaries:
            all_metrics.update(s.metric_means.keys())
        all_metrics = sorted(all_metrics)

        # Header
        model_names = [s.model_name for s in summaries]
        header = f"{'Metric':<20}"
        for name in model_names:
            header += f" {name:>15}"
        header += f" {'Best':>10}"

        lines = [header, "-" * len(header)]

        # Rows
        for metric in all_metrics:
            row = f"{metric:<20}"
            values = []
            for s in summaries:
                val = s.metric_means.get(metric, 0.0)
                std = s.metric_stds.get(metric, 0.0)
                values.append((s.model_name, val))
                row += f" {val:>7.3f}±{std:.3f}"

            best_model = max(values, key=lambda x: x[1])[0]
            row += f" {best_model:>10}"
            lines.append(row)

        # Latency row
        row = f"{'latency_ms':<20}"
        for s in summaries:
            row += f" {s.avg_latency_ms:>15.0f}"
        lines.append(row)

        # Total cases
        row = f"{'total_cases':<20}"
        for s in summaries:
            row += f" {s.total_cases:>15}"
        lines.append(row)

        return "\n".join(lines)


class FailureAnalyzer:
    """Identify and analyze failure cases."""

    def __init__(self, results: list[EvalResult]):
        self.results = results

    def find_failures(
        self,
        metric: str,
        threshold: float = 0.5,
        model: Optional[str] = None,
    ) -> list[EvalResult]:
        """Find cases where a metric is below threshold."""
        failures = []
        for r in self.results:
            if model and r.model != model:
                continue
            score = r.scores.get(metric, 1.0)
            if score < threshold:
                failures.append(r)
        return failures

    def find_worst_categories(
        self, metric: str, model: Optional[str] = None
    ) -> list[tuple[str, float]]:
        """Find categories with lowest average scores."""
        category_scores = defaultdict(list)
        for r in self.results:
            if model and r.model != model:
                continue
            score = r.scores.get(metric, 0.0)
            category_scores[r.category].append(score)

        avgs = [
            (cat, statistics.mean(scores))
            for cat, scores in category_scores.items()
        ]
        return sorted(avgs, key=lambda x: x[1])

    def failure_summary(self, metric: str, top_n: int = 10) -> str:
        """Generate a failure summary report."""
        lines = [f"Failure Analysis for '{metric}'", "=" * 50, ""]

        models = set(r.model for r in self.results)
        for model in sorted(models):
            failures = self.find_failures(metric, 0.5, model)
            lines.append(f"Model: {model}")
            lines.append(f"  Failures: {len(failures)} / {sum(1 for r in self.results if r.model == model)}")
            lines.append(f"  Failure Rate: {len(failures) / max(1, sum(1 for r in self.results if r.model == model)):.1%}")

            worst_cats = self.find_worst_categories(metric, model)
            if worst_cats:
                lines.append("  Worst Categories:")
                for cat, avg in worst_cats[:3]:
                    lines.append(f"    {cat}: {avg:.3f}")
            lines.append("")

        return "\n".join(lines)


class ReportGenerator:
    """Generate markdown evaluation reports."""

    def __init__(self, results: list[EvalResult]):
        self.results = results
        self.loader = EvalResultLoader()
        self.stats = StatisticsCalculator()
        self.failure_analyzer = FailureAnalyzer(results)

    def generate_summary(self) -> str:
        """Generate summary statistics."""
        models = defaultdict(list)
        for r in self.results:
            models[r.model].append(r)

        lines = ["# Evaluation Summary", ""]
        lines.append(f"**Total Results**: {len(self.results)}")
        lines.append(f"**Models Evaluated**: {len(models)}")
        lines.append(f"**Generated**: {datetime.now().isoformat()}")
        lines.append("")

        for model_name, model_results in sorted(models.items()):
            summary = self.stats.compute_summary(model_results)
            lines.append(f"## {model_name}")
            lines.append(f"- Total Cases: {summary.total_cases}")
            lines.append(f"- Avg Latency: {summary.avg_latency_ms:.0f}ms")
            lines.append("")
            lines.append("| Metric | Mean | Std | Min | Max |")
            lines.append("|--------|------|-----|-----|-----|")
            for metric in sorted(summary.metric_means.keys()):
                lines.append(
                    f"| {metric} "
                    f"| {summary.metric_means[metric]:.3f} "
                    f"| {summary.metric_stds[metric]:.3f} "
                    f"| {summary.metric_mins[metric]:.3f} "
                    f"| {summary.metric_maxs[metric]:.3f} |"
                )
            lines.append("")

        return "\n".join(lines)

    def generate_comparison(self) -> str:
        """Generate model comparison table."""
        models = defaultdict(list)
        for r in self.results:
            models[r.model].append(r)

        summaries = [
            self.stats.compute_summary(results)
            for results in models.values()
        ]

        lines = ["# Model Comparison", ""]
        lines.append("```")
        lines.append(self.stats.compare_models(summaries))
        lines.append("```")
        lines.append("")

        # Per-category comparison
        all_categories = set()
        for s in summaries:
            all_categories.update(s.category_scores.keys())

        if all_categories:
            lines.append("## Per-Category Comparison")
            lines.append("")

            for category in sorted(all_categories):
                lines.append(f"### {category}")
                lines.append("")
                all_metrics = set()
                for s in summaries:
                    if category in s.category_scores:
                        all_metrics.update(s.category_scores[category].keys())

                header = f"{'Metric':<20}"
                for s in summaries:
                    header += f" {s.model_name:>15}"
                lines.append(header)
                lines.append("-" * len(header))

                for metric in sorted(all_metrics):
                    row = f"{metric:<20}"
                    for s in summaries:
                        val = s.category_scores.get(category, {}).get(metric, 0.0)
                        row += f" {val:>15.3f}"
                    lines.append(row)
                lines.append("")

        return "\n".join(lines)

    def generate_failure_report(self, metric: str = "accuracy") -> str:
        """Generate failure analysis report."""
        return self.failure_analyzer.failure_summary(metric)

    def generate_full_report(self) -> str:
        """Generate complete evaluation report."""
        sections = [
            self.generate_summary(),
            "",
            "---",
            "",
            self.generate_comparison(),
            "",
            "---",
            "",
            self.generate_failure_report(),
        ]
        return "\n".join(sections)

    def save_report(self, path: str, format: str = "markdown"):
        """Save report to file."""
        content = self.generate_full_report()
        Path(path).write_text(content)


class VisualizationHelpers:
    """Helpers for creating text-based visualizations."""

    @staticmethod
    def bar_chart(
        data: dict[str, float],
        title: str = "",
        width: int = 40,
    ) -> str:
        """Create a text-based bar chart."""
        if not data:
            return ""

        max_val = max(data.values()) if data.values() else 1
        lines = []
        if title:
            lines.append(title)
            lines.append("-" * len(title))

        for label, value in data.items():
            bar_len = int((value / max_val) * width) if max_val > 0 else 0
            bar = "#" * bar_len
            lines.append(f"  {label:<20} {bar} {value:.3f}")

        return "\n".join(lines)

    @staticmethod
    def score_distribution(
        scores: list[float],
        bins: int = 10,
        title: str = "",
    ) -> str:
        """Create a text-based histogram of scores."""
        if not scores:
            return ""

        min_s = min(scores)
        max_s = max(scores)
        bin_width = (max_s - min_s) / bins if bins > 0 else 1

        counts = [0] * bins
        for s in scores:
            idx = min(int((s - min_s) / bin_width), bins - 1) if bin_width > 0 else 0
            counts[idx] += 1

        max_count = max(counts) if counts else 1
        lines = []
        if title:
            lines.append(title)
            lines.append("-" * len(title))

        for i in range(bins):
            low = min_s + i * bin_width
            high = low + bin_width
            bar_len = int((counts[i] / max_count) * 30) if max_count > 0 else 0
            bar = "#" * bar_len
            lines.append(f"  [{low:.1f}-{high:.1f}] {bar} {counts[i]}")

        return "\n".join(lines)

    @staticmethod
    def comparison_chart(
        model_scores: dict[str, dict[str, float]],
        metric: str,
    ) -> str:
        """Compare models on a single metric."""
        data = {
            model: scores.get(metric, 0.0)
            for model, scores in model_scores.items()
        }
        return VisualizationHelpers.bar_chart(
            data,
            title=f"Model Comparison: {metric}",
        )

    @staticmethod
    def radar_text(
        metrics: dict[str, float],
        width: int = 30,
    ) -> str:
        """Simple text-based radar/spider chart."""
        lines = ["Radar Chart:"]
        for metric, value in metrics.items():
            filled = int(value * width)
            empty = width - filled
            bar = "#" * filled + "." * empty
            lines.append(f"  {metric:<15} [{bar}] {value:.3f}")
        return "\n".join(lines)


def demo():
    """Demo the evaluation report generator."""
    print("Evaluation Report Generator Demo")
    print("=" * 50)

    # Create sample results
    results = []
    models = ["gpt-4", "gpt-3.5-turbo", "claude-3-opus"]
    categories = ["factual", "creative", "technical", "conversational"]

    import random
    random.seed(42)

    for model in models:
        base_quality = {"gpt-4": 0.85, "gpt-3.5-turbo": 0.72, "claude-3-opus": 0.88}[model]
        base_latency = {"gpt-4": 800, "gpt-3.5-turbo": 400, "claude-3-opus": 900}[model]

        for i in range(20):
            category = random.choice(categories)
            quality_adj = {"factual": 0.05, "creative": -0.05, "technical": 0.02, "conversational": 0.0}[category]
            results.append(EvalResult(
                case_id=f"{model}-case-{i:03d}",
                model=model,
                scores={
                    "accuracy": min(1.0, max(0.0, base_quality + quality_adj + random.gauss(0, 0.1))),
                    "fluency": min(1.0, max(0.0, base_quality + 0.05 + random.gauss(0, 0.08))),
                    "relevance": min(1.0, max(0.0, base_quality - 0.02 + random.gauss(0, 0.1))),
                },
                input_text=f"Sample input {i}",
                output_text=f"Sample output {i}",
                expected_text=f"Expected output {i}",
                category=category,
                latency_ms=base_latency + random.gauss(0, 100),
            ))

    # Generate report
    generator = ReportGenerator(results)

    print("\n--- Summary ---")
    print(generator.generate_summary()[:800] + "...")

    print("\n--- Comparison ---")
    print(generator.generate_comparison()[:800] + "...")

    print("\n--- Failure Analysis ---")
    print(generator.generate_failure_report("accuracy")[:500] + "...")

    # Visualization demos
    print("\n--- Visualizations ---")
    viz = VisualizationHelpers()

    print(viz.bar_chart(
        {"gpt-4": 0.87, "gpt-3.5": 0.73, "claude-3": 0.89},
        title="Model Accuracy",
    ))

    print()
    print(viz.score_distribution(
        [r.scores["accuracy"] for r in results],
        bins=8,
        title="Accuracy Distribution (All Models)",
    ))

    print()
    print(viz.radar_text({
        "accuracy": 0.85,
        "fluency": 0.88,
        "relevance": 0.82,
        "safety": 0.95,
    }))

    # Save report
    report_path = Path("eval_report.md")
    generator.save_report(str(report_path))
    print(f"\nFull report saved to {report_path}")

    # Export raw results
    raw_data = [
        {
            "case_id": r.case_id,
            "model": r.model,
            "scores": r.scores,
            "category": r.category,
            "latency_ms": r.latency_ms,
        }
        for r in results
    ]
    raw_path = Path("eval_results_raw.json")
    raw_path.write_text(json.dumps(raw_data, indent=2))
    print(f"Raw results saved to {raw_path}")


if __name__ == "__main__":
    demo()
