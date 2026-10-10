"""Human Evaluation Framework.

Provides tools for collecting, aggregating, and analyzing human evaluations
with inter-annotator agreement metrics.
"""

import json
import math
import random
import statistics
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class EvalCriteria:
    """Definition of an evaluation criterion."""
    name: str
    description: str
    scale: list[str] = field(default_factory=lambda: ["1", "2", "3", "4", "5"])
    scale_labels: dict[int, str] = field(default_factory=dict)

    def __post_init__(self):
        if not self.scale_labels and self.scale:
            self.scale_labels = {
                i + 1: label for i, label in enumerate(self.scale)
            }


@dataclass
class EvalTemplate:
    """Template for a human evaluation session."""
    task_description: str
    criteria: list[EvalCriteria]
    examples: list[dict] = field(default_factory=list)
    output_format: str = "json"

    def format_criteria(self) -> str:
        lines = []
        for c in self.criteria:
            scale_str = " | ".join(
                f"{i+1}={label}" for i, label in enumerate(c.scale)
            )
            lines.append(f"  - {c.name}: {c.description}\n    Scale: [{scale_str}]")
        return "\n".join(lines)


@dataclass
class Annotation:
    """A single annotation from one annotator."""
    annotator_id: str
    case_id: str
    scores: dict[str, int]
    comments: str = ""
    timestamp: str = ""


@dataclass
class EvalItem:
    """An item to be evaluated."""
    id: str
    input_text: str
    output_text: str
    reference_text: str = ""
    metadata: dict = field(default_factory=dict)


def cohens_kappa(annotations_a: list[int], annotations_b: list[int]) -> float:
    """Compute Cohen's kappa for two annotators.

    Args:
        annotations_a: List of ratings from annotator A
        annotations_b: List of ratings from annotator B (same items, same order)

    Returns:
        Cohen's kappa coefficient
    """
    if len(annotations_a) != len(annotations_b):
        raise ValueError("Annotation lists must have the same length")

    n = len(annotations_a)
    if n == 0:
        return 0.0

    categories = sorted(set(annotations_a) | set(annotations_b))

    # Build confusion matrix
    matrix = Counter()
    for a, b in zip(annotations_a, annotations_b):
        matrix[(a, b)] += 1

    # Compute observed agreement
    po = sum(matrix[(c, c)] for c in categories) / n

    # Compute expected agreement
    pe = 0.0
    for c in categories:
        count_a = sum(1 for x in annotations_a if x == c)
        count_b = sum(1 for x in annotations_b if x == c)
        pe += (count_a / n) * (count_b / n)

    if pe == 1.0:
        return 1.0

    return (po - pe) / (1 - pe)


def fleiss_kappa(ratings: list[list[int]], num_categories: int = 5) -> float:
    """Compute Fleiss' kappa for multiple annotators.

    Args:
        ratings: List of lists. Each inner list contains ratings for one item
                 from all annotators. ratings[item][annotator] = category index.
        num_categories: Number of rating categories (e.g., 5 for 1-5 scale)

    Returns:
        Fleiss' kappa coefficient
    """
    n_items = len(ratings)
    if n_items == 0:
        return 0.0

    n_annotators = len(ratings[0])
    if n_annotators < 2:
        return 0.0

    # Ensure all ratings have same number of annotators
    if not all(len(r) == n_annotators for r in ratings):
        raise ValueError("All items must have the same number of annotators")

    total_ratings = n_items * n_annotators

    # Count category assignments per item
    category_counts = []
    for item_ratings in ratings:
        counts = [0] * num_categories
        for r in item_ratings:
            if 0 <= r < num_categories:
                counts[r] += 1
        category_counts.append(counts)

    # Compute P_i (proportion of agreeing pairs for each item)
    p_items = []
    for counts in category_counts:
        n_ij = sum(counts)
        p_i = (sum(c * c for c in counts) - n_ij) / (n_ij * (n_ij - 1)) if n_ij > 1 else 0
        p_items.append(p_i)

    p_bar = statistics.mean(p_items)

    # Compute p_j (proportion of items assigned to each category)
    p_categories = [0.0] * num_categories
    for counts in category_counts:
        for j, c in enumerate(counts):
            p_categories[j] += c
    p_categories = [p / total_ratings for p in p_categories]

    p_e = sum(p * p for p in p_categories)

    if p_e == 1.0:
        return 1.0

    return (p_bar - p_e) / (1 - p_e)


class AnnotationInterface:
    """CLI-based annotation interface."""

    def __init__(self, template: EvalTemplate):
        self.template = template
        self.annotations: list[Annotation] = []

    def display_item(self, item: EvalItem):
        """Display an item for annotation."""
        print("\n" + "=" * 60)
        print(f"Item ID: {item.id}")
        print("-" * 60)
        print(f"Input:\n  {item.input_text}")
        print(f"\nOutput:\n  {item.output_text}")
        if item.reference_text:
            print(f"\nReference:\n  {item.reference_text}")
        print("-" * 60)
        print(f"\nCriteria:\n{self.template.format_criteria()}")

    def get_scores(self, item: EvalItem) -> dict[str, int]:
        """Get scores from user input."""
        scores = {}
        for criterion in self.template.criteria:
            max_val = len(criterion.scale)
            while True:
                try:
                    raw = input(f"  {criterion.name} (1-{max_val}): ").strip()
                    score = int(raw)
                    if 1 <= score <= max_val:
                        scores[criterion.name] = score
                        break
                    print(f"  Please enter a number between 1 and {max_val}")
                except ValueError:
                    print("  Please enter a valid integer")
        return scores

    def annotate_item(self, item: EvalItem, annotator_id: str) -> Annotation:
        """Annotate a single item."""
        self.display_item(item)
        scores = self.get_scores(item)
        comments = input("  Comments (optional): ").strip()

        annotation = Annotation(
            annotator_id=annotator_id,
            case_id=item.id,
            scores=scores,
            comments=comments,
        )
        self.annotations.append(annotation)
        return annotation

    def annotate_batch(
        self, items: list[EvalItem], annotator_id: str
    ) -> list[Annotation]:
        """Annotate a batch of items."""
        annotations = []
        for i, item in enumerate(items):
            print(f"\nAnnotating item {i+1}/{len(items)}")
            ann = self.annotate_item(item, annotator_id)
            annotations.append(ann)
        return annotations


class AgreementAnalyzer:
    """Analyze inter-annotator agreement."""

    def __init__(self, annotations: list[Annotation]):
        self.annotations = annotations

    def _group_by_item(self) -> dict[str, dict[str, Annotation]]:
        """Group annotations by item and annotator."""
        grouped = {}
        for ann in self.annotations:
            if ann.case_id not in grouped:
                grouped[ann.case_id] = {}
            grouped[ann.case_id][ann.annotator_id] = ann
        return grouped

    def _get_aligned_scores(
        self, criterion: str
    ) -> tuple[list[int], list[int], list[str]]:
        """Get aligned scores for two annotators on a criterion."""
        grouped = self._group_by_item()
        annotators = set()
        for item_anns in grouped.values():
            annotators.update(item_anns.keys())

        annotators = sorted(annotators)
        if len(annotators) < 2:
            return [], [], annotators[:2]

        a_id, b_id = annotators[0], annotators[1]
        scores_a, scores_b = [], []
        item_ids = []

        for item_id, item_anns in grouped.items():
            if a_id in item_anns and b_id in item_anns:
                scores_a.append(item_anns[a_id].scores.get(criterion, 0))
                scores_b.append(item_anns[b_id].scores.get(criterion, 0))
                item_ids.append(item_id)

        return scores_a, scores_b, [a_id, b_id]

    def cohens_kappa_for_criterion(self, criterion: str) -> dict:
        """Compute Cohen's kappa for a specific criterion."""
        scores_a, scores_b, annotators = self._get_aligned_scores(criterion)

        if not scores_a:
            return {"criterion": criterion, "kappa": 0.0, "n": 0, "error": "No aligned annotations"}

        kappa = cohens_kappa(scores_a, scores_b)
        return {
            "criterion": criterion,
            "kappa": kappa,
            "n": len(scores_a),
            "annotators": annotators,
            "interpretation": _interpret_kappa(kappa),
        }

    def fleiss_kappa_for_criterion(self, criterion: str) -> dict:
        """Compute Fleiss' kappa for a specific criterion."""
        grouped = self._group_by_item()
        max_categories = 5

        annotators = set()
        for item_anns in grouped.values():
            annotators.update(item_anns.keys())
        annotators = sorted(annotators)

        if len(annotators) < 2:
            return {"criterion": criterion, "kappa": 0.0, "n": 0, "error": "Need at least 2 annotators"}

        ratings = []
        for item_id in sorted(grouped.keys()):
            item_ratings = []
            for ann_id in annotators:
                if ann_id in grouped[item_id]:
                    score = grouped[item_id][ann_id].scores.get(criterion, 0)
                    item_ratings.append(score - 1)
                else:
                    item_ratings.append(0)
            ratings.append(item_ratings)

        kappa = fleiss_kappa(ratings, max_categories)
        return {
            "criterion": criterion,
            "kappa": kappa,
            "n": len(ratings),
            "n_annotators": len(annotators),
            "interpretation": _interpret_kappa(kappa),
        }

    def agreement_report(self) -> str:
        """Generate a full agreement report."""
        grouped = self._group_by_item()
        all_criteria = set()
        for item_anns in grouped.values():
            for ann in item_anns.values():
                all_criteria.update(ann.scores.keys())

        lines = [
            "Inter-Annotator Agreement Report",
            "=" * 50,
            f"Total annotations: {len(self.annotations)}",
            f"Total items: {len(grouped)}",
            "",
        ]

        for criterion in sorted(all_criteria):
            result = self.cohens_kappa_for_criterion(criterion)
            kappa = result["kappa"]
            lines.append(f"Criterion: {criterion}")
            lines.append(f"  Cohen's Kappa: {kappa:.3f} ({result['interpretation']})")
            lines.append(f"  N: {result['n']}")
            lines.append("")

        return "\n".join(lines)


def _interpret_kappa(kappa: float) -> str:
    """Interpret kappa value."""
    if kappa < 0:
        return "Poor (below chance)"
    elif kappa < 0.20:
        return "Slight agreement"
    elif kappa < 0.40:
        return "Fair agreement"
    elif kappa < 0.60:
        return "Moderate agreement"
    elif kappa < 0.80:
        return "Substantial agreement"
    else:
        return "Almost perfect agreement"


class ScoreAggregator:
    """Aggregate scores from multiple annotators."""

    def __init__(self, annotations: list[Annotation]):
        self.annotations = annotations

    def mean_scores(self) -> dict[str, dict[str, float]]:
        """Compute mean scores per item per criterion."""
        grouped = {}
        for ann in self.annotations:
            if ann.case_id not in grouped:
                grouped[ann.case_id] = {}
            for criterion, score in ann.scores.items():
                if criterion not in grouped[ann.case_id]:
                    grouped[ann.case_id][criterion] = []
                grouped[ann.case_id][criterion].append(score)

        result = {}
        for case_id, criteria in grouped.items():
            result[case_id] = {
                c: statistics.mean(scores) for c, scores in criteria.items()
            }
        return result

    def majority_vote(self) -> dict[str, dict[str, int]]:
        """Compute majority vote per item per criterion."""
        grouped = {}
        for ann in self.annotations:
            if ann.case_id not in grouped:
                grouped[ann.case_id] = {}
            for criterion, score in ann.scores.items():
                if criterion not in grouped[ann.case_id]:
                    grouped[ann.case_id][criterion] = []
                grouped[ann.case_id][criterion].append(score)

        result = {}
        for case_id, criteria in grouped.items():
            result[case_id] = {}
            for c, scores in criteria.items():
                counter = Counter(scores)
                result[case_id][c] = counter.most_common(1)[0][0]
        return result

    def disagreeing_items(self) -> list[dict]:
        """Find items where annotators disagree."""
        grouped = {}
        for ann in self.annotations:
            if ann.case_id not in grouped:
                grouped[ann.case_id] = {}
            for criterion, score in ann.scores.items():
                if criterion not in grouped[ann.case_id]:
                    grouped[ann.case_id][criterion] = []
                grouped[ann.case_id][criterion].append(score)

        disagreements = []
        for case_id, criteria in grouped.items():
            for c, scores in criteria.items():
                if len(set(scores)) > 1:
                    disagreements.append({
                        "case_id": case_id,
                        "criterion": c,
                        "scores": scores,
                        "range": max(scores) - min(scores),
                    })
        return sorted(disagreements, key=lambda x: x["range"], reverse=True)


class DisagreementResolver:
    """Resolve disagreements between annotators."""

    def __init__(self, annotations: list[Annotation]):
        self.annotations = annotations
        self.resolutions: list[dict] = []

    def resolve_by_majority(self) -> list[dict]:
        """Resolve all disagreements by majority vote."""
        aggregator = ScoreAggregator(self.annotations)
        return aggregator.disagreeing_items()

    def resolve_by_discussion(
        self, item_id: str, criterion: str, resolution: int
    ) -> Annotation:
        """Record a resolution for a specific disagreement."""
        resolution_data = {
            "case_id": item_id,
            "criterion": criterion,
            "resolution_score": resolution,
            "method": "discussion",
        }
        self.resolutions.append(resolution_data)

        return Annotation(
            annotator_id="resolved",
            case_id=item_id,
            scores={criterion: resolution},
            comments="Resolved by discussion",
        )

    def export_resolutions(self, path: str):
        """Export resolutions to JSON."""
        Path(path).write_text(json.dumps(self.resolutions, indent=2))


def demo():
    """Demo the human evaluation framework."""
    print("Human Evaluation Framework Demo")
    print("=" * 50)

    # Create evaluation template
    template = EvalTemplate(
        task_description="Evaluate the quality of AI-generated summaries.",
        criteria=[
            EvalCriteria(
                name="relevance",
                description="Does the summary capture the main points?",
                scale=["Not relevant", "Slightly", "Moderately", "Mostly", "Fully"],
            ),
            EvalCriteria(
                name="fluency",
                description="Is the summary well-written and readable?",
                scale=["Very poor", "Poor", "OK", "Good", "Excellent"],
            ),
            EvalCriteria(
                name="conciseness",
                description="Is the summary appropriately brief?",
                scale=["Too long", "Long", "OK", "Brief", "Too brief"],
            ),
        ],
    )

    print(f"Task: {template.task_description}")
    print(f"\nCriteria:\n{template.format_criteria()}")

    # Create evaluation items
    items = [
        EvalItem(
            id="sum-001",
            input_text="The quick brown fox jumps over the lazy dog.",
            output_text="A fox jumps over a dog.",
            reference_text="The fox jumps over the lazy dog.",
        ),
        EvalItem(
            id="sum-002",
            input_text="Python is a programming language known for its simplicity.",
            output_text="Python is simple and readable.",
            reference_text="Python emphasizes simplicity and readability.",
        ),
        EvalItem(
            id="sum-003",
            input_text="Climate change is causing global temperatures to rise.",
            output_text="Temperatures are increasing due to climate change.",
            reference_text="Climate change drives rising global temperatures.",
        ),
    ]

    # Simulate annotations from two annotators
    print("\n\nSimulating annotations from 2 annotators...")
    random.seed(42)

    annotations = []
    for item in items:
        for annotator_id in ["annotator_1", "annotator_2"]:
            scores = {}
            for criterion in template.criteria:
                if annotator_id == "annotator_1":
                    scores[criterion.name] = random.randint(3, 5)
                else:
                    scores[criterion.name] = random.randint(2, 5)
            annotations.append(Annotation(
                annotator_id=annotator_id,
                case_id=item.id,
                scores=scores,
            ))

    # Agreement analysis
    print("\n\nInter-Annotator Agreement Analysis")
    print("-" * 40)
    analyzer = AgreementAnalyzer(annotations)
    print(analyzer.agreement_report())

    # Score aggregation
    print("\n\nScore Aggregation")
    print("-" * 40)
    aggregator = ScoreAggregator(annotations)

    print("Mean Scores:")
    means = aggregator.mean_scores()
    for case_id, scores in means.items():
        print(f"  {case_id}: {scores}")

    print("\nMajority Votes:")
    votes = aggregator.majority_vote()
    for case_id, scores in votes.items():
        print(f"  {case_id}: {scores}")

    disagreements = aggregator.disagreeing_items()
    print(f"\nDisagreements Found: {len(disagreements)}")
    for d in disagreements[:3]:
        print(f"  {d['case_id']} on {d['criterion']}: scores={d['scores']}")

    # Disagreement resolution
    print("\n\nDisagreement Resolution")
    print("-" * 40)
    resolver = DisagreementResolver(annotations)
    for d in disagreements[:2]:
        resolved = resolver.resolve_by_discussion(
            d["case_id"], d["criterion"], 3
        )
        print(f"  Resolved {d['case_id']}/{d['criterion']}: {resolved.scores}")

    print("\nDemo complete. In production, use AnnotationInterface for real annotations.")


if __name__ == "__main__":
    demo()
