"""
RAG Evaluation with RAGAS

Evaluates RAG pipeline quality using faithfulness, answer relevancy,
context precision, and context recall metrics.
"""

import json
from typing import List, Dict, Optional
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class EvaluationSample:
    question: str
    answer: str
    contexts: List[str]
    ground_truth: str


@dataclass
class EvaluationResult:
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float
    overall_score: float
    details: Dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# 1. Create Evaluation Dataset
# ---------------------------------------------------------------------------

def create_evaluation_dataset() -> List[EvaluationSample]:
    """
    Create a sample evaluation dataset.
    In production, this would come from human annotations.
    """
    return [
        EvaluationSample(
            question="What is the Transformer architecture?",
            answer="The Transformer is a neural network architecture that uses self-attention mechanisms to process sequences in parallel. It was introduced in the paper 'Attention Is All You Need' in 2017.",
            contexts=[
                "The Transformer architecture uses self-attention mechanisms to process sequences in parallel, unlike RNNs which process sequentially.",
                "Transformers were introduced in the 2017 paper 'Attention Is All You Need' by Vaswani et al.",
                "Self-attention allows each token in a sequence to attend to all other tokens, capturing long-range dependencies.",
            ],
            ground_truth="The Transformer is a neural network architecture based on self-attention mechanisms, introduced in 2017, that processes sequences in parallel and captures long-range dependencies.",
        ),
        EvaluationSample(
            question="How does transfer learning work in NLP?",
            answer="Transfer learning in NLP involves pre-training a model on a large corpus and then fine-tuning it on specific downstream tasks.",
            contexts=[
                "Transfer learning reuses pre-trained models for new tasks, reducing training time and data requirements.",
                "BERT and GPT are pre-trained on large text corpora and can be fine-tuned for tasks like classification and question answering.",
                "Fine-tuning adjusts the pre-trained model weights on task-specific data.",
            ],
            ground_truth="Transfer learning in NLP pre-trains models on large corpora then fine-tunes them on specific tasks, leveraging learned representations.",
        ),
        EvaluationSample(
            question="What are the main differences between BERT and GPT?",
            answer="BERT uses bidirectional attention and is pre-trained with masked language modeling, while GPT uses unidirectional (left-to-right) attention and is trained on next token prediction.",
            contexts=[
                "BERT is a bidirectional transformer that uses masked language modeling for pre-training.",
                "GPT models are autoregressive transformers trained to predict the next token in a sequence.",
                "BERT uses encoder-only architecture while GPT uses decoder-only architecture.",
            ],
            ground_truth="BERT is bidirectional (encoder-only) with masked language modeling, GPT is autoregressive (decoder-only) with next token prediction.",
        ),
        EvaluationSample(
            question="What is gradient descent and how does it relate to backpropagation?",
            answer="Gradient descent is an optimization algorithm that minimizes a loss function by iteratively moving in the direction of steepest descent. Backpropagation computes the gradients needed for gradient descent in neural networks.",
            contexts=[
                "Gradient descent is an optimization algorithm that adjusts model parameters to minimize the loss function.",
                "Backpropagation efficiently computes gradients through the chain rule of calculus in neural networks.",
                "SGD, Adam, and RMSprop are variants of gradient descent used in deep learning.",
            ],
            ground_truth="Gradient descent minimizes loss by following negative gradients. Backpropagation computes those gradients efficiently in neural networks using the chain rule.",
        ),
        EvaluationSample(
            question="How do convolutional neural networks process images?",
            answer="CNNs use convolutional layers with learnable filters that slide across the image to detect features like edges, textures, and objects. Pooling layers reduce spatial dimensions.",
            contexts=[
                "Convolutional neural networks use filters/kernels to detect local features in images.",
                "Pooling layers reduce the spatial dimensions of feature maps, providing translation invariance.",
                "CNNs achieve translation invariance through weight sharing in convolutional layers.",
            ],
            ground_truth="CNNs apply learnable filters to detect features, use pooling for dimensionality reduction, and share weights for translation invariance.",
        ),
    ]


# ---------------------------------------------------------------------------
# 2. RAGAS-style Evaluation (simplified implementation)
# ---------------------------------------------------------------------------

class RAGASEvaluator:
    """
    Simplified RAGAS evaluation.

    For production use, install the full RAGAS library:
        pip install ragas datasets

    This implementation provides the core metric logic without external dependencies.
    """

    def __init__(self):
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                from openai import OpenAI
                self._client = OpenAI()
            except Exception:
                return None
        return self._client

    def evaluate(self, samples: List[EvaluationSample]) -> EvaluationResult:
        """Run full RAGAS evaluation on a dataset."""
        faithfulness_scores = []
        relevancy_scores = []
        precision_scores = []
        recall_scores = []

        for sample in samples:
            f = self._evaluate_faithfulness(sample)
            r = self._evaluate_answer_relevancy(sample)
            p = self._evaluate_context_precision(sample)
            rc = self._evaluate_context_recall(sample)

            faithfulness_scores.append(f)
            relevancy_scores.append(r)
            precision_scores.append(p)
            recall_scores.append(rc)

        avg_f = sum(faithfulness_scores) / len(faithfulness_scores) if faithfulness_scores else 0
        avg_r = sum(relevancy_scores) / len(relevancy_scores) if relevancy_scores else 0
        avg_p = sum(precision_scores) / len(precision_scores) if precision_scores else 0
        avg_rc = sum(recall_scores) / len(recall_scores) if recall_scores else 0
        overall = (avg_f + avg_r + avg_p + avg_rc) / 4

        return EvaluationResult(
            faithfulness=round(avg_f, 3),
            answer_relevancy=round(avg_r, 3),
            context_precision=round(avg_p, 3),
            context_recall=round(avg_rc, 3),
            overall_score=round(overall, 3),
            details={
                "num_samples": len(samples),
                "per_sample": {
                    "faithfulness": [round(s, 3) for s in faithfulness_scores],
                    "answer_relevancy": [round(s, 3) for s in relevancy_scores],
                    "context_precision": [round(s, 3) for s in precision_scores],
                    "context_recall": [round(s, 3) for s in recall_scores],
                },
            },
        )

    def _evaluate_faithfulness(self, sample: EvaluationSample) -> float:
        """
        Faithfulness: Are all claims in the answer supported by the context?

        Method: Extract claims from answer, check each against context.
        """
        client = self._get_client()
        if client is None:
            return self._heuristic_faithfulness(sample)

        prompt = f"""Evaluate the faithfulness of this answer to the given context.

Answer: {sample.answer}

Context:
{chr(10).join(sample.contexts)}

Score from 0.0 to 1.0:
- 1.0: Every claim in the answer is fully supported by the context
- 0.7-0.9: Most claims are supported, minor unsupported details
- 0.4-0.6: Some claims are supported, some are not
- 0.0-0.3: Most claims are unsupported or contradicted

Return ONLY a JSON: {{"score": <float>, "reasoning": "<brief>"}}"""

        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
            )
            result = json.loads(response.choices[0].message.content.strip())
            return float(result["score"])
        except Exception:
            return self._heuristic_faithfulness(sample)

    def _evaluate_answer_relevancy(self, sample: EvaluationSample) -> float:
        """
        Answer Relevancy: Does the answer address the question?

        Method: Generate questions from the answer, compare to original.
        """
        client = self._get_client()
        if client is None:
            return self._heuristic_relevancy(sample)

        prompt = f"""Given this answer, generate 3 questions that it directly answers.

Answer: {sample.answer}

Original question: {sample.question}

Return ONLY a JSON: {{"generated_questions": ["q1", "q2", "q3"]}}"""

        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
            )
            result = json.loads(response.choices[0].message.content.strip())
            generated = result["generated_questions"]

            # Compare generated questions to original
            original_words = set(sample.question.lower().split())
            match_scores = []
            for q in generated:
                q_words = set(q.lower().split())
                overlap = len(original_words & q_words) / max(len(original_words), 1)
                match_scores.append(min(overlap * 1.5, 1.0))

            return sum(match_scores) / len(match_scores) if match_scores else 0.5
        except Exception:
            return self._heuristic_relevancy(sample)

    def _evaluate_context_precision(self, sample: EvaluationSample) -> float:
        """
        Context Precision: Are the retrieved contexts relevant to the answer?

        Method: Score each context for relevance to the answer.
        """
        client = self._get_client()
        if client is None:
            return self._heuristic_precision(sample)

        prompt = f"""Evaluate which of these contexts are relevant to the answer.

Answer: {sample.answer}

Contexts:
""" + "\n".join(f"{i+1}. {c}" for i, c in enumerate(sample.contexts)) + """

Return ONLY a JSON: {{"relevant_indices": [<list of 1-indexed relevant context numbers>], "reasoning": "<brief>"}}"""

        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
            )
            result = json.loads(response.choices[0].message.content.strip())
            relevant_count = len(result["relevant_indices"])
            return min(relevant_count / max(len(sample.contexts), 1), 1.0)
        except Exception:
            return self._heuristic_precision(sample)

    def _evaluate_context_recall(self, sample: EvaluationSample) -> float:
        """
        Context Recall: Does the context contain the information needed to answer?

        Method: Check if ground truth claims are covered by context.
        """
        client = self._get_client()
        if client is None:
            return self._heuristic_recall(sample)

        prompt = f"""Evaluate if the context contains the information needed to support this ground truth answer.

Ground Truth: {sample.ground_truth}

Context:
{chr(10).join(sample.contexts)}

Score from 0.0 to 1.0:
- 1.0: All information in ground truth is covered by context
- 0.5-0.9: Most information is covered
- 0.0-0.4: Key information is missing

Return ONLY a JSON: {{"score": <float>, "reasoning": "<brief>"}}"""

        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
            )
            result = json.loads(response.choices[0].message.content.strip())
            return float(result["score"])
        except Exception:
            return self._heuristic_recall(sample)

    # --- Heuristic fallbacks ---

    def _heuristic_faithfulness(self, sample: EvaluationSample) -> float:
        answer_words = set(sample.answer.lower().split())
        context_text = " ".join(sample.contexts).lower()
        context_words = set(context_text.split())
        overlap = len(answer_words & context_words) / max(len(answer_words), 1)
        return min(overlap * 1.2, 1.0)

    def _heuristic_relevancy(self, sample: EvaluationSample) -> float:
        q_words = set(sample.question.lower().split())
        a_words = set(sample.answer.lower().split())
        overlap = len(q_words & a_words) / max(len(q_words), 1)
        return min(overlap * 1.3, 1.0)

    def _heuristic_precision(self, sample: EvaluationSample) -> float:
        answer_words = set(sample.answer.lower().split())
        relevant = 0
        for ctx in sample.contexts:
            ctx_words = set(ctx.lower().split())
            if len(answer_words & ctx_words) > 2:
                relevant += 1
        return relevant / max(len(sample.contexts), 1)

    def _heuristic_recall(self, sample: EvaluationSample) -> float:
        gt_words = set(sample.ground_truth.lower().split())
        ctx_text = " ".join(sample.contexts).lower()
        ctx_words = set(ctx_text.split())
        overlap = len(gt_words & ctx_words) / max(len(gt_words), 1)
        return min(overlap * 1.2, 1.0)


# ---------------------------------------------------------------------------
# 3. Results Interpretation
# ---------------------------------------------------------------------------

def interpret_results(result: EvaluationResult) -> str:
    """Generate human-readable interpretation of evaluation results."""
    lines = []
    lines.append("=" * 60)
    lines.append("RAGAS EVALUATION RESULTS")
    lines.append("=" * 60)
    lines.append(f"\nOverall Score: {result.overall_score:.3f}/1.000")
    lines.append(f"  {'█' * int(result.overall_score * 20)}{'░' * (20 - int(result.overall_score * 20))}")

    metrics = [
        ("Faithfulness", result.faithfulness, "Answer is grounded in retrieved context"),
        ("Answer Relevancy", result.answer_relevancy, "Answer addresses the question"),
        ("Context Precision", result.context_precision, "Retrieved contexts are relevant"),
        ("Context Recall", result.context_recall, "All needed info was retrieved"),
    ]

    lines.append("\nMetric Breakdown:")
    lines.append("-" * 60)
    for name, score, desc in metrics:
        bar = "█" * int(score * 20) + "░" * (20 - int(score * 20))
        lines.append(f"  {name:<20} {bar} {score:.3f}")
        lines.append(f"    {'↳ ' + desc}")

    # Improvement suggestions
    lines.append("\n" + "=" * 60)
    lines.append("IMPROVEMENT SUGGESTIONS")
    lines.append("=" * 60)

    if result.faithfulness < 0.7:
        lines.append("\n⚠ Low Faithfulness:")
        lines.append("  → The model may be hallucinating. Suggestions:")
        lines.append("    - Add system prompt: 'Answer only from the provided context'")
        lines.append("    - Increase context chunk size to provide more grounding")
        lines.append("    - Use a more conservative generation temperature")

    if result.answer_relevancy < 0.7:
        lines.append("\n⚠ Low Answer Relevancy:")
        lines.append("  → The answer doesn't address the question well. Suggestions:")
        lines.append("    - Use query transformation (HyDE, multi-query)")
        lines.append("    - Improve prompt engineering to focus on the question")
        lines.append("    - Rerank retrieved documents by relevance to query")

    if result.context_precision < 0.7:
        lines.append("\n⚠ Low Context Precision:")
        lines.append("  → Retrieved chunks aren't relevant. Suggestions:")
        lines.append("    - Switch to hybrid search (BM25 + dense)")
        lines.append("    - Use reranking (cross-encoder)")
        lines.append("    - Improve chunking strategy (semantic chunking)")

    if result.context_recall < 0.7:
        lines.append("\n⚠ Low Context Recall:")
        lines.append("  → Missing important information. Suggestions:")
        lines.append("    - Increase top_k for retrieval")
        lines.append("    - Use multi-query retrieval to cast a wider net")
        lines.append("    - Implement graph RAG for relationship-heavy queries")

    if result.overall_score >= 0.8:
        lines.append("\n✓ Good overall performance. Consider:")
        lines.append("  - A/B testing with production traffic")
        lines.append("  - Monitoring metrics over time")
        lines.append("  - Expanding evaluation dataset")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 4. Quick Evaluation (no API)
# ---------------------------------------------------------------------------

def quick_heuristic_evaluation(samples: List[EvaluationSample]) -> EvaluationResult:
    """Run evaluation using heuristics only (no API calls)."""
    evaluator = RAGASEvaluator()

    faith_scores = [evaluator._heuristic_faithfulness(s) for s in samples]
    rel_scores = [evaluator._heuristic_relevancy(s) for s in samples]
    prec_scores = [evaluator._heuristic_precision(s) for s in samples]
    recall_scores = [evaluator._heuristic_recall(s) for s in samples]

    avg_f = sum(faith_scores) / len(faith_scores)
    avg_r = sum(rel_scores) / len(rel_scores)
    avg_p = sum(prec_scores) / len(prec_scores)
    avg_rc = sum(recall_scores) / len(recall_scores)

    return EvaluationResult(
        faithfulness=round(avg_f, 3),
        answer_relevancy=round(avg_r, 3),
        context_precision=round(avg_p, 3),
        context_recall=round(avg_rc, 3),
        overall_score=round((avg_f + avg_r + avg_p + avg_rc) / 4, 3),
        details={"num_samples": len(samples), "method": "heuristic"},
    )


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def demo():
    """Run the RAGAS evaluation demo."""
    print("=" * 70)
    print("RAGAS EVALUATION DEMO")
    print("=" * 70)

    samples = create_evaluation_dataset()
    print(f"Created {len(samples)} evaluation samples\n")

    # Quick heuristic evaluation
    print("Running heuristic evaluation (no API)...")
    result = quick_heuristic_evaluation(samples)
    print(interpret_results(result))

    # Full LLM-based evaluation (if API available)
    print("\n\n" + "=" * 70)
    print("Attempting full LLM-based evaluation...")
    print("=" * 70)

    evaluator = RAGASEvaluator()
    if evaluator._get_client():
        full_result = evaluator.evaluate(samples)
        print(interpret_results(full_result))
    else:
        print("OpenAI API not available. Showing heuristic results above.")
        print("Set OPENAI_API_KEY environment variable for full evaluation.")


if __name__ == "__main__":
    demo()
