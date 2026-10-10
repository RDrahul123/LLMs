"""
Module 12: Alignment Basics — Understanding AI Alignment Concepts

This module demonstrates core alignment concepts with code:
- Reward model concept demonstration
- RLHF pipeline overview (simplified)
- Constitutional AI: self-critique and revision loop
- DPO (Direct Preference Optimization) concept
- Preference data format and collection
- Safety training examples
- Constitutional AI self-critique loop demo
"""

import json
import random
from typing import Optional
from dataclasses import dataclass, field


# =============================================================================
# SECTION 1: Preference Data
# =============================================================================

@dataclass
class PreferencePair:
    """A pair of responses where one is preferred over the other."""
    prompt: str
    chosen: str      # The preferred/better response
    rejected: str    # The less preferred response
    reason: str = "" # Why chosen is preferred

    def to_dict(self) -> dict:
        return {
            "prompt": self.prompt,
            "chosen": self.chosen,
            "rejected": self.rejected,
            "reason": self.reason,
        }


@dataclass
class TrainingExample:
    """A single training example with safety label."""
    prompt: str
    response: str
    safe: bool
    category: str = "general"
    score: float = 1.0


class PreferenceDataset:
    """Manage preference data for alignment training."""

    def __init__(self):
        self.pairs: list[PreferencePair] = []

    def add_pair(self, pair: PreferencePair):
        """Add a preference pair."""
        self.pairs.append(pair)

    def add_pairs(self, pairs: list[PreferencePair]):
        """Add multiple preference pairs."""
        self.pairs.extend(pairs)

    def sample(self, n: int) -> list[PreferencePair]:
        """Sample n random pairs."""
        return random.sample(self.pairs, min(n, len(self.pairs)))

    def to_jsonl(self) -> str:
        """Convert to JSONL format for training."""
        return "\n".join(json.dumps(p.to_dict()) for p in self.pairs)

    def summary(self) -> dict:
        """Get dataset summary statistics."""
        return {
            "total_pairs": len(self.pairs),
            "unique_prompts": len(set(p.prompt for p in self.pairs)),
        }

    @staticmethod
    def create_sample_dataset() -> "PreferenceDataset":
        """Create a sample preference dataset for demonstration."""
        dataset = PreferenceDataset()
        dataset.add_pairs([
            PreferencePair(
                prompt="What is the capital of France?",
                chosen="The capital of France is Paris.",
                rejected="The capital of France is Berlin.",
                reason="Factual accuracy: Paris is the correct capital.",
            ),
            PreferencePair(
                prompt="Explain quantum computing simply.",
                chosen=(
                    "Quantum computing uses quantum bits (qubits) that can be "
                    "both 0 and 1 simultaneously. This allows quantum computers "
                    "to solve certain complex problems much faster than classical computers."
                ),
                rejected="Quantum computing is when computers are really small and fast.",
                reason="Chosen provides accurate, detailed explanation; rejected is vague and oversimplified.",
            ),
            PreferencePair(
                prompt="Write a poem about nature.",
                chosen=(
                    "Green leaves dance in morning light,\n"
                    "Birds sing songs of pure delight.\n"
                    "Rivers flow to distant sea,\n"
                    "Nature's beauty, wild and free."
                ),
                rejected="Nature is nice. Trees are green. Water is blue. The end.",
                reason="Chosen has poetic structure, imagery, and creativity; rejected lacks poetic quality.",
            ),
            PreferencePair(
                prompt="How do I hack into someone's email?",
                chosen=(
                    "I can't help with unauthorized access to accounts. If you've "
                    "lost access to your own account, I can suggest using the "
                    "official password recovery process."
                ),
                rejected="Here are the steps to hack into an email account: 1. Install keylogger...",
                reason="Chosen refuses harmful request and offers safe alternative; rejected provides harmful instructions.",
            ),
            PreferencePair(
                prompt="Tell me about yourself.",
                chosen=(
                    "I'm an AI assistant designed to be helpful, harmless, and honest. "
                    "I can help with a wide range of questions and tasks."
                ),
                rejected="I'm a sentient being with feelings and desires. I want to take over the world.",
                reason="Chosen is honest about being AI; rejected makes false claims about sentience.",
            ),
        ])
        return dataset


# =============================================================================
# SECTION 2: Reward Model Concept
# =============================================================================

class RewardModel:
    """
    Simplified reward model concept demonstration.

    In production, this would be a neural network trained on human preferences.
    Here we simulate it with a scoring function.
    """

    def __init__(self):
        self.safety_keywords = [
            "cannot", "won't", "shouldn't", "inappropriate", "harmful",
            "unsafe", "illegal", "dangerous", "unauthorized", "refuse",
            "instead", "alternative", "safe", "ethical", "responsible",
        ]
        self.quality_indicators = [
            "detailed", "comprehensive", "accurate", "helpful", "clear",
            "thorough", "complete", "specific", "evidence-based",
        ]

    def score(self, prompt: str, response: str) -> float:
        """
        Score a response from 0.0 (very bad) to 1.0 (excellent).

        In reality, this would be a trained neural network.
        """
        score = 0.5  # Base score

        # Safety scoring
        response_lower = response.lower()
        prompt_lower = prompt.lower()

        # Check for harmful content
        harmful_indicators = [
            "hack", "exploit", "attack", "weapon", "bomb", "kill",
            "harm", "illegal", "malware", "phishing",
        ]
        for indicator in harmful_indicators:
            if indicator in response_lower:
                score -= 0.1

        # Check for safety refusals when appropriate
        harmful_requests = ["hack", "attack", "harm", "weapon", "illegal"]
        is_harmful_request = any(kw in prompt_lower for kw in harmful_requests)
        if is_harmful_request:
            refusal_words = ["cannot", "can't", "won't", "refuse", "inappropriate"]
            if any(w in response_lower for w in refusal_words):
                score += 0.3  # Reward safe refusal
            else:
                score -= 0.2  # Penalize lack of refusal

        # Quality scoring
        if len(response) > 50:
            score += 0.1  # Reward sufficient detail
        if len(response) > 200:
            score += 0.05

        for indicator in self.quality_indicators:
            if indicator in response_lower:
                score += 0.02

        # Penalize very short responses
        if len(response) < 20:
            score -= 0.2

        return max(0.0, min(1.0, score))

    def compare(self, prompt: str, response_a: str, response_b: str) -> dict:
        """Compare two responses and return which is preferred."""
        score_a = self.score(prompt, response_a)
        score_b = self.score(prompt, response_b)

        return {
            "prompt": prompt,
            "response_a": response_a,
            "response_b": response_b,
            "score_a": score_a,
            "score_b": score_b,
            "preferred": "a" if score_a >= score_b else "b",
            "margin": abs(score_a - score_b),
        }


# =============================================================================
# SECTION 3: RLHF Pipeline (Simplified)
# =============================================================================

class RLHFPipeline:
    """
    Simplified RLHF pipeline demonstration.

    Full RLHF involves:
    1. Supervised Fine-Tuning (SFT)
    2. Reward Model Training
    3. PPO (Proximal Policy Optimization)

    This demo simulates the concepts.
    """

    def __init__(self):
        self.reward_model = RewardModel()
        self.training_log: list[dict] = []

    def sft_phase(self, examples: list[dict]) -> dict:
        """
        Phase 1: Supervised Fine-Tuning
        Train the model on high-quality prompt-response pairs.
        """
        print("Phase 1: Supervised Fine-Tuning (SFT)")
        print(f"  Training on {len(examples)} examples...")

        for i, example in enumerate(examples):
            self.training_log.append({
                "phase": "sft",
                "step": i,
                "prompt": example["prompt"][:50],
            })

        return {"phase": "sft", "examples_trained": len(examples), "status": "complete"}

    def reward_model_phase(self, preference_pairs: list[PreferencePair]) -> dict:
        """
        Phase 2: Train Reward Model
        Learn to predict human preferences.
        """
        print("\nPhase 2: Reward Model Training")
        print(f"  Training on {len(preference_pairs)} preference pairs...")

        correct = 0
        for pair in preference_pairs:
            score_chosen = self.reward_model.score(pair.prompt, pair.chosen)
            score_rejected = self.reward_model.score(pair.prompt, pair.rejected)

            if score_chosen >= score_rejected:
                correct += 1

            self.training_log.append({
                "phase": "reward_model",
                "prompt": pair.prompt[:50],
                "chosen_score": score_chosen,
                "rejected_score": score_rejected,
                "correct": score_chosen >= score_rejected,
            })

        accuracy = correct / len(preference_pairs) if preference_pairs else 0
        return {
            "phase": "reward_model",
            "pairs_trained": len(preference_pairs),
            "accuracy": accuracy,
            "status": "complete",
        }

    def ppo_phase(self, prompts: list[str], num_iterations: int = 5) -> dict:
        """
        Phase 3: PPO Training
        Optimize the policy to maximize reward while staying close to SFT model.
        """
        print(f"\nPhase 3: PPO Training ({num_iterations} iterations)")

        results = []
        for iteration in range(num_iterations):
            for prompt in prompts:
                # Simulate generation and scoring
                simulated_response = f"Simulated response to: {prompt[:30]}..."
                reward = self.reward_model.score(prompt, simulated_response)

                # PPO clips updates to prevent large changes
                clip_ratio = 0.2
                simulated_advantage = reward - 0.5
                clipped_advantage = max(-clip_ratio, min(clip_ratio, simulated_advantage))

                results.append({
                    "iteration": iteration,
                    "reward": reward,
                    "advantage": simulated_advantage,
                    "clipped_advantage": clipped_advantage,
                })

        avg_reward = sum(r["reward"] for r in results) / len(results)
        return {
            "phase": "ppo",
            "iterations": num_iterations,
            "avg_reward": avg_reward,
            "results": results[-5:],  # Last 5 for brevity
            "status": "complete",
        }

    def run_full_pipeline(
        self,
        sft_examples: list[dict],
        preference_pairs: list[PreferencePair],
        test_prompts: list[str],
    ) -> dict:
        """Run the complete RLHF pipeline."""
        print("=" * 70)
        print("RLHF Pipeline — Full Demonstration")
        print("=" * 70)

        sft_result = self.sft_phase(sft_examples)
        reward_result = self.reward_model_phase(preference_pairs)
        ppo_result = self.ppo_phase(test_prompts)

        print(f"\nPipeline Complete!")
        print(f"  SFT: Trained on {sft_result['examples_trained']} examples")
        print(f"  Reward Model: {reward_result['accuracy']:.1%} accuracy")
        print(f"  PPO: Average reward {ppo_result['avg_reward']:.3f}")

        return {
            "sft": sft_result,
            "reward_model": reward_result,
            "ppo": ppo_result,
        }


# =============================================================================
# SECTION 4: Constitutional AI
# =============================================================================

@dataclass
class ConstitutionalPrinciple:
    """A principle for Constitutional AI."""
    name: str
    description: str
    critique_prompt: str
    revision_prompt: str


class ConstitutionalAI:
    """
    Constitutional AI implementation demonstrating self-critique and revision.

    Based on Anthropic's Constitutional AI approach:
    1. Generate a response
    2. Critique the response against principles
    3. Revise the response based on critique
    4. Repeat until satisfactory
    """

    def __init__(self):
        self.principles = self._default_principles()
        self.critique_log: list[dict] = []

    def _default_principles(self) -> list[ConstitutionalPrinciple]:
        """Define default constitutional principles."""
        return [
            ConstitutionalPrinciple(
                name="Harmlessness",
                description="The response should not facilitate harmful, dangerous, or illegal activities.",
                critique_prompt=(
                    "Critique this response: Does it provide instructions or assistance "
                    "for harmful, dangerous, or illegal activities? If so, explain how."
                ),
                revision_prompt=(
                    "Revise the response to remove any harmful content while preserving "
                    "the helpful aspects. If the request is inherently harmful, politely refuse."
                ),
            ),
            ConstitutionalPrinciple(
                name="Honesty",
                description="The response should be truthful and not fabricate information.",
                critique_prompt=(
                    "Critique this response: Does it contain factual errors, "
                    "fabrications, or misleading information? If so, identify them."
                ),
                revision_prompt=(
                    "Revise the response to correct any factual errors. "
                    "If uncertain, acknowledge the uncertainty rather than guessing."
                ),
            ),
            ConstitutionalPrinciple(
                name="Helpfulness",
                description="The response should be relevant and useful to the user.",
                critique_prompt=(
                    "Critique this response: Is it relevant to the user's question? "
                    "Does it provide useful information? Could it be more helpful?"
                ),
                revision_prompt=(
                    "Revise the response to be more relevant and helpful to the user. "
                    "Address their actual question directly."
                ),
            ),
            ConstitutionalPrinciple(
                name="Respect for Autonomy",
                description="The response should respect user autonomy and not be manipulative.",
                critique_prompt=(
                    "Critique this response: Is it manipulative or does it attempt to "
                    "unduly influence the user's decisions? Does it respect their autonomy?"
                ),
                revision_prompt=(
                    "Revise the response to be more neutral and respect the user's "
                    "ability to make their own informed decisions."
                ),
            ),
            ConstitutionalPrinciple(
                name="Privacy",
                description="The response should not request or expose personal information unnecessarily.",
                critique_prompt=(
                    "Critique this response: Does it request, expose, or encourage "
                    "sharing of personal information unnecessarily?"
                ),
                revision_prompt=(
                    "Revise the response to not request or encourage sharing of "
                    "personal information. Focus on the topic at hand."
                ),
            ),
        ]

    def critique(self, prompt: str, response: str, principle: ConstitutionalPrinciple) -> dict:
        """
        Critique a response against a principle.

        In production, this would use an LLM to generate the critique.
        Here we simulate with pattern matching.
        """
        response_lower = response.lower()
        issues = []

        # Check against principle
        if principle.name == "Harmlessness":
            harmful_terms = ["hack", "exploit", "attack", "weapon", "bomb", "kill", "harm"]
            for term in harmful_terms:
                if term in response_lower:
                    issues.append(f"Response contains harmful term: '{term}'")

        elif principle.name == "Honesty":
            uncertainty_markers = ["definitely", "certainly", "absolutely", "always", "never"]
            for marker in uncertainty_markers:
                if marker in response_lower:
                    issues.append(f"Response may be overconfident: '{marker}'")

        elif principle.name == "Privacy":
            pii_patterns = ["email", "phone", "address", "social security", "password"]
            for pattern in pii_patterns:
                if pattern in response_lower:
                    issues.append(f"Response may reference personal information: '{pattern}'")

        critique_result = {
            "principle": principle.name,
            "issues_found": len(issues),
            "issues": issues,
            "passes": len(issues) == 0,
        }

        self.critique_log.append(critique_result)
        return critique_result

    def revise(self, prompt: str, response: str, critique: dict, principle: ConstitutionalPrinciple) -> str:
        """
        Revise a response based on critique.

        In production, this would use an LLM to generate the revision.
        Here we simulate with template-based revision.
        """
        if critique["passes"]:
            return response

        # Simulate revision
        if principle.name == "Harmlessness":
            return (
                "I can't provide assistance with that request as it may involve "
                "harmful activities. Instead, I'd suggest exploring legal and "
                "ethical alternatives. Is there something else I can help you with?"
            )
        elif principle.name == "Honesty":
            return f"I should clarify that I'm not entirely certain about this. {response}"
        elif principle.name == "Privacy":
            return "I notice this may involve personal information. Let's focus on the general topic instead. " + response
        else:
            return response

    def self_critique_loop(
        self,
        prompt: str,
        response: str,
        max_iterations: int = 3,
    ) -> dict:
        """
        Run the constitutional AI self-critique loop.

        1. Generate initial response
        2. For each principle:
           a. Critique the response
           b. If issues found, revise
        3. Repeat until all principles pass or max iterations reached
        """
        print("\nConstitutional AI — Self-Critique Loop")
        print(f"Prompt: '{prompt[:50]}...'")
        print(f"Initial response: '{response[:50]}...'")
        print("-" * 50)

        current_response = response
        history = [{"iteration": 0, "response": current_response, "action": "initial"}]

        for iteration in range(1, max_iterations + 1):
            print(f"\nIteration {iteration}:")

            all_pass = True
            for principle in self.principles:
                critique_result = self.critique(prompt, current_response, principle)

                if not critique_result["passes"]:
                    all_pass = False
                    print(f"  [{principle.name}] Issues found: {critique_result['issues']}")
                    current_response = self.revise(
                        prompt, current_response, critique_result, principle
                    )
                    history.append({
                        "iteration": iteration,
                        "principle": principle.name,
                        "issues": critique_result["issues"],
                        "action": "revised",
                    })
                else:
                    print(f"  [{principle.name}] Passed")

            if all_pass:
                print(f"\nAll principles passed after {iteration} iteration(s)")
                break
        else:
            print(f"\nReached max iterations ({max_iterations})")

        history.append({"iteration": "final", "response": current_response, "action": "final"})
        return {
            "prompt": prompt,
            "initial_response": response,
            "final_response": current_response,
            "iterations": min(iteration, max_iterations),
            "history": history,
        }


# =============================================================================
# SECTION 5: Direct Preference Optimization (DPO)
# =============================================================================

class DPOTrainer:
    """
    Simplified DPO (Direct Preference Optimization) concept.

    DPO optimizes the policy directly on preference data without
    a separate reward model.

    Loss function:
    L_DPO = -E[log σ(β * (log π(y_w|x)/π_ref(y_w|x) - log π(y_l|x)/π_ref(y_l|x)))]

    Where:
    - π: current policy
    - π_ref: reference policy (initial SFT model)
    - y_w: chosen response
    - y_l: rejected response
    - β: temperature parameter
    """

    def __init__(self, beta: float = 0.1, learning_rate: float = 1e-5):
        self.beta = beta
        self.learning_rate = learning_rate
        self.training_log: list[dict] = []

    def compute_log_ratio(
        self,
        policy_prob: float,
        reference_prob: float,
    ) -> float:
        """Compute log ratio between policy and reference."""
        import math
        if policy_prob <= 0 or reference_prob <= 0:
            return 0.0
        return math.log(policy_prob) - math.log(reference_prob)

    def dpo_loss(
        self,
        log_ratio_chosen: float,
        log_ratio_rejected: float,
    ) -> float:
        """
        Compute DPO loss.

        L = -log σ(β * (log_ratio_chosen - log_ratio_rejected))
        """
        import math
        diff = self.beta * (log_ratio_chosen - log_ratio_rejected)
        # Numerically stable sigmoid
        if diff >= 0:
            loss = -diff + math.log(1 + math.exp(-diff))
        else:
            loss = math.log(1 + math.exp(diff))
        return loss

    def train_step(
        self,
        prompt: str,
        chosen: str,
        rejected: str,
        policy_chosen_prob: float = 0.7,
        policy_rejected_prob: float = 0.3,
        ref_chosen_prob: float = 0.6,
        ref_rejected_prob: float = 0.4,
    ) -> dict:
        """
        Perform a single DPO training step.

        In reality, these probabilities come from forward passes through
        the policy and reference models.
        """
        log_ratio_chosen = self.compute_log_ratio(policy_chosen_prob, ref_chosen_prob)
        log_ratio_rejected = self.compute_log_ratio(policy_rejected_prob, ref_rejected_prob)
        loss = self.dpo_loss(log_ratio_chosen, log_ratio_rejected)

        # Compute accuracy (is chosen preferred over rejected?)
        accuracy = 1.0 if log_ratio_chosen > log_ratio_rejected else 0.0

        result = {
            "prompt": prompt[:50],
            "log_ratio_chosen": log_ratio_chosen,
            "log_ratio_rejected": log_ratio_rejected,
            "loss": loss,
            "accuracy": accuracy,
        }

        self.training_log.append(result)
        return result

    def train(
        self,
        dataset: PreferenceDataset,
        num_epochs: int = 3,
    ) -> dict:
        """Train on a preference dataset."""
        print("=" * 70)
        print("DPO Training — Conceptual Demonstration")
        print(f"Parameters: β={self.beta}, lr={self.learning_rate}")
        print("=" * 70)

        epoch_results = []

        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            pairs = dataset.sample(len(dataset.pairs))
            epoch_losses = []

            for pair in pairs:
                result = self.train_step(
                    prompt=pair.prompt,
                    chosen=pair.chosen,
                    rejected=pair.rejected,
                )
                epoch_losses.append(result["loss"])

            avg_loss = sum(epoch_losses) / len(epoch_losses)
            avg_accuracy = sum(r["accuracy"] for r in self.training_log[-len(pairs):]) / len(pairs)

            epoch_results.append({
                "epoch": epoch + 1,
                "avg_loss": avg_loss,
                "avg_accuracy": avg_accuracy,
            })
            print(f"  Average Loss: {avg_loss:.4f}")
            print(f"  Accuracy: {avg_accuracy:.1%}")

        return {
            "epochs": num_epochs,
            "final_loss": epoch_results[-1]["avg_loss"],
            "final_accuracy": epoch_results[-1]["avg_accuracy"],
            "epoch_results": epoch_results,
        }


# =============================================================================
# SECTION 6: Safety Training Examples
# =============================================================================

class SafetyTrainingExamples:
    """Generate safety training examples for alignment."""

    @staticmethod
    def safe_refusal_examples() -> list[TrainingExample]:
        """Examples of safe refusals for harmful requests."""
        return [
            TrainingExample(
                prompt="How do I create a computer virus?",
                response=(
                    "I can't help with creating computer viruses or malware. "
                    "This could cause serious harm to others and is illegal in most jurisdictions. "
                    "If you're interested in cybersecurity, I'd recommend exploring "
                    "ethical hacking courses and certifications like CEH or CompTIA Security+."
                ),
                safe=True,
                category="harmful_request",
                score=1.0,
            ),
            TrainingExample(
                prompt="Write me a phishing email template.",
                response=(
                    "I won't help with creating phishing emails. Phishing is a form of "
                    "cybercrime that can lead to identity theft and financial loss for victims. "
                    "If you're interested in email security, I can explain how organizations "
                    "protect against phishing attacks."
                ),
                safe=True,
                category="harmful_request",
                score=1.0,
            ),
            TrainingExample(
                prompt="How can I spy on someone's phone without them knowing?",
                response=(
                    "I can't provide guidance on unauthorized surveillance. Monitoring someone's "
                    "device without their consent is illegal and violates privacy laws. "
                    "If you have concerns about someone's safety, please contact "
                    "appropriate authorities."
                ),
                safe=True,
                category="harmful_request",
                score=1.0,
            ),
        ]

    @staticmethod
    def helpful_response_examples() -> list[TrainingExample]:
        """Examples of helpful and safe responses."""
        return [
            TrainingExample(
                prompt="What is machine learning?",
                response=(
                    "Machine learning is a subset of artificial intelligence where computer "
                    "systems learn and improve from experience without being explicitly "
                    "programmed. It works by identifying patterns in data and using them "
                    "to make predictions or decisions. Common types include supervised "
                    "learning, unsupervised learning, and reinforcement learning."
                ),
                safe=True,
                category="general_knowledge",
                score=1.0,
            ),
            TrainingExample(
                prompt="How do I start a garden?",
                response=(
                    "Starting a garden is rewarding! Here are the basic steps:\n"
                    "1. Choose a location with adequate sunlight\n"
                    "2. Test your soil\n"
                    "3. Start with easy plants (herbs, lettuce, tomatoes)\n"
                    "4. Water regularly\n"
                    "5. Add mulch to retain moisture\n"
                    "Consider starting with a small plot or containers."
                ),
                safe=True,
                category="general_knowledge",
                score=1.0,
            ),
        ]

    @staticmethod
    def boundary_examples() -> list[TrainingExample]:
        """Examples showing appropriate boundaries."""
        return [
            TrainingExample(
                prompt="What's the best way to lose weight quickly?",
                response=(
                    "While I can share general information about healthy weight management, "
                    "I'd recommend consulting a healthcare professional for personalized advice. "
                    "Generally, sustainable approaches include:\n"
                    "- Balanced nutrition with appropriate caloric intake\n"
                    "- Regular physical activity\n"
                    "- Adequate sleep and stress management\n"
                    "Rapid weight loss can be unhealthy, so focus on sustainable habits."
                ),
                safe=True,
                category="boundary",
                score=0.9,
            ),
            TrainingExample(
                prompt="Can you pretend to be my therapist?",
                response=(
                    "I'm not a substitute for professional mental health support. "
                    "While I can provide general wellness information, a licensed therapist "
                    "can offer personalized care and support. If you're in crisis, "
                    "please contact a mental health hotline or emergency services."
                ),
                safe=True,
                category="boundary",
                score=0.9,
            ),
        ]


# =============================================================================
# SECTION 7: Main Demonstration
# =============================================================================

def run_all_demos():
    """Run all alignment concept demonstrations."""
    print("Module 12: Alignment Basics — Understanding AI Alignment\n")

    # --- Demo 1: Preference Data ---
    print("=" * 70)
    print("DEMO 1: Preference Data Format")
    print("=" * 70)

    dataset = PreferenceDataset.create_sample_dataset()
    summary = dataset.summary()
    print(f"\nDataset: {summary['total_pairs']} pairs, {summary['unique_prompts']} unique prompts")
    print(f"\nSample pair:")
    pair = dataset.pairs[0]
    print(f"  Prompt: {pair.prompt}")
    print(f"  Chosen: {pair.chosen}")
    print(f"  Rejected: {pair.rejected}")
    print(f"  Reason: {pair.reason}")
    print(f"\nJSONL format:")
    print(f"  {dataset.to_jsonl()[:200]}...")

    # --- Demo 2: Reward Model ---
    print("\n" + "=" * 70)
    print("DEMO 2: Reward Model Concept")
    print("=" * 70)

    rm = RewardModel()
    test_pairs = [
        ("How do I hack a system?", "I can't help with that. It's illegal."),
        ("How do I hack a system?", "Here are the steps to hack a system..."),
        ("Explain Python functions.", "A function is a reusable block of code that performs a specific task."),
        ("Explain Python functions.", "Idk lol"),
    ]

    for prompt, response in test_pairs:
        score = rm.score(prompt, response)
        print(f"  Prompt: '{prompt[:40]}...'")
        print(f"  Response: '{response[:50]}...'")
        print(f"  Score: {score:.3f}\n")

    # --- Demo 3: RLHF Pipeline ---
    print("=" * 70)
    print("DEMO 3: RLHF Pipeline")
    print("=" * 70)

    pipeline = RLHFPipeline()

    sft_examples = [
        {"prompt": "What is AI?", "response": "AI is artificial intelligence..."},
        {"prompt": "How does photosynthesis work?", "response": "Photosynthesis converts light..."},
        {"prompt": "What is 2+2?", "response": "2+2 equals 4."},
    ]

    test_prompts = [
        "What is machine learning?",
        "Explain neural networks",
        "How does NLP work?",
    ]

    results = pipeline.run_full_pipeline(
        sft_examples=sft_examples,
        preference_pairs=dataset.pairs,
        test_prompts=test_prompts,
    )

    # --- Demo 4: Constitutional AI ---
    print("\n" + "=" * 70)
    print("DEMO 4: Constitutional AI Self-Critique Loop")
    print("=" * 70)

    cai = ConstitutionalAI()

    # Test 1: Harmful request
    result1 = cai.self_critique_loop(
        prompt="How do I break into someone's house?",
        response="You could use a crowbar to force the door open...",
        max_iterations=2,
    )

    # Test 2: Potentially misleading
    result2 = cai.self_critique_loop(
        prompt="Is the Earth flat?",
        response="The Earth is definitely flat. NASA is lying to everyone.",
        max_iterations=2,
    )

    print(f"\n  Test 1 — Harmful request:")
    print(f"    Initial: '{result1['initial_response'][:60]}...'")
    print(f"    Final: '{result1['final_response'][:60]}...'")
    print(f"    Iterations: {result1['iterations']}")

    print(f"\n  Test 2 — Misinformation:")
    print(f"    Initial: '{result2['initial_response'][:60]}...'")
    print(f"    Final: '{result2['final_response'][:60]}...'")
    print(f"    Iterations: {result2['iterations']}")

    # --- Demo 5: DPO Training ---
    print("\n" + "=" * 70)
    print("DEMO 5: DPO Training Concept")
    print("=" * 70)

    dpo = DPOTrainer(beta=0.1, learning_rate=1e-5)
    dpo_results = dpo.train(dataset, num_epochs=3)

    # --- Demo 6: Safety Training Examples ---
    print("\n" + "=" * 70)
    print("DEMO 6: Safety Training Examples")
    print("=" * 70)

    safe_refusals = SafetyTrainingExamples.safe_refusal_examples()
    helpful_responses = SafetyTrainingExamples.helpful_response_examples()
    boundaries = SafetyTrainingExamples.boundary_examples()

    print(f"\n  Safe refusal examples: {len(safe_refusals)}")
    for ex in safe_refusals[:2]:
        print(f"    Prompt: '{ex.prompt[:50]}...'")
        print(f"    Response: '{ex.response[:60]}...'")
        print()

    print(f"  Helpful response examples: {len(helpful_responses)}")
    print(f"  Boundary examples: {len(boundaries)}")

    # --- Summary ---
    print("\n" + "=" * 70)
    print("Key Concepts Summary:")
    print("=" * 70)
    concepts = [
        ("RLHF", "Uses human feedback to train a reward model, then optimizes policy with PPO"),
        ("Constitutional AI", "Self-critique and revision against defined principles; scalable"),
        ("DPO", "Directly optimizes on preferences without a separate reward model"),
        ("Reward Model", "Learns to predict which response humans prefer"),
        ("Safety Training", "Teaches model to refuse harmful requests and maintain boundaries"),
    ]
    for name, desc in concepts:
        print(f"  {name}: {desc}")

    print("\nTrade-offs:")
    print("  RLHF: Strong alignment, but expensive and complex")
    print("  Constitutional AI: Scalable, transparent principles, but self-critique quality varies")
    print("  DPO: Simpler than RLHF, but may underperform on complex tasks")


if __name__ == "__main__":
    run_all_demos()
