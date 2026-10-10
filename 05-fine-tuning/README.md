# Module 6: Fine-Tuning

> **Level:** Advanced | **Time:** ~3 hours | **Prerequisites:** Modules 1-5 | **Code:** finetune_openai.py, prepare_dataset.py, sample_data.jsonl

---

## What is Fine-Tuning?

Fine-tuning is the process of **further training** a pre-trained LLM on a smaller, task-specific dataset. Instead of learning language from scratch, the model adapts its existing knowledge to your specific use case.

```
Pre-training:         Fine-tuning:
Internet text → LLM   Your data → Specialized LLM
(General knowledge)   (Domain expertise)
```

---

## When to Fine-Tune?

### Decision Table

| Approach | Best When | Pros | Cons |
|----------|-----------|------|------|
| **Prompt Engineering** | Simple tasks, few examples | No training needed, fast iteration | Limited by prompt length, no customization |
| **RAG** | Need access to specific data | No retraining, always up-to-date | Retrieval quality varies, slower |
| **Fine-Tuning** | Consistent format/style, domain expertise | Deep customization, faster inference | Requires data, expensive, maintenance |

**Rule of thumb:** Try prompt engineering first → then RAG → then fine-tune.

---

## Dataset Preparation

### JSONL Format

OpenAI fine-tuning expects **JSONL** (JSON Lines) files with `messages` arrays:

```jsonl
{"messages": [{"role": "system", "content": "You are a helpful customer support agent."}, {"role": "user", "content": "How do I reset my password?"}, {"role": "assistant", "content": "To reset your password:\n1. Go to Settings\n2. Click 'Security'\n3. Select 'Reset Password'\n4. Check your email for the reset link"}]}
{"messages": [{"role": "system", "content": "You are a helpful customer support agent."}, {"role": "user", "content": "Can I change my email address?"}, {"role": "assistant", "content": "Yes! To change your email:\n1. Go to Settings > Profile\n2. Click 'Edit' next to your email\n3. Enter your new email\n4. Verify the new email address"}]}
```

### Data Quality Tips

1. **Consistency** — Use the same format across all examples
2. **Coverage** — Include edge cases and common questions
3. **Quality** — Proofread; the model learns your mistakes too
4. **Volume** — Minimum 10 examples, 50-100+ for good results
5. **Balance** — Don't overrepresent any single pattern

### sample_data.jsonl

See `05-fine-tuning/sample_data.jsonl` for 5 customer support training examples.

---

## OpenAI Fine-Tuning API

### Full Pipeline

```python
from openai import OpenAI
client = OpenAI()

# Step 1: Upload training data
file = client.files.create(
    file=open("train.jsonl", "rb"),
    purpose="fine-tune"
)

# Step 2: Create fine-tuning job
job = client.fine_tuning.jobs.create(
    training_file=file.id,
    model="gpt-4o-mini-2024-07-18",
    hyperparameters={"n_epochs": 3}
)

# Step 3: Monitor progress
status = client.fine_tuning.jobs.retrieve(job.id)
print(f"Status: {status.status}")

# Step 4: Use the fine-tuned model
response = client.chat.completions.create(
    model="ft:gpt-4o-mini-2024-07-18:org-id:custom-suffix:id",
    messages=[
        {"role": "user", "content": "How do I reset my password?"}
    ]
)
```

### Cost Estimation

OpenAI fine-tuning costs (approximate):
- **Training:** $3.00 / 1M tokens (gpt-4o-mini)
- **Inference:** ~1.5× the base model cost
- **Example:** 100 examples × 500 tokens = 50K tokens training ≈ $0.15

### Time

- Small datasets (< 100 examples): 5-10 minutes
- Large datasets (1000+ examples): 1-4 hours

---

## HuggingFace / TRL Fine-Tuning

For open-source models, use HuggingFace Transformers + TRL:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer, SFTConfig

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B")

config = SFTConfig(
    output_dir="./output",
    per_device_train_batch_size=4,
    num_train_epochs=3,
    learning_rate=2e-5,
    bf16=True,
)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    args=config,
)
trainer.train()
```

### Requirements

- GPU with 16GB+ VRAM (for 8B models)
- `pip install transformers trl peft accelerate bitsandbytes`
- For larger models: use QLoRA (4-bit quantization)

---

## Evaluation

### Before Deploying

1. **Compare outputs** — Side-by-side with base model
2. **Test on held-out data** — Never seen during training
3. **Edge cases** — Unusual inputs, adversarial prompts
4. **Regression testing** — Fine-tuning shouldn't break existing capabilities

### Metrics

- **Accuracy** — For classification tasks
- **ROUGE/BLEU** — For summarization/translation
- **Human evaluation** — Best for open-ended generation
- **A/B testing** — Real-world comparison

---

## Cost Summary

| Approach | Setup Cost | Per-Query Cost | Maintenance |
|----------|-----------|----------------|-------------|
| Prompt Engineering | $0 | $0.001-0.01 | Low |
| RAG | $0-100 | $0.01-0.05 | Medium |
| Fine-Tuning (OpenAI) | $50-500 | $0.002-0.02 | Medium |
| Fine-Tuning (Self-hosted) | $100-1000 | $0.001-0.01 | High |

---

## Try It Yourself

```bash
# Prepare training data
python 05-fine-tuning/prepare_dataset.py

# Fine-tune with OpenAI
python 05-fine-tuning/finetune_openai.py --data sample_data.jsonl

# Check training status
python 05-fine-tuning/finetune_openai.py --status <job-id>
```

---

## Course Complete!

Congratulations! You've covered the full LLM development pipeline:

1. **Introduction** — What LLMs are and how they work
2. **Transformers** — The architecture powering it all
3. **Prompt Engineering** — Communicating effectively with LLMs
4. **LLM APIs** — Building with OpenAI, Claude, and more
5. **RAG** — Giving LLMs access to your data
6. **Fine-Tuning** — Customizing models for your use case

### What's Next?

Explore the advanced modules:
- **Module 7:** Advanced Prompting Techniques
- **Module 8:** AI Agents & Tool Use
- **Module 9:** Advanced RAG Strategies
- **Module 10:** Evaluation & Testing
- **Module 11:** Deployment
- **Module 12:** Safety & Ethics

---

## Further Reading

- [OpenAI Fine-Tuning Guide](https://platform.openai.com/docs/guides/fine-tuning)
- [HuggingFace TRL Documentation](https://huggingface.co/docs/trl/)
- [LoRA: Low-Rank Adaptation](https://arxiv.org/abs/2106.09685)
- [QLoRA Paper](https://arxiv.org/abs/2305.14314)

## Visual Deep Dive

Continue to **[Module 6 Deep Dive](deep-dive.md)** for choosing between prompts, RAG, and fine-tuning; SFT objectives; LoRA intuition; data leakage; and release evaluation.
