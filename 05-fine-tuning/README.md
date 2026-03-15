# Module 5 — Fine-Tuning LLMs 🔴

> **Level:** Advanced | **Time:** ~4 hours | **Code:** Python

---

## What is Fine-Tuning?

Fine-tuning takes a pre-trained LLM and trains it further on **your specific data** to make it better at a particular task.

```
Base Model (GPT-4o-mini)
        +
Your Custom Data (JSONL)
        =
Fine-Tuned Model (your-model-ft-xyz)
```

---

## Fine-Tune vs. Prompt Engineer vs. RAG

This is the most important decision before you start:

| Approach | When to Use | Cost | Complexity |
|---|---|---|---|
| **Prompt Engineering** | General tasks, quick wins | Low | Low |
| **RAG** | Need access to your data/docs | Medium | Medium |
| **Fine-Tuning** | Specific style/format/behavior | High | High |

### Use Fine-Tuning When:
- ✅ You need a very specific output format consistently
- ✅ You want to match your brand's writing style exactly
- ✅ The task is very domain-specific (medical, legal, code)
- ✅ You want faster/cheaper inference than few-shot prompting
- ✅ Prompt engineering has hit a ceiling

### Don't Use Fine-Tuning When:
- ❌ You need fresh/recent information → use RAG
- ❌ You just need a one-time task → use prompting
- ❌ You have less than ~50 examples → not enough data
- ❌ You're still exploring the task → start with prompts first

---

## The Fine-Tuning Process

```
1. Prepare Dataset  →  JSONL format with prompt/completion pairs
2. Upload to API    →  Training file gets validated
3. Start Job        →  Training takes minutes to hours
4. Evaluate         →  Test the fine-tuned model
5. Deploy           →  Use your model via API
```

---

## Step 1: Prepare Your Dataset

### The JSONL Format
Each line is a complete training example with `messages` (same format as chat API):

```json
{"messages": [{"role": "system", "content": "You are a helpful customer support agent for AcmeCorp."}, {"role": "user", "content": "My order hasn't arrived."}, {"role": "assistant", "content": "I'm sorry to hear that! Let me look into your order right away. Could you please share your order number so I can check the status for you?"}]}
{"messages": [{"role": "system", "content": "You are a helpful customer support agent for AcmeCorp."}, {"role": "user", "content": "How do I cancel my subscription?"}, {"role": "assistant", "content": "I can help you cancel your subscription. You can do this in two ways: (1) Go to Settings → Subscription → Cancel, or (2) Reply here and I'll process it for you. Which would you prefer?"}]}
```

### Dataset Size Guidelines
| Dataset Size | Expected Improvement |
|---|---|
| 10–50 examples | Small style/format improvements |
| 50–200 examples | Noticeable behavior changes |
| 200–1000 examples | Strong task-specific improvements |
| 1000+ examples | Significant capability changes |

### Quality > Quantity
50 high-quality examples beats 500 mediocre ones. Each example should:
- Represent the task you want to improve
- Show the exact output format/style you want
- Be reviewed by a human for correctness

---

## Step 2: Validate Your Dataset

```python
import json

def validate_jsonl(filepath: str):
    errors = []
    with open(filepath, "r") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                example = json.loads(line)
                if "messages" not in example:
                    errors.append(f"Line {i}: Missing 'messages' key")
                    continue
                for msg in example["messages"]:
                    if "role" not in msg or "content" not in msg:
                        errors.append(f"Line {i}: Message missing 'role' or 'content'")
                    if msg["role"] not in ("system", "user", "assistant"):
                        errors.append(f"Line {i}: Invalid role '{msg['role']}'")
            except json.JSONDecodeError as e:
                errors.append(f"Line {i}: Invalid JSON — {e}")

    if errors:
        print("❌ Validation failed:")
        for error in errors:
            print(f"  • {error}")
    else:
        print("✅ Dataset is valid!")

    return len(errors) == 0
```

---

## Step 3: Fine-Tune with OpenAI API

```python
from openai import OpenAI
import time

client = OpenAI()

# Upload training file
print("📤 Uploading training file...")
with open("training_data.jsonl", "rb") as f:
    upload = client.files.create(file=f, purpose="fine-tune")

file_id = upload.id
print(f"✅ File uploaded: {file_id}")

# Start fine-tuning job
print("🚀 Starting fine-tuning job...")
job = client.fine_tuning.jobs.create(
    training_file=file_id,
    model="gpt-4o-mini-2024-07-18",  # base model to fine-tune
    hyperparameters={
        "n_epochs": 3,  # number of training passes
    }
)

job_id = job.id
print(f"✅ Job started: {job_id}")
print("⏳ Training in progress (this takes 10–60 minutes)...")

# Poll for completion
while True:
    job_status = client.fine_tuning.jobs.retrieve(job_id)
    status = job_status.status

    print(f"  Status: {status}", end="\r")

    if status == "succeeded":
        model_id = job_status.fine_tuned_model
        print(f"\n✅ Training complete! Model: {model_id}")
        break
    elif status in ("failed", "cancelled"):
        print(f"\n❌ Training {status}: {job_status.error}")
        break

    time.sleep(30)
```

---

## Step 4: Evaluate Your Fine-Tuned Model

Compare the base model vs. fine-tuned model on a test set:

```python
def evaluate_models(test_cases: list[dict], fine_tuned_model: str):
    base_model = "gpt-4o-mini"
    results = []

    for test in test_cases:
        question = test["question"]
        expected = test["expected"]

        # Base model response
        base_response = client.chat.completions.create(
            model=base_model,
            messages=[{"role": "user", "content": question}]
        ).choices[0].message.content

        # Fine-tuned model response
        ft_response = client.chat.completions.create(
            model=fine_tuned_model,
            messages=[{"role": "user", "content": question}]
        ).choices[0].message.content

        results.append({
            "question": question,
            "expected": expected,
            "base_model": base_response,
            "fine_tuned": ft_response,
        })

    return results
```

---

## Step 5: Use Your Fine-Tuned Model

```python
model_id = "ft:gpt-4o-mini:your-org:custom-name:abc123"

response = client.chat.completions.create(
    model=model_id,
    messages=[
        {"role": "system", "content": "You are a helpful customer support agent."},
        {"role": "user", "content": "I need help with my account."}
    ]
)

print(response.choices[0].message.content)
```

---

## Open-Source Fine-Tuning with HuggingFace

For free fine-tuning on open-source models (requires GPU):

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import SFTTrainer
from datasets import load_dataset

# Load base model (Mistral 7B example)
model_name = "mistralai/Mistral-7B-v0.1"
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Load your dataset
dataset = load_dataset("json", data_files="training_data.jsonl")

# Configure training
training_args = TrainingArguments(
    output_dir="./fine_tuned_model",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    save_strategy="epoch",
    logging_steps=10,
)

# Train with SFT (Supervised Fine-Tuning)
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    dataset_text_field="text",
    max_seq_length=2048,
)

trainer.train()
trainer.save_model("./my_fine_tuned_model")
```

> 💡 Use **Google Colab** (free T4 GPU) for small models. For production, use **RunPod** or **Lambda Labs**.

---

## Cost Estimation (OpenAI)

| Model | Training Cost | Inference Cost |
|---|---|---|
| gpt-4o-mini | $0.30 / 1M tokens | $0.60 input / $2.40 output per 1M |

**Example:** 100 examples × 500 tokens each = 50,000 tokens = **$0.015** to train

---

## 📁 Files in This Module

| File | Description |
|---|---|
| [`prepare_dataset.py`](./prepare_dataset.py) | Generate and validate JSONL datasets |
| [`finetune_openai.py`](./finetune_openai.py) | Full OpenAI fine-tuning pipeline |
| [`evaluate.py`](./evaluate.py) | Compare base vs fine-tuned model |
| [`sample_data.jsonl`](./sample_data.jsonl) | Sample training data to get started |

---

## 🎉 Course Complete!

You've covered:
- ✅ What LLMs are and how they work
- ✅ Prompt Engineering techniques
- ✅ Using LLM APIs (OpenAI, Claude, Gemini, Ollama)
- ✅ Building RAG pipelines
- ✅ Fine-tuning models on custom data

**What to build next?**
- A customer support chatbot with RAG
- A code review tool fine-tuned on your codebase
- A document Q&A system for your company
