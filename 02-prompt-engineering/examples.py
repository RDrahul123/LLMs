"""
Module 2: Prompt Engineering Examples
======================================
Run each example individually by calling the function.
Make sure to set OPENAI_API_KEY in your .env file.
"""

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI()

# ─────────────────────────────────────────────
# 1. Zero-Shot Prompting
# ─────────────────────────────────────────────

def zero_shot_sentiment():
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": (
                    "Classify the sentiment of this review as Positive, Negative, or Neutral:\n"
                    "'The product works fine but the packaging was damaged.'"
                ),
            }
        ],
    )
    print("Zero-Shot Result:", response.choices[0].message.content)


# ─────────────────────────────────────────────
# 2. Few-Shot Prompting
# ─────────────────────────────────────────────

def few_shot_sentiment():
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": """Classify sentiment. Examples:

Review: 'Amazing product, love it!' → Positive
Review: 'Terrible, broke after one day.' → Negative
Review: 'It arrived on time.' → Neutral

Now classify:
Review: 'Good quality but a bit expensive.'""",
            }
        ],
    )
    print("Few-Shot Result:", response.choices[0].message.content)


# ─────────────────────────────────────────────
# 3. Chain-of-Thought Prompting
# ─────────────────────────────────────────────

def chain_of_thought_math():
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": (
                    "A store sells apples for $0.75 each and oranges for $1.20 each. "
                    "If I buy 4 apples and 3 oranges, how much do I spend in total?\n\n"
                    "Think step by step."
                ),
            }
        ],
    )
    print("CoT Result:", response.choices[0].message.content)


# ─────────────────────────────────────────────
# 4. Role Prompting
# ─────────────────────────────────────────────

def role_prompting_code_review():
    code = """
def calculate_average(numbers):
    total = 0
    for n in numbers:
        total = total + n
    avg = total / len(numbers)
    return avg
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a senior Python developer with 10 years of experience. "
                    "You write clean, Pythonic code following PEP 8 standards. "
                    "When reviewing code, explain WHY something should change, not just that it should."
                ),
            },
            {
                "role": "user",
                "content": f"Review this Python function and suggest improvements:\n```python{code}```",
            },
        ],
    )
    print("Code Review:", response.choices[0].message.content)


# ─────────────────────────────────────────────
# 5. Prompt Templates
# ─────────────────────────────────────────────

def summarize(text: str, audience: str = "general public", bullets: int = 3) -> str:
    """Reusable summarization prompt template."""
    prompt = f"""Summarize the following text in {bullets} bullet points.
Target audience: {audience}
Be concise and focus on actionable insights.

Text:
{text}"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content


def translate(text: str, target_language: str) -> str:
    """Reusable translation prompt template."""
    prompt = f"""Translate the following text to {target_language}.
Preserve the original tone and formatting.

Text:
{text}"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content


def extract_info(text: str, fields: list[str]) -> str:
    """Extract structured information from unstructured text."""
    fields_str = "\n".join(f"- {f}" for f in fields)
    prompt = f"""Extract the following information from the text below.
Return the result as JSON with these exact keys:
{fields_str}

If a field is not found, use null.

Text:
{text}"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
    )
    return response.choices[0].message.content


# ─────────────────────────────────────────────
# Run All Examples
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 50)
    print("1. ZERO-SHOT PROMPTING")
    print("=" * 50)
    zero_shot_sentiment()

    print("\n" + "=" * 50)
    print("2. FEW-SHOT PROMPTING")
    print("=" * 50)
    few_shot_sentiment()

    print("\n" + "=" * 50)
    print("3. CHAIN-OF-THOUGHT PROMPTING")
    print("=" * 50)
    chain_of_thought_math()

    print("\n" + "=" * 50)
    print("4. ROLE PROMPTING")
    print("=" * 50)
    role_prompting_code_review()

    print("\n" + "=" * 50)
    print("5. PROMPT TEMPLATES")
    print("=" * 50)
    sample_text = (
        "The James Webb Space Telescope has captured new images of a galaxy 13.4 billion "
        "light-years away, giving astronomers a glimpse into the early universe. "
        "The findings challenge existing models of galaxy formation."
    )
    print(summarize(sample_text, audience="high school students", bullets=3))

    print("\nExtraction Example:")
    contact_text = "Hi, I'm Sarah Johnson. You can reach me at sarah@example.com or call +1-555-0199."
    print(extract_info(contact_text, ["name", "email", "phone"]))
