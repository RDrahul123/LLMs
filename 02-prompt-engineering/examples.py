"""
Prompt Engineering Examples
===========================
Runnable examples of prompting techniques using the OpenAI API.

Install: pip install openai
Setup:   export OPENAI_API_KEY="your-key-here"

Usage:   python examples.py
"""

import os
import sys

try:
    from openai import OpenAI
except ImportError:
    print("Install openai: pip install openai")
    sys.exit(1)

client = OpenAI()  # Uses OPENAI_API_KEY env var

MODEL = "gpt-4o-mini"


def zero_shot_sentiment():
    """Classify sentiment with no examples."""
    print("\n" + "=" * 50)
    print("ZERO-SHOT SENTIMENT CLASSIFICATION")
    print("=" * 50)

    review = "The food was amazing, but the service was painfully slow."
    prompt = f"Classify the sentiment of this review as positive, negative, or neutral:\n\n\"{review}\""

    print(f"\nReview: {review}")
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    print(f"Result: {response.choices[0].message.content}")


def few_shot_sentiment():
    """Classify sentiment with examples."""
    print("\n" + "=" * 50)
    print("FEW-SHOT SENTIMENT CLASSIFICATION")
    print("=" * 50)

    prompt = """Classify the sentiment of each review as positive, negative, or neutral.

Review: "Absolutely love this product!" → Positive
Review: "Broke after one day, total waste." → Negative
Review: "It works fine, nothing special." → Neutral

Review: "The food was amazing, but the service was painfully slow." →"""

    print(f"\nPrompt:\n{prompt}")
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    print(f"\nResult: {response.choices[0].message.content}")


def chain_of_thought_math():
    """Solve a math problem using chain-of-thought."""
    print("\n" + "=" * 50)
    print("CHAIN-OF-THOUGHT MATH")
    print("=" * 50)

    prompt = """Solve this step by step:

A store sells t-shirts for $25 each.
If you buy 3 or more, you get a 20% discount.
Tax is 8% on the discounted price.
How much do 5 t-shirts cost (including tax)?

Show your work step by step."""

    print(f"\nPrompt:\n{prompt}")
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    print(f"\nResult:\n{response.choices[0].message.content}")


def role_prompting_code_review():
    """Use role prompting for a code review."""
    print("\n" + "=" * 50)
    print("ROLE PROMPTING — CODE REVIEW")
    print("=" * 50)

    system = "You are a senior security engineer with 15 years of experience. You focus on finding security vulnerabilities in code."

    code = """
def authenticate(username, password):
    query = f"SELECT * FROM users WHERE name='{username}' AND pass='{password}'"
    result = db.execute(query)
    if result:
        return create_session(username)
    return None
"""

    user = f"Review this code for security issues:\n\n```python\n{code}\n```"

    print(f"\nSystem: {system}")
    print(f"Code:\n{code}")
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.3,
    )
    print(f"\nReview:\n{response.choices[0].message.content}")


def summarize():
    """Template function for summarization."""
    print("\n" + "=" * 50)
    print("SUMMARIZATION TEMPLATE")
    print("=" * 50)

    text = """
    Large language models have transformed the field of natural language processing.
    These models, trained on vast corpora of text, can perform a wide range of tasks
    including translation, summarization, question answering, and code generation.
    The Transformer architecture, introduced in 2017, forms the foundation of all
    modern LLMs. Key innovations include self-attention mechanisms, positional
    embeddings, and pre-training followed by fine-tuning. Recent trends show a move
    toward larger models with more parameters, longer context windows, and multimodal
    capabilities. Companies like OpenAI, Anthropic, Google, and Meta are leading the
    development of increasingly capable models.
    """

    prompt = f"Summarize the following text in exactly 3 bullet points:\n\n{text}"

    print(f"\nText: {text[:100]}...")
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )
    print(f"\nSummary:\n{response.choices[0].message.content}")


def translate():
    """Template function for translation."""
    print("\n" + "=" * 50)
    print("TRANSLATION TEMPLATE")
    print("=" * 50)

    prompt = """Translate the following English text to Spanish, French, and Japanese.
Return the translations as a numbered list.

Text: "The weather is beautiful today. I think I'll go for a walk in the park."
"""

    print(f"\nPrompt:\n{prompt}")
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )
    print(f"Translations:\n{response.choices[0].message.content}")


def extract_info():
    """Template function for information extraction."""
    print("\n" + "=" * 50)
    print("INFORMATION EXTRACTION TEMPLATE")
    print("=" * 50)

    text = """
    Hi, I'm John Smith and I'd like to schedule a meeting.
    My email is john.smith@company.com and my phone is 555-0123.
    I'm available on Tuesday or Thursday afternoon next week.
    I'd like to discuss the Q4 budget proposal.
    """

    prompt = f"""Extract structured information from this message as JSON:

{text}

Return JSON with these fields:
- name
- email
- phone
- available_days
- preferred_time
- topic
"""

    print(f"\nMessage:\n{text}")
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        response_format={"type": "json_object"},
    )
    print(f"\nExtracted:\n{response.choices[0].message.content}")


def main():
    if not os.environ.get("OPENAI_API_KEY"):
        print("Set OPENAI_API_KEY environment variable first.")
        print("  export OPENAI_API_KEY='your-key-here'")
        sys.exit(1)

    print("Prompt Engineering Examples")
    print("Using model:", MODEL)

    zero_shot_sentiment()
    few_shot_sentiment()
    chain_of_thought_math()
    role_prompting_code_review()
    summarize()
    translate()
    extract_info()

    print("\n" + "=" * 50)
    print("All demos complete!")
    print("=" * 50)


if __name__ == "__main__":
    main()
