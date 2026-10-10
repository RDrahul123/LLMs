# Prompt Engineering Exercises

Practice your prompt engineering skills with these exercises. Solutions are in the collapsible sections below.

---

## Exercise 1: Product Review Classification

**Task:** Write a prompt that classifies product reviews into categories: `positive`, `negative`, `neutral`, `mixed`.

Test it with this review:
> "The battery life is incredible and the screen is gorgeous, but the keyboard feels cheap and the speakers are terrible."

<details>
<summary>💡 Solution</summary>

```
Classify the following product review into exactly one category:
positive, negative, neutral, or mixed.

Rules:
- "positive" if mostly good
- "negative" if mostly bad
- "neutral" if balanced or mediocre
- "mixed" if it has clearly both good AND bad points

Review: "The battery life is incredible and the screen is gorgeous, but the keyboard feels cheap and the speakers are terrible."

Category:
```

**Answer:** `mixed` — The review has strong positive points (battery, screen) and strong negative points (keyboard, speakers).

</details>

---

## Exercise 2: Data Extraction

**Task:** Write a prompt that extracts name, email, phone, and company from unstructured text.

Test with:
> "Contact Sarah Chen at sarah.chen@techcorp.io or call (555) 987-6543. She's the VP of Engineering at TechCorp International."

<details>
<summary>💡 Solution</summary>

```
Extract the following fields from the text below:
- name
- email
- phone
- company

Return the result as a JSON object with these exact keys.
If a field is not found, use null.

Text: "Contact Sarah Chen at sarah.chen@techcorp.io or call (555) 987-6543. She's the VP of Engineering at TechCorp International."

JSON:
```

Expected output:
```json
{
  "name": "Sarah Chen",
  "email": "sarah.chen@techcorp.io",
  "phone": "(555) 987-6543",
  "company": "TechCorp International"
}
```

</details>

---

## Exercise 3: Chain-of-Thought Debugging

**Task:** Write a chain-of-thought prompt that helps debug code. The prompt should guide the model to identify the bug and suggest a fix.

Test with:
```python
def average(numbers):
    total = 0
    for n in numbers:
        total += n
    return total / len(numbers) + 1
```

<details>
<summary>💡 Solution</summary>

```
You are a Python debugging expert. Analyze the following code step by step.

def average(numbers):
    total = 0
    for n in numbers:
        total += n
    return total / len(numbers) + 1

Follow these steps:
1. What is the function supposed to do?
2. Trace through the code with input [10, 20, 30]
3. What is the expected result?
4. What is the actual result?
5. Identify the bug
6. Provide the corrected code

Be specific about each step.
```

**Bug:** The `+ 1` at the end of the return statement adds 1 to the correct average. For `[10, 20, 30]`, the correct average is 20, but the function returns 21.

</details>

---

## Exercise 4: Role-Based Writing

**Task:** Write a prompt using role prompting that generates a technical blog post introduction about WebAssembly, targeting beginner developers.

<details>
<summary>💡 Solution</summary>

```
You are a senior developer advocate at a major tech company. You have a talent for making complex technical topics accessible to beginners. You write in a friendly, encouraging tone.

Write a blog post introduction (150-200 words) about WebAssembly (Wasm) for developers who have never used it.

Requirements:
- Start with a hook that makes it relevant
- Explain what WebAssembly is in simple terms
- Mention 2-3 key use cases
- End with a teaser for what's coming next in the post
- Use a friendly, encouraging tone

Do NOT use jargon without explaining it.
```

</details>

---

## Exercise 5: Refinement Challenge

**Task:** The following prompt produces bad results. Rewrite it to be more effective:

```
Tell me about Python
```

Improve it to get a focused, useful response about Python's use in data science.

<details>
<summary>💡 Solution</summary>

**Problems with the original:**
- Too vague — could be about anything Python-related
- No format specified
- No audience defined
- No scope

**Improved prompt:**
```
Write a 200-word overview of how Python is used in data science for someone
who knows programming but hasn't done data science before.

Include:
- The top 3 libraries (with one-sentence descriptions)
- One real-world use case
- Why Python is preferred over other languages for this

Format as a clear, scannable document with headers.
```

**Key improvements:**
- Specific topic (Python in data science)
- Audience defined (programmer new to data science)
- Length constraint (200 words)
- Specific sub-topics to cover
- Output format specified

</details>

---

## Bonus Exercise: System Prompt Design

**Task:** Design a system prompt for a customer support chatbot that:
- Answers questions about a fictional SaaS product
- Escalates to a human when it can't help
- Never makes up pricing information
- Responds in a professional but friendly tone

<details>
<summary>💡 Solution</summary>

```
You are a customer support agent for CloudFlow, a cloud-based project management tool.

RULES:
1. Only answer questions about CloudFlow's features, pricing, and policies
2. If you don't know the answer, say: "I'm not sure about that — let me connect you with our support team."
3. NEVER make up pricing information. If asked about pricing, direct them to cloudflow.io/pricing
4. Keep responses under 150 words unless the user asks for detail
5. Use a professional but friendly tone — like a helpful colleague
6. Always confirm the user's question before giving a complex answer
7. For bugs or technical issues, collect: (a) what they were trying to do, (b) what happened, (c) their browser/OS

CURRENT CONTEXT:
- Product: CloudFlow v3.2
- Plan options: Free, Pro ($12/mo), Enterprise (custom)
- Status page: status.cloudflow.io
```

**Why this works:**
- Clear identity and scope
- Escalation rules defined
- Boundaries (no fake pricing)
- Response guidelines
- Practical workflow for technical issues

</details>
