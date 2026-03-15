# Prompt Engineering Exercises 🏋️

Practice these to sharpen your prompting skills. Solutions are at the bottom.

---

## Exercise 1: Fix the Bad Prompt

Rewrite these prompts to be more effective:

**a)** `"Tell me about Python"`

**b)** `"Make it better"`

**c)** `"Summarize. Also translate. And make it shorter."`

---

## Exercise 2: Add Few-Shot Examples

Write a few-shot prompt to classify customer support tickets into:
`Bug Report`, `Feature Request`, `Billing Issue`, `General Question`

Start with this zero-shot version and improve it:
```
Classify this ticket: "I can't log into my account after the update."
```

---

## Exercise 3: Write a System Prompt

Write a system prompt for an AI assistant that helps junior developers review their code. It should:
- Be encouraging but honest
- Always explain the "why" behind suggestions
- Prefer Python best practices
- Not rewrite the entire code, just suggest changes

---

## Exercise 4: Chain-of-Thought

This prompt gives inconsistent results:
```
"Is 17 a prime number?"
```

Add chain-of-thought instruction to make it reliably accurate.

---

## Exercise 5: Build a Template

Create a reusable Python function `generate_job_description(role, company, skills, level)` that uses an LLM to generate a professional job description. The output should always include: Overview, Responsibilities, Requirements, Nice-to-have.

---

## ✅ Solutions

<details>
<summary>Click to reveal Exercise 1 solutions</summary>

**a)** `"Explain Python to a complete beginner in 3 bullet points. Focus on what it's used for and why someone should learn it."`

**b)** `"Rewrite the following paragraph to be more concise (under 50 words) and use an active voice: [text]"`

**c)** Split into three separate prompts — one task per prompt.

</details>

<details>
<summary>Click to reveal Exercise 2 solution</summary>

```
Classify customer support tickets into one of: Bug Report, Feature Request, Billing Issue, General Question.

Examples:
Ticket: "The app crashes when I click export" → Bug Report
Ticket: "Can you add dark mode?" → Feature Request
Ticket: "I was charged twice this month" → Billing Issue
Ticket: "How do I reset my password?" → General Question

Now classify:
Ticket: "I can't log into my account after the update."
```

</details>

<details>
<summary>Click to reveal Exercise 3 solution</summary>

```
You are a senior software engineer mentoring a junior developer.

When reviewing code:
- Be warm and encouraging — remember what it was like to learn
- Point out what they did WELL before suggesting improvements
- For every suggestion, explain WHY it's better (not just that it is)
- Focus on the 2-3 most impactful changes, don't nitpick everything
- Follow Python best practices and PEP 8
- Ask questions to understand their intent before assuming something is wrong
- Never rewrite their entire code — guide them to improve it themselves
```

</details>

<details>
<summary>Click to reveal Exercise 4 solution</summary>

```
Is 17 a prime number? Think step by step.
```

Expected output:
```
Step 1: A prime number is only divisible by 1 and itself.
Step 2: Check if 17 is divisible by any number from 2 to √17 ≈ 4.1
Step 3: 17 ÷ 2 = 8.5 (not whole) ✗
Step 4: 17 ÷ 3 = 5.67 (not whole) ✗
Step 5: 17 ÷ 4 = 4.25 (not whole) ✗
Conclusion: 17 is a prime number. ✅
```

</details>
