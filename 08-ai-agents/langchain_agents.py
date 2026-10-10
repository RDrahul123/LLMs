"""
LangChain Agent Examples

Demonstrates LangChain's agent capabilities:
- Simple ReAct agent
- Custom tools
- Conversation memory
- AgentExecutor with error handling
- Comparison with from-scratch agents

Requirements:
    pip install langchain langchain-openai langchain-anthropic

Usage:
    export OPENAI_API_KEY="your-key"
    export ANTHROPIC_API_KEY="your-key"
    python langchain_agents.py
"""

from __future__ import annotations

import json
import math
import os
import re
from typing import Any, Callable


# ---------------------------------------------------------------------------
# Installation Notes (printed at runtime)
# ---------------------------------------------------------------------------

INSTALL_NOTES = """
LangChain Agent Setup:
  pip install langchain langchain-openai langchain-anthropic

For Anthropic support:
  pip install langchain-anthropic

Environment variables:
  export OPENAI_API_KEY="your-key"
  export ANTHROPIC_API_KEY="your-key"
"""


# ---------------------------------------------------------------------------
# Simple ReAct Agent with LangChain
# ---------------------------------------------------------------------------


def simple_react_agent() -> None:
    """Build and run a basic ReAct agent using LangChain."""
    try:
        from langchain_openai import ChatOpenAI
        from langchain_core.tools import tool
        from langchain.agents import create_tool_calling_agent, AgentExecutor
        from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
    except ImportError:
        print("[Simple ReAct] Skipping — langchain packages not installed.")
        print(INSTALL_NOTES)
        return

    if not os.environ.get("OPENAI_API_KEY"):
        print("[Simple ReAct] Skipping — OPENAI_API_KEY not set.")
        return

    print("=" * 60)
    print("SIMPLE REACT AGENT (LangChain)")
    print("=" * 60)

    @tool
    def calculator(expression: str) -> str:
        """Evaluate a mathematical expression. Example: '2 + 3 * 4'"""
        allowed = {k: v for k, v in math.__dict__.items() if not k.startswith("_")}
        allowed.update({"abs": abs, "round": round, "min": min, "max": max})
        sanitized = re.sub(r"[^0-9+\-*/().,%s ]" % "".join(allowed.keys()), "", expression)
        try:
            return str(eval(sanitized, {"__builtins__": {}}, allowed))
        except Exception as e:
            return f"Error: {e}"

    @tool
    def word_count(text: str) -> str:
        """Count the number of words in a text."""
        words = text.split()
        return f"Word count: {len(words)}"

    @tool
    def get_length(text: str) -> str:
        """Get the character length of a text string."""
        return f"Character count: {len(text)}"

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant. Use tools when needed."),
        MessagesPlaceholder(variable_name="chat_history", optional=True),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    tools = [calculator, word_count, get_length]
    agent = create_tool_calling_agent(llm, tools, prompt)
    executor = AgentExecutor(agent=agent, tools=tools, verbose=True, max_iterations=5)

    queries = [
        "What is 15 * 37 + 128?",
        "How many words are in 'The quick brown fox jumps over the lazy dog'?",
        "Calculate sqrt(144) * 3 and also count the characters in 'Hello World'",
    ]

    for query in queries:
        print(f"\n{'=' * 50}")
        print(f"Query: {query}")
        result = executor.invoke({"input": query})
        print(f"Answer: {result['output']}")


# ---------------------------------------------------------------------------
# Agent with Custom Tools
# ---------------------------------------------------------------------------


def custom_tools_agent() -> None:
    """Build an agent with custom-decorated tools."""
    try:
        from langchain_openai import ChatOpenAI
        from langchain_core.tools import tool
        from langchain.agents import create_tool_calling_agent, AgentExecutor
        from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
    except ImportError:
        print("[Custom Tools] Skipping — langchain packages not installed.")
        return

    if not os.environ.get("OPENAI_API_KEY"):
        print("[Custom Tools] Skipping — OPENAI_API_KEY not set.")
        return

    print("\n" + "=" * 60)
    print("AGENT WITH CUSTOM TOOLS")
    print("=" * 60)

    @tool
    def analyze_sentiment(text: str) -> str:
        """Analyze the sentiment of a text. Returns positive, negative, or neutral with confidence."""
        positive_words = {"good", "great", "excellent", "amazing", "love", "happy", "wonderful", "fantastic"}
        negative_words = {"bad", "terrible", "awful", "hate", "sad", "horrible", "poor", "worst"}

        words = set(text.lower().split())
        pos_count = len(words & positive_words)
        neg_count = len(words & negative_words)
        total = pos_count + neg_count

        if total == 0:
            return json.dumps({"sentiment": "neutral", "confidence": 0.5})

        if pos_count > neg_count:
            conf = pos_count / total
            return json.dumps({"sentiment": "positive", "confidence": round(conf, 2)})
        elif neg_count > pos_count:
            conf = neg_count / total
            return json.dumps({"sentiment": "negative", "confidence": round(conf, 2)})
        else:
            return json.dumps({"sentiment": "neutral", "confidence": 0.5})

    @tool
    def generate_summary(text: str, max_sentences: int = 3) -> str:
        """Generate a summary of the given text by extracting key sentences."""
        sentences = re.split(r"[.!?]+", text)
        sentences = [s.strip() for s in sentences if s.strip()]

        if len(sentences) <= max_sentences:
            return text

        scored = []
        word_freq: dict[str, int] = {}
        for sentence in sentences:
            for word in sentence.lower().split():
                word_freq[word] = word_freq.get(word, 0) + 1

        for sentence in sentences:
            score = sum(word_freq.get(w.lower(), 0) for w in sentence.split())
            scored.append((score, sentence))

        scored.sort(key=lambda x: x[0], reverse=True)
        top_sentences = [s for _, s in scored[:max_sentences]]
        return ". ".join(top_sentences) + "."

    @tool
    def extract_keywords(text: str, top_n: int = 5) -> str:
        """Extract the most important keywords from text based on frequency."""
        stop_words = {"the", "a", "an", "is", "are", "was", "were", "in", "on", "at",
                       "to", "for", "of", "with", "by", "and", "or", "but", "not", "this",
                       "that", "it", "from", "as", "be", "has", "have", "had", "do", "does"}

        words = re.findall(r"\b[a-z]+\b", text.lower())
        filtered = [w for w in words if w not in stop_words and len(w) > 2]

        freq: dict[str, int] = {}
        for word in filtered:
            freq[word] = freq.get(word, 0) + 1

        sorted_words = sorted(freq.items(), key=lambda x: x[1], reverse=True)[:top_n]
        return json.dumps([{"keyword": w, "count": c} for w, c in sorted_words])

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a text analysis assistant. Use tools to analyze text."),
        MessagesPlaceholder(variable_name="chat_history", optional=True),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    tools = [analyze_sentiment, generate_summary, extract_keywords]
    agent = create_tool_calling_agent(llm, tools, prompt)
    executor = AgentExecutor(agent=agent, tools=tools, verbose=True, max_iterations=5)

    sample_text = (
        "Artificial intelligence has made great progress in recent years. "
        "Machine learning models are amazing at pattern recognition. However, "
        "there are concerns about AI safety and bias. The good applications "
        "include healthcare and scientific research, but bad actors could "
        "use AI for harmful purposes. Overall, AI is a wonderful tool when "
        "used responsibly."
    )

    queries = [
        f"Analyze the sentiment of this text: '{sample_text}'",
        f"Summarize this text in 2 sentences: '{sample_text}'",
        f"Extract keywords from: '{sample_text}'",
    ]

    for query in queries:
        print(f"\n{'=' * 50}")
        print(f"Query: {query[:80]}...")
        result = executor.invoke({"input": query})
        print(f"Answer: {result['output']}")


# ---------------------------------------------------------------------------
# Conversation Memory with LangChain
# ---------------------------------------------------------------------------


def conversation_memory_demo() -> None:
    """Demonstrate conversation memory with LangChain agents."""
    try:
        from langchain_openai import ChatOpenAI
        from langchain_core.tools import tool
        from langchain.agents import create_tool_calling_agent, AgentExecutor
        from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
        from langchain.memory import ConversationBufferMemory
    except ImportError:
        print("[Conversation Memory] Skipping — langchain packages not installed.")
        return

    if not os.environ.get("OPENAI_API_KEY"):
        print("[Conversation Memory] Skipping — OPENAI_API_KEY not set.")
        return

    print("\n" + "=" * 60)
    print("CONVERSATION MEMORY")
    print("=" * 60)

    @tool
    def get_time() -> str:
        """Get the current time."""
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    @tool
    def calculator(expression: str) -> str:
        """Evaluate a mathematical expression."""
        allowed = {k: v for k, v in math.__dict__.items() if not k.startswith("_")}
        allowed.update({"abs": abs, "round": round})
        sanitized = re.sub(r"[^0-9+\-*/().,%s ]" % "".join(allowed.keys()), "", expression)
        try:
            return str(eval(sanitized, {"__builtins__": {}}, allowed))
        except Exception as e:
            return f"Error: {e}"

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant with memory."),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    tools = [get_time, calculator]
    agent = create_tool_calling_agent(llm, tools, prompt)
    executor = AgentExecutor(
        agent=agent,
        tools=tools,
        memory=memory,
        verbose=True,
        max_iterations=5,
    )

    conversation = [
        "My name is Alice and I'm a software engineer.",
        "What's the current time?",
        "Can you calculate 25 * 17?",
        "What was my name again?",
        "Can you calculate the result of that multiplication plus 100?",
    ]

    for message in conversation:
        print(f"\n{'=' * 50}")
        print(f"User: {message}")
        result = executor.invoke({"input": message})
        print(f"Assistant: {result['output']}")

    print(f"\nMemory contents ({len(memory.chat_memory.messages)} messages):")
    for msg in memory.chat_memory.messages:
        print(f"  {msg.type}: {msg.content[:80]}...")


# ---------------------------------------------------------------------------
# AgentExecutor with Error Handling
# ---------------------------------------------------------------------------


def agent_with_error_handling() -> None:
    """Demonstrate error handling patterns with AgentExecutor."""
    try:
        from langchain_openai import ChatOpenAI
        from langchain_core.tools import tool
        from langchain.agents import create_tool_calling_agent, AgentExecutor
        from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
    except ImportError:
        print("[Error Handling] Skipping — langchain packages not installed.")
        return

    if not os.environ.get("OPENAI_API_KEY"):
        print("[Error Handling] Skipping — OPENAI_API_KEY not set.")
        return

    print("\n" + "=" * 60)
    print("AGENT ERROR HANDLING")
    print("=" * 60)

    @tool
    def risky_division(a: float, b: float) -> str:
        """Divide a by b. Will fail if b is zero."""
        if b == 0:
            raise ValueError("Cannot divide by zero")
        return str(a / b)

    @tool
    def always_fails(query: str) -> str:
        """A tool that always raises an error (for testing)."""
        raise RuntimeError("This tool intentionally fails")

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant."),
        MessagesPlaceholder(variable_name="chat_history", optional=True),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    tools = [risky_division, always_fails]
    agent = create_tool_calling_agent(llm, tools, prompt)

    executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        max_iterations=3,
        handle_parsing_errors=True,
        return_intermediate_steps=True,
    )

    queries = [
        "Divide 100 by 0",
        "Divide 100 by 5",
    ]

    for query in queries:
        print(f"\n{'=' * 50}")
        print(f"Query: {query}")
        try:
            result = executor.invoke({"input": query})
            print(f"Answer: {result.get('output', 'No output')}")
            if "intermediate_steps" in result:
                print(f"Steps taken: {len(result['intermediate_steps'])}")
        except Exception as e:
            print(f"Agent error caught: {type(e).__name__}: {e}")


# ---------------------------------------------------------------------------
# Comparison: LangChain vs From-Scratch
# ---------------------------------------------------------------------------


def comparison_summary() -> None:
    """Print a comparison of LangChain agents vs from-scratch agents."""
    print("\n" + "=" * 60)
    print("COMPARISON: LangChain vs From-Scratch Agent")
    print("=" * 60)

    comparison = """
┌─────────────────────┬──────────────────────┬──────────────────────┐
│ Feature             │ LangChain            │ From-Scratch         │
├─────────────────────┼──────────────────────┼──────────────────────┤
│ Setup time          │ Minutes              │ Hours                │
│ Boilerplate         │ Minimal              │ Significant          │
│ Tool integration    │ @tool decorator      │ Manual schema + exec │
│ Memory              │ Built-in classes     │ Custom implementation│
│ Error handling      │ AgentExecutor        │ Manual try/except    │
│ Streaming           │ Built-in             │ Manual               │
│ Debugging           │ Verbose mode, LangSmith│ Print statements   │
│ Dependencies        │ langchain ecosystem  │ Only openai          │
│ Control             │ Framework conventions│ Full control         │
│ Breaking changes    │ API evolves rapidly  │ Stable               │
│ Learning value      │ Higher-level concepts│ Deep understanding   │
│ Production readiness│ Mature ecosystem     │ Depends on impl      │
│ Community           │ Large, active        │ N/A                  │
│ Customization       │ Override callbacks   │ Unlimited            │
│ Token efficiency    │ Good                 │ Optimizable          │
└─────────────────────┴──────────────────────┴──────────────────────┘

When to use LangChain:
  - Rapid prototyping
  - Team projects with multiple developers
  - Need for complex chains and memory patterns
  - Integration with LangSmith for observability

When to build from scratch:
  - Learning how agents work internally
  - Production systems needing fine-grained control
  - Minimal dependency requirements
  - Custom agent architectures
  - When framework overhead is unacceptable
"""
    print(comparison)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Run all LangChain agent examples."""
    print("LangChain Agent Examples")
    print("=" * 60)
    print(INSTALL_NOTES)

    simple_react_agent()
    custom_tools_agent()
    conversation_memory_demo()
    agent_with_error_handling()
    comparison_summary()

    print("\n" + "=" * 60)
    print("All LangChain agent examples completed.")


if __name__ == "__main__":
    main()
