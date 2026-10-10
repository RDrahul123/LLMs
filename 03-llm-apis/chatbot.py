"""
Terminal Chatbot
================
A multi-turn chatbot with streaming responses.

Install: pip install openai

Usage:
    python chatbot.py
    python chatbot.py --model gpt-4o
    python chatbot.py --system "You are a pirate. Respond in pirate dialect."
    python chatbot.py --model llama3.1:8b  # with Ollama running
"""

import argparse
import sys

try:
    from openai import OpenAI
except ImportError:
    print("Install openai: pip install openai")
    sys.exit(1)


def create_client(model: str) -> OpenAI:
    """Create OpenAI client. Uses Ollama if model looks local."""
    if ":" in model or model.startswith("llama"):
        return OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
    return OpenAI()


def chat(client: OpenAI, model: str, system_prompt: str, history: list):
    """Send a message and stream the response."""
    messages = [{"role": "system", "content": system_prompt}] + history

    try:
        stream = client.chat.completions.create(
            model=model,
            messages=messages,
            stream=True,
        )

        response = ""
        for chunk in stream:
            if chunk.choices[0].delta.content:
                text = chunk.choices[0].delta.content
                print(text, end="", flush=True)
                response += text
        print()
        return response
    except Exception as e:
        print(f"\nError: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Terminal chatbot with streaming")
    parser.add_argument("--model", type=str, default="gpt-4o-mini", help="Model to use")
    parser.add_argument("--system", type=str, default="You are a helpful assistant.", help="System prompt")
    args = parser.parse_args()

    client = create_client(args.model)
    history = []

    print("=" * 50)
    print("  TERMINAL CHATBOT")
    print(f"  Model: {args.model}")
    print("  Commands: quit, exit, clear, history")
    print("=" * 50)
    print()

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        if user_input.lower() in ("quit", "exit"):
            print("Goodbye!")
            break

        if user_input.lower() == "clear":
            history.clear()
            print("[Conversation cleared]\n")
            continue

        if user_input.lower() == "history":
            print()
            for msg in history:
                role = "You" if msg["role"] == "user" else "Bot"
                print(f"  {role}: {msg['content'][:80]}...")
            print()
            continue

        history.append({"role": "user", "content": user_input})

        print("Bot: ", end="", flush=True)
        response = chat(client, args.model, args.system, history)

        if response:
            history.append({"role": "assistant", "content": response})
        else:
            history.pop()  # Remove failed user message

        print()


if __name__ == "__main__":
    main()
