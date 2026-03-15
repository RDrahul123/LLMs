"""
Module 3: Simple Terminal Chatbot
===================================
A multi-turn chatbot that remembers conversation history.

Usage:
    python chatbot.py
    python chatbot.py --model gpt-4o --system "You are a pirate."
"""

import argparse
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI()


def run_chatbot(model: str = "gpt-4o-mini", system_prompt: str = "You are a helpful assistant."):
    """Run an interactive terminal chatbot."""
    messages = [{"role": "system", "content": system_prompt}]

    print(f"\n🤖 Chatbot ready! (model: {model})")
    print(f"📝 System: {system_prompt}")
    print("Type 'quit' or 'exit' to stop. Type 'clear' to reset conversation.\n")
    print("-" * 50)

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\nGoodbye!")
            break

        if not user_input:
            continue

        if user_input.lower() in ("quit", "exit"):
            print("\nGoodbye!")
            break

        if user_input.lower() == "clear":
            messages = [{"role": "system", "content": system_prompt}]
            print("🔄 Conversation cleared.\n")
            continue

        if user_input.lower() == "history":
            print("\n📜 Conversation History:")
            for msg in messages[1:]:  # skip system message
                role = "You" if msg["role"] == "user" else "AI"
                print(f"  {role}: {msg['content'][:100]}...")
            continue

        # Add user message to history
        messages.append({"role": "user", "content": user_input})

        # Get response (streaming)
        print("\nAI: ", end="", flush=True)
        full_response = ""

        try:
            stream = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.7,
                stream=True,
            )

            for chunk in stream:
                if chunk.choices[0].delta.content:
                    token = chunk.choices[0].delta.content
                    print(token, end="", flush=True)
                    full_response += token

        except Exception as e:
            print(f"\n❌ Error: {e}")
            messages.pop()  # remove the failed user message
            continue

        print()  # newline after response

        # Add AI response to history
        messages.append({"role": "assistant", "content": full_response})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLM Terminal Chatbot")
    parser.add_argument(
        "--model", default="gpt-4o-mini",
        help="Model to use (default: gpt-4o-mini)"
    )
    parser.add_argument(
        "--system", default="You are a helpful assistant.",
        help="System prompt"
    )
    args = parser.parse_args()

    run_chatbot(model=args.model, system_prompt=args.system)
