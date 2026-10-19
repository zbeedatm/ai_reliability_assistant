#!/usr/bin/env python3
from __future__ import annotations
import argparse
import json
import sys
import re
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

from data_loader import DataLoader
from vector_store import VectorStore
from agent.orchestrator import Orchestrator

OUTPUT_DIR = Path(__file__).parent / "output"


def _save_response(question: str, data: dict) -> Path:
    OUTPUT_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    slug = re.sub(r"[^a-z0-9]+", "_", question.lower())[:40].strip("_")
    path = OUTPUT_DIR / f"{timestamp}_{slug}.json"
    path.write_text(json.dumps({"question": question, "response": data}, indent=2))
    return path


def run_question(orchestrator: Orchestrator, question: str) -> None:
    print(f"\nQuestion: {question}")
    print("-" * 60)
    response = orchestrator.run(question)
    data = response.model_dump()
    print(json.dumps(data, indent=2))
    saved = _save_response(question, data)
    print(f"\nSaved to: {saved}")


def interactive_mode(orchestrator: Orchestrator) -> None:
    print("AI Reliability Assistant (type 'quit' to exit)\n")
    while True:
        try:
            question = input("Question: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break
        if not question:
            continue
        if question.lower() in ("quit", "exit", "q"):
            print("Goodbye.")
            break
        run_question(orchestrator, question)
        print()


def main() -> None:
    parser = argparse.ArgumentParser(description="AI Reliability Assistant")
    parser.add_argument("question", nargs="?", help="Natural language question")
    parser.add_argument("--interactive", "-i", action="store_true", help="Interactive REPL mode")
    args = parser.parse_args()

    data_loader = DataLoader()
    orchestrator = Orchestrator(data_loader, VectorStore(data_loader))

    if args.interactive:
        interactive_mode(orchestrator)
    elif args.question:
        run_question(orchestrator, args.question)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
