#!/usr/bin/env python3
"""
Evaluation harness for the AI Reliability Assistant.
Run: python eval.py
"""
from __future__ import annotations
import sys
from dotenv import load_dotenv

load_dotenv()

from data_loader import DataLoader
from vector_store import VectorStore
from agent.orchestrator import Orchestrator
from agent.models import AssistantResponse

CASES = [
    {
        "id": "prioritize_this_week",
        "question": "Which machines should maintenance prioritize this week, and why?",
        "must_contain": ["M-006", "M-021", "overdue"],
        "expected_machines": ["M-006", "M-021"],
        "expected_confidence": None,
        "out_of_scope": False,
    },
    {
        "id": "machine_specific_m006",
        "question": "What should we do about machine M-006?",
        "must_contain": ["M-006", "overheating", "bearing"],
        "expected_machines": ["M-006"],
        "expected_confidence": None,
        "out_of_scope": False,
    },
    {
        "id": "site_compare",
        "question": "Compare Site A and Site B and recommend where maintenance should focus first.",
        "must_contain": ["Site A", "Site B"],
        "expected_machines": [],
        "expected_confidence": None,
        "out_of_scope": False,
    },
    {
        "id": "heat_related",
        "question": "Are there any machines with heat- or temperature-related issues we should look at?",
        "must_contain": ["M-006"],
        "expected_machines": ["M-006"],
        "expected_confidence": None,
        "out_of_scope": False,
    },
    {
        "id": "out_of_scope_cyber",
        "question": "Which machine has the best cybersecurity posture?",
        "must_contain": [],
        "expected_machines": [],
        "expected_confidence": "low",
        "out_of_scope": True,
    },
    {
        "id": "data_limited_energy_consumption",
        "question": "What is the energy consumption trend for all motors over the last 6 months?",
        "must_contain": ["energy consumption"],
        "expected_machines": [],
        "expected_confidence": "low",
        "out_of_scope": False,
    },
]

HEDGE_WORDS = ["outside", "scope", "cannot", "not available", "no data", "not support"]


def evaluate_case(case: dict, response: AssistantResponse) -> tuple[bool, list[str]]:
    failures = []
    combined_text = (response.answer + " " + response.reasoning).lower()

    for phrase in case["must_contain"]:
        if phrase.lower() not in combined_text:
            failures.append(f"Missing required phrase: '{phrase}'")

    for mid in case["expected_machines"]:
        if mid not in response.machines_referenced:
            if mid.lower() not in combined_text:
                failures.append(f"Expected machine {mid} not referenced")

    if case["expected_confidence"] and response.confidence != case["expected_confidence"]:
        failures.append(
            f"Expected confidence='{case['expected_confidence']}', got '{response.confidence}'"
        )

    if case["out_of_scope"]:
        has_hedge = any(w in combined_text for w in HEDGE_WORDS)
        if not has_hedge and response.confidence != "low":
            failures.append("Expected out-of-scope hedge or confidence=low")
        has_invented_machines = len(response.machines_referenced) > 0
        if has_invented_machines:
            failures.append(
                f"Expected no machines referenced for out-of-scope, got: {response.machines_referenced}"
            )

    return len(failures) == 0, failures


def main() -> None:
    print("Initializing AI Reliability Assistant...")
    data_loader = DataLoader()
    orchestrator = Orchestrator(data_loader, VectorStore(data_loader))
    print("Ready.\n")

    results = []
    for case in CASES:
        print(f"Running: {case['id']}...")
        try:
            response = orchestrator.run(case["question"])
            passed, failures = evaluate_case(case, response)
        except Exception as e:
            passed = False
            failures = [f"Exception: {e}"]

        results.append((case["id"], passed, failures))

    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    all_passed = True
    for case_id, passed, failures in results:
        status = "[PASS]" if passed else "[FAIL]"
        print(f"{status} {case_id}")
        if not passed:
            all_passed = False
            for f in failures:
                print(f"       ✗ {f}")

    print("=" * 60)
    passed_count = sum(1 for _, p, _ in results if p)
    print(f"Result: {passed_count}/{len(results)} passed")
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
