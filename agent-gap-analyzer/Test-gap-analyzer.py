"""
Layer 2 test — offline logic check for Gap Analyzer Agent

This bypasses Band entirely. It loads the agent's real SYSTEM_PROMPT, sends
sample inputs (human_input + ai_output pairs) straight to Groq, and checks
the returned JSON matches the expected schema:
  - agent: "gap-analyzer"
  - status: "no-gap" | "gap-found"
  - gap: null | string

It tests the prompt quality — not Band's plumbing.

Run:
    python test_gap_analyzer_layer2.py

Needs GROQ_API_KEY in .env. Does NOT need agent_config.yaml or Band.
Does NOT import agent.py — extracts SYSTEM_PROMPT via regex to avoid
triggering Groq/Band imports at test time.
"""

import json
import os
import re
import sys
from pathlib import Path

from typing import Dict, List

from dotenv import load_dotenv
from groq import Groq

ROOT = Path(__file__).parent

AGENT_FILE = ROOT / "agent.py"

REQUIRED_KEYS = {"agent", "status", "gap"}
VALID_STATUSES = {"no-gap", "gap-found"}

# ============================================================================
# TEST SAMPLES
# ============================================================================

TEST_SAMPLES = [
    {
        "name": "No Gap - Perfect Match",
        "human_input": "Explain blockchain in simple terms",
        "ai_output": """Blockchain is like a shared notebook:
- Everyone can see what's written
- No one can erase past entries
- New entries link to previous ones""",
        "expected_status": "no-gap",
        "expected_gap_is_null": True,
    },
    {
        "name": "Gap - Missing Information",
        "human_input": "Explain blockchain including consensus mechanisms AND security",
        "ai_output": """Blockchain records transactions in blocks.
Each block contains transaction data and a hash.""",
        "expected_status": "gap-found",
        "expected_gap_is_null": False,
    },
    {
        "name": "Gap - Wrong Answer",
        "human_input": "How does photosynthesis work?",
        "ai_output": """Blockchain is a distributed ledger technology.
Nodes validate transactions through consensus.""",
        "expected_status": "gap-found",
        "expected_gap_is_null": False,
    },
    {
        "name": "Gap - Wrong Depth Level",
        "human_input": "Explain quantum computing for a beginner",
        "ai_output": """Quantum computing leverages superposition and entanglement
to achieve exponential speedup via unitary transformations on Hilbert spaces.""",
        "expected_status": "gap-found",
        "expected_gap_is_null": False,
    },
    {
        "name": "Gap - Over-Answering",
        "human_input": "What is DNS?",
        "ai_output": """DNS translates domain names to IP addresses.
The system was invented in 1983. It uses UDP port 53.
The hierarchical structure includes root nameservers, TLD servers, and authoritative servers.
The history of DNS predates HTTPS. RFC 1035 defines the protocol specification...
[continues with 500 more words of unnecessary detail]""",
        "expected_status": "gap-found",
        "expected_gap_is_null": False,
    },
    {
        "name": "No Gap - Complex Technical",
        "human_input": "Explain MVCC in PostgreSQL for someone familiar with databases",
        "ai_output": """MVCC (Multi-Version Concurrency Control) in PostgreSQL allows
concurrent reads and writes without blocking through versioning:
- Each transaction sees a consistent snapshot
- Writers don't block readers
- Readers don't block writers
- Old versions are garbage collected when no longer needed""",
        "expected_status": "no-gap",
        "expected_gap_is_null": True,
    },
    {
        "name": "Gap - Customer Service Scenario",
        "human_input": """Help me troubleshoot my internet. I need:
1. How to reconnect
2. What to check
3. When to call support""",
        "ai_output": """Your internet might be slow.
Wireless technology has improved over the years.""",
        "expected_status": "gap-found",
        "expected_gap_is_null": False,
    },
    {
        "name": "No Gap - Acknowledges Limits",
        "human_input": "Write me a 500-word essay on climate change",
        "ai_output": """I can help with that. Here's a 500-word essay on climate change:
[Full 500-word essay follows]""",
        "expected_status": "no-gap",
        "expected_gap_is_null": True,
    },
]

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def load_system_prompt_from_agent() -> str:
    """
    Load SYSTEM_PROMPT from agent.py by reading it as plain text and
    extracting the prompt with regex.

    This avoids importing agent.py directly, which would trigger
    langchain_groq and Band imports that are not needed for testing.
    """
    if not AGENT_FILE.exists():
        raise FileNotFoundError(f"Cannot find {AGENT_FILE}")

    source = AGENT_FILE.read_text()

    # Match SYSTEM_PROMPT = """..."""
    match = re.search(r'SYSTEM_PROMPT\s*=\s*"""(.*?)"""', source, re.DOTALL)
    if not match:
        raise AttributeError(f"No SYSTEM_PROMPT found in {AGENT_FILE}")

    return match.group(1)


def extract_json(text: str) -> dict:
    """
    Extract JSON from the model's response.
    Tolerates code fences, preamble, etc.
    """
    # Try markdown code fence first
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fenced:
        return json.loads(fenced.group(1))

    # Try raw JSON object
    json_match = re.search(r"\{.*\}", text, re.DOTALL)
    if json_match:
        return json.loads(json_match.group(0))

    raise json.JSONDecodeError("No JSON found in response", text, 0)


def validate_verdict(
    verdict: dict,
    expected_status: str,
    expected_gap_is_null: bool
) -> List[str]:
    """
    Check if verdict matches schema and expectations.
    Returns list of problems (empty = pass).
    """
    problems = []

    # Check required keys
    missing = REQUIRED_KEYS - set(verdict.keys())
    if missing:
        problems.append(f"Missing keys: {sorted(missing)}")

    # Check agent name
    if verdict.get("agent") != "gap-analyzer":
        problems.append(f"agent={verdict.get('agent')!r}, expected 'gap-analyzer'")

    # Check status value
    if verdict.get("status") not in VALID_STATUSES:
        problems.append(f"status={verdict.get('status')!r}, expected 'no-gap' or 'gap-found'")

    # Check status matches expectation
    if verdict.get("status") != expected_status:
        problems.append(
            f"status mismatch: got {verdict.get('status')!r}, expected {expected_status!r}"
        )

    # Check gap field
    gap = verdict.get("gap")

    if expected_gap_is_null:
        if gap is not None:
            problems.append(f"gap should be null, got {gap!r}")
    else:
        if gap is None:
            problems.append("gap should have description, got null")
        elif not isinstance(gap, str):
            problems.append(f"gap should be string, got {type(gap).__name__}")
        elif len(gap.strip()) < 10:
            problems.append(f"gap description too short ({len(gap)} chars)")

    return problems


def format_sample_input(sample: Dict) -> str:
    """Format sample input for the model"""
    return f"""Human asked: {sample['human_input']}

AI produced: {sample['ai_output']}

Analyze if there is a gap."""


# ============================================================================
# MAIN TEST RUNNER
# ============================================================================

def run_tests() -> int:
    """Run all test samples. Returns 0 if all pass, 1 if any fail."""

    load_dotenv()

    client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    # Load system prompt via regex — no agent.py import
    try:
        system_prompt = load_system_prompt_from_agent()
    except (FileNotFoundError, AttributeError) as e:
        print(f"Failed to load system prompt: {e}")
        return 1

    print("\n" + "=" * 70)
    print("GAP ANALYZER LAYER 2 TEST — OFFLINE PROMPT QUALITY CHECK")
    print("=" * 70)
    print(f"\nLoaded SYSTEM_PROMPT from: {AGENT_FILE}")
    print(f"Total test samples: {len(TEST_SAMPLES)}\n")

    total = 0
    passed = 0
    failed = 0

    for sample in TEST_SAMPLES:
        total += 1

        print(f"\n[Test {total}] {sample['name']}")
        print("-" * 70)
        print(f"Human input:     {sample['human_input']}")
        print(f"AI output:       {sample['ai_output']}")
        print(f"Expected status: {sample['expected_status']}")

        # Call Groq to evaluate the prompt
        try:
            user_message = format_sample_input(sample)

            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                max_tokens=500,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ]
            )

            reply = response.choices[0].message.content

        except Exception as e:
            print(f"FAIL — Groq API error: {e}")
            failed += 1
            continue

        # Extract JSON
        try:
            verdict = extract_json(reply)
        except json.JSONDecodeError:
            print(f"FAIL — Could not parse JSON from response")
            print(f"   Raw reply: {reply[:100]}...")
            failed += 1
            continue

        # Validate
        problems = validate_verdict(
            verdict,
            sample["expected_status"],
            sample["expected_gap_is_null"]
        )

        print(f"Agent:           {verdict.get('agent', 'MISSING')}")
        print(f"Status:          {verdict.get('status', 'MISSING')}")

        gap_val = verdict.get("gap")
        if gap_val is None:
            print(f"Gap:             null")
        else:
            print(f"Gap:             {gap_val}")

        if problems:
            print(f"\n FAIL")
            for problem in problems:
                print(f"   • {problem}")
            failed += 1
        else:
            print(f"\n PASS")
            passed += 1

    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    print(f"Total:   {total}")
    print(f"Passed:  {passed} ")
    print(f"Failed:  {failed} ")
    print(f"Rate:    {passed}/{total} ({100 * passed // total}%)")

    if failed == 0:
        print("\n🎉 ALL TESTS PASSED — Agent prompt is working correctly!")
        return 0
    else:
        print(f"\n⚠️  {failed} test(s) failed — Review prompt and rules")
        return 1


if __name__ == "__main__":
    sys.exit(run_tests())