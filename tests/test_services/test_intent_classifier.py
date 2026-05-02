"""
Unit tests for the intent classification heuristic layer.

Only tests Layer 1 (keyword matching). Does NOT invoke any LLM.
The heuristic is extracted as a pure function to test independently.
"""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.services.llm_provider import classify_intent

# ── Extract the heuristic logic as a testable pure function ──

def _heuristic_classify(query: str) -> bool | None:
    """Reproduce the heuristic layer from classify_intent().

    Returns:
        False — heuristic caught this as simple chat (no LLM needed)
        None  — heuristic did NOT catch this, would fall through to LLM
    """
    query_lower = query.lower().strip()
    query_words = query_lower.split()

    if not query_words:
        return None  # empty query passes to LLM

    SIMPLE_PATTERNS = {
        "hi", "hello", "hey", "howdy", "yo", "sup",
        "thanks", "thank you", "thx", "ty",
        "bye", "goodbye", "see ya", "later",
        "good morning", "good evening", "good night",
        "how are you", "what's up", "whats up",
        "ok", "okay", "sure", "yes", "no", "yep", "nope",
        "lol", "haha", "nice", "cool", "great", "awesome",
    }

    GREETING_FIRST_WORDS = {
        "hi", "hello", "hey", "thanks", "bye", "ok", "okay", "yo", "sup",
    }

    if query_lower in SIMPLE_PATTERNS:
        return False

    if len(query_words) <= 3 and query_words[0] in GREETING_FIRST_WORDS:
        return False

    return None  # Not caught — would proceed to LLM


# ── Fixture-driven tests ────────────────────────────────────

FIXTURE_PATH = Path("tests/test_eval/fixtures/intent_cases.json")
FIXTURE_CASES = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


@pytest.mark.parametrize(
    "case",
    [c for c in FIXTURE_CASES if c["expect_heuristic"] is False],
    ids=[c["input"][:40] for c in FIXTURE_CASES if c["expect_heuristic"] is False],
)
def test_heuristic_catches_simple_chat(case: dict):
    """These queries should be caught by the heuristic as simple chat."""
    result = _heuristic_classify(case["input"])
    assert result is False, (
        f"Heuristic should catch '{case['input']}' as chat "
        f"(reason: {case['reason']}), but it didn't"
    )


@pytest.mark.parametrize(
    "case",
    [c for c in FIXTURE_CASES if c["expect_heuristic"] is None],
    ids=[c["input"][:40] for c in FIXTURE_CASES if c["expect_heuristic"] is None],
)
def test_heuristic_passes_knowledge_queries(case: dict):
    """These queries should NOT be caught by the heuristic — they need LLM classification."""
    result = _heuristic_classify(case["input"])
    assert result is None, (
        f"Heuristic should NOT catch '{case['input']}' "
        f"(reason: {case['reason']}), but it returned {result}"
    )


# ── Explicit edge cases ─────────────────────────────────────

def test_case_insensitivity():
    """Heuristic should be case-insensitive."""
    assert _heuristic_classify("HI") is False
    assert _heuristic_classify("Hello") is False
    assert _heuristic_classify("THANKS") is False


def test_whitespace_handling():
    """Leading/trailing whitespace should not affect classification."""
    assert _heuristic_classify("  hi  ") is False
    assert _heuristic_classify("  hello  ") is False
    assert _heuristic_classify("   ") is None


@pytest.mark.asyncio
async def test_classify_intent_handles_whitespace_only_query(monkeypatch):
    """Whitespace-only queries should not crash before the LLM fallback."""

    class FakeCompletions:
        async def create(self, **kwargs):
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(content="CHAT"),
                    )
                ]
            )

    fake_client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=FakeCompletions(),
        )
    )
    monkeypatch.setattr("app.services.llm_provider.get_async_llm_client", lambda: fake_client)
    monkeypatch.setattr("app.services.llm_provider._get_model_name", lambda: "test-model")

    assert await classify_intent("   ") is False


def test_four_word_greeting_passes_to_llm():
    """Greetings with 4+ words should NOT be caught by heuristic."""
    result = _heuristic_classify("hey can you help me")
    assert result is None, "4+ word greeting should pass to LLM"


def test_three_word_greeting_caught():
    """Greetings with <=3 words and greeting first word should be caught."""
    assert _heuristic_classify("hi there friend") is False
    assert _heuristic_classify("hey how are") is False


def test_knowledge_query_never_caught():
    """Explicit knowledge queries must never be caught as simple chat."""
    knowledge_queries = [
        "What is machine learning?",
        "Explain the difference between SQL and NoSQL",
        "How does a neural network learn?",
        "Compare Python and JavaScript for backend development",
        "What are the ACID properties in databases?",
    ]
    for q in knowledge_queries:
        result = _heuristic_classify(q)
        assert result is None, f"Knowledge query should not be caught: '{q}'"
