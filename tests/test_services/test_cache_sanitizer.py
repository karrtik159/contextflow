"""
Unit tests for the PII sanitization and query normalization pipeline.

These tests are fully deterministic — no LLM, no database, no network calls.
They validate the regex-based PII detection, masking, filler stripping,
and isolation flag logic in cache_sanitizer.py.
"""

import json
from pathlib import Path

import pytest

from app.services.cache_sanitizer import sanitize_query


# ── Fixture-driven PII tests ────────────────────────────────

FIXTURE_PATH = Path("tests/test_eval/fixtures/sanitizer_cases.json")
FIXTURE_CASES = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


@pytest.mark.parametrize(
    "case",
    [c for c in FIXTURE_CASES if not c.get("skip")],
    ids=[c["input"][:50] for c in FIXTURE_CASES if not c.get("skip")],
)
def test_pii_detection(case: dict):
    """Each fixture case specifies expected PII types and isolation flag."""
    result = sanitize_query(case["input"])

    assert result.requires_isolation == case["expected_isolation"], (
        f"Isolation mismatch for '{case['input']}': "
        f"got {result.requires_isolation}, expected {case['expected_isolation']}"
    )

    assert sorted(result.detected_pii_types) == sorted(case["expected_pii"]), (
        f"PII types mismatch for '{case['input']}': "
        f"got {result.detected_pii_types}, expected {case['expected_pii']}"
    )


@pytest.mark.parametrize(
    "case",
    [c for c in FIXTURE_CASES if "expected_normalized" in c],
    ids=[c["input"][:50] for c in FIXTURE_CASES if "expected_normalized" in c],
)
def test_normalization_exact(case: dict):
    """Cases with expected_normalized check the full output string."""
    result = sanitize_query(case["input"])
    assert result.normalized_query == case["expected_normalized"]


@pytest.mark.parametrize(
    "case",
    [c for c in FIXTURE_CASES if "expected_normalized_contains" in c],
    ids=[c["input"][:50] for c in FIXTURE_CASES if "expected_normalized_contains" in c],
)
def test_normalization_contains_tag(case: dict):
    """Cases with expected_normalized_contains check that the PII tag is present."""
    result = sanitize_query(case["input"])
    assert case["expected_normalized_contains"] in result.normalized_query


# ── Explicit edge case tests ────────────────────────────────

def test_original_query_preserved():
    """Sanitization must never modify the original_query field."""
    raw = "Email me at test@example.com"
    result = sanitize_query(raw)
    assert result.original_query == raw


def test_multiple_pii_types_detected():
    """Multiple distinct PII types in one query should all be detected."""
    result = sanitize_query("Email john@test.com from IP 10.0.0.1")
    assert "EMAIL" in result.detected_pii_types
    assert "IPV4" in result.detected_pii_types
    assert result.requires_isolation is True


def test_pii_tags_survive_lowercasing():
    """PII tags like [EMAIL] must stay lowercase-insensitive identifiable after .lower()."""
    result = sanitize_query("Contact admin@corp.io")
    # After lowercasing, [EMAIL] becomes [email] — this is the expected behavior
    assert "[email]" in result.normalized_query


def test_whitespace_normalization():
    """Multiple spaces, tabs, and newlines should collapse to single spaces."""
    result = sanitize_query("  what   is    machine   learning  ")
    assert result.normalized_query == "what is machine learning"


def test_filler_prefix_stripping():
    """Conversational fillers at the start should be removed."""
    cases = [
        ("hey, what is AI?", "what is ai?"),
        ("hi, tell me about GPT", "about gpt"),
        ("please explain RAG", "rag"),
        ("can you tell me about vectors", "about vectors"),
        ("i want to know about embeddings", "about embeddings"),
    ]
    for raw, expected in cases:
        result = sanitize_query(raw)
        assert result.normalized_query == expected, f"Failed for '{raw}'"


def test_no_pii_means_global_cache():
    """A clean query without PII should have requires_isolation=False."""
    result = sanitize_query("What is the capital of France?")
    assert result.requires_isolation is False
    assert result.detected_pii_types == []


def test_idempotent_normalization():
    """Running sanitize_query twice on the normalized output should be stable."""
    first = sanitize_query("Hey, what is AI?")
    second = sanitize_query(first.normalized_query)
    assert second.normalized_query == first.normalized_query
    assert second.requires_isolation == first.requires_isolation
