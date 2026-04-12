import re
from dataclasses import dataclass


@dataclass
class SanitizedQuery:
    original_query: str
    normalized_query: str
    requires_isolation: bool
    detected_pii_types: list[str]


# Common native Regex mappings for high-quality PII detection
PII_REGEX_PATTERNS = {
    "EMAIL": r"(?i)\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b",
    "PHONE": r"\b(?:\+?1[-.\s]?)?\(?[2-9][0-8][0-9]\)?[-.\s]?[2-9][0-9]{2}[-.\s]?[0-9]{4}\b",
    "SSN": r"\b(?!000|666)[0-8][0-9]{2}-(?!00)[0-9]{2}-(?!0000)[0-9]{4}\b",
    "CREDIT_CARD": r"\b(?:4[0-9]{12}(?:[0-9]{3})?|5[1-5][0-9]{14}|3[47][0-9]{13}|3(?:0[0-5]|[68][0-9])[0-9]{11}|6(?:011|5[0-9]{2})[0-9]{12}|(?:2131|1800|35\d{3})\d{11})\b",
    "IPV4": r"\b(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\b",
}


def sanitize_query(query: str) -> SanitizedQuery:
    """
    Evaluates a raw user query against robust PII regex mappings.
    If PII is found, it masks the entity (e.g. [EMAIL]), normalizes the string,
    and flags `requires_isolation=True` to sandbox this entry to a per-user semantic cache.
    """
    # 1. Normalize whitespace and lowercase FIRST so regex patterns work uniformly
    normalized = query.strip()
    normalized = re.sub(r"\s+", " ", normalized)
    
    detected_pii = []
    requires_isolation = False

    # 2. Run PII detection on the original-cased string, then mask
    for pii_type, pattern in PII_REGEX_PATTERNS.items():
        if re.search(pattern, normalized):
            detected_pii.append(pii_type)
            requires_isolation = True
            normalized = re.sub(pattern, f"[{pii_type}]", normalized)

    # 3. Lowercase AFTER masking so PII tags stay uppercase ([EMAIL] not [email])
    normalized = normalized.lower()
    
    # 4. Strip conversational filler prefixes to tighten embedding similarity
    query_fillers = [
        r"^hey,?\s*",
        r"^hi,?\s*",
        r"^hello,?\s*",
        r"^can you tell me\s+",
        r"^tell me\s+",
        r"^i want to know\s+",
        r"^please explain\s+",
        r"^please\s+",
    ]
    for filler in query_fillers:
        normalized = re.sub(filler, "", normalized)

    return SanitizedQuery(
        original_query=query,
        normalized_query=normalized.strip(),
        requires_isolation=requires_isolation,
        detected_pii_types=detected_pii,
    )
