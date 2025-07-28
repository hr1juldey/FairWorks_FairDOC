"""
Unit tests for NICELookupService (V2)
Keeps <200 LOC while exercising positive & negative paths.
File: src/tests/unit/test_nice_lookup.py
"""
import pytest 

from src.app2.services.context.nice_lookup import NICELookupService

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def nice_lookup():
    """Return a NICELookupService loaded with default seed data."""
    return NICELookupService()

# ---------------------------------------------------------------------------
# Tests: matching keywords
# ---------------------------------------------------------------------------

@pytest.parametrize(
    ("query", "expected_code"),
    [
        ("My head has been hurting – maybe a headache?", "NG127_HEADACHE"),
        ("I feel crushing chest_pain when I walk upstairs", "CG95_CHEST_PAIN"),
    ],
)
async def test_find_relevant_protocols_positive(nice_lookup, query, expected_code):
    """Ensure known keywords map to correct NICE protocol codes."""
    result = nice_lookup.find_relevant_protocols(query)
    assert result["protocol_code"] == expected_code
    # Sanity-check that protocol text is non-empty
    assert result["protocol_text"].startswith("Condition:")

# ---------------------------------------------------------------------------
# Tests: case-insensitivity & punctuation handling
# ---------------------------------------------------------------------------
async def test_protocol_lookup_case_punctuation(nice_lookup):
    """Lookup should ignore case and punctuation characters."""
    text = "  HEADACHE!!!   "
    result = nice_lookup.find_relevant_protocols(text)
    assert result["protocol_code"] == "NG127_HEADACHE"

# ---------------------------------------------------------------------------
# Tests: no match returns NONE
# ---------------------------------------------------------------------------
async def test_find_relevant_protocols_no_match(nice_lookup):
    """Unknown symptom strings should yield protocol_code == 'NONE'."""
    result = nice_lookup.find_relevant_protocols("I have an itchy elbow")
    assert result["protocol_code"] == "NONE"
    assert result["protocol_text"] == ""
