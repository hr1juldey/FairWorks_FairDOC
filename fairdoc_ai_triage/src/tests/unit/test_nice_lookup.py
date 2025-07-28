"""
Unit tests for NICELookupService (V2)
Keeps <200 LOC while exercising positive & negative paths.
"""
from pytest import mark

from src.app2.services.context.nice_lookup import NICELookupService

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@mark.fixture(scope="module")
def nice_lookup():
    """Return a NICELookupService loaded with default seed data."""
    return NICELookupService()

# ---------------------------------------------------------------------------
# Tests: matching keywords
# ---------------------------------------------------------------------------

@mark.parametrize(
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
# Tests: no match returns NONE
# ---------------------------------------------------------------------------

async def test_find_relevant_protocols_no_match(nice_lookup):
    """Unknown symptom strings should yield protocol_code == 'NONE'."""
    result = nice_lookup.find_relevant_protocols("I have an itchy elbow")
    assert result["protocol_code"] == "NONE"
    assert result["protocol_text"] == ""
