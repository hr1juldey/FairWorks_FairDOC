import pytest
import asyncio
from test_run_poc import individual_tests

@pytest.mark.asyncio
async def test_individual_poc_components():
    """Runs the individual component tests from run_poc.py as a pytest test."""
    await individual_tests()
