"""
Unit tests for StakeholderRouter (V2)
Tests message routing between patients, doctors, admin, and Fairdoc agent
Keeps under 200 LOC while covering all routing scenarios
"""
import pytest
from unittest.mock import AsyncMock, patch

from src.app2.services.chat.stakeholder_router import (
    StakeholderRouter, 
    MessageRoute, 
    StakeholderType
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def stakeholder_router():
    """Return a StakeholderRouter instance for testing."""
    return StakeholderRouter()


@pytest.fixture
def sample_conversation_id():
    """Return a sample conversation ID."""
    return "conv_test_123456789"


# ---------------------------------------------------------------------------
# Tests: Patient message routing
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_patient_message_routes_to_agent(stakeholder_router, sample_conversation_id):
    """Patient messages should always route to Fairdoc agent first."""
    routes = await stakeholder_router.route_message(
        conversation_id=sample_conversation_id,
        from_stakeholder="patient",
        message="I have a headache",
        medical_outcome=None
    )
    
    assert len(routes) == 1
    assert routes[0].from_stakeholder == "patient"
    assert routes[0].to_stakeholder == "fairdoc_agent" 
    assert routes[0].priority == "medium"
    assert not routes[0].requires_human_review


@pytest.mark.asyncio
async def test_patient_emergency_routes_to_agent_and_doctor(stakeholder_router, sample_conversation_id):
    """Emergency from patient should route to both agent and doctor immediately."""
    routes = await stakeholder_router.route_message(
        conversation_id=sample_conversation_id,
        from_stakeholder="patient", 
        message="I have severe chest pain and can't breathe",
        medical_outcome="emergency"
    )
    
    assert len(routes) == 2
    
    # First route: patient -> agent
    patient_route = routes[0]
    assert patient_route.from_stakeholder == "patient" 
    assert patient_route.to_stakeholder == "fairdoc_agent"
    assert patient_route.priority == "medium"
    
    # Second route: agent -> doctor (emergency alert)
    emergency_route = routes[1]
    assert emergency_route.from_stakeholder == "fairdoc_agent"
    assert emergency_route.to_stakeholder == "doctor"
    assert emergency_route.priority == "emergency"
    assert emergency_route.requires_human_review
    assert "🚨 EMERGENCY" in emergency_route.message_content


# ---------------------------------------------------------------------------
# Tests: Agent response routing
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_agent_response_routes_to_patient(stakeholder_router, sample_conversation_id):
    """Agent responses should route back to patient."""
    routes = await stakeholder_router.route_message(
        conversation_id=sample_conversation_id,
        from_stakeholder="fairdoc_agent",
        message="Can you describe the pain location?",
        medical_outcome="inconclusive"
    )
    
    assert len(routes) == 1
    assert routes[0].from_stakeholder == "fairdoc_agent"
    assert routes[0].to_stakeholder == "patient"
    assert routes[0].priority == "medium"


@pytest.mark.asyncio
async def test_agent_routine_doctor_routes_to_patient_and_doctor(stakeholder_router, sample_conversation_id):
    """Agent recommending routine doctor should notify both patient and doctor."""
    routes = await stakeholder_router.route_message(
        conversation_id=sample_conversation_id,
        from_stakeholder="fairdoc_agent",
        message="Based on your symptoms, I recommend seeing a doctor.",
        medical_outcome="routine_doctor"
    )
    
    assert len(routes) == 2
    
    # Route back to patient
    patient_route = routes[0]
    assert patient_route.to_stakeholder == "patient"
    assert patient_route.priority == "medium"
    
    # Route to doctor for consultation
    doctor_route = routes[1] 
    assert doctor_route.to_stakeholder == "doctor"
    assert doctor_route.priority == "medium"
    assert doctor_route.requires_human_review
    assert "📋 New patient consultation needed" in doctor_route.message_content


# ---------------------------------------------------------------------------
# Tests: Doctor and admin routing
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_doctor_message_routes_to_patient(stakeholder_router, sample_conversation_id):
    """Doctor messages should route to patient with high priority."""
    routes = await stakeholder_router.route_message(
        conversation_id=sample_conversation_id,
        from_stakeholder="doctor",
        message="Please come in for further tests.",
        medical_outcome=None
    )
    
    assert len(routes) == 1
    assert routes[0].from_stakeholder == "doctor"
    assert routes[0].to_stakeholder == "patient" 
    assert routes[0].priority == "high"
    assert "👩‍⚕️ Doctor:" in routes[0].message_content


@pytest.mark.asyncio
async def test_admin_message_routes_to_patient(stakeholder_router, sample_conversation_id):
    """Admin messages should route to patient with low priority."""
    routes = await stakeholder_router.route_message(
        conversation_id=sample_conversation_id,
        from_stakeholder="admin",
        message="Your appointment has been scheduled.",
        medical_outcome=None
    )
    
    assert len(routes) == 1
    assert routes[0].from_stakeholder == "admin"
    assert routes[0].to_stakeholder == "patient"
    assert routes[0].priority == "low"
    assert "🏥 Fairdoc Admin:" in routes[0].message_content


# ---------------------------------------------------------------------------
# Tests: Edge cases and error handling
# ---------------------------------------------------------------------------
@pytest.mark.asyncio 
async def test_empty_message_handling(stakeholder_router, sample_conversation_id):
    """Should handle empty messages gracefully."""
    routes = await stakeholder_router.route_message(
        conversation_id=sample_conversation_id,
        from_stakeholder="patient",
        message="",
        medical_outcome=None
    )
    
    assert len(routes) == 1
    assert routes[0].message_content == ""


@pytest.mark.asyncio
async def test_get_conversation_stakeholders_default(stakeholder_router, sample_conversation_id):
    """Should return default stakeholders for any conversation."""
    stakeholders = await stakeholder_router.get_conversation_stakeholders(sample_conversation_id)
    
    assert len(stakeholders) == 2
    assert "patient" in stakeholders
    assert "fairdoc_agent" in stakeholders


@pytest.mark.asyncio
async def test_message_route_logging(stakeholder_router, sample_conversation_id):
    """Should log message routing for monitoring."""
    with patch('src.app2.services.chat.stakeholder_router.logger') as mock_logger:
        await stakeholder_router.route_message(
            conversation_id=sample_conversation_id,
            from_stakeholder="patient",
            message="Test message",
            medical_outcome="emergency"
        )
        
        # Verify logging was called
        mock_logger.info.assert_called_once()
        call_args = mock_logger.info.call_args[1]
        assert call_args['conversation_id'] == sample_conversation_id
        assert call_args['from_stakeholder'] == "patient"
        assert call_args['medical_outcome'] == "emergency"


# ---------------------------------------------------------------------------
# Tests: Message priority assignment
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(("outcome", "expected_routes", "expected_priorities"), [
    ("emergency", 2, ["medium", "emergency"]),
    ("routine_doctor", 2, ["medium", "medium"]), 
    ("self_care", 1, ["medium"]),
    ("inconclusive", 1, ["medium"]),
    (None, 1, ["medium"])
])
@pytest.mark.asyncio
async def test_priority_assignment_by_outcome(
    stakeholder_router, sample_conversation_id, outcome, expected_routes, expected_priorities
):
    """Test that different medical outcomes produce correct routing priorities."""
    routes = await stakeholder_router.route_message(
        conversation_id=sample_conversation_id,
        from_stakeholder="patient",
        message="Test symptoms",
        medical_outcome=outcome
    )
    
    assert len(routes) == expected_routes
    for i, expected_priority in enumerate(expected_priorities):
        assert routes[i].priority == expected_priority
