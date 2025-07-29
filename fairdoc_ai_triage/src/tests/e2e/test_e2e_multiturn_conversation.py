"""
E2E Test: Multi-turn Medical Conversation Flow

Tests complete multi-turn conversations from initial vague symptoms through
progressive questioning to final medical outcome, including conversation
state management, NICE protocol integration, and proper workflow completion.

File: src/tests/e2e/test_e2e_multiturn_conversation.py
"""

import pytest
import asyncio
import json
import time
from datetime import datetime
from httpx import AsyncClient
from unittest.mock import patch, AsyncMock
import uuid

from src.app2.main_v2 import app
from src.app2.models.schemas.multiturn_chat import StakeholderRole, ChatProvider

class TestMultiturnConversationE2E:
    """Complete multi-turn conversation workflow testing"""
    
    @pytest.fixture(autouse=True)
    async def setup_conversation_environment(self):
        """Setup environment for multi-turn conversation testing"""
        self.test_session_id = f"multiturn_test_{int(time.time())}"
        self.conversation_log = {
            "test_start": datetime.utcnow().isoformat(),
            "conversations": [],
            "state_transitions": [],
            "nice_protocol_usage": [],
            "performance_metrics": {}
        }
        
        # Track NICE protocol lookups
        self.nice_lookups = []
        
        def track_nice_lookup(self, symptoms):
            lookup = {
                "timestamp": datetime.utcnow().isoformat(),
                "symptoms": symptoms,
                "protocol_found": "NG127_HEADACHE" if "headache" in symptoms.lower() else "NONE"
            }
            self.nice_lookups.append(lookup)
            return {
                "protocol_code": lookup["protocol_found"],
                "protocol_text": "NICE Guidelines for headache assessment" if lookup["protocol_found"] != "NONE" else ""
            }
        
        with patch('src.app2.services.context.nice_lookup.NICELookupService.find_relevant_protocols',
                   side_effect=track_nice_lookup):
            yield
    
    @pytest.mark.asyncio
    async def test_headache_progression_multiturn_flow(self):
        """Test complete headache conversation from vague to specific symptoms"""
        async with AsyncClient(app=app, base_url="http://test") as client:
            
            patient_id = f"{self.test_session_id}_headache_patient"
            conversation_id = None
            turn_count = 0
            
            # === TURN 1: Initial vague symptom ===
            turn_count += 1
            initial_payload = {
                "user_message": "I have a headache",
                "stakeholder_role": StakeholderRole.PATIENT.value,
                "stakeholder_id": patient_id,
                "chat_provider": ChatProvider.API_DIRECT.value
            }
            
            start_time = time.time()
            response1 = await client.post("/api/v2/medical/chat", json=initial_payload)
            turn1_time = time.time() - start_time
            
            assert response1.status_code == 200
            data1 = response1.json()
            conversation_id = data1["conversation_id"]
            
            # Should be inconclusive and ask follow-up
            assert data1["medical_outcome"] in ["need_more_questions", "inconclusive"]
            assert data1["is_conversation_complete"] is False
            assert data1["next_question"] is not None
            assert data1["turn_count"] == turn_count
            
            self.conversation_log["conversations"].append({
                "turn": turn_count,
                "user_input": initial_payload["user_message"],
                "agent_response": data1.get("agent_response", ""),
                "outcome": data1["medical_outcome"],
                "confidence": data1["confidence_score"],
                "next_question": data1.get("next_question"),
                "response_time": turn1_time
            })
            
            print(f"Turn {turn_count}: '{initial_payload['user_message']}'")
            print(f"  → Agent: {data1.get('next_question', 'No question')}")
            print(f"  → Outcome: {data1['medical_outcome']} ({data1['confidence_score']}%)")
            
            # === TURN 2: Duration information ===
            turn_count += 1
            duration_payload = {
                "conversation_id": conversation_id,
                "user_message": "It started this morning and it's getting worse",
                "stakeholder_role": StakeholderRole.PATIENT.value,
                "stakeholder_id": patient_id,
                "chat_provider": ChatProvider.API_DIRECT.value
            }
            
            start_time = time.time()
            response2 = await client.post("/api/v2/medical/chat", json=duration_payload)
            turn2_time = time.time() - start_time
            
            assert response2.status_code == 200
            data2 = response2.json()
            
            # Verify conversation continuity
            assert data2["conversation_id"] == conversation_id
            assert data2["turn_count"] == turn_count
            
            self.conversation_log["conversations"].append({
                "turn": turn_count,
                "user_input": duration_payload["user_message"],
                "agent_response": data2.get("agent_response", ""),
                "outcome": data2["medical_outcome"],
                "confidence": data2["confidence_score"],
                "next_question": data2.get("next_question"),
                "response_time": turn2_time
            })
            
            print(f"Turn {turn_count}: '{duration_payload['user_message']}'")
            print(f"  → Agent: {data2.get('next_question', data2.get('agent_response', ''))}")
            print(f"  → Outcome: {data2['medical_outcome']} ({data2['confidence_score']}%)")
            
            # === TURN 3: Severity and associated symptoms ===
            if not data2["is_conversation_complete"]:
                turn_count += 1
                severity_payload = {
                    "conversation_id": conversation_id,
                    "user_message": "It's very severe, about 8/10, and I feel nauseous and sensitive to light",
                    "stakeholder_role": StakeholderRole.PATIENT.value,
                    "stakeholder_id": patient_id,
                    "chat_provider": ChatProvider.API_DIRECT.value
                }
                
                start_time = time.time()
                response3 = await client.post("/api/v2/medical/chat", json=severity_payload)
                turn3_time = time.time() - start_time
                
                assert response3.status_code == 200
                data3 = response3.json()
                
                # Severe headache with red flags should escalate
                assert data3["medical_outcome"] in ["routine_doctor_consultation", "emergency"]
                
                # If red flags present, should complete conversation
                if any(flag in data3.get("red_flags", []) for flag in ["photophobia", "nausea", "severe"]):
                    assert data3["confidence_score"] >= 75
                
                self.conversation_log["conversations"].append({
                    "turn": turn_count,
                    "user_input": severity_payload["user_message"],
                    "agent_response": data3.get("agent_response", ""),
                    "outcome": data3["medical_outcome"],
                    "confidence": data3["confidence_score"],
                    "red_flags": data3.get("red_flags", []),
                    "response_time": turn3_time,
                    "conversation_complete": data3["is_conversation_complete"]
                })
                
                print(f"Turn {turn_count}: '{severity_payload['user_message']}'")
                print(f"  → Final Outcome: {data3['medical_outcome']} ({data3['confidence_score']}%)")
                print(f"  → Red Flags: {data3.get('red_flags', [])}")
                print(f"  → Complete: {data3['is_conversation_complete']}")
            
            # === VERIFY CONVERSATION STATE MANAGEMENT ===
            state_response = await client.get(f"/api/v2/medical/chat/{conversation_id}/state")
            assert state_response.status_code == 200
            
            state_data = state_response.json()
            assert state_data["conversation_id"] == conversation_id
            assert state_data["turn_count"] == turn_count
            assert state_data["user_id"] == patient_id
            
            self.conversation_log["state_transitions"].append({
                "conversation_id": conversation_id,
                "final_turn_count": turn_count,
                "final_status": state_data.get("status"),
                "final_outcome": state_data.get("current_outcome")
            })
            
            # === VERIFY CONVERSATION HISTORY ===
            history_response = await client.get(f"/api/v2/medical/chat/{conversation_id}/history")
            assert history_response.status_code == 200
            
            history_data = history_response.json()
            assert history_data["conversation_id"] == conversation_id
            assert len(history_data["turns"]) == turn_count
            
            # Verify each turn is properly recorded
            for i, turn in enumerate(history_data["turns"], 1):
                assert "user_message" in turn or "user_response" in turn
                assert turn.get("turn_number", i) == i
            
            print(f"\n✅ Multi-turn conversation completed with {turn_count} turns")
            print(f"📊 Final outcome: {state_data.get('current_outcome')}")
    
    @pytest.mark.asyncio
    async def test_conversation_branching_logic(self):
        """Test different conversation paths based on symptom severity"""
        async with AsyncClient(app=app, base_url="http://test") as client:
            
            # Test mild symptoms path
            mild_payload = {
                "user_message": "I have a very mild headache, barely noticeable",
                "stakeholder_role": StakeholderRole.PATIENT.value,
                "stakeholder_id": f"{self.test_session_id}_mild_patient",
                "chat_provider": ChatProvider.API_DIRECT.value
            }
            
            mild_response = await client.post("/api/v2/medical/chat", json=mild_payload)
            mild_data = mild_response.json()
            
            # Mild symptoms might lead to self-care or need more questions
            assert mild_data["medical_outcome"] in ["self_care_advice", "need_more_questions"]
            
            # Test severe symptoms path
            severe_payload = {
                "user_message": "I have the worst headache of my life, sudden onset, with neck stiffness",
                "stakeholder_role": StakeholderRole.PATIENT.value,
                "stakeholder_id": f"{self.test_session_id}_severe_patient",
                "chat_provider": ChatProvider.API_DIRECT.value
            }
            
            severe_response = await client.post("/api/v2/medical/chat", json=severe_payload)
            severe_data = severe_response.json()
            
            # Severe symptoms with red flags should trigger emergency
            assert severe_data["medical_outcome"] in ["emergency", "routine_doctor_consultation"]
            assert severe_data["confidence_score"] >= 80
            
            # Should have red flags for severe presentation
            red_flags = severe_data.get("red_flags", [])
            assert len(red_flags) > 0
            
            self.conversation_log["nice_protocol_usage"] = self.nice_lookups
            
            print(f"Mild symptoms outcome: {mild_data['medical_outcome']}")
            print(f"Severe symptoms outcome: {severe_data['medical_outcome']}")
            print(f"NICE protocol lookups: {len(self.nice_lookups)}")
    
    @pytest.mark.asyncio
    async def test_conversation_timeout_and_limits(self):
        """Test conversation handles maximum turn limits and timeout scenarios"""
        async with AsyncClient(app=app, base_url="http://test") as client:
            
            patient_id = f"{self.test_session_id}_limit_test"
            conversation_id = None
            max_turns = 8  # Based on FAIRDOC_V2_MAX_CONVERSATION_TURNS
            
            # Start with vague symptoms
            initial_payload = {
                "user_message": "I don't feel well",
                "stakeholder_role": StakeholderRole.PATIENT.value,
                "stakeholder_id": patient_id,
                "chat_provider": ChatProvider.API_DIRECT.value
            }
            
            response = await client.post("/api/v2/medical/chat", json=initial_payload)
            data = response.json()
            conversation_id = data["conversation_id"]
            
            turn_count = 1
            
            # Continue conversation until completion or limit
            while turn_count < max_turns and not data.get("is_conversation_complete", False):
                turn_count += 1
                
                # Provide progressively vague responses to test limit handling
                vague_responses = [
                    "I still don't feel right",
                    "It's hard to describe",
                    "Maybe a little better",
                    "Not sure, still the same",
                    "I think it might be nothing",
                    "Could be stress",
                    "I'm not sure what's wrong"
                ]
                
                next_message = vague_responses[min(turn_count - 2, len(vague_responses) - 1)]
                
                payload = {
                    "conversation_id": conversation_id,
                    "user_message": next_message,
                    "stakeholder_role": StakeholderRole.PATIENT.value,
                    "stakeholder_id": patient_id,
                    "chat_provider": ChatProvider.API_DIRECT.value
                }
                
                response = await client.post("/api/v2/medical/chat", json=payload)
                
                if response.status_code != 200:
                    break
                    
                data = response.json()
                
                print(f"Turn {turn_count}: '{next_message}' → {data['medical_outcome']}")
            
            # Verify conversation handling at limits
            if turn_count >= max_turns:
                # Should either complete or handle gracefully
                assert data["is_conversation_complete"] or data["medical_outcome"] != "need_more_questions"
                print(f"✅ Conversation properly handled at {turn_count} turns")
            
            self.conversation_log["performance_metrics"]["max_turns_test"] = {
                "turns_reached": turn_count,
                "conversation_completed": data.get("is_conversation_complete", False),
                "final_outcome": data.get("medical_outcome")
            }
    
    def teardown_method(self):
        """Print comprehensive conversation analysis"""
        self.conversation_log["test_end"] = datetime.utcnow().isoformat()
        
        print(f"\n{'=' * 80}")
        print("MULTI-TURN CONVERSATION E2E TEST ANALYSIS")
        print(f"{'=' * 80}")
        print(f"Session ID: {self.test_session_id}")
        
        if self.conversation_log["conversations"]:
            total_turns = len(self.conversation_log["conversations"])
            avg_response_time = sum(conv.get("response_time", 0) for conv in self.conversation_log["conversations"]) / total_turns
            
            print(f"Total Conversation Turns: {total_turns}")
            print(f"Average Response Time: {avg_response_time:.3f}s")
            
            print("\nCONVERSATION FLOW:")
            for conv in self.conversation_log["conversations"]:
                print(f"  Turn {conv['turn']}: {conv['user_input'][:50]}...")
                print(f"    → {conv['outcome']} ({conv['confidence']}%) - {conv.get('response_time', 0):.3f}s")
                if conv.get('red_flags'):
                    print(f"    🚩 Red flags: {conv['red_flags']}")
        
        if self.conversation_log["state_transitions"]:
            print("\nSTATE TRANSITIONS:")
            for state in self.conversation_log["state_transitions"]:
                print(f"  {state['conversation_id']}: {state['final_turn_count']} turns → {state['final_outcome']}")
        
        if self.nice_lookups:
            print(f"\nNICE PROTOCOL LOOKUPS: {len(self.nice_lookups)}")
            for lookup in self.nice_lookups:
                print(f"  {lookup['symptoms'][:40]}... → {lookup['protocol_found']}")
        
        print(f"{'=' * 80}\n")

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--asyncio-mode=auto", "-s"])
