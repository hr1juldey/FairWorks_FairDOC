"""
DSPy Evaluation Optimizer for Fairdoc AI V2
================================================
Keeps ≤200 LOC while providing a minimal yet complete evaluation
pipeline that replays gold-standard dialogues through the
`MedicalTriageAgent`, then scores predicted outcomes with
DSPy’s `SemanticF1` metric.

Usage
-----
>>> optimiser = EvaluationOptimizer()
>>> asyncio.run(optimiser.evaluate(limit=10))
{'examples': 10, 'average_score': 0.82}
"""
from __future__ import annotations

import asyncio
from typing import List, Dict, Any

import structlog
import dspy
from dspy.evaluate import SemanticF1

from src.app2.models.database.gold_standards import GoldStandardDialogue
from src.app2.services.dspy.medical_agent import MedicalTriageAgent
from src.app2.core.database_v2 import get_async_session  # async session factory

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Main Optimiser class
# ---------------------------------------------------------------------------
class EvaluationOptimizer:  # pylint: disable=too-few-public-methods
    """Run evaluation (and future fine-tuning) over gold standards."""

    def __init__(self, model_name: str = "deepseek-r1:8b") -> None:
        self.agent = MedicalTriageAgent(model_name=model_name)
        self.metric = SemanticF1()  # semantic F1 over free-text labels
        logger.info("📊 Evaluation optimiser initialised", model=model_name)

    # ---------------------------------------------------------------------
    # Public helpers
    # ---------------------------------------------------------------------
    async def evaluate(self, *, limit: int = 50) -> Dict[str, Any]:
        """Replay *limit* gold-standard dialogues and return average score."""
        examples = await self._load_gold_standards(limit)
        if not examples:
            logger.warning("⚠️ No gold standards available – skipping eval")
            return {"examples": 0, "average_score": 0.0}

        scores: List[float] = []
        for gs in examples:
            prediction = await self._run_dialogue(gs)
            score = self.metric(prediction["medical_outcome"], gs.expected_outcome.value)
            scores.append(float(score))

        avg_score = sum(scores) / len(scores)
        logger.info("✅ Evaluation complete", n=len(scores), avg=avg_score)
        return {"examples": len(scores), "average_score": round(avg_score, 4)}

    async def optimise(self, *, limit: int = 50) -> Dict[str, Any]:
        """Placeholder for future fine-tuning – currently just evaluates."""
        logger.info("⚙️ Optimise not implemented – delegating to evaluate()")
        return await self.evaluate(limit=limit)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    async def _load_gold_standards(self, limit: int) -> List[GoldStandardDialogue]:
        """Fetch active gold standards from PostgreSQL (async)."""
        async for session in get_async_session():
            return GoldStandardDialogue.get_training_set(session, limit=limit)
        return []

    async def _run_dialogue(self, gs: GoldStandardDialogue) -> Dict[str, Any]:
        """Replay a gold-standard conversation through the triage agent."""
        self.agent.reset_conversation()
        next_q = None
        for turn in gs.conversation_dialogue:
            user_msg: str = turn["user_message"]
            result = await self.agent.process_turn(
                symptoms=user_msg,
                nice_context="\n".join(gs.relevant_protocols) or ""
            )
            next_q = result.get("next_question")
            # Stop early if agent is satisfied or GS dialogue ends
            if result["is_complete"] or not next_q:
                break
        return {
            "medical_outcome": result["outcome"],
            "confidence": result["confidence"],
        }

# Convenience singleton
evaluation_optimizer = EvaluationOptimizer()

# ---------------------------------------------------------------------------
# CLI helper (optional) – `python -m evaluation_optimizer`
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import asyncio
    report = asyncio.run(evaluation_optimizer.evaluate(limit=10))
    print(report)
