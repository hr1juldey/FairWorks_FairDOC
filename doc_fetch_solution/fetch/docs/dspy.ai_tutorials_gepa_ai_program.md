# Program

**Source:** https://dspy.ai/tutorials/gepa/ai/program
**Fetched:** 2025-08-24 17:10:31
**Status:** Success

---

[

](https://github.com/stanfordnlp/dspy/blob/main/docs/docs/tutorials/gepa_ai_program/index.md)
# Reflective Prompt Evolution with GEPA
This section introduces GEPA, a reflective prompt optimizer for DSPy. GEPA works by leveraging LM's ability to reflect on the DSPy program's trajectory, identifying what went well, what didn't, and what can be improved. Based on this reflection, GEPA proposes new prompts, building a tree of evolved prompt candidates, accumulating improvements as the optimization progresses. Since GEPA can leverage domain-specific text feedback (as opposed to only the scalar metric), GEPA can often propose high performing prompts in very few rollouts. GEPA was introduced in the paper GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning and available as dspy.GEPA which internally uses the GEPA implementation provided in gepa-ai/gepa.
## dspy.GEPA Tutorials
### GEPA for AIME (Math)
This tutorial explores how GEPA can optimize a single dspy.ChainOfThought based program to achieve 10% gains on AIME 2025 with GPT-4.1 Mini!
### GEPA for Structured Information Extraction for Enterprise Tasks
This tutorial explores how GEPA leverages predictor-level feedback to improve GPT-4.1 Nano's performance on a three-part task for structured information extraction and classification in an enterprise setting.
### GEPA for Privacy-Conscious Delegation
This tutorial explores how GEPA can improve rapidly in as few as 1 iteration, while leveraging a simple feedback provided by a LLM-as-a-judge metric. The tutorial also explores how GEPA benefits from the textual feedback showing a breakdown of aggregate metrics into sub-components, allowing the reflection LM to identify what aspects of the task need improvement.