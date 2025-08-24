# Papillon

**Source:** https://dspy.ai/tutorials/papillon
**Fetched:** 2025-08-24 17:10:37
**Status:** Success

---

[

](https://github.com/stanfordnlp/dspy/blob/main/docs/docs/tutorials/papillon/index.md)
# Privacy-Conscious Delegation
Please refer to this tutorial from the PAPILLON authors using DSPy.
This tutorial demonstrates a few aspects of using DSPy in a more advanced context:
1. It builds a multi-stage dspy.Module that involves a small local LM using an external tool.
2. It builds a multi-stage judge in DSPy, and uses it as a metric for evaluation.
3. It uses this judge for optimizing the dspy.Module, using a large model as a teacher for a small local LM.