# Match

**Source:** https://dspy.ai/api/evaluation/answer/exact/match
**Fetched:** 2025-08-24 17:10:33
**Status:** Success

---

[

](https://github.com/stanfordnlp/dspy/blob/main/docs/docs/api/evaluation/answer_exact_match.md)
# dspy.evaluate.answer_exact_match
## 
 dspy.evaluate.answer_exact_match(example, pred, trace=None, frac=1.0)
Source code in
dspy/evaluate/metrics.py
```
[141](#__codelineno-0-141)
[142](#__codelineno-0-142)
[143](#__codelineno-0-143)
[144](#__codelineno-0-144)
[145](#__codelineno-0-145)
[146](#__codelineno-0-146)
[147](#__codelineno-0-147)
```
```
def answer_exact_match(example, pred, trace=None, frac=1.0):
    if isinstance(example.answer, str):
        return _answer_match(pred.answer, [example.answer], frac=frac)
    elif isinstance(example.answer, list):
        return _answer_match(pred.answer, example.answer, frac=frac)

    raise ValueError(f"Invalid answer type: {type(example.answer)}")

```
:::