# Match

**Source:** https://dspy.ai/api/evaluation/answer/passage/match
**Fetched:** 2025-08-24 17:10:32
**Status:** Success

---

[

](https://github.com/stanfordnlp/dspy/blob/main/docs/docs/api/evaluation/answer_passage_match.md)
# dspy.evaluate.answer_passage_match
## 
 dspy.evaluate.answer_passage_match(example, pred, trace=None)
Source code in
dspy/evaluate/metrics.py
```
[150](#__codelineno-0-150)
[151](#__codelineno-0-151)
[152](#__codelineno-0-152)
[153](#__codelineno-0-153)
[154](#__codelineno-0-154)
[155](#__codelineno-0-155)
[156](#__codelineno-0-156)
```
```
def answer_passage_match(example, pred, trace=None):
    if isinstance(example.answer, str):
        return _passage_match(pred.context, [example.answer])
    elif isinstance(example.answer, list):
        return _passage_match(pred.context, example.answer)

    raise ValueError(f"Invalid answer type: {type(example.answer)}")

```
:::