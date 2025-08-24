# History

**Source:** https://dspy.ai/api/utils/inspect/history
**Fetched:** 2025-08-24 17:10:34
**Status:** Success

---

[

](https://github.com/stanfordnlp/dspy/blob/main/docs/docs/api/utils/inspect_history.md)
# dspy.inspect_history
## 
 dspy.inspect_history(n: int = 1)
The global history shared across all LMs.
Source code in
dspy/clients/base_lm.py
```
[167](#__codelineno-0-167)
[168](#__codelineno-0-168)
[169](#__codelineno-0-169)
```
```
def inspect_history(n: int = 1):
    """The global history shared across all LMs."""
    return pretty_print_history(GLOBAL_HISTORY, n)

```
:::