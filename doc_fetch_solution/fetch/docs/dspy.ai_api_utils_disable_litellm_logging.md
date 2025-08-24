# Logging

**Source:** https://dspy.ai/api/utils/disable/litellm/logging
**Fetched:** 2025-08-24 17:10:32
**Status:** Success

---

[

](https://github.com/stanfordnlp/dspy/blob/main/docs/docs/api/utils/disable_litellm_logging.md)
# dspy.disable_litellm_logging
## 
 dspy.disable_litellm_logging()
Source code in
dspy/clients/__init__.py
```
[110](#__codelineno-0-110)
[111](#__codelineno-0-111)
[112](#__codelineno-0-112)
```
```
def disable_litellm_logging():
    litellm.suppress_debug_info = True
    configure_litellm_logging("ERROR")

```
:::