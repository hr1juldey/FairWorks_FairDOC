# Logging

**Source:** https://dspy.ai/api/utils/enable/litellm/logging
**Fetched:** 2025-08-24 17:10:36
**Status:** Success

---

[

](https://github.com/stanfordnlp/dspy/blob/main/docs/docs/api/utils/enable_litellm_logging.md)
# dspy.enable_litellm_logging
## 
 dspy.enable_litellm_logging()
Source code in
dspy/clients/__init__.py
```
[105](#__codelineno-0-105)
[106](#__codelineno-0-106)
[107](#__codelineno-0-107)
```
```
def enable_litellm_logging():
    litellm.suppress_debug_info = False
    configure_litellm_logging("DEBUG")

```
:::