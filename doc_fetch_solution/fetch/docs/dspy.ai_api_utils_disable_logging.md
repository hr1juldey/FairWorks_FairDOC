# Logging

**Source:** https://dspy.ai/api/utils/disable/logging
**Fetched:** 2025-08-24 17:10:36
**Status:** Success

---

[

](https://github.com/stanfordnlp/dspy/blob/main/docs/docs/api/utils/disable_logging.md)
# dspy.disable_logging
## 
 dspy.disable_logging()
Disables the DSPyLoggingStream used by event logging APIs throughout DSPy
(eprint(), logger.info(), etc), silencing all subsequent event logs.
Source code in
dspy/utils/logging_utils.py
```
[40](#__codelineno-0-40)
[41](#__codelineno-0-41)
[42](#__codelineno-0-42)
[43](#__codelineno-0-43)
[44](#__codelineno-0-44)
[45](#__codelineno-0-45)
```
```
def disable_logging():
    """
    Disables the `DSPyLoggingStream` used by event logging APIs throughout DSPy
    (`eprint()`, `logger.info()`, etc), silencing all subsequent event logs.
    """
    DSPY_LOGGING_STREAM.enabled = False

```
:::