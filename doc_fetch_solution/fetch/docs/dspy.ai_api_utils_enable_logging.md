# Logging

**Source:** https://dspy.ai/api/utils/enable/logging
**Fetched:** 2025-08-24 17:10:37
**Status:** Success

---

[

](https://github.com/stanfordnlp/dspy/blob/main/docs/docs/api/utils/enable_logging.md)
# dspy.enable_logging
## 
 dspy.enable_logging()
Enables the DSPyLoggingStream used by event logging APIs throughout DSPy
(eprint(), logger.info(), etc), emitting all subsequent event logs. This
reverses the effects of disable_logging().
Source code in
dspy/utils/logging_utils.py
```
[48](#__codelineno-0-48)
[49](#__codelineno-0-49)
[50](#__codelineno-0-50)
[51](#__codelineno-0-51)
[52](#__codelineno-0-52)
[53](#__codelineno-0-53)
[54](#__codelineno-0-54)
```
```
def enable_logging():
    """
    Enables the `DSPyLoggingStream` used by event logging APIs throughout DSPy
    (`eprint()`, `logger.info()`, etc), emitting all subsequent event logs. This
    reverses the effects of `disable_logging()`.
    """
    DSPY_LOGGING_STREAM.enabled = True

```
:::