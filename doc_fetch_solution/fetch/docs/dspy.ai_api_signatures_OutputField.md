# Outputfield

**Source:** https://dspy.ai/api/signatures/OutputField
**Fetched:** 2025-08-24 17:10:32
**Status:** Success

---

[

](https://github.com/stanfordnlp/dspy/blob/main/docs/docs/api/signatures/OutputField.md)
# dspy.OutputField
## 
 dspy.OutputField(**kwargs)
Source code in
dspy/signatures/field.py
```
[58](#__codelineno-0-58)
[59](#__codelineno-0-59)
```
```
def OutputField(**kwargs): # noqa: N802
    return pydantic.Field(**move_kwargs(**kwargs, __dspy_field_type="output"))

```
:::