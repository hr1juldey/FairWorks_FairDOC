# Inputfield

**Source:** https://dspy.ai/api/signatures/InputField
**Fetched:** 2025-08-24 17:10:33
**Status:** Success

---

[

](https://github.com/stanfordnlp/dspy/blob/main/docs/docs/api/signatures/InputField.md)
# dspy.InputField
## 
 dspy.InputField(**kwargs)
Source code in
dspy/signatures/field.py
```
[54](#__codelineno-0-54)
[55](#__codelineno-0-55)
```
```
def InputField(**kwargs): # noqa: N802
    return pydantic.Field(**move_kwargs(**kwargs, __dspy_field_type="input"))

```
:::