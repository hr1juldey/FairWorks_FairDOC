# Ensemble

**Source:** https://dspy.ai/api/optimizers/Ensemble
**Fetched:** 2025-08-24 17:10:36
**Status:** Success

---

[

](https://github.com/stanfordnlp/dspy/blob/main/docs/docs/api/optimizers/Ensemble.md)
# dspy.Ensemble
## 
 dspy.Ensemble(*, reduce_fn=None, size=None, deterministic=False)
Bases: Teleprompter
A common reduce_fn is dspy.majority.
Source code in
dspy/teleprompt/ensemble.py
```
[11](#__codelineno-0-11)
[12](#__codelineno-0-12)
[13](#__codelineno-0-13)
[14](#__codelineno-0-14)
[15](#__codelineno-0-15)
[16](#__codelineno-0-16)
[17](#__codelineno-0-17)
[18](#__codelineno-0-18)
```
```
def __init__(self, *, reduce_fn=None, size=None, deterministic=False):
    """A common reduce_fn is dspy.majority."""

    assert deterministic is False, "TODO: Implement example hashing for deterministic ensemble."

    self.reduce_fn = reduce_fn
    self.size = size
    self.deterministic = deterministic

```
### Functions
#### 
 compile(programs)
Source code in
dspy/teleprompt/ensemble.py
```
[20](#__codelineno-0-20)
[21](#__codelineno-0-21)
[22](#__codelineno-0-22)
[23](#__codelineno-0-23)
[24](#__codelineno-0-24)
[25](#__codelineno-0-25)
[26](#__codelineno-0-26)
[27](#__codelineno-0-27)
[28](#__codelineno-0-28)
[29](#__codelineno-0-29)
[30](#__codelineno-0-30)
[31](#__codelineno-0-31)
[32](#__codelineno-0-32)
[33](#__codelineno-0-33)
[34](#__codelineno-0-34)
[35](#__codelineno-0-35)
[36](#__codelineno-0-36)
[37](#__codelineno-0-37)
[38](#__codelineno-0-38)
[39](#__codelineno-0-39)
[40](#__codelineno-0-40)
```
```
def compile(self, programs):
    size = self.size
    reduce_fn = self.reduce_fn

    import dspy

    class EnsembledProgram(dspy.Module):
        def __init__(self):
            super().__init__()
            self.programs = programs

        def forward(self, *args, **kwargs):
            programs = random.sample(self.programs, size) if size else self.programs
            outputs = [prog(*args, **kwargs) for prog in programs]

            if reduce_fn:
                return reduce_fn(outputs)

            return outputs

    return EnsembledProgram()

```
#### 
 get_params() -> dict[str, Any]
Get the parameters of the teleprompter.
Returns:
Type
Description
dict
[
str
,
Any
]
The parameters of the teleprompter.
Source code in
dspy/teleprompt/teleprompt.py
```
[25](#__codelineno-0-25)
[26](#__codelineno-0-26)
[27](#__codelineno-0-27)
[28](#__codelineno-0-28)
[29](#__codelineno-0-29)
[30](#__codelineno-0-30)
[31](#__codelineno-0-31)
[32](#__codelineno-0-32)
```
```
def get_params(self) -> dict[str, Any]:
    """
    Get the parameters of the teleprompter.

    Returns:
        The parameters of the teleprompter.
    """
    return self.__dict__

```
:::