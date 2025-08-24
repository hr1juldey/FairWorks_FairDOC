# Labeledfewshot

**Source:** https://dspy.ai/api/optimizers/LabeledFewShot
**Fetched:** 2025-08-24 17:10:32
**Status:** Success

---

[

](https://github.com/stanfordnlp/dspy/blob/main/docs/docs/api/optimizers/LabeledFewShot.md)
# dspy.LabeledFewShot
## 
 dspy.LabeledFewShot(k=16)
Bases: Teleprompter
Source code in
dspy/teleprompt/vanilla.py
```
[7](#__codelineno-0-7)
[8](#__codelineno-0-8)
```
```
def __init__(self, k=16):
    self.k = k

```
### Functions
#### 
 compile(student, *, trainset, sample=True)
Source code in
dspy/teleprompt/vanilla.py
```
[10](#__codelineno-0-10)
[11](#__codelineno-0-11)
[12](#__codelineno-0-12)
[13](#__codelineno-0-13)
[14](#__codelineno-0-14)
[15](#__codelineno-0-15)
[16](#__codelineno-0-16)
[17](#__codelineno-0-17)
[18](#__codelineno-0-18)
[19](#__codelineno-0-19)
[20](#__codelineno-0-20)
[21](#__codelineno-0-21)
[22](#__codelineno-0-22)
[23](#__codelineno-0-23)
[24](#__codelineno-0-24)
[25](#__codelineno-0-25)
```
```
def compile(self, student, *, trainset, sample=True):
    self.student = student.reset_copy()
    self.trainset = trainset

    if len(self.trainset) == 0:
        return self.student

    rng = random.Random(0)

    for predictor in self.student.predictors():
        if sample:
            predictor.demos = rng.sample(self.trainset, min(self.k, len(self.trainset)))
        else:
            predictor.demos = self.trainset[: min(self.k, len(self.trainset))]

    return self.student

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