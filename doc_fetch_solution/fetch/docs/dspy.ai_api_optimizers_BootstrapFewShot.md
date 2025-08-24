# Bootstrapfewshot

**Source:** https://dspy.ai/api/optimizers/BootstrapFewShot
**Fetched:** 2025-08-24 17:10:33
**Status:** Success

---

[

](https://github.com/stanfordnlp/dspy/blob/main/docs/docs/api/optimizers/BootstrapFewShot.md)
# dspy.BootstrapFewShot
## 
 dspy.BootstrapFewShot(metric=None, metric_threshold=None, teacher_settings: dict | None = None, max_bootstrapped_demos=4, max_labeled_demos=16, max_rounds=1, max_errors=None)
Bases: Teleprompter
A Teleprompter class that composes a set of demos/examples to go into a predictor's prompt.
These demos come from a combination of labeled examples in the training set, and bootstrapped demos.
Parameters:
Name
Type
Description
Default
metric
Callable
A function that compares an expected value and predicted value,
outputting the result of that comparison.
None
metric_threshold
float
If the metric yields a numerical value, then check it
against this threshold when deciding whether or not to accept a bootstrap example.
Defaults to None.
None
teacher_settings
dict
Settings for the teacher model.
Defaults to None.
None
max_bootstrapped_demos
int
Maximum number of bootstrapped demonstrations to include.
Defaults to 4.
4
max_labeled_demos
int
Maximum number of labeled demonstrations to include.
Defaults to 16.
16
max_rounds
int
Number of iterations to attempt generating the required bootstrap
examples. If unsuccessful after max_rounds, the program ends. Defaults to 1.
1
max_errors
Optional
[
int
]
Maximum number of errors until program ends.
If None, inherits from dspy.settings.max_errors.
None
Source code in
dspy/teleprompt/bootstrap.py
```
[37](#__codelineno-0-37)
[38](#__codelineno-0-38)
[39](#__codelineno-0-39)
[40](#__codelineno-0-40)
[41](#__codelineno-0-41)
[42](#__codelineno-0-42)
[43](#__codelineno-0-43)
[44](#__codelineno-0-44)
[45](#__codelineno-0-45)
[46](#__codelineno-0-46)
[47](#__codelineno-0-47)
[48](#__codelineno-0-48)
[49](#__codelineno-0-49)
[50](#__codelineno-0-50)
[51](#__codelineno-0-51)
[52](#__codelineno-0-52)
[53](#__codelineno-0-53)
[54](#__codelineno-0-54)
[55](#__codelineno-0-55)
[56](#__codelineno-0-56)
[57](#__codelineno-0-57)
[58](#__codelineno-0-58)
[59](#__codelineno-0-59)
[60](#__codelineno-0-60)
[61](#__codelineno-0-61)
[62](#__codelineno-0-62)
[63](#__codelineno-0-63)
[64](#__codelineno-0-64)
[65](#__codelineno-0-65)
[66](#__codelineno-0-66)
[67](#__codelineno-0-67)
[68](#__codelineno-0-68)
[69](#__codelineno-0-69)
[70](#__codelineno-0-70)
[71](#__codelineno-0-71)
[72](#__codelineno-0-72)
[73](#__codelineno-0-73)
[74](#__codelineno-0-74)
[75](#__codelineno-0-75)
[76](#__codelineno-0-76)
```
```
def __init__(
    self,
    metric=None,
    metric_threshold=None,
    teacher_settings: dict | None = None,
    max_bootstrapped_demos=4,
    max_labeled_demos=16,
    max_rounds=1,
    max_errors=None,
):
    """A Teleprompter class that composes a set of demos/examples to go into a predictor's prompt.
    These demos come from a combination of labeled examples in the training set, and bootstrapped demos.

    Args:
        metric (Callable): A function that compares an expected value and predicted value,
            outputting the result of that comparison.
        metric_threshold (float, optional): If the metric yields a numerical value, then check it
            against this threshold when deciding whether or not to accept a bootstrap example.
            Defaults to None.
        teacher_settings (dict, optional): Settings for the `teacher` model.
            Defaults to None.
        max_bootstrapped_demos (int): Maximum number of bootstrapped demonstrations to include.
            Defaults to 4.
        max_labeled_demos (int): Maximum number of labeled demonstrations to include.
            Defaults to 16.
        max_rounds (int): Number of iterations to attempt generating the required bootstrap
            examples. If unsuccessful after `max_rounds`, the program ends. Defaults to 1.
        max_errors (Optional[int]): Maximum number of errors until program ends.
            If ``None``, inherits from ``dspy.settings.max_errors``.
    """
    self.metric = metric
    self.metric_threshold = metric_threshold
    self.teacher_settings = {} if teacher_settings is None else teacher_settings

    self.max_bootstrapped_demos = max_bootstrapped_demos
    self.max_labeled_demos = max_labeled_demos
    self.max_rounds = max_rounds
    self.max_errors = max_errors
    self.error_count = 0
    self.error_lock = threading.Lock()

```
### Functions
#### 
 compile(student, *, teacher=None, trainset)
Source code in
dspy/teleprompt/bootstrap.py
```
[78](#__codelineno-0-78)
[79](#__codelineno-0-79)
[80](#__codelineno-0-80)
[81](#__codelineno-0-81)
[82](#__codelineno-0-82)
[83](#__codelineno-0-83)
[84](#__codelineno-0-84)
[85](#__codelineno-0-85)
[86](#__codelineno-0-86)
[87](#__codelineno-0-87)
[88](#__codelineno-0-88)
```
```
def compile(self, student, *, teacher=None, trainset):
    self.trainset = trainset

    self._prepare_student_and_teacher(student, teacher)
    self._prepare_predictor_mappings()
    self._bootstrap()

    self.student = self._train()
    self.student._compiled = True

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