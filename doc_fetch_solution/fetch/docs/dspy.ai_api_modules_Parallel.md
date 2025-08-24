# Parallel

**Source:** https://dspy.ai/api/modules/Parallel
**Fetched:** 2025-08-24 17:10:32
**Status:** Success

---

[

](https://github.com/stanfordnlp/dspy/blob/main/docs/docs/api/modules/Parallel.md)
# dspy.Parallel
## 
 dspy.Parallel(num_threads: int | None = None, max_errors: int | None = None, access_examples: bool = True, return_failed_examples: bool = False, provide_traceback: bool | None = None, disable_progress_bar: bool = False)
Source code in
dspy/predict/parallel.py
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
[26](#__codelineno-0-26)
[27](#__codelineno-0-27)
[28](#__codelineno-0-28)
[29](#__codelineno-0-29)
[30](#__codelineno-0-30)
[31](#__codelineno-0-31)
```
```
def __init__(
    self,
    num_threads: int | None = None,
    max_errors: int | None = None,
    access_examples: bool = True,
    return_failed_examples: bool = False,
    provide_traceback: bool | None = None,
    disable_progress_bar: bool = False,
):
    super().__init__()
    self.num_threads = num_threads or settings.num_threads
    self.max_errors = settings.max_errors if max_errors is None else max_errors
    self.access_examples = access_examples
    self.return_failed_examples = return_failed_examples
    self.provide_traceback = provide_traceback
    self.disable_progress_bar = disable_progress_bar

    self.error_count = 0
    self.error_lock = threading.Lock()
    self.cancel_jobs = threading.Event()
    self.failed_examples = []
    self.exceptions = []

```
### Functions
#### 
 __call__(*args: Any, **kwargs: Any) -> Any
Source code in
dspy/predict/parallel.py
```
[72](#__codelineno-0-72)
[73](#__codelineno-0-73)
```
```
def __call__(self, *args: Any, **kwargs: Any) -> Any:
    return self.forward(*args, **kwargs)

```
#### 
 forward(exec_pairs: list[tuple[Any, Example]], num_threads: int | None = None) -> list[Any]
Source code in
dspy/predict/parallel.py
```
[33](#__codelineno-0-33)
[34](#__codelineno-0-34)
[35](#__codelineno-0-35)
[36](#__codelineno-0-36)
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
```
```
def forward(self, exec_pairs: list[tuple[Any, Example]], num_threads: int | None = None) -> list[Any]:
    num_threads = num_threads if num_threads is not None else self.num_threads

    executor = ParallelExecutor(
        num_threads=num_threads,
        max_errors=self.max_errors,
        provide_traceback=self.provide_traceback,
        disable_progress_bar=self.disable_progress_bar,
    )

    def process_pair(pair):
        result = None
        module, example = pair

        if isinstance(example, Example):
            if self.access_examples:
                result = module(**example.inputs())
            else:
                result = module(example)
        elif isinstance(example, dict):
            result = module(**example)
        elif isinstance(example, list) and module.__class__.__name__ == "Parallel":
            result = module(example)
        elif isinstance(example, tuple):
            result = module(*example)
        else:
            raise ValueError(
                f"Invalid example type: {type(example)}, only supported types are Example, dict, list and tuple"
            )
        return result

    # Execute the processing function over the execution pairs
    results = executor.execute(process_pair, exec_pairs)

    if self.return_failed_examples:
        return results, self.failed_examples, self.exceptions
    else:
        return results

```
:::