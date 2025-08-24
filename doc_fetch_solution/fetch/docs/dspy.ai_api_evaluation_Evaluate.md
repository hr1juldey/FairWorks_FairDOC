# Evaluate

**Source:** https://dspy.ai/api/evaluation/Evaluate
**Fetched:** 2025-08-24 17:10:33
**Status:** Success

---

[

](https://github.com/stanfordnlp/dspy/blob/main/docs/docs/api/evaluation/Evaluate.md)
# dspy.Evaluate
## 
 dspy.Evaluate(*, devset: list[dspy.Example], metric: Callable | None = None, num_threads: int | None = None, display_progress: bool = False, display_table: bool | int = False, max_errors: int | None = None, provide_traceback: bool | None = None, failure_score: float = 0.0, **kwargs)
DSPy Evaluate class.
This class is used to evaluate the performance of a DSPy program. Users need to provide a evaluation dataset and
a metric function in order to use this class. This class supports parallel evaluation on the provided dataset.
Parameters:
Name
Type
Description
Default
devset
list
[
[Example](../../primitives/Example/#dspy.Example)
]
the evaluation dataset.
required
metric
Callable
The metric function to use for evaluation.
None
num_threads
Optional
[
int
]
The number of threads to use for parallel evaluation.
None
display_progress
bool
Whether to display progress during evaluation.
False
display_table
Union
[
bool
,
int
]
Whether to display the evaluation results in a table.
If a number is passed, the evaluation results will be truncated to that number before displayed.
False
max_errors
Optional
[
int
]
The maximum number of errors to allow before
stopping evaluation. If None, inherits from dspy.settings.max_errors.
None
provide_traceback
Optional
[
bool
]
Whether to provide traceback information during evaluation.
None
failure_score
float
The default score to use if evaluation fails due to an exception.
0.0
Source code in
dspy/evaluate/evaluate.py
```
[ 69](#__codelineno-0-69)
[ 70](#__codelineno-0-70)
[ 71](#__codelineno-0-71)
[ 72](#__codelineno-0-72)
[ 73](#__codelineno-0-73)
[ 74](#__codelineno-0-74)
[ 75](#__codelineno-0-75)
[ 76](#__codelineno-0-76)
[ 77](#__codelineno-0-77)
[ 78](#__codelineno-0-78)
[ 79](#__codelineno-0-79)
[ 80](#__codelineno-0-80)
[ 81](#__codelineno-0-81)
[ 82](#__codelineno-0-82)
[ 83](#__codelineno-0-83)
[ 84](#__codelineno-0-84)
[ 85](#__codelineno-0-85)
[ 86](#__codelineno-0-86)
[ 87](#__codelineno-0-87)
[ 88](#__codelineno-0-88)
[ 89](#__codelineno-0-89)
[ 90](#__codelineno-0-90)
[ 91](#__codelineno-0-91)
[ 92](#__codelineno-0-92)
[ 93](#__codelineno-0-93)
[ 94](#__codelineno-0-94)
[ 95](#__codelineno-0-95)
[ 96](#__codelineno-0-96)
[ 97](#__codelineno-0-97)
[ 98](#__codelineno-0-98)
[ 99](#__codelineno-0-99)
[100](#__codelineno-0-100)
[101](#__codelineno-0-101)
[102](#__codelineno-0-102)
[103](#__codelineno-0-103)
[104](#__codelineno-0-104)
[105](#__codelineno-0-105)
```
```
def __init__(
    self,
    *,
    devset: list["dspy.Example"],
    metric: Callable | None = None,
    num_threads: int | None = None,
    display_progress: bool = False,
    display_table: bool | int = False,
    max_errors: int | None = None,
    provide_traceback: bool | None = None,
    failure_score: float = 0.0,
    **kwargs,
):
    """
    Args:
        devset (list[dspy.Example]): the evaluation dataset.
        metric (Callable): The metric function to use for evaluation.
        num_threads (Optional[int]): The number of threads to use for parallel evaluation.
        display_progress (bool): Whether to display progress during evaluation.
        display_table (Union[bool, int]): Whether to display the evaluation results in a table.
            If a number is passed, the evaluation results will be truncated to that number before displayed.
        max_errors (Optional[int]): The maximum number of errors to allow before
            stopping evaluation. If ``None``, inherits from ``dspy.settings.max_errors``.
        provide_traceback (Optional[bool]): Whether to provide traceback information during evaluation.
        failure_score (float): The default score to use if evaluation fails due to an exception.
    """
    self.devset = devset
    self.metric = metric
    self.num_threads = num_threads
    self.display_progress = display_progress
    self.display_table = display_table
    self.max_errors = max_errors
    self.provide_traceback = provide_traceback
    self.failure_score = failure_score

    if "return_outputs" in kwargs:
        raise ValueError("`return_outputs` is no longer supported. Results are always returned inside the `results` field of the `EvaluationResult` object.")

```
### Functions
#### 
 __call__(program: dspy.Module, metric: Callable | None = None, devset: list[dspy.Example] | None = None, num_threads: int | None = None, display_progress: bool | None = None, display_table: bool | int | None = None, callback_metadata: dict[str, Any] | None = None) -> EvaluationResult
Parameters:
Name
Type
Description
Default
program
[Module](../../modules/Module/#dspy.Module)
The DSPy program to evaluate.
required
metric
Callable
The metric function to use for evaluation. if not provided, use self.metric.
None
devset
list
[
[Example](../../primitives/Example/#dspy.Example)
]
the evaluation dataset. if not provided, use self.devset.
None
num_threads
Optional
[
int
]
The number of threads to use for parallel evaluation. if not provided, use
self.num_threads.
None
display_progress
bool
Whether to display progress during evaluation. if not provided, use
self.display_progress.
None
display_table
Union
[
bool
,
int
]
Whether to display the evaluation results in a table. if not provided, use
self.display_table. If a number is passed, the evaluation results will be truncated to that number before displayed.
None
callback_metadata
dict
Metadata to be used for evaluate callback handlers.
None
Returns:
Type
Description
EvaluationResult
The evaluation results are returned as a dspy.EvaluationResult object containing the following attributes:
EvaluationResult
- score: A float percentage score (e.g., 67.30) representing overall performance
EvaluationResult
- results: a list of (example, prediction, score) tuples for each example in devset
Source code in
dspy/evaluate/evaluate.py
```
[107](#__codelineno-0-107)
[108](#__codelineno-0-108)
[109](#__codelineno-0-109)
[110](#__codelineno-0-110)
[111](#__codelineno-0-111)
[112](#__codelineno-0-112)
[113](#__codelineno-0-113)
[114](#__codelineno-0-114)
[115](#__codelineno-0-115)
[116](#__codelineno-0-116)
[117](#__codelineno-0-117)
[118](#__codelineno-0-118)
[119](#__codelineno-0-119)
[120](#__codelineno-0-120)
[121](#__codelineno-0-121)
[122](#__codelineno-0-122)
[123](#__codelineno-0-123)
[124](#__codelineno-0-124)
[125](#__codelineno-0-125)
[126](#__codelineno-0-126)
[127](#__codelineno-0-127)
[128](#__codelineno-0-128)
[129](#__codelineno-0-129)
[130](#__codelineno-0-130)
[131](#__codelineno-0-131)
[132](#__codelineno-0-132)
[133](#__codelineno-0-133)
[134](#__codelineno-0-134)
[135](#__codelineno-0-135)
[136](#__codelineno-0-136)
[137](#__codelineno-0-137)
[138](#__codelineno-0-138)
[139](#__codelineno-0-139)
[140](#__codelineno-0-140)
[141](#__codelineno-0-141)
[142](#__codelineno-0-142)
[143](#__codelineno-0-143)
[144](#__codelineno-0-144)
[145](#__codelineno-0-145)
[146](#__codelineno-0-146)
[147](#__codelineno-0-147)
[148](#__codelineno-0-148)
[149](#__codelineno-0-149)
[150](#__codelineno-0-150)
[151](#__codelineno-0-151)
[152](#__codelineno-0-152)
[153](#__codelineno-0-153)
[154](#__codelineno-0-154)
[155](#__codelineno-0-155)
[156](#__codelineno-0-156)
[157](#__codelineno-0-157)
[158](#__codelineno-0-158)
[159](#__codelineno-0-159)
[160](#__codelineno-0-160)
[161](#__codelineno-0-161)
[162](#__codelineno-0-162)
[163](#__codelineno-0-163)
[164](#__codelineno-0-164)
[165](#__codelineno-0-165)
[166](#__codelineno-0-166)
[167](#__codelineno-0-167)
[168](#__codelineno-0-168)
[169](#__codelineno-0-169)
[170](#__codelineno-0-170)
[171](#__codelineno-0-171)
[172](#__codelineno-0-172)
[173](#__codelineno-0-173)
[174](#__codelineno-0-174)
[175](#__codelineno-0-175)
[176](#__codelineno-0-176)
[177](#__codelineno-0-177)
[178](#__codelineno-0-178)
[179](#__codelineno-0-179)
[180](#__codelineno-0-180)
[181](#__codelineno-0-181)
[182](#__codelineno-0-182)
[183](#__codelineno-0-183)
[184](#__codelineno-0-184)
[185](#__codelineno-0-185)
```
```
@with_callbacks
def __call__(
    self,
    program: "dspy.Module",
    metric: Callable | None = None,
    devset: list["dspy.Example"] | None = None,
    num_threads: int | None = None,
    display_progress: bool | None = None,
    display_table: bool | int | None = None,
    callback_metadata: dict[str, Any] | None = None,
) -> EvaluationResult:
    """
    Args:
        program (dspy.Module): The DSPy program to evaluate.
        metric (Callable): The metric function to use for evaluation. if not provided, use `self.metric`.
        devset (list[dspy.Example]): the evaluation dataset. if not provided, use `self.devset`.
        num_threads (Optional[int]): The number of threads to use for parallel evaluation. if not provided, use
            `self.num_threads`.
        display_progress (bool): Whether to display progress during evaluation. if not provided, use
            `self.display_progress`.
        display_table (Union[bool, int]): Whether to display the evaluation results in a table. if not provided, use
            `self.display_table`. If a number is passed, the evaluation results will be truncated to that number before displayed.
        callback_metadata (dict): Metadata to be used for evaluate callback handlers.

    Returns:
        The evaluation results are returned as a dspy.EvaluationResult object containing the following attributes:

        - score: A float percentage score (e.g., 67.30) representing overall performance

        - results: a list of (example, prediction, score) tuples for each example in devset
    """
    metric = metric if metric is not None else self.metric
    devset = devset if devset is not None else self.devset
    num_threads = num_threads if num_threads is not None else self.num_threads
    display_progress = display_progress if display_progress is not None else self.display_progress
    display_table = display_table if display_table is not None else self.display_table

    if callback_metadata:
        logger.debug(f"Evaluate is called with callback metadata: {callback_metadata}")

    tqdm.tqdm._instances.clear()

    executor = ParallelExecutor(
        num_threads=num_threads,
        disable_progress_bar=not display_progress,
        max_errors=(self.max_errors if self.max_errors is not None else dspy.settings.max_errors),
        provide_traceback=self.provide_traceback,
        compare_results=True,
    )

    def process_item(example):
        prediction = program(**example.inputs())
        score = metric(example, prediction)
        return prediction, score

    results = executor.execute(process_item, devset)
    assert len(devset) == len(results)

    results = [((dspy.Prediction(), self.failure_score) if r is None else r) for r in results]
    results = [(example, prediction, score) for example, (prediction, score) in zip(devset, results, strict=False)]
    ncorrect, ntotal = sum(score for *_, score in results), len(devset)

    logger.info(f"Average Metric: {ncorrect} / {ntotal} ({round(100 * ncorrect / ntotal, 1)}%)")

    if display_table:
        if importlib.util.find_spec("pandas") is not None:
            # Rename the 'correct' column to the name of the metric object
            metric_name = metric.__name__ if isinstance(metric, types.FunctionType) else metric.__class__.__name__
            # Construct a pandas DataFrame from the results
            result_df = self._construct_result_table(results, metric_name)

            self._display_result_table(result_df, display_table, metric_name)
        else:
            logger.warning("Skipping table display since `pandas` is not installed.")

    return EvaluationResult(
        score=round(100 * ncorrect / ntotal, 2),
        results=results,
    )

```
:::