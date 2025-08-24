# Bettertogether

**Source:** https://dspy.ai/api/optimizers/BetterTogether
**Fetched:** 2025-08-24 17:10:35
**Status:** Success

---

[

](https://github.com/stanfordnlp/dspy/blob/main/docs/docs/api/optimizers/BetterTogether.md)
# dspy.BetterTogether
## 
 dspy.BetterTogether(metric: Callable, prompt_optimizer: Teleprompter | None = None, weight_optimizer: Teleprompter | None = None, seed: int | None = None)
Bases: Teleprompter
Source code in
dspy/teleprompt/bettertogether.py
```
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
```
```
def __init__(self,
    metric: Callable,
    prompt_optimizer: Teleprompter | None = None,
    weight_optimizer: Teleprompter | None = None,
    seed: int | None = None,
  ):
    if not dspy.settings.experimental:
        raise ValueError("This is an experimental optimizer. Set `dspy.settings.experimental` to `True` to use it.")

    # TODO: Note that the BetterTogether optimizer is meaningful when
    # BootstrapFinetune uses a metric to filter the training data before
    # fine-tuning. However, one can also choose to run this optimizer with
    # a BootstrapFinetune without a metric, say, if there aren't labels
    # available for the training data. Should this be noted somewhere?
    # TODO: We should re-consider if the metric should be required.
    self.prompt_optimizer = prompt_optimizer if prompt_optimizer else BootstrapFewShotWithRandomSearch(metric=metric)
    self.weight_optimizer = weight_optimizer if weight_optimizer else BootstrapFinetune(metric=metric)

    is_supported_prompt = isinstance(self.prompt_optimizer, BootstrapFewShotWithRandomSearch)
    is_supported_weight = isinstance(self.weight_optimizer, BootstrapFinetune)
    if not is_supported_prompt or not is_supported_weight:
        raise ValueError(
            "The BetterTogether optimizer only supports the following optimizers for now: BootstrapFinetune, "
            "BootstrapFewShotWithRandomSearch."
        )

    self.rng = random.Random(seed)

```
### Functions
#### 
 compile(student: Module, trainset: list[Example], strategy: str = 'p -> w -> p', valset_ratio=0.1) -> Module
Source code in
dspy/teleprompt/bettertogether.py
```
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
[77](#__codelineno-0-77)
[78](#__codelineno-0-78)
[79](#__codelineno-0-79)
[80](#__codelineno-0-80)
[81](#__codelineno-0-81)
[82](#__codelineno-0-82)
[83](#__codelineno-0-83)
[84](#__codelineno-0-84)
```
```
def compile(
    self,
    student: Module,
    trainset: list[Example],
    strategy: str = "p -> w -> p",
    valset_ratio = 0.1,
) -> Module:
    # TODO: We could record acc on a different valset to pick the best
    # strategy within the provided strategy
    logger.info("Validating the strategy")
    parsed_strategy = strategy.lower().split(self.STRAT_SEP)

    if not all(s in ["p", "w"] for s in parsed_strategy):
        raise ValueError(
            f"The strategy should be a sequence of 'p' and 'w' separated by '{self.STRAT_SEP}', but "
            f"found: {strategy}"
        )

    logger.info("Preparing the student program...")
    # TODO: Prepare student returns student.reset_copy(), which is what gets
    # optimized. We should make this clear in the doc comments.
    student = prepare_student(student)
    all_predictors_have_lms(student)

    # Make a shallow copy of the trainset, so that we don't change the order
    # of the examples in the original trainset
    trainset = trainset[:]
    logger.info("Compiling the student program...")
    student = self._run_strategies(parsed_strategy, student, trainset, valset_ratio)

    logger.info("BetterTogether has finished compiling the student program")
    return student

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