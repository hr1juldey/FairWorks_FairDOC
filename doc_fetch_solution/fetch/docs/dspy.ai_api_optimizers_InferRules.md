# Inferrules

**Source:** https://dspy.ai/api/optimizers/InferRules
**Fetched:** 2025-08-24 17:10:30
**Status:** Success

---

[

](https://github.com/stanfordnlp/dspy/blob/main/docs/docs/api/optimizers/InferRules.md)
# dspy.InferRules
## 
 dspy.InferRules(num_candidates=10, num_rules=10, num_threads=None, teacher_settings=None, **kwargs)
Bases: BootstrapFewShot
Source code in
dspy/teleprompt/infer_rules.py
```
[14](#__codelineno-0-14)
[15](#__codelineno-0-15)
[16](#__codelineno-0-16)
[17](#__codelineno-0-17)
[18](#__codelineno-0-18)
[19](#__codelineno-0-19)
[20](#__codelineno-0-20)
[21](#__codelineno-0-21)
[22](#__codelineno-0-22)
```
```
def __init__(self, num_candidates=10, num_rules=10, num_threads=None, teacher_settings=None, **kwargs):
    super().__init__(teacher_settings=teacher_settings, **kwargs)

    self.num_candidates = num_candidates
    self.num_rules = num_rules
    self.num_threads = num_threads
    self.rules_induction_program = RulesInductionProgram(num_rules, teacher_settings=teacher_settings)
    self.metric = kwargs.get("metric")
    self.max_errors = kwargs.get("max_errors")

```
### Functions
#### 
 compile(student, *, teacher=None, trainset, valset=None)
Source code in
dspy/teleprompt/infer_rules.py
```
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
```
```
def compile(self, student, *, teacher=None, trainset, valset=None):
    if valset is None:
        train_size = int(0.5 * len(trainset))
        trainset, valset = trainset[:train_size], trainset[train_size:]

    super().compile(student, teacher=teacher, trainset=trainset)

    original_program = self.student.deepcopy()
    all_predictors = [p for p in original_program.predictors() if hasattr(p, "signature")]
    instructions_list = [p.signature.instructions for p in all_predictors]

    best_score = -np.inf
    best_program = None

    for candidate_idx in range(self.num_candidates):
        candidate_program = original_program.deepcopy()
        candidate_predictors = [p for p in candidate_program.predictors() if hasattr(p, "signature")]

        for i, predictor in enumerate(candidate_predictors):
            predictor.signature.instructions = instructions_list[i]

        for i, predictor in enumerate(candidate_predictors):
            rules = self.induce_natural_language_rules(predictor, trainset)
            predictor.signature.instructions = instructions_list[i]
            self.update_program_instructions(predictor, rules)

        score = self.evaluate_program(candidate_program, valset)

        if score > best_score:
            best_score = score
            best_program = candidate_program

        logger.info(f"Evaluated Candidate {candidate_idx + 1} with score {score}. Current best score: {best_score}")

    logger.info(f"Final best score: {best_score}")

    return best_program

```
#### 
 evaluate_program(program, dataset)
Source code in
dspy/teleprompt/infer_rules.py
```
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
```
```
def evaluate_program(self, program, dataset):
    effective_max_errors = (
        self.max_errors if self.max_errors is not None else dspy.settings.max_errors
    )
    evaluate = Evaluate(
        devset=dataset,
        metric=self.metric,
        num_threads=self.num_threads,
        max_errors=effective_max_errors,
        display_table=False,
        display_progress=True,
    )
    score = evaluate(program, metric=self.metric).score
    return score

```
#### 
 format_examples(demos, signature)
Source code in
dspy/teleprompt/infer_rules.py
```
[89](#__codelineno-0-89)
[90](#__codelineno-0-90)
[91](#__codelineno-0-91)
[92](#__codelineno-0-92)
[93](#__codelineno-0-93)
[94](#__codelineno-0-94)
[95](#__codelineno-0-95)
[96](#__codelineno-0-96)
[97](#__codelineno-0-97)
```
```
def format_examples(self, demos, signature):
    examples_text = ""
    for demo in demos:
        input_fields = {k: v for k, v in demo.items() if k in signature.input_fields}
        output_fields = {k: v for k, v in demo.items() if k in signature.output_fields}
        input_text = "\n".join(f"{k}: {v}" for k, v in input_fields.items())
        output_text = "\n".join(f"{k}: {v}" for k, v in output_fields.items())
        examples_text += f"Input Fields:\n{input_text}\n\n=========\nOutput Fields:\n{output_text}\n\n"
    return examples_text

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
#### 
 get_predictor_demos(trainset, predictor)
Source code in
dspy/teleprompt/infer_rules.py
```
[ 99](#__codelineno-0-99)
[100](#__codelineno-0-100)
[101](#__codelineno-0-101)
[102](#__codelineno-0-102)
[103](#__codelineno-0-103)
[104](#__codelineno-0-104)
[105](#__codelineno-0-105)
[106](#__codelineno-0-106)
[107](#__codelineno-0-107)
[108](#__codelineno-0-108)
[109](#__codelineno-0-109)
```
```
def get_predictor_demos(self, trainset, predictor):
    # TODO: Consider how this handled "incomplete" demos.
    signature = predictor.signature
    return [
        {
            key: value
            for key, value in example.items()
            if key in signature.input_fields or key in signature.output_fields
        }
        for example in trainset
    ]

```
#### 
 induce_natural_language_rules(predictor, trainset)
Source code in
dspy/teleprompt/infer_rules.py
```
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
```
```
def induce_natural_language_rules(self, predictor, trainset):
    demos = self.get_predictor_demos(trainset, predictor)
    signature = predictor.signature
    while True:
        examples_text = self.format_examples(demos, signature)
        try:
            return self.rules_induction_program(examples_text)
        except Exception as e:
            assert (
                isinstance(e, ValueError)
                or e.__class__.__name__ == "BadRequestError"
                or "ContextWindowExceededError" in str(e)
            )
            if len(demos) > 1:
                demos = demos[:-1]
            else:
                raise RuntimeError(
                    "Failed to generate natural language rules since a single example couldn't fit in the model's "
                    "context window."
                ) from e

```
#### 
 update_program_instructions(predictor, natural_language_rules)
Source code in
dspy/teleprompt/infer_rules.py
```
[83](#__codelineno-0-83)
[84](#__codelineno-0-84)
[85](#__codelineno-0-85)
[86](#__codelineno-0-86)
[87](#__codelineno-0-87)
```
```
def update_program_instructions(self, predictor, natural_language_rules):
    predictor.signature.instructions = (
        f"{predictor.signature.instructions}\n\n"
        f"Please adhere to the following rules when making your prediction:\n{natural_language_rules}"
    )

```
:::