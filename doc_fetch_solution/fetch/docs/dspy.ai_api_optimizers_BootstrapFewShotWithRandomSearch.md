# Bootstrapfewshotwithrandomsearch

**Source:** https://dspy.ai/api/optimizers/BootstrapFewShotWithRandomSearch
**Fetched:** 2025-08-24 17:10:32
**Status:** Success

---

[

](https://github.com/stanfordnlp/dspy/blob/main/docs/docs/api/optimizers/BootstrapFewShotWithRandomSearch.md)
# dspy.BootstrapFewShotWithRandomSearch
## 
 dspy.BootstrapFewShotWithRandomSearch(metric, teacher_settings=None, max_bootstrapped_demos=4, max_labeled_demos=16, max_rounds=1, num_candidate_programs=16, num_threads=None, max_errors=None, stop_at_score=None, metric_threshold=None)
Bases: Teleprompter
Source code in
dspy/teleprompt/random_search.py
```
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
```
```
def __init__(
    self,
    metric,
    teacher_settings=None,
    max_bootstrapped_demos=4,
    max_labeled_demos=16,
    max_rounds=1,
    num_candidate_programs=16,
    num_threads=None,
    max_errors=None,
    stop_at_score=None,
    metric_threshold=None,
):
    self.metric = metric
    self.teacher_settings = teacher_settings or {}
    self.max_rounds = max_rounds

    self.num_threads = num_threads
    self.stop_at_score = stop_at_score
    self.metric_threshold = metric_threshold
    self.min_num_samples = 1
    self.max_num_samples = max_bootstrapped_demos
    self.max_errors = max_errors
    self.num_candidate_sets = num_candidate_programs
    self.max_labeled_demos = max_labeled_demos

    print(f"Going to sample between {self.min_num_samples} and {self.max_num_samples} traces per predictor.")
    print(f"Will attempt to bootstrap {self.num_candidate_sets} candidate sets.")

```
### Functions
#### 
 compile(student, *, teacher=None, trainset, valset=None, restrict=None, labeled_sample=True)
Source code in
dspy/teleprompt/random_search.py
```
[ 57](#__codelineno-0-57)
[ 58](#__codelineno-0-58)
[ 59](#__codelineno-0-59)
[ 60](#__codelineno-0-60)
[ 61](#__codelineno-0-61)
[ 62](#__codelineno-0-62)
[ 63](#__codelineno-0-63)
[ 64](#__codelineno-0-64)
[ 65](#__codelineno-0-65)
[ 66](#__codelineno-0-66)
[ 67](#__codelineno-0-67)
[ 68](#__codelineno-0-68)
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
[106](#__codelineno-0-106)
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
```
```
def compile(self, student, *, teacher=None, trainset, valset=None, restrict=None, labeled_sample=True):
    self.trainset = trainset
    self.valset = valset or trainset  # TODO: FIXME: Note this choice.

    effective_max_errors = self.max_errors if self.max_errors is not None else dspy.settings.max_errors

    scores = []
    all_subscores = []
    score_data = []

    for seed in range(-3, self.num_candidate_sets):
        if (restrict is not None) and (seed not in restrict):
            continue

        trainset_copy = list(self.trainset)

        if seed == -3:
            # zero-shot
            program = student.reset_copy()

        elif seed == -2:
            # labels only
            teleprompter = LabeledFewShot(k=self.max_labeled_demos)
            program = teleprompter.compile(student, trainset=trainset_copy, sample=labeled_sample)

        elif seed == -1:
            # unshuffled few-shot
            optimizer = BootstrapFewShot(
                metric=self.metric,
                metric_threshold=self.metric_threshold,
                max_bootstrapped_demos=self.max_num_samples,
                max_labeled_demos=self.max_labeled_demos,
                teacher_settings=self.teacher_settings,
                max_rounds=self.max_rounds,
                max_errors=effective_max_errors,
            )
            program = optimizer.compile(student, teacher=teacher, trainset=trainset_copy)

        else:
            assert seed >= 0, seed

            random.Random(seed).shuffle(trainset_copy)
            size = random.Random(seed).randint(self.min_num_samples, self.max_num_samples)

            optimizer = BootstrapFewShot(
                metric=self.metric,
                metric_threshold=self.metric_threshold,
                max_bootstrapped_demos=size,
                max_labeled_demos=self.max_labeled_demos,
                teacher_settings=self.teacher_settings,
                max_rounds=self.max_rounds,
                max_errors=effective_max_errors,
            )

            program = optimizer.compile(student, teacher=teacher, trainset=trainset_copy)

        evaluate = Evaluate(
            devset=self.valset,
            metric=self.metric,
            num_threads=self.num_threads,
            max_errors=effective_max_errors,
            display_table=False,
            display_progress=True,
        )

        result = evaluate(program)

        score, subscores = result.score, [output[2] for output in result.results]

        all_subscores.append(subscores)

        if len(scores) == 0 or score > max(scores):
            print("New best score:", score, "for seed", seed)
            best_program = program

        scores.append(score)
        print(f"Scores so far: {scores}")
        print(f"Best score so far: {max(scores)}")

        score_data.append({"score": score, "subscores": subscores, "seed": seed, "program": program})

        if self.stop_at_score is not None and score >= self.stop_at_score:
            print(f"Stopping early because score {score} is >= stop_at_score {self.stop_at_score}")
            break

    # To best program, attach all program candidates in decreasing average score
    best_program.candidate_programs = score_data
    best_program.candidate_programs = sorted(
        best_program.candidate_programs, key=lambda x: x["score"], reverse=True
    )

    print(f"{len(best_program.candidate_programs)} candidate programs found.")

    return best_program

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