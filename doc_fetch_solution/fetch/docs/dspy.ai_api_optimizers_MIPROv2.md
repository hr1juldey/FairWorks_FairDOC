# Miprov2

**Source:** https://dspy.ai/api/optimizers/MIPROv2
**Fetched:** 2025-08-24 17:10:37
**Status:** Success

---

[

](https://github.com/stanfordnlp/dspy/blob/main/docs/docs/api/optimizers/MIPROv2.md)
# dspy.MIPROv2
MIPROv2 (Multiprompt Instruction PRoposal Optimizer Version 2) is an prompt optimizer capable of optimizing both instructions and few-shot examples jointly. It does this by bootstrapping few-shot example candidates, proposing instructions grounded in different dynamics of the task, and finding an optimized combination of these options using Bayesian Optimization. It can be used for optimizing few-shot examples & instructions jointly, or just instructions for 0-shot optimization.
## 
 dspy.MIPROv2(metric: Callable, prompt_model: Any | None = None, task_model: Any | None = None, teacher_settings: dict | None = None, max_bootstrapped_demos: int = 4, max_labeled_demos: int = 4, auto: Literal['light', 'medium', 'heavy'] | None = 'light', num_candidates: int | None = None, num_threads: int | None = None, max_errors: int | None = None, seed: int = 9, init_temperature: float = 0.5, verbose: bool = False, track_stats: bool = True, log_dir: str | None = None, metric_threshold: float | None = None)
Bases: Teleprompter
Source code in
dspy/teleprompt/mipro_optimizer_v2.py
```
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
[77](#__codelineno-0-77)
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
[89](#__codelineno-0-89)
[90](#__codelineno-0-90)
[91](#__codelineno-0-91)
```
```
def __init__(
    self,
    metric: Callable,
    prompt_model: Any | None = None,
    task_model: Any | None = None,
    teacher_settings: dict | None = None,
    max_bootstrapped_demos: int = 4,
    max_labeled_demos: int = 4,
    auto: Literal["light", "medium", "heavy"] | None = "light",
    num_candidates: int | None = None,
    num_threads: int | None = None,
    max_errors: int | None = None,
    seed: int = 9,
    init_temperature: float = 0.5,
    verbose: bool = False,
    track_stats: bool = True,
    log_dir: str | None = None,
    metric_threshold: float | None = None,
):
    # Validate 'auto' parameter
    allowed_modes = {None, "light", "medium", "heavy"}
    if auto not in allowed_modes:
        raise ValueError(f"Invalid value for auto: {auto}. Must be one of {allowed_modes}.")
    self.auto = auto
    self.num_fewshot_candidates = num_candidates
    self.num_instruct_candidates = num_candidates
    self.num_candidates = num_candidates
    self.metric = metric
    self.init_temperature = init_temperature
    self.task_model = task_model if task_model else dspy.settings.lm
    self.prompt_model = prompt_model if prompt_model else dspy.settings.lm
    self.max_bootstrapped_demos = max_bootstrapped_demos
    self.max_labeled_demos = max_labeled_demos
    self.verbose = verbose
    self.track_stats = track_stats
    self.log_dir = log_dir
    self.teacher_settings = teacher_settings or {}
    self.prompt_model_total_calls = 0
    self.total_calls = 0
    self.num_threads = num_threads
    self.max_errors = max_errors
    self.metric_threshold = metric_threshold
    self.seed = seed
    self.rng = None

```
### Functions
#### 
 compile(student: Any, *, trainset: list, teacher: Any = None, valset: list | None = None, num_trials: int | None = None, max_bootstrapped_demos: int | None = None, max_labeled_demos: int | None = None, seed: int | None = None, minibatch: bool = True, minibatch_size: int = 35, minibatch_full_eval_steps: int = 5, program_aware_proposer: bool = True, data_aware_proposer: bool = True, view_data_batch_size: int = 10, tip_aware_proposer: bool = True, fewshot_aware_proposer: bool = True, requires_permission_to_run: bool | None = None, provide_traceback: bool | None = None) -> Any
Source code in
dspy/teleprompt/mipro_optimizer_v2.py
```
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
[186](#__codelineno-0-186)
[187](#__codelineno-0-187)
[188](#__codelineno-0-188)
[189](#__codelineno-0-189)
[190](#__codelineno-0-190)
[191](#__codelineno-0-191)
[192](#__codelineno-0-192)
[193](#__codelineno-0-193)
[194](#__codelineno-0-194)
[195](#__codelineno-0-195)
[196](#__codelineno-0-196)
[197](#__codelineno-0-197)
[198](#__codelineno-0-198)
[199](#__codelineno-0-199)
[200](#__codelineno-0-200)
[201](#__codelineno-0-201)
[202](#__codelineno-0-202)
[203](#__codelineno-0-203)
[204](#__codelineno-0-204)
[205](#__codelineno-0-205)
[206](#__codelineno-0-206)
[207](#__codelineno-0-207)
[208](#__codelineno-0-208)
[209](#__codelineno-0-209)
[210](#__codelineno-0-210)
[211](#__codelineno-0-211)
[212](#__codelineno-0-212)
[213](#__codelineno-0-213)
[214](#__codelineno-0-214)
```
```
def compile(
    self,
    student: Any,
    *,
    trainset: list,
    teacher: Any = None,
    valset: list | None = None,
    num_trials: int | None = None,
    max_bootstrapped_demos: int | None = None,
    max_labeled_demos: int | None = None,
    seed: int | None = None,
    minibatch: bool = True,
    minibatch_size: int = 35,
    minibatch_full_eval_steps: int = 5,
    program_aware_proposer: bool = True,
    data_aware_proposer: bool = True,
    view_data_batch_size: int = 10,
    tip_aware_proposer: bool = True,
    fewshot_aware_proposer: bool = True,
    requires_permission_to_run: bool | None = None, # deprecated
    provide_traceback: bool | None = None,
) -> Any:
    if requires_permission_to_run == False:
        logger.warning(
            "'requires_permission_to_run' is deprecated and will be removed in a future version."
        )
    elif requires_permission_to_run == True:
        raise ValueError("User confirmation is removed from MIPROv2. Please remove the 'requires_permission_to_run' argument.")

    effective_max_errors = (
        self.max_errors
        if self.max_errors is not None
        else dspy.settings.max_errors
    )
    zeroshot_opt = (self.max_bootstrapped_demos == 0) and (self.max_labeled_demos == 0)

    # If auto is None, and num_trials is not provided (but num_candidates is), raise an error that suggests a good num_trials value
    if self.auto is None and (self.num_candidates is not None and num_trials is None):
        raise ValueError(
            f"If auto is None, num_trials must also be provided. Given num_candidates={self.num_candidates}, we'd recommend setting num_trials to ~{self._set_num_trials_from_num_candidates(student, zeroshot_opt, self.num_candidates)}."
        )

    # If auto is None, and num_candidates or num_trials is None, raise an error
    if self.auto is None and (self.num_candidates is None or num_trials is None):
        raise ValueError("If auto is None, num_candidates must also be provided.")

    # If auto is provided, and either num_candidates or num_trials is not None, raise an error
    if self.auto is not None and (self.num_candidates is not None or num_trials is not None):
        raise ValueError(
            "If auto is not None, num_candidates and num_trials cannot be set, since they would be overridden by the auto settings. Please either set auto to None, or do not specify num_candidates and num_trials."
        )

    # Set random seeds
    seed = seed or self.seed
    self._set_random_seeds(seed)

    # Update max demos if specified
    if max_bootstrapped_demos is not None:
        self.max_bootstrapped_demos = max_bootstrapped_demos
    if max_labeled_demos is not None:
        self.max_labeled_demos = max_labeled_demos

    # Set training & validation sets
    trainset, valset = self._set_and_validate_datasets(trainset, valset)

    # Set hyperparameters based on run mode (if set)
    num_trials, valset, minibatch = self._set_hyperparams_from_run_mode(
        student, num_trials, minibatch, zeroshot_opt, valset
    )

    if self.auto:
        self._print_auto_run_settings(num_trials, minibatch, valset)

    if minibatch and minibatch_size > len(valset):
        raise ValueError(f"Minibatch size cannot exceed the size of the valset. Valset size: {len(valset)}.")

    # Initialize program and evaluator
    program = student.deepcopy()
    evaluate = Evaluate(
        devset=valset,
        metric=self.metric,
        num_threads=self.num_threads,
        max_errors=effective_max_errors,
        display_table=False,
        display_progress=True,
        provide_traceback=provide_traceback,
    )

    # Step 1: Bootstrap few-shot examples
    demo_candidates = self._bootstrap_fewshot_examples(program, trainset, seed, teacher)

    # Step 2: Propose instruction candidates
    instruction_candidates = self._propose_instructions(
        program,
        trainset,
        demo_candidates,
        view_data_batch_size,
        program_aware_proposer,
        data_aware_proposer,
        tip_aware_proposer,
        fewshot_aware_proposer,
    )

    # If zero-shot, discard demos
    if zeroshot_opt:
        demo_candidates = None

    # Step 3: Find optimal prompt parameters
    best_program = self._optimize_prompt_parameters(
        program,
        instruction_candidates,
        demo_candidates,
        evaluate,
        valset,
        num_trials,
        minibatch,
        minibatch_size,
        minibatch_full_eval_steps,
        seed,
    )

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
## Example Usage
The program below shows optimizing a math program with MIPROv2
```
import dspy
from dspy.datasets.gsm8k import GSM8K, gsm8k_metric

# Import the optimizer
from dspy.teleprompt import MIPROv2

# Initialize the LM
lm = dspy.LM('openai/gpt-4o-mini', api_key='YOUR_OPENAI_API_KEY')
dspy.configure(lm=lm)

# Initialize optimizer
teleprompter = MIPROv2(
    metric=gsm8k_metric,
    auto="medium", # Can choose between light, medium, and heavy optimization runs
)

# Optimize program
print(f"Optimizing program with MIPROv2...")
gsm8k = GSM8K()
optimized_program = teleprompter.compile(
    dspy.ChainOfThought("question -> answer"),
    trainset=gsm8k.train,
)

# Save optimize program for future use
optimized_program.save(f"optimized.json")

```
## How MIPROv2 works
At a high level, MIPROv2 works by creating both few-shot examples and new instructions for each predictor in your LM program, and then searching over these using Bayesian Optimization to find the best combination of these variables for your program.  If you want a visual explanation check out this twitter thread.
These steps are broken down in more detail below:
1) Bootstrap Few-Shot Examples: Randomly samples examples from your training set, and run them through your LM program. If the output from the program is correct for this example, it is kept as a valid few-shot example candidate. Otherwise, we try another example until we've curated the specified amount of few-shot example candidates. This step creates num_candidates sets of max_bootstrapped_demos bootstrapped examples and max_labeled_demos basic examples sampled from the training set.
2) Propose Instruction Candidates. The instruction proposer includes (1) a generated summary of properties of the training dataset, (2) a generated summary of your LM program's code and the specific predictor that an instruction is being generated for, (3) the previously bootstrapped few-shot examples to show reference inputs / outputs for a given predictor and (4) a randomly sampled tip for generation (i.e. "be creative", "be concise", etc.) to help explore the feature space of potential instructions.  This context is provided to a prompt_model which writes high quality instruction candidates.
3) Find an Optimized Combination of Few-Shot Examples & Instructions. Finally, we use Bayesian Optimization to choose which combinations of instructions and demonstrations work best for each predictor in our program. This works by running a series of num_trials trials, where a new set of prompts are evaluated over our validation set at each trial. The new set of prompts are only evaluated on a minibatch of size minibatch_size at each trial (when minibatch=True). The best averaging set of prompts is then evaluated on the full validation set every minibatch_full_eval_steps. At the end of the optimization process, the LM program with the set of prompts that performed best on the full validation set is returned.
For those interested in more details, more information on MIPROv2 along with a study on MIPROv2 compared with other DSPy optimizers can be found in this paper.