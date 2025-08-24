# Gepa

**Source:** https://dspy.ai/api/optimizers/GEPA
**Fetched:** 2025-08-24 17:10:34
**Status:** Success

---

[

](https://github.com/stanfordnlp/dspy/blob/main/docs/docs/api/optimizers/GEPA.md)
# dspy.GEPA: Reflective Prompt Optimizer
GEPA (Genetic-Pareto) is a reflective optimizer proposed in "GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning" (Agrawal et al., 2025, arxiv:2507.19457), that adaptively evolves textual components (such as prompts) of arbitrary systems. In addition to scalar scores returned by metrics, users can also provide GEPA with a text feedback to guide the optimization process. Such textual feedback provides GEPA more visibility into why the system got the score that it did, and then GEPA can introspect to identify how to improve the score. This allows GEPA to propose high performing prompts in very few rollouts.
## 
 dspy.GEPA(metric: GEPAFeedbackMetric, *, auto: Literal['light', 'medium', 'heavy'] | None = None, max_full_evals: int | None = None, max_metric_calls: int | None = None, reflection_minibatch_size: int = 3, candidate_selection_strategy: Literal['pareto', 'current_best'] = 'pareto', reflection_lm: LM | None = None, skip_perfect_score: bool = True, add_format_failure_as_feedback: bool = False, use_merge: bool = True, max_merge_invocations: int | None = 5, num_threads: int | None = None, failure_score: float = 0.0, perfect_score: float = 1.0, log_dir: str = None, track_stats: bool = False, use_wandb: bool = False, wandb_api_key: str | None = None, wandb_init_kwargs: dict[str, Any] | None = None, track_best_outputs: bool = False, seed: int | None = 0)
Bases: Teleprompter
GEPA is an evolutionary optimizer, which uses reflection to evolve text components
of complex systems. GEPA is proposed in the paper GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning.
The GEPA optimization engine is provided by the gepa package, available from https://github.com/gepa-ai/gepa.
GEPA captures full traces of the DSPy module's execution, identifies the parts of the trace
corresponding to a specific predictor, and reflects on the behaviour of the predictor to
propose a new instruction for the predictor. GEPA allows users to provide textual feedback
to the optimizer, which is used to guide the evolution of the predictor. The textual feedback
can be provided at the granularity of individual predictors, or at the level of the entire system's
execution.
To provide feedback to the GEPA optimizer, implement a metric as follows:
def metric(
    gold: Example,
    pred: Prediction,
    trace: Optional[DSPyTrace] = None,
    pred_name: Optional[str] = None,
    pred_trace: Optional[DSPyTrace] = None,
) -> float | ScoreWithFeedback:
    """
    This function is called with the following arguments:
    - gold: The gold example.
    - pred: The predicted output.
    - trace: Optional. The trace of the program's execution.
    - pred_name: Optional. The name of the target predictor currently being optimized by GEPA, for which 
        the feedback is being requested.
    - pred_trace: Optional. The trace of the target predictor's execution GEPA is seeking feedback for.

    Note the `pred_name` and `pred_trace` arguments. During optimization, GEPA will call the metric to obtain
    feedback for individual predictors being optimized. GEPA provides the name of the predictor in `pred_name`
    and the sub-trace (of the trace) corresponding to the predictor in `pred_trace`.
    If available at the predictor level, the metric should return {'score': float, 'feedback': str} corresponding 
    to the predictor.
    If not available at the predictor level, the metric can also return a text feedback at the program level
    (using just the gold, pred and trace).
    If no feedback is returned, GEPA will use a simple text feedback consisting of just the score: 
    f"This trajectory got a score of {score}."
    """
    ...
GEPA can also be used as a batch inference-time search strategy, by passing valset=trainset, track_stats=True, track_best_outputs=True, and using the
detailed_results attribute of the optimized program (returned by compile) to get the Pareto frontier of the batch. optimized_program.detailed_results.best_outputs_valset will contain the best outputs for each task in the batch.
Example:
gepa = GEPA(metric=metric, track_stats=True)
batch_of_tasks = [dspy.Example(...) for task in tasks]
new_prog = gepa.compile(student, trainset=trainset, valset=batch_of_tasks)
pareto_frontier = new_prog.detailed_results.val_aggregate_scores
# pareto_frontier is a list of scores, one for each task in the batch.
Parameters:
Name
Type
Description
Default
-
metric
The metric function to use for feedback and evaluation.
required
Budget
configuration (exactly one of the following must be provided
required
-
auto
The auto budget to use for the run.
required
-
max_full_evals
The maximum number of full evaluations to perform.
required
-
max_metric_calls
The maximum number of metric calls to perform.
required
Reflection
based configuration
required
-
reflection_minibatch_size
The number of examples to use for reflection in a single GEPA step.
required
-
candidate_selection_strategy
The strategy to use for candidate selection. Default is "pareto", which stochastically selects candidates from the Pareto frontier of all validation scores.
required
-
reflection_lm
[Required] The language model to use for reflection. GEPA benefits from a strong reflection model, and you can use dspy.LM(model='gpt-5', temperature=1.0, max_tokens=32000) to get a good reflection model.
required
Merge-based
configuration
required
-
use_merge
Whether to use merge-based optimization. Default is True.
required
-
max_merge_invocations
The maximum number of merge invocations to perform. Default is 5.
required
Evaluation
configuration
required
-
num_threads
The number of threads to use for evaluation with Evaluate
required
-
failure_score
The score to assign to failed examples. Default is 0.0.
required
-
perfect_score
The maximum score achievable by the metric. Default is 1.0. Used by GEPA to determine if all examples in a minibatch are perfect.
required
Logging
configuration
required
-
log_dir
The directory to save the logs. GEPA saves elaborate logs, along with all the candidate programs, in this directory. Running GEPA with the same log_dir will resume the run from the last checkpoint.
required
-
track_stats
Whether to return detailed results and all proposed programs in the detailed_results attribute of the optimized program. Default is False.
required
-
use_wandb
Whether to use wandb for logging. Default is False.
required
-
wandb_api_key
The API key to use for wandb. If not provided, wandb will use the API key from the environment variable WANDB_API_KEY.
required
-
wandb_init_kwargs
Additional keyword arguments to pass to wandb.init.
required
-
track_best_outputs
Whether to track the best outputs on the validation set. track_stats must be True if track_best_outputs is True. optimized_program.detailed_results.best_outputs_valset will contain the best outputs for each task in the validation set.
required
Reproducibility
required
-
seed
The random seed to use for reproducibility. Default is 0.
required
Source code in
dspy/teleprompt/gepa/gepa.py
```
[232](#__codelineno-0-232)
[233](#__codelineno-0-233)
[234](#__codelineno-0-234)
[235](#__codelineno-0-235)
[236](#__codelineno-0-236)
[237](#__codelineno-0-237)
[238](#__codelineno-0-238)
[239](#__codelineno-0-239)
[240](#__codelineno-0-240)
[241](#__codelineno-0-241)
[242](#__codelineno-0-242)
[243](#__codelineno-0-243)
[244](#__codelineno-0-244)
[245](#__codelineno-0-245)
[246](#__codelineno-0-246)
[247](#__codelineno-0-247)
[248](#__codelineno-0-248)
[249](#__codelineno-0-249)
[250](#__codelineno-0-250)
[251](#__codelineno-0-251)
[252](#__codelineno-0-252)
[253](#__codelineno-0-253)
[254](#__codelineno-0-254)
[255](#__codelineno-0-255)
[256](#__codelineno-0-256)
[257](#__codelineno-0-257)
[258](#__codelineno-0-258)
[259](#__codelineno-0-259)
[260](#__codelineno-0-260)
[261](#__codelineno-0-261)
[262](#__codelineno-0-262)
[263](#__codelineno-0-263)
[264](#__codelineno-0-264)
[265](#__codelineno-0-265)
[266](#__codelineno-0-266)
[267](#__codelineno-0-267)
[268](#__codelineno-0-268)
[269](#__codelineno-0-269)
[270](#__codelineno-0-270)
[271](#__codelineno-0-271)
[272](#__codelineno-0-272)
[273](#__codelineno-0-273)
[274](#__codelineno-0-274)
[275](#__codelineno-0-275)
[276](#__codelineno-0-276)
[277](#__codelineno-0-277)
[278](#__codelineno-0-278)
[279](#__codelineno-0-279)
[280](#__codelineno-0-280)
[281](#__codelineno-0-281)
[282](#__codelineno-0-282)
[283](#__codelineno-0-283)
[284](#__codelineno-0-284)
[285](#__codelineno-0-285)
[286](#__codelineno-0-286)
[287](#__codelineno-0-287)
[288](#__codelineno-0-288)
[289](#__codelineno-0-289)
[290](#__codelineno-0-290)
[291](#__codelineno-0-291)
[292](#__codelineno-0-292)
[293](#__codelineno-0-293)
[294](#__codelineno-0-294)
[295](#__codelineno-0-295)
[296](#__codelineno-0-296)
[297](#__codelineno-0-297)
[298](#__codelineno-0-298)
[299](#__codelineno-0-299)
[300](#__codelineno-0-300)
[301](#__codelineno-0-301)
[302](#__codelineno-0-302)
[303](#__codelineno-0-303)
[304](#__codelineno-0-304)
[305](#__codelineno-0-305)
[306](#__codelineno-0-306)
[307](#__codelineno-0-307)
[308](#__codelineno-0-308)
[309](#__codelineno-0-309)
[310](#__codelineno-0-310)
[311](#__codelineno-0-311)
```
```
def __init__(
    self,
    metric: GEPAFeedbackMetric,
    *,
    # Budget configuration
    auto: Literal["light", "medium", "heavy"] | None = None,
    max_full_evals: int | None = None,
    max_metric_calls: int | None = None,
    # Reflection based configuration
    reflection_minibatch_size: int = 3,
    candidate_selection_strategy: Literal["pareto", "current_best"] = "pareto",
    reflection_lm: LM | None = None,
    skip_perfect_score: bool = True,
    add_format_failure_as_feedback: bool = False,
    # Merge-based configuration
    use_merge: bool = True,
    max_merge_invocations: int | None = 5,
    # Evaluation configuration
    num_threads: int | None = None,
    failure_score: float = 0.0,
    perfect_score: float = 1.0,
    # Logging
    log_dir: str = None,
    track_stats: bool = False,
    use_wandb: bool = False,
    wandb_api_key: str | None = None,
    wandb_init_kwargs: dict[str, Any] | None = None,
    track_best_outputs: bool = False,
    # Reproducibility
    seed: int | None = 0,
):
    self.metric_fn = metric

    # Budget configuration
    assert (
        (max_metric_calls is not None) +
        (max_full_evals is not None) +
        (auto is not None)
        == 1
    ), (
        "Exactly one of max_metric_calls, max_full_evals, auto must be set. "
        f"You set max_metric_calls={max_metric_calls}, "
        f"max_full_evals={max_full_evals}, "
        f"auto={auto}."
    )
    self.auto = auto
    self.max_full_evals = max_full_evals
    self.max_metric_calls = max_metric_calls

    # Reflection based configuration
    self.reflection_minibatch_size = reflection_minibatch_size
    self.candidate_selection_strategy = candidate_selection_strategy
    # self.reflection_lm = reflection_lm
    assert reflection_lm is not None, "GEPA requires a reflection language model to be provided. Typically, you can use `dspy.LM(model='gpt-5', temperature=1.0, max_tokens=32000)` to get a good reflection model. Reflection LM is used by GEPA to reflect on the behavior of the program and propose new instructions, and will benefit from a strong model."
    self.reflection_lm = lambda x: reflection_lm(x)[0]
    self.skip_perfect_score = skip_perfect_score
    self.add_format_failure_as_feedback = add_format_failure_as_feedback

    # Merge-based configuration
    self.use_merge = use_merge
    self.max_merge_invocations = max_merge_invocations

    # Evaluation Configuration
    self.num_threads = num_threads
    self.failure_score = failure_score
    self.perfect_score = perfect_score

    # Logging configuration
    self.log_dir = log_dir
    self.track_stats = track_stats
    self.use_wandb = use_wandb
    self.wandb_api_key = wandb_api_key
    self.wandb_init_kwargs = wandb_init_kwargs

    if track_best_outputs:
        assert track_stats, "track_stats must be True if track_best_outputs is True."
    self.track_best_outputs = track_best_outputs

    # Reproducibility
    self.seed = seed

```
### Functions
#### 
 compile(student: Module, *, trainset: list[Example], teacher: Module | None = None, valset: list[Example] | None = None) -> Module
GEPA uses the trainset to perform reflective updates to the prompt, but uses the valset for tracking Pareto scores.
If no valset is provided, GEPA will use the trainset for both.
Parameters:
- student: The student module to optimize.
- trainset: The training set to use for reflective updates.
- valset: The validation set to use for tracking Pareto scores. If not provided, GEPA will use the trainset for both.
Source code in
dspy/teleprompt/gepa/gepa.py
```
[344](#__codelineno-0-344)
[345](#__codelineno-0-345)
[346](#__codelineno-0-346)
[347](#__codelineno-0-347)
[348](#__codelineno-0-348)
[349](#__codelineno-0-349)
[350](#__codelineno-0-350)
[351](#__codelineno-0-351)
[352](#__codelineno-0-352)
[353](#__codelineno-0-353)
[354](#__codelineno-0-354)
[355](#__codelineno-0-355)
[356](#__codelineno-0-356)
[357](#__codelineno-0-357)
[358](#__codelineno-0-358)
[359](#__codelineno-0-359)
[360](#__codelineno-0-360)
[361](#__codelineno-0-361)
[362](#__codelineno-0-362)
[363](#__codelineno-0-363)
[364](#__codelineno-0-364)
[365](#__codelineno-0-365)
[366](#__codelineno-0-366)
[367](#__codelineno-0-367)
[368](#__codelineno-0-368)
[369](#__codelineno-0-369)
[370](#__codelineno-0-370)
[371](#__codelineno-0-371)
[372](#__codelineno-0-372)
[373](#__codelineno-0-373)
[374](#__codelineno-0-374)
[375](#__codelineno-0-375)
[376](#__codelineno-0-376)
[377](#__codelineno-0-377)
[378](#__codelineno-0-378)
[379](#__codelineno-0-379)
[380](#__codelineno-0-380)
[381](#__codelineno-0-381)
[382](#__codelineno-0-382)
[383](#__codelineno-0-383)
[384](#__codelineno-0-384)
[385](#__codelineno-0-385)
[386](#__codelineno-0-386)
[387](#__codelineno-0-387)
[388](#__codelineno-0-388)
[389](#__codelineno-0-389)
[390](#__codelineno-0-390)
[391](#__codelineno-0-391)
[392](#__codelineno-0-392)
[393](#__codelineno-0-393)
[394](#__codelineno-0-394)
[395](#__codelineno-0-395)
[396](#__codelineno-0-396)
[397](#__codelineno-0-397)
[398](#__codelineno-0-398)
[399](#__codelineno-0-399)
[400](#__codelineno-0-400)
[401](#__codelineno-0-401)
[402](#__codelineno-0-402)
[403](#__codelineno-0-403)
[404](#__codelineno-0-404)
[405](#__codelineno-0-405)
[406](#__codelineno-0-406)
[407](#__codelineno-0-407)
[408](#__codelineno-0-408)
[409](#__codelineno-0-409)
[410](#__codelineno-0-410)
[411](#__codelineno-0-411)
[412](#__codelineno-0-412)
[413](#__codelineno-0-413)
[414](#__codelineno-0-414)
[415](#__codelineno-0-415)
[416](#__codelineno-0-416)
[417](#__codelineno-0-417)
[418](#__codelineno-0-418)
[419](#__codelineno-0-419)
[420](#__codelineno-0-420)
[421](#__codelineno-0-421)
[422](#__codelineno-0-422)
[423](#__codelineno-0-423)
[424](#__codelineno-0-424)
[425](#__codelineno-0-425)
[426](#__codelineno-0-426)
[427](#__codelineno-0-427)
[428](#__codelineno-0-428)
[429](#__codelineno-0-429)
[430](#__codelineno-0-430)
[431](#__codelineno-0-431)
[432](#__codelineno-0-432)
[433](#__codelineno-0-433)
[434](#__codelineno-0-434)
[435](#__codelineno-0-435)
[436](#__codelineno-0-436)
[437](#__codelineno-0-437)
[438](#__codelineno-0-438)
[439](#__codelineno-0-439)
[440](#__codelineno-0-440)
[441](#__codelineno-0-441)
[442](#__codelineno-0-442)
[443](#__codelineno-0-443)
[444](#__codelineno-0-444)
[445](#__codelineno-0-445)
[446](#__codelineno-0-446)
[447](#__codelineno-0-447)
[448](#__codelineno-0-448)
[449](#__codelineno-0-449)
[450](#__codelineno-0-450)
[451](#__codelineno-0-451)
[452](#__codelineno-0-452)
[453](#__codelineno-0-453)
[454](#__codelineno-0-454)
[455](#__codelineno-0-455)
[456](#__codelineno-0-456)
[457](#__codelineno-0-457)
[458](#__codelineno-0-458)
[459](#__codelineno-0-459)
[460](#__codelineno-0-460)
[461](#__codelineno-0-461)
[462](#__codelineno-0-462)
[463](#__codelineno-0-463)
[464](#__codelineno-0-464)
[465](#__codelineno-0-465)
[466](#__codelineno-0-466)
[467](#__codelineno-0-467)
[468](#__codelineno-0-468)
[469](#__codelineno-0-469)
```
```
def compile(
    self,
    student: Module,
    *,
    trainset: list[Example],
    teacher: Module | None = None,
    valset: list[Example] | None = None,
) -> Module:
    """
    GEPA uses the trainset to perform reflective updates to the prompt, but uses the valset for tracking Pareto scores.
    If no valset is provided, GEPA will use the trainset for both.

    Parameters:
    - student: The student module to optimize.
    - trainset: The training set to use for reflective updates.
    - valset: The validation set to use for tracking Pareto scores. If not provided, GEPA will use the trainset for both.
    """
    from gepa import GEPAResult, optimize

    from dspy.teleprompt.gepa.gepa_utils import DspyAdapter, LoggerAdapter

    assert trainset is not None and len(trainset) > 0, "Trainset must be provided and non-empty"
    assert teacher is None, "Teacher is not supported in DspyGEPA yet."

    if self.auto is not None:
        self.max_metric_calls = self.auto_budget(
            num_preds=len(student.predictors()),
            num_candidates=AUTO_RUN_SETTINGS[self.auto]["n"],
            valset_size=len(valset) if valset is not None else len(trainset),
        )
    elif self.max_full_evals is not None:
        self.max_metric_calls = self.max_full_evals * (len(trainset) + (len(valset) if valset is not None else 0))
    else:
        assert self.max_metric_calls is not None, "Either auto, max_full_evals, or max_metric_calls must be set."

    logger.info(f"Running GEPA for approx {self.max_metric_calls} metric calls of the program. This amounts to {self.max_metric_calls / len(trainset) if valset is None else self.max_metric_calls / (len(trainset) + len(valset)):.2f} full evals on the {'train' if valset is None else 'train+val'} set.")

    valset = valset or trainset
    logger.info(f"Using {len(valset)} examples for tracking Pareto scores. You can consider using a smaller sample of the valset to allow GEPA to explore more diverse solutions within the same budget.")

    rng = random.Random(self.seed)

    def feedback_fn_creator(pred_name: str, predictor) -> "PredictorFeedbackFn":
        def feedback_fn(
            predictor_output: dict[str, Any],
            predictor_inputs: dict[str, Any],
            module_inputs: Example,
            module_outputs: Prediction,
            captured_trace: "DSPyTrace",
        ) -> "ScoreWithFeedback":
            trace_for_pred = [(predictor, predictor_inputs, predictor_output)]
            o = self.metric_fn(
                module_inputs,
                module_outputs,
                captured_trace,
                pred_name,
                trace_for_pred,
            )
            if hasattr(o, "feedback"):
                if o["feedback"] is None:
                    o["feedback"] = f"This trajectory got a score of {o['score']}."
                return o
            else:
                return dict(score=o, feedback=f"This trajectory got a score of {o}.")
        return feedback_fn

    feedback_map = {
        k: feedback_fn_creator(k, v)
        for k, v in student.named_predictors()
    }

    # Build the DSPy adapter that encapsulates evaluation, trace capture, feedback extraction, and instruction proposal
    adapter = DspyAdapter(
        student_module=student,
        metric_fn=self.metric_fn,
        feedback_map=feedback_map,
        failure_score=self.failure_score,
        num_threads=self.num_threads,
        add_format_failure_as_feedback=self.add_format_failure_as_feedback,
        rng=rng,
    )

    reflection_lm = self.reflection_lm

    # Instantiate GEPA with the simpler adapter-based API
    base_program = {name: pred.signature.instructions for name, pred in student.named_predictors()}
    gepa_result: GEPAResult = optimize(
        seed_candidate=base_program,
        trainset=trainset,
        valset=valset,
        adapter=adapter,

        # Reflection-based configuration
        reflection_lm=reflection_lm,
        candidate_selection_strategy=self.candidate_selection_strategy,
        skip_perfect_score=self.skip_perfect_score,
        reflection_minibatch_size=self.reflection_minibatch_size,

        perfect_score=self.perfect_score,

        # Merge-based configuration
        use_merge=self.use_merge,
        max_merge_invocations=self.max_merge_invocations,

        # Budget
        max_metric_calls=self.max_metric_calls,

        # Logging
        logger=LoggerAdapter(logger),
        run_dir=self.log_dir,
        use_wandb=self.use_wandb,
        wandb_api_key=self.wandb_api_key,
        wandb_init_kwargs=self.wandb_init_kwargs,
        track_best_outputs=self.track_best_outputs,

        # Reproducibility
        seed=self.seed,
    )

    new_prog = adapter.build_program(gepa_result.best_candidate)

    if self.track_stats:
        dspy_gepa_result = DspyGEPAResult.from_gepa_result(gepa_result, adapter)
        new_prog.detailed_results = dspy_gepa_result

    return new_prog

```
:::
One of the key insights behind GEPA is its ability to leverage domain-specific textual feedback. Users should provide a feedback function as the GEPA metric, which has the following call signature:
## 
 dspy.teleprompt.gepa.gepa.GEPAFeedbackMetric
Bases: Protocol
### Functions
#### 
 __call__(gold: Example, pred: Prediction, trace: Optional[DSPyTrace], pred_name: str | None, pred_trace: Optional[DSPyTrace]) -> Union[float, ScoreWithFeedback]
This function is called with the following arguments:
- gold: The gold example.
- pred: The predicted output.
- trace: Optional. The trace of the program's execution.
- pred_name: Optional. The name of the target predictor currently being optimized by GEPA, for which 
    the feedback is being requested.
- pred_trace: Optional. The trace of the target predictor's execution GEPA is seeking feedback for.
Note the pred_name and pred_trace arguments. During optimization, GEPA will call the metric to obtain
feedback for individual predictors being optimized. GEPA provides the name of the predictor in pred_name
and the sub-trace (of the trace) corresponding to the predictor in pred_trace.
If available at the predictor level, the metric should return dspy.Prediction(score: float, feedback: str) corresponding 
to the predictor.
If not available at the predictor level, the metric can also return a text feedback at the program level
(using just the gold, pred and trace).
If no feedback is returned, GEPA will use a simple text feedback consisting of just the score: 
f"This trajectory got a score of {score}."
Source code in
dspy/teleprompt/gepa/gepa.py
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
```
```
def __call__(
    gold: Example,
    pred: Prediction,
    trace: Optional["DSPyTrace"],
    pred_name: str | None,
    pred_trace: Optional["DSPyTrace"],
) -> Union[float, "ScoreWithFeedback"]:
    """
    This function is called with the following arguments:
    - gold: The gold example.
    - pred: The predicted output.
    - trace: Optional. The trace of the program's execution.
    - pred_name: Optional. The name of the target predictor currently being optimized by GEPA, for which 
        the feedback is being requested.
    - pred_trace: Optional. The trace of the target predictor's execution GEPA is seeking feedback for.

    Note the `pred_name` and `pred_trace` arguments. During optimization, GEPA will call the metric to obtain
    feedback for individual predictors being optimized. GEPA provides the name of the predictor in `pred_name`
    and the sub-trace (of the trace) corresponding to the predictor in `pred_trace`.
    If available at the predictor level, the metric should return dspy.Prediction(score: float, feedback: str) corresponding 
    to the predictor.
    If not available at the predictor level, the metric can also return a text feedback at the program level
    (using just the gold, pred and trace).
    If no feedback is returned, GEPA will use a simple text feedback consisting of just the score: 
    f"This trajectory got a score of {score}."
    """
    ...

```
:::
When track_stats=True, GEPA returns detailed results about all of the proposed candidates, and metadata about the optimization run. The results are available in the detailed_results attribute of the optimized program returned by GEPA, and has the following type:
## 
 dspy.teleprompt.gepa.gepa.DspyGEPAResult(candidates: list[Module], parents: list[list[int | None]], val_aggregate_scores: list[float], val_subscores: list[list[float]], per_val_instance_best_candidates: list[set[int]], discovery_eval_counts: list[int], best_outputs_valset: list[list[tuple[int, list[Prediction]]]] | None = None, total_metric_calls: int | None = None, num_full_val_evals: int | None = None, log_dir: str | None = None, seed: int | None = None)

dataclass
Additional data related to the GEPA run.
Fields:
- candidates: list of proposed candidates (component_name -> component_text)
- parents: lineage info; for each candidate i, parents[i] is a list of parent indices or None
- val_aggregate_scores: per-candidate aggregate score on the validation set (higher is better)
- val_subscores: per-candidate per-instance scores on the validation set (len == num_val_instances)
- per_val_instance_best_candidates: for each val instance t, a set of candidate indices achieving the best score on t
- discovery_eval_counts: Budget (number of metric calls / rollouts) consumed up to the discovery of each candidate
- total_metric_calls: total number of metric calls made across the run
- num_full_val_evals: number of full validation evaluations performed
- log_dir: where artifacts were written (if any)
- 
seed: RNG seed for reproducibility (if known)

- 
best_idx: candidate index with the highest val_aggregate_scores

- best_candidate: the program text mapping for best_idx
### Attributes
#### 
 candidates: list[Module]

instance-attribute
#### 
 parents: list[list[int | None]]

instance-attribute
#### 
 val_aggregate_scores: list[float]

instance-attribute
#### 
 val_subscores: list[list[float]]

instance-attribute
#### 
 per_val_instance_best_candidates: list[set[int]]

instance-attribute
#### 
 discovery_eval_counts: list[int]

instance-attribute
#### 
 best_outputs_valset: list[list[tuple[int, list[Prediction]]]] | None = None

class-attribute
instance-attribute
#### 
 total_metric_calls: int | None = None

class-attribute
instance-attribute
#### 
 num_full_val_evals: int | None = None

class-attribute
instance-attribute
#### 
 log_dir: str | None = None

class-attribute
instance-attribute
#### 
 seed: int | None = None

class-attribute
instance-attribute
#### 
 best_idx: int

property
#### 
 best_candidate: dict[str, str]

property
#### 
 highest_score_achieved_per_val_task: list[float]

property
### Functions
#### 
 to_dict() -> dict[str, Any]
Source code in
dspy/teleprompt/gepa/gepa.py
```
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
```
```
def to_dict(self) -> dict[str, Any]:
    cands = [
        {k: v for k, v in cand.items()}
        for cand in self.candidates
    ]

    return dict(
        candidates=cands,
        parents=self.parents,
        val_aggregate_scores=self.val_aggregate_scores,
        best_outputs_valset=self.best_outputs_valset,
        val_subscores=self.val_subscores,
        per_val_instance_best_candidates=[list(s) for s in self.per_val_instance_best_candidates],
        discovery_eval_counts=self.discovery_eval_counts,
        total_metric_calls=self.total_metric_calls,
        num_full_val_evals=self.num_full_val_evals,
        log_dir=self.log_dir,
        seed=self.seed,
        best_idx=self.best_idx,
    )

```
#### 
 from_gepa_result(gepa_result: GEPAResult, adapter: DspyAdapter) -> DspyGEPAResult

staticmethod
Source code in
dspy/teleprompt/gepa/gepa.py
```
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
```
```
@staticmethod
def from_gepa_result(gepa_result: "GEPAResult", adapter: "DspyAdapter") -> "DspyGEPAResult":
    return DspyGEPAResult(
        candidates=[adapter.build_program(c) for c in gepa_result.candidates],
        parents=gepa_result.parents,
        val_aggregate_scores=gepa_result.val_aggregate_scores,
        best_outputs_valset=gepa_result.best_outputs_valset,
        val_subscores=gepa_result.val_subscores,
        per_val_instance_best_candidates=gepa_result.per_val_instance_best_candidates,
        discovery_eval_counts=gepa_result.discovery_eval_counts,
        total_metric_calls=gepa_result.total_metric_calls,
        num_full_val_evals=gepa_result.num_full_val_evals,
        log_dir=gepa_result.run_dir,
        seed=gepa_result.seed,
    )

```
:::
## Usage Examples
See GEPA usage tutorials in GEPA Tutorials.
### Inference-Time Search
GEPA can act as a test-time/inference search mechanism. By setting your valset to your evaluation batch and using track_best_outputs=True, GEPA produces for each batch element the highest-scoring outputs found during the evolutionary search.
```
gepa = dspy.GEPA(metric=metric, track_stats=True, ...)
new_prog = gepa.compile(student, trainset=my_tasks, valset=my_tasks)
highest_score_achieved_per_task = new_prog.detailed_results.highest_score_achieved_per_val_task
best_outputs = new_prog.detailed_results.best_outputs_valset

```
## How Does GEPA Work?
### 1. Reflective Prompt Mutation
GEPA uses LLMs to reflect on structured execution traces (inputs, outputs, failures, feedback), targeting a chosen module and proposing a new instruction/program text tailored to real observed failures and rich textual/environmental feedback.
### 2. Rich Textual Feedback as Optimization Signal
GEPA can leverage any textual feedback available—not just scalar rewards. This includes evaluation logs, code traces, failed parses, constraint violations, error message strings, or even isolated submodule-specific feedback. This allows actionable, domain-aware optimization.
### 3. Pareto-based Candidate Selection
Rather than evolving just the best global candidate (which leads to local optima or stagnation), GEPA maintains a Pareto frontier: the set of candidates which achieve the highest score on at least one evaluation instance. In each iteration, the next candidate to mutate is sampled (with probability proportional to coverage) from this frontier, guaranteeing both exploration and robust retention of complementary strategies.
### Algorithm Summary
1. Initialize the candidate pool with the the unoptimized program.
2. Iterate:
3. Sample a candidate (from Pareto frontier).
4. Sample a minibatch from the train set.
5. Collect execution traces + feedbacks for module rollout on minibatch.
6. Select a module of the candidate for targeted improvement.
7. LLM Reflection: Propose a new instruction/prompt for the targeted module using reflective meta-prompting and the gathered feedback.
8. Roll out the new candidate on the minibatch; if improved, evaluate on Pareto validation set.
9. Update the candidate pool/Pareto frontier.
10. [Optionally] System-aware merge/crossover: Combine best-performing modules from distinct lineages.
11. Continue until rollout or metric budget is exhausted. 
12. Return candidate with best aggregate performance on validation.
## Implementing Feedback Metrics
A well-designed metric is central to GEPA's sample efficiency and learning signal richness. GEPA expects the metric to returns a dspy.Prediction(score=..., feedback=...). GEPA leverages natural language traces from LLM-based workflows for optimization, preserving intermediate trajectories and errors in plain text rather than reducing them to numerical rewards. This mirrors human diagnostic processes, enabling clearer identification of system behaviors and bottlenecks.
Practical Recipe for GEPA-Friendly Feedback:
- Leverage Existing Artifacts: Use logs, unit tests, evaluation scripts, and profiler outputs; surfacing these often suffices.
- Decompose Outcomes: Break scores into per-objective components (e.g., correctness, latency, cost, safety) and attribute errors to steps.
- Expose Trajectories: Label pipeline stages, reporting pass/fail with salient errors (e.g., in code generation pipelines).
- Ground in Checks: Employ automatic validators (unit tests, schemas, simulators) or LLM-as-a-judge for non-verifiable tasks (as in PUPA).
- Prioritize Clarity: Focus on error coverage and decision points over technical complexity.
### Examples
- Document Retrieval (e.g., HotpotQA): List correctly retrieved, incorrect, or missed documents, beyond mere Recall/F1 scores.
- Multi-Objective Tasks (e.g., PUPA): Decompose aggregate scores to reveal contributions from each objective, highlighting tradeoffs (e.g., quality vs. privacy).
- Stacked Pipelines (e.g., code generation: parse → compile → run → profile → evaluate): Expose stage-specific failures; natural-language traces often suffice for LLM self-correction.
## Further Reading
- [GEPA Paper: arxiv:2507.19457](https://arxiv.org/abs/2507.19457)
- [GEPA Github](https://github.com/gepa-ai/gepa) - This repository provides the core GEPA evolution pipeline used by dspy.GEPA optimizer.
- [DSPy Tutorials](../../../tutorials/gepa_ai_program/)