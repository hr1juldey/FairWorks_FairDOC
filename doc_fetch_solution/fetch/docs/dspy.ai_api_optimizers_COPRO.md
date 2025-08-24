# Copro

**Source:** https://dspy.ai/api/optimizers/COPRO
**Fetched:** 2025-08-24 17:10:34
**Status:** Success

---

[

](https://github.com/stanfordnlp/dspy/blob/main/docs/docs/api/optimizers/COPRO.md)
# dspy.COPRO
## 
 dspy.COPRO(prompt_model=None, metric=None, breadth=10, depth=3, init_temperature=1.4, track_stats=False, **_kwargs)
Bases: Teleprompter
Source code in
dspy/teleprompt/copro_optimizer.py
```
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
```
```
def __init__(
    self,
    prompt_model=None,
    metric=None,
    breadth=10,
    depth=3,
    init_temperature=1.4,
    track_stats=False,
    **_kwargs,
):
    if breadth <= 1:
        raise ValueError("Breadth must be greater than 1")
    self.metric = metric
    self.breadth = breadth
    self.depth = depth
    self.init_temperature = init_temperature
    self.prompt_model = prompt_model
    self.track_stats = track_stats

```
### Functions
#### 
 compile(student, *, trainset, eval_kwargs)
optimizes signature of student program - note that it may be zero-shot or already pre-optimized (demos already chosen - demos != [])
parameters:
student: program to optimize and left modified.
trainset: iterable of Examples
eval_kwargs: optional, dict
   Additional keywords to go into Evaluate for the metric.
Returns optimized version of student.
Source code in
dspy/teleprompt/copro_optimizer.py
```
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
[215](#__codelineno-0-215)
[216](#__codelineno-0-216)
[217](#__codelineno-0-217)
[218](#__codelineno-0-218)
[219](#__codelineno-0-219)
[220](#__codelineno-0-220)
[221](#__codelineno-0-221)
[222](#__codelineno-0-222)
[223](#__codelineno-0-223)
[224](#__codelineno-0-224)
[225](#__codelineno-0-225)
[226](#__codelineno-0-226)
[227](#__codelineno-0-227)
[228](#__codelineno-0-228)
[229](#__codelineno-0-229)
[230](#__codelineno-0-230)
[231](#__codelineno-0-231)
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
[312](#__codelineno-0-312)
[313](#__codelineno-0-313)
[314](#__codelineno-0-314)
[315](#__codelineno-0-315)
[316](#__codelineno-0-316)
[317](#__codelineno-0-317)
[318](#__codelineno-0-318)
[319](#__codelineno-0-319)
[320](#__codelineno-0-320)
[321](#__codelineno-0-321)
[322](#__codelineno-0-322)
[323](#__codelineno-0-323)
[324](#__codelineno-0-324)
[325](#__codelineno-0-325)
[326](#__codelineno-0-326)
[327](#__codelineno-0-327)
[328](#__codelineno-0-328)
[329](#__codelineno-0-329)
[330](#__codelineno-0-330)
[331](#__codelineno-0-331)
[332](#__codelineno-0-332)
[333](#__codelineno-0-333)
[334](#__codelineno-0-334)
[335](#__codelineno-0-335)
[336](#__codelineno-0-336)
[337](#__codelineno-0-337)
[338](#__codelineno-0-338)
[339](#__codelineno-0-339)
[340](#__codelineno-0-340)
[341](#__codelineno-0-341)
[342](#__codelineno-0-342)
[343](#__codelineno-0-343)
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
```
```
def compile(self, student, *, trainset, eval_kwargs):
    """
    optimizes `signature` of `student` program - note that it may be zero-shot or already pre-optimized (demos already chosen - `demos != []`)

    parameters:
    student: program to optimize and left modified.
    trainset: iterable of `Example`s
    eval_kwargs: optional, dict
       Additional keywords to go into `Evaluate` for the metric.

    Returns optimized version of `student`.
    """
    module = student.deepcopy()
    evaluate = Evaluate(devset=trainset, metric=self.metric, **eval_kwargs)
    total_calls = 0
    results_best = {
        id(p): {"depth": [], "max": [], "average": [], "min": [], "std": []} for p in module.predictors()
    }
    results_latest = {
        id(p): {"depth": [], "max": [], "average": [], "min": [], "std": []} for p in module.predictors()
    }

    if self.track_stats:
        import numpy as np

    candidates = {}
    evaluated_candidates = defaultdict(dict)

    # Seed the prompt optimizer zero shot with just the instruction, generate BREADTH new prompts
    for predictor in module.predictors():
        basic_instruction = None
        basic_prefix = None
        *_, last_key = self._get_signature(predictor).fields.keys()
        basic_instruction = self._get_signature(predictor).instructions
        basic_prefix = self._get_signature(predictor).fields[last_key].json_schema_extra["prefix"]
        if self.prompt_model:
            with dspy.settings.context(lm=self.prompt_model):
                instruct = dspy.Predict(
                    BasicGenerateInstruction,
                    n=self.breadth - 1,
                    temperature=self.init_temperature,
                )(basic_instruction=basic_instruction)
        else:
            instruct = dspy.Predict(
                BasicGenerateInstruction,
                n=self.breadth - 1,
                temperature=self.init_temperature,
            )(basic_instruction=basic_instruction)
        # Add in our initial prompt as a candidate as well
        instruct.completions.proposed_instruction.append(basic_instruction)
        instruct.completions.proposed_prefix_for_output_field.append(basic_prefix)
        candidates[id(predictor)] = instruct.completions
        evaluated_candidates[id(predictor)] = {}

    if self.prompt_model:
        logger.debug(f"{self.prompt_model.inspect_history(n=1)}")

    latest_candidates = candidates
    all_candidates = candidates

    module_clone = module.deepcopy()

    # For each iteration in depth...
    for d in range(
        self.depth,
    ):  # TODO: fix this so that we eval the new batch of predictors with the new best following predictors
        logger.info(f"Iteration Depth: {d+1}/{self.depth}.")

        latest_scores = []

        # Go through our module's predictors
        for p_i, (p_old, p_new) in enumerate(zip(module.predictors(), module_clone.predictors(), strict=False)):
            candidates_ = latest_candidates[id(p_old)]  # Use the most recently generated candidates for evaluation
            if len(module.predictors()) > 1:
                # Unless our program has multiple predictors, in which case we need to reevaluate all prompts with
                # the new prompt(s) for the other predictor(s).
                candidates_ = all_candidates[
                    id(p_old)
                ]

            # For each candidate
            for c_i, c in enumerate(candidates_):
                # Get the candidate instruction and prefix
                instruction, prefix = (
                    c.proposed_instruction.strip('"').strip(),
                    c.proposed_prefix_for_output_field.strip('"').strip(),
                )

                # Set this new module with our instruction / prefix
                *_, last_key = self._get_signature(p_new).fields.keys()
                updated_signature = (
                    self._get_signature(p_new)
                    .with_instructions(instruction)
                    .with_updated_fields(last_key, prefix=prefix)
                )
                self._set_signature(p_new, updated_signature)

                # Score the instruction / prefix
                for i, predictor in enumerate(module_clone.predictors()):
                    logger.debug(f"Predictor {i+1}")
                    self._print_signature(predictor)
                logger.info(
                    f"At Depth {d+1}/{self.depth}, Evaluating Prompt Candidate #{c_i+1}/{len(candidates_)} for "
                    f"Predictor {p_i+1} of {len(module.predictors())}.",
                )
                score = evaluate(module_clone, devset=trainset, **eval_kwargs).score
                if self.prompt_model:
                    logger.debug(f"prompt_model.inspect_history(n=1) {self.prompt_model.inspect_history(n=1)}")
                total_calls += 1

                replace_entry = True
                logger.debug(f"(instruction, prefix) {(instruction, prefix)}")
                if (instruction, prefix) in evaluated_candidates[id(p_old)]:
                    if evaluated_candidates[id(p_old)][(instruction, prefix)]["score"] >= score:
                        replace_entry = False

                if replace_entry:
                    # Add it to our evaluated candidates list
                    evaluated_candidates[id(p_old)][(instruction, prefix)] = {
                        "score": score,
                        "program": module_clone.deepcopy(),
                        "instruction": instruction,
                        "prefix": prefix,
                        "depth": d,
                    }

                if len(candidates_) - self.breadth <= c_i:
                    latest_scores.append(score)

            if self.track_stats:
                results_latest[id(p_old)]["depth"].append(d)
                results_latest[id(p_old)]["max"].append(max(latest_scores))
                results_latest[id(p_old)]["average"].append(sum(latest_scores) / len(latest_scores))
                results_latest[id(p_old)]["min"].append(min(latest_scores))
                results_latest[id(p_old)]["std"].append(np.std(latest_scores))

            # Now that we've evaluated the candidates, set this predictor to the best performing version
            # to ensure the next round of scores reflect the best possible version
            best_candidate = max(evaluated_candidates[id(p_old)].values(), key=lambda candidate: candidate["score"])
            *_, last_key = self._get_signature(p_old).fields.keys()
            updated_signature = (
                self._get_signature(p_new)
                .with_instructions(best_candidate["instruction"])
                .with_updated_fields(last_key, prefix=best_candidate["prefix"])
            )
            self._set_signature(p_new, updated_signature)

            logger.debug(
                f"Updating Predictor {id(p_old)} to:\ni: {best_candidate['instruction']}\n"
                f"p: {best_candidate['prefix']}",
            )
            logger.debug("Full predictor with update: ")
            for i, predictor in enumerate(module_clone.predictors()):
                logger.debug(f"Predictor {i}")
                self._print_signature(predictor)

        if d == self.depth - 1:
            break

        new_candidates = {}
        for p_base in module.predictors():
            # Build Few-Shot Example of Optimized Prompts
            attempts = []
            shortest_len = self.breadth
            shortest_len = min(len(evaluated_candidates[id(p_base)]), shortest_len)
            best_predictors = list(evaluated_candidates[id(p_base)].values())

            # best_predictors = evaluated_candidates[id(p_base)].values()[:]
            best_predictors.sort(key=lambda x: x["score"], reverse=True)

            if self.track_stats:
                scores = [x["score"] for x in best_predictors][:10]
                results_best[id(p_base)]["depth"].append(d)
                results_best[id(p_base)]["max"].append(max(scores))
                results_best[id(p_base)]["average"].append(sum(scores) / len(scores))
                results_best[id(p_base)]["min"].append(min(scores))
                results_best[id(p_base)]["std"].append(np.std(scores))

            for i in range(shortest_len - 1, -1, -1):
                # breakpoint()
                attempts.append(f'Instruction #{shortest_len-i}: {best_predictors[i]["instruction"]}')
                attempts.append(f'Prefix #{shortest_len-i}: {best_predictors[i]["prefix"]}')
                attempts.append(f'Resulting Score #{shortest_len-i}: {best_predictors[i]["score"]}')

            # Generate next batch of potential prompts to optimize, with previous attempts as input
            if self.prompt_model:
                with dspy.settings.context(lm=self.prompt_model):
                    instr = dspy.Predict(
                        GenerateInstructionGivenAttempts,
                        n=self.breadth,
                        temperature=self.init_temperature,
                    )(attempted_instructions=attempts)
            else:
                instr = dspy.Predict(
                    GenerateInstructionGivenAttempts,
                    n=self.breadth,
                    temperature=self.init_temperature,
                )(attempted_instructions=attempts)

            # Get candidates for each predictor
            new_candidates[id(p_base)] = instr.completions
            all_candidates[id(p_base)].proposed_instruction.extend(instr.completions.proposed_instruction)
            all_candidates[id(p_base)].proposed_prefix_for_output_field.extend(
                instr.completions.proposed_prefix_for_output_field,
            )

        latest_candidates = new_candidates

    candidates = []
    for predictor in module.predictors():
        candidates.extend(list(evaluated_candidates[id(predictor)].values()))

        if self.track_stats:
            best_predictors = list(evaluated_candidates[id(predictor)].values())
            best_predictors.sort(key=lambda x: x["score"], reverse=True)

            scores = [x["score"] for x in best_predictors][:10]
            results_best[id(predictor)]["depth"].append(d)
            results_best[id(predictor)]["max"].append(max(scores))
            results_best[id(predictor)]["average"].append(sum(scores) / len(scores))
            results_best[id(predictor)]["min"].append(min(scores))
            results_best[id(predictor)]["std"].append(np.std(scores))

    candidates.sort(key=lambda x: x["score"], reverse=True)

    candidates = self._drop_duplicates(candidates)

    best_program = candidates[0]["program"]
    best_program.candidate_programs = candidates
    best_program.total_calls = total_calls
    if self.track_stats:
        best_program.results_best = results_best
        best_program.results_latest = results_latest

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