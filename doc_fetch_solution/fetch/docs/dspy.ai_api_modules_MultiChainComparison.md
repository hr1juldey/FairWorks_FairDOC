# Multichaincomparison

**Source:** https://dspy.ai/api/modules/MultiChainComparison
**Fetched:** 2025-08-24 17:10:31
**Status:** Success

---

[

](https://github.com/stanfordnlp/dspy/blob/main/docs/docs/api/modules/MultiChainComparison.md)
# dspy.MultiChainComparison
## 
 dspy.MultiChainComparison(signature, M=3, temperature=0.7, **config)
Bases: Module
Source code in
dspy/predict/multi_chain_comparison.py
```
[ 8](#__codelineno-0-8)
[ 9](#__codelineno-0-9)
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
[32](#__codelineno-0-32)
[33](#__codelineno-0-33)
```
```
def __init__(self, signature, M=3, temperature=0.7, **config):  # noqa: N803
    super().__init__()

    self.M = M
    signature = ensure_signature(signature)

    *_, self.last_key = signature.output_fields.keys()

    for idx in range(M):
        signature = signature.append(
            f"reasoning_attempt_{idx+1}",
            InputField(
                prefix=f"Student Attempt #{idx+1}:",
                desc="${reasoning attempt}",
            ),
        )

    signature = signature.prepend(
        "rationale",
        OutputField(
            prefix="Accurate Reasoning: Thank you everyone. Let's now holistically",
            desc="${corrected reasoning}",
        ),
    )

    self.predict = Predict(signature, temperature=temperature, **config)

```
### Functions
#### 
 __call__(*args, **kwargs) -> Prediction
Source code in
dspy/primitives/module.py
```
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
```
```
@with_callbacks
def __call__(self, *args, **kwargs) -> Prediction:
    caller_modules = settings.caller_modules or []
    caller_modules = list(caller_modules)
    caller_modules.append(self)

    with settings.context(caller_modules=caller_modules):
        if settings.track_usage and thread_local_overrides.get().get("usage_tracker") is None:
            with track_usage() as usage_tracker:
                output = self.forward(*args, **kwargs)
            output.set_lm_usage(usage_tracker.get_total_tokens())
            return output

        return self.forward(*args, **kwargs)

```
#### 
 acall(*args, **kwargs) -> Prediction

async
Source code in
dspy/primitives/module.py
```
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
[92](#__codelineno-0-92)
[93](#__codelineno-0-93)
```
```
@with_callbacks
async def acall(self, *args, **kwargs) -> Prediction:
    caller_modules = settings.caller_modules or []
    caller_modules = list(caller_modules)
    caller_modules.append(self)

    with settings.context(caller_modules=caller_modules):
        if settings.track_usage and thread_local_overrides.get().get("usage_tracker") is None:
            with track_usage() as usage_tracker:
                output = await self.aforward(*args, **kwargs)
                output.set_lm_usage(usage_tracker.get_total_tokens())
                return output

        return await self.aforward(*args, **kwargs)

```
#### 
 batch(examples: list[Example], num_threads: int | None = None, max_errors: int | None = None, return_failed_examples: bool = False, provide_traceback: bool | None = None, disable_progress_bar: bool = False) -> list[Example] | tuple[list[Example], list[Example], list[Exception]]
Processes a list of dspy.Example instances in parallel using the Parallel module.
Parameters:
Name
Type
Description
Default
examples
list
[
[Example](../../primitives/Example/#dspy.Example)
]
List of dspy.Example instances to process.
required
num_threads
int
| None
Number of threads to use for parallel processing.
None
max_errors
int
| None
Maximum number of errors allowed before stopping execution.
If None, inherits from dspy.settings.max_errors.
None
return_failed_examples
bool
Whether to return failed examples and exceptions.
False
provide_traceback
bool
| None
Whether to include traceback information in error logs.
None
disable_progress_bar
bool
Whether to display the progress bar.
False
Returns:
Type
Description
list
[
[Example](../../primitives/Example/#dspy.Example)
] |
tuple
[
list
[
[Example](../../primitives/Example/#dspy.Example)
],
list
[
[Example](../../primitives/Example/#dspy.Example)
],
list
[
Exception
]]
List of results, and optionally failed examples and exceptions.
Source code in
dspy/primitives/module.py
```
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
```
```
def batch(
    self,
    examples: list[Example],
    num_threads: int | None = None,
    max_errors: int | None = None,
    return_failed_examples: bool = False,
    provide_traceback: bool | None = None,
    disable_progress_bar: bool = False,
) -> list[Example] | tuple[list[Example], list[Example], list[Exception]]:
    """
    Processes a list of dspy.Example instances in parallel using the Parallel module.

    Args:
        examples: List of dspy.Example instances to process.
        num_threads: Number of threads to use for parallel processing.
        max_errors: Maximum number of errors allowed before stopping execution.
            If ``None``, inherits from ``dspy.settings.max_errors``.
        return_failed_examples: Whether to return failed examples and exceptions.
        provide_traceback: Whether to include traceback information in error logs.
        disable_progress_bar: Whether to display the progress bar.

    Returns:
        List of results, and optionally failed examples and exceptions.
    """
    # Create a list of execution pairs (self, example)
    exec_pairs = [(self, example.inputs()) for example in examples]

    # Create an instance of Parallel
    parallel_executor = Parallel(
        num_threads=num_threads,
        max_errors=max_errors,
        return_failed_examples=return_failed_examples,
        provide_traceback=provide_traceback,
        disable_progress_bar=disable_progress_bar,
    )

    # Execute the forward method of Parallel
    if return_failed_examples:
        results, failed_examples, exceptions = parallel_executor.forward(exec_pairs)
        return results, failed_examples, exceptions
    else:
        results = parallel_executor.forward(exec_pairs)
        return results

```
#### 
 deepcopy()
Deep copy the module.
This is a tweak to the default python deepcopy that only deep copies self.parameters(), and for other
attributes, we just do the shallow copy.
Source code in
dspy/primitives/base_module.py
```
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
```
```
def deepcopy(self):
    """Deep copy the module.

    This is a tweak to the default python deepcopy that only deep copies `self.parameters()`, and for other
    attributes, we just do the shallow copy.
    """
    try:
        # If the instance itself is copyable, we can just deep copy it.
        # Otherwise we will have to create a new instance and copy over the attributes one by one.
        return copy.deepcopy(self)
    except Exception:
        pass

    # Create an empty instance.
    new_instance = self.__class__.__new__(self.__class__)
    # Set attribuetes of the copied instance.
    for attr, value in self.__dict__.items():
        if isinstance(value, BaseModule):
            setattr(new_instance, attr, value.deepcopy())
        else:
            try:
                # Try to deep copy the attribute
                setattr(new_instance, attr, copy.deepcopy(value))
            except Exception:
                logging.warning(
                    f"Failed to deep copy attribute '{attr}' of {self.__class__.__name__}, "
                    "falling back to shallow copy or reference copy."
                )
                try:
                    # Fallback to shallow copy if deep copy fails
                    setattr(new_instance, attr, copy.copy(value))
                except Exception:
                    # If even the shallow copy fails, we just copy over the reference.
                    setattr(new_instance, attr, value)

    return new_instance

```
#### 
 dump_state()
Source code in
dspy/primitives/base_module.py
```
[156](#__codelineno-0-156)
[157](#__codelineno-0-157)
```
```
def dump_state(self):
    return {name: param.dump_state() for name, param in self.named_parameters()}

```
#### 
 forward(completions, **kwargs)
Source code in
dspy/predict/multi_chain_comparison.py
```
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
```
```
def forward(self, completions, **kwargs):
    attempts = []

    for c in completions:
        rationale = c.get("rationale", c.get("reasoning")).strip().split("\n")[0].strip()
        answer = str(c[self.last_key]).strip().split("\n")[0].strip()
        attempts.append(
            f"«I'm trying to {rationale} I'm not sure but my prediction is {answer}»",
        )

    assert (
        len(attempts) == self.M
    ), f"The number of attempts ({len(attempts)}) doesn't match the expected number M ({self.M}). Please set the correct value for M when initializing MultiChainComparison."

    kwargs = {
        **{f"reasoning_attempt_{idx+1}": attempt for idx, attempt in enumerate(attempts)},
        **kwargs,
    }
    return self.predict(**kwargs)

```
#### 
 get_lm()
Source code in
dspy/primitives/module.py
```
[107](#__codelineno-0-107)
[108](#__codelineno-0-108)
[109](#__codelineno-0-109)
[110](#__codelineno-0-110)
[111](#__codelineno-0-111)
[112](#__codelineno-0-112)
[113](#__codelineno-0-113)
```
```
def get_lm(self):
    all_used_lms = [param.lm for _, param in self.named_predictors()]

    if len(set(all_used_lms)) == 1:
        return all_used_lms[0]

    raise ValueError("Multiple LMs are being used in the module. There's no unique LM to return.")

```
#### 
 inspect_history(n: int = 1)
Source code in
dspy/primitives/module.py
```
[129](#__codelineno-0-129)
[130](#__codelineno-0-130)
```
```
def inspect_history(self, n: int = 1):
    return pretty_print_history(self.history, n)

```
#### 
 load(path)
Load the saved module. You may also want to check out dspy.load, if you want to
load an entire program, not just the state for an existing program.
Parameters:
Name
Type
Description
Default
path
str
Path to the saved state file, which should be a .json or a .pkl file
required
Source code in
dspy/primitives/base_module.py
```
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
```
```
def load(self, path):
    """Load the saved module. You may also want to check out dspy.load, if you want to
    load an entire program, not just the state for an existing program.

    Args:
        path (str): Path to the saved state file, which should be a .json or a .pkl file
    """
    path = Path(path)

    if path.suffix == ".json":
        with open(path, encoding="utf-8") as f:
            state = ujson.loads(f.read())
    elif path.suffix == ".pkl":
        with open(path, "rb") as f:
            state = cloudpickle.load(f)
    else:
        raise ValueError(f"`path` must end with `.json` or `.pkl`, but received: {path}")

    dependency_versions = get_dependency_versions()
    saved_dependency_versions = state["metadata"]["dependency_versions"]
    for key, saved_version in saved_dependency_versions.items():
        if dependency_versions[key] != saved_version:
            logger.warning(
                f"There is a mismatch of {key} version between saved model and current environment. "
                f"You saved with `{key}=={saved_version}`, but now you have "
                f"`{key}=={dependency_versions[key]}`. This might cause errors or performance downgrade "
                "on the loaded model, please consider loading the model in the same environment as the "
                "saving environment."
            )
    self.load_state(state)

```
#### 
 load_state(state)
Source code in
dspy/primitives/base_module.py
```
[159](#__codelineno-0-159)
[160](#__codelineno-0-160)
[161](#__codelineno-0-161)
```
```
def load_state(self, state):
    for name, param in self.named_parameters():
        param.load_state(state[name])

```
#### 
 map_named_predictors(func)
Applies a function to all named predictors.
Source code in
dspy/primitives/module.py
```
[123](#__codelineno-0-123)
[124](#__codelineno-0-124)
[125](#__codelineno-0-125)
[126](#__codelineno-0-126)
[127](#__codelineno-0-127)
```
```
def map_named_predictors(self, func):
    """Applies a function to all named predictors."""
    for name, predictor in self.named_predictors():
        set_attribute_by_name(self, name, func(predictor))
    return self

```
#### 
 named_parameters()
Unlike PyTorch, handles (non-recursive) lists of parameters too.
Source code in
dspy/primitives/base_module.py
```
[23](#__codelineno-0-23)
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
[61](#__codelineno-0-61)
[62](#__codelineno-0-62)
[63](#__codelineno-0-63)
[64](#__codelineno-0-64)
[65](#__codelineno-0-65)
[66](#__codelineno-0-66)
[67](#__codelineno-0-67)
```
```
def named_parameters(self):
    """
    Unlike PyTorch, handles (non-recursive) lists of parameters too.
    """

    import dspy
    from dspy.predict.parameter import Parameter

    visited = set()
    named_parameters = []

    def add_parameter(param_name, param_value):
        if isinstance(param_value, Parameter):
            if id(param_value) not in visited:
                visited.add(id(param_value))
                named_parameters.append((param_name, param_value))

        elif isinstance(param_value, dspy.Module):
            # When a sub-module is pre-compiled, keep it frozen.
            if not getattr(param_value, "_compiled", False):
                for sub_name, param in param_value.named_parameters():
                    add_parameter(f"{param_name}.{sub_name}", param)

    if isinstance(self, Parameter):
        add_parameter("self", self)

    for name, value in self.__dict__.items():
        if isinstance(value, Parameter):
            add_parameter(name, value)

        elif isinstance(value, dspy.Module):
            # When a sub-module is pre-compiled, keep it frozen.
            if not getattr(value, "_compiled", False):
                for sub_name, param in value.named_parameters():
                    add_parameter(f"{name}.{sub_name}", param)

        elif isinstance(value, (list, tuple)):
            for idx, item in enumerate(value):
                add_parameter(f"{name}[{idx}]", item)

        elif isinstance(value, dict):
            for key, item in value.items():
                add_parameter(f"{name}['{key}']", item)

    return named_parameters

```
#### 
 named_predictors()
Source code in
dspy/primitives/module.py
```
[95](#__codelineno-0-95)
[96](#__codelineno-0-96)
[97](#__codelineno-0-97)
[98](#__codelineno-0-98)
```
```
def named_predictors(self):
    from dspy.predict.predict import Predict

    return [(name, param) for name, param in self.named_parameters() if isinstance(param, Predict)]

```
#### 
 named_sub_modules(type_=None, skip_compiled=False) -> Generator[tuple[str, BaseModule], None, None]
Find all sub-modules in the module, as well as their names.
Say self.children[4]['key'].sub_module is a sub-module. Then the name will be
children[4]['key'].sub_module. But if the sub-module is accessible at different
paths, only one of the paths will be returned.
Source code in
dspy/primitives/base_module.py
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
def named_sub_modules(self, type_=None, skip_compiled=False) -> Generator[tuple[str, "BaseModule"], None, None]:
    """Find all sub-modules in the module, as well as their names.

    Say `self.children[4]['key'].sub_module` is a sub-module. Then the name will be
    `children[4]['key'].sub_module`. But if the sub-module is accessible at different
    paths, only one of the paths will be returned.
    """
    if type_ is None:
        type_ = BaseModule

    queue = deque([("self", self)])
    seen = {id(self)}

    def add_to_queue(name, item):
        if id(item) not in seen:
            seen.add(id(item))
            queue.append((name, item))

    while queue:
        name, item = queue.popleft()

        if isinstance(item, type_):
            yield name, item

        if isinstance(item, BaseModule):
            if skip_compiled and getattr(item, "_compiled", False):
                continue
            for sub_name, sub_item in item.__dict__.items():
                add_to_queue(f"{name}.{sub_name}", sub_item)

        elif isinstance(item, (list, tuple)):
            for i, sub_item in enumerate(item):
                add_to_queue(f"{name}[{i}]", sub_item)

        elif isinstance(item, dict):
            for key, sub_item in item.items():
                add_to_queue(f"{name}[{key}]", sub_item)

```
#### 
 parameters()
Source code in
dspy/primitives/base_module.py
```
[107](#__codelineno-0-107)
[108](#__codelineno-0-108)
```
```
def parameters(self):
    return [param for _, param in self.named_parameters()]

```
#### 
 predictors()
Source code in
dspy/primitives/module.py
```
[100](#__codelineno-0-100)
[101](#__codelineno-0-101)
```
```
def predictors(self):
    return [param for _, param in self.named_predictors()]

```
#### 
 reset_copy()
Deep copy the module and reset all parameters.
Source code in
dspy/primitives/base_module.py
```
[147](#__codelineno-0-147)
[148](#__codelineno-0-148)
[149](#__codelineno-0-149)
[150](#__codelineno-0-150)
[151](#__codelineno-0-151)
[152](#__codelineno-0-152)
[153](#__codelineno-0-153)
[154](#__codelineno-0-154)
```
```
def reset_copy(self):
    """Deep copy the module and reset all parameters."""
    new_instance = self.deepcopy()

    for param in new_instance.parameters():
        param.reset()

    return new_instance

```
#### 
 save(path, save_program=False, modules_to_serialize=None)
Save the module.
Save the module to a directory or a file. There are two modes:
- save_program=False: Save only the state of the module to a json or pickle file, based on the value of
    the file extension.
- save_program=True: Save the whole module to a directory via cloudpickle, which contains both the state and
    architecture of the model.
If save_program=True and modules_to_serialize are provided, it will register those modules for serialization 
with cloudpickle's register_pickle_by_value. This causes cloudpickle to serialize the module by value rather 
than by reference, ensuring the module is fully preserved along with the saved program. This is useful 
when you have custom modules that need to be serialized alongside your program. If None, then no modules 
will be registered for serialization.
We also save the dependency versions, so that the loaded model can check if there is a version mismatch on
critical dependencies or DSPy version.
Parameters:
Name
Type
Description
Default
path
str
Path to the saved state file, which should be a .json or .pkl file when save_program=False,
and a directory when save_program=True.
required
save_program
bool
If True, save the whole module to a directory via cloudpickle, otherwise only save
the state.
False
modules_to_serialize
list
A list of modules to serialize with cloudpickle's register_pickle_by_value.
If None, then no modules will be registered for serialization.
None
Source code in
dspy/primitives/base_module.py
```
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
```
```
def save(self, path, save_program=False, modules_to_serialize=None):
    """Save the module.

    Save the module to a directory or a file. There are two modes:
    - `save_program=False`: Save only the state of the module to a json or pickle file, based on the value of
        the file extension.
    - `save_program=True`: Save the whole module to a directory via cloudpickle, which contains both the state and
        architecture of the model.

    If `save_program=True` and `modules_to_serialize` are provided, it will register those modules for serialization 
    with cloudpickle's `register_pickle_by_value`. This causes cloudpickle to serialize the module by value rather 
    than by reference, ensuring the module is fully preserved along with the saved program. This is useful 
    when you have custom modules that need to be serialized alongside your program. If None, then no modules 
    will be registered for serialization.

    We also save the dependency versions, so that the loaded model can check if there is a version mismatch on
    critical dependencies or DSPy version.

    Args:
        path (str): Path to the saved state file, which should be a .json or .pkl file when `save_program=False`,
            and a directory when `save_program=True`.
        save_program (bool): If True, save the whole module to a directory via cloudpickle, otherwise only save
            the state.
        modules_to_serialize (list): A list of modules to serialize with cloudpickle's `register_pickle_by_value`.
            If None, then no modules will be registered for serialization.

    """
    metadata = {}
    metadata["dependency_versions"] = get_dependency_versions()
    path = Path(path)

    if save_program:
        if path.suffix:
            raise ValueError(
                f"`path` must point to a directory without a suffix when `save_program=True`, but received: {path}"
            )
        if path.exists() and not path.is_dir():
            raise NotADirectoryError(f"The path '{path}' exists but is not a directory.")

        if not path.exists():
            # Create the directory (and any parent directories)
            path.mkdir(parents=True)

        try:
            modules_to_serialize = modules_to_serialize or []
            for module in modules_to_serialize:
                cloudpickle.register_pickle_by_value(module)

            with open(path / "program.pkl", "wb") as f:
                cloudpickle.dump(self, f)
        except Exception as e:
            raise RuntimeError(
                f"Saving failed with error: {e}. Please remove the non-picklable attributes from your DSPy program, "
                "or consider using state-only saving by setting `save_program=False`."
            )
        with open(path / "metadata.json", "w", encoding="utf-8") as f:
            ujson.dump(metadata, f, indent=2, ensure_ascii=False)

        return

    state = self.dump_state()
    state["metadata"] = metadata
    if path.suffix == ".json":
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write(ujson.dumps(state, indent=2 , ensure_ascii=False))
        except Exception as e:
            raise RuntimeError(
                f"Failed to save state to {path} with error: {e}. Your DSPy program may contain non "
                "json-serializable objects, please consider saving the state in .pkl by using `path` ending "
                "with `.pkl`, or saving the whole program by setting `save_program=True`."
            )
    elif path.suffix == ".pkl":
        with open(path, "wb") as f:
            cloudpickle.dump(state, f)
    else:
        raise ValueError(f"`path` must end with `.json` or `.pkl` when `save_program=False`, but received: {path}")

```
#### 
 set_lm(lm)
Source code in
dspy/primitives/module.py
```
[103](#__codelineno-0-103)
[104](#__codelineno-0-104)
[105](#__codelineno-0-105)
```
```
def set_lm(self, lm):
    for _, param in self.named_predictors():
        param.lm = lm

```
:::