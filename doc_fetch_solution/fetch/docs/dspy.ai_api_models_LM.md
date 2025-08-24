# Lm

**Source:** https://dspy.ai/api/models/LM
**Fetched:** 2025-08-24 17:10:36
**Status:** Success

---

[

](https://github.com/stanfordnlp/dspy/blob/main/docs/docs/api/models/LM.md)
# dspy.LM
## 
 dspy.LM(model: str, model_type: Literal['chat', 'text'] = 'chat', temperature: float = 0.0, max_tokens: int = 4000, cache: bool = True, cache_in_memory: bool = True, callbacks: list[BaseCallback] | None = None, num_retries: int = 3, provider: Provider | None = None, finetuning_model: str | None = None, launch_kwargs: dict[str, Any] | None = None, train_kwargs: dict[str, Any] | None = None, **kwargs)
Bases: BaseLM
A language model supporting chat or text completion requests for use with DSPy modules.
Create a new language model instance for use with DSPy modules and programs.
Parameters:
Name
Type
Description
Default
model
str
The model to use. This should be a string of the form "llm_provider/llm_name"
   supported by LiteLLM. For example, "openai/gpt-4o".
required
model_type
Literal
['chat', 'text']
The type of the model, either "chat" or "text".
'chat'
temperature
float
The sampling temperature to use when generating responses.
0.0
max_tokens
int
The maximum number of tokens to generate per response.
4000
cache
bool
Whether to cache the model responses for reuse to improve performance
   and reduce costs.
True
cache_in_memory
deprecated
To enable additional caching with LRU in memory.
True
callbacks
list
[
BaseCallback
] | None
A list of callback functions to run before and after each request.
None
num_retries
int
The number of times to retry a request if it fails transiently due to
         network error, rate limiting, etc. Requests are retried with exponential
         backoff.
3
provider
Provider
| None
The provider to use. If not specified, the provider will be inferred from the model.
None
finetuning_model
str
| None
The model to finetune. In some providers, the models available for finetuning is different
from the models available for inference.
None
Source code in
dspy/clients/lm.py
```
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
[92](#__codelineno-0-92)
```
```
def __init__(
    self,
    model: str,
    model_type: Literal["chat", "text"] = "chat",
    temperature: float = 0.0,
    max_tokens: int = 4000,
    cache: bool = True,
    cache_in_memory: bool = True,
    callbacks: list[BaseCallback] | None = None,
    num_retries: int = 3,
    provider: Provider | None = None,
    finetuning_model: str | None = None,
    launch_kwargs: dict[str, Any] | None = None,
    train_kwargs: dict[str, Any] | None = None,
    **kwargs,
):
    """
    Create a new language model instance for use with DSPy modules and programs.

    Args:
        model: The model to use. This should be a string of the form ``"llm_provider/llm_name"``
               supported by LiteLLM. For example, ``"openai/gpt-4o"``.
        model_type: The type of the model, either ``"chat"`` or ``"text"``.
        temperature: The sampling temperature to use when generating responses.
        max_tokens: The maximum number of tokens to generate per response.
        cache: Whether to cache the model responses for reuse to improve performance
               and reduce costs.
        cache_in_memory (deprecated): To enable additional caching with LRU in memory.
        callbacks: A list of callback functions to run before and after each request.
        num_retries: The number of times to retry a request if it fails transiently due to
                     network error, rate limiting, etc. Requests are retried with exponential
                     backoff.
        provider: The provider to use. If not specified, the provider will be inferred from the model.
        finetuning_model: The model to finetune. In some providers, the models available for finetuning is different
            from the models available for inference.
    """
    # Remember to update LM.copy() if you modify the constructor!
    self.model = model
    self.model_type = model_type
    self.cache = cache
    self.cache_in_memory = cache_in_memory
    self.provider = provider or self.infer_provider()
    self.callbacks = callbacks or []
    self.history = []
    self.num_retries = num_retries
    self.finetuning_model = finetuning_model
    self.launch_kwargs = launch_kwargs or {}
    self.train_kwargs = train_kwargs or {}

    # Handle model-specific configuration for different model families
    model_family = model.split("/")[-1].lower() if "/" in model else model.lower()

    # Match pattern: o[1,3,4] at the start, optionally followed by -mini and anything else
    model_pattern = re.match(r"^(?:o([1345])|gpt-(5))(?:-mini)?", model_family)

    if model_pattern:
        if max_tokens < 20000 or temperature != 1.0:
            raise ValueError(
                "OpenAI's reasoning models require passing temperature=1.0 and max_tokens >= 20000 to "
                "`dspy.LM(...)`, e.g., dspy.LM('openai/gpt-5', temperature=1.0, max_tokens=20000)"
            )
        self.kwargs = dict(temperature=temperature, max_completion_tokens=max_tokens, **kwargs)
    else:
        self.kwargs = dict(temperature=temperature, max_tokens=max_tokens, **kwargs)

```
### Functions
#### 
 __call__(prompt=None, messages=None, **kwargs)
Source code in
dspy/clients/base_lm.py
```
[92](#__codelineno-0-92)
[93](#__codelineno-0-93)
[94](#__codelineno-0-94)
[95](#__codelineno-0-95)
[96](#__codelineno-0-96)
[97](#__codelineno-0-97)
```
```
@with_callbacks
def __call__(self, prompt=None, messages=None, **kwargs):
    response = self.forward(prompt=prompt, messages=messages, **kwargs)
    outputs = self._process_lm_response(response, prompt, messages, **kwargs)

    return outputs

```
#### 
 acall(prompt=None, messages=None, **kwargs)

async
Source code in
dspy/clients/base_lm.py
```
[ 99](#__codelineno-0-99)
[100](#__codelineno-0-100)
[101](#__codelineno-0-101)
[102](#__codelineno-0-102)
[103](#__codelineno-0-103)
```
```
@with_callbacks
async def acall(self, prompt=None, messages=None, **kwargs):
    response = await self.aforward(prompt=prompt, messages=messages, **kwargs)
    outputs = self._process_lm_response(response, prompt, messages, **kwargs)
    return outputs

```
#### 
 aforward(prompt=None, messages=None, **kwargs)

async
Source code in
dspy/clients/lm.py
```
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
```
```
async def aforward(self, prompt=None, messages=None, **kwargs):
    # Build the request.
    cache = kwargs.pop("cache", self.cache)
    enable_memory_cache = kwargs.pop("cache_in_memory", self.cache_in_memory)

    messages = messages or [{"role": "user", "content": prompt}]
    kwargs = {**self.kwargs, **kwargs}

    completion = alitellm_completion if self.model_type == "chat" else alitellm_text_completion
    completion, litellm_cache_args = self._get_cached_completion_fn(completion, cache, enable_memory_cache)

    results = await completion(
        request=dict(model=self.model, messages=messages, **kwargs),
        num_retries=self.num_retries,
        cache=litellm_cache_args,
    )

    if any(c.finish_reason == "length" for c in results["choices"]):
        logger.warning(
            f"LM response was truncated due to exceeding max_tokens={self.kwargs['max_tokens']}. "
            "You can inspect the latest LM interactions with `dspy.inspect_history()`. "
            "To avoid truncation, consider passing a larger max_tokens when setting up dspy.LM. "
            f"You may also consider increasing the temperature (currently {self.kwargs['temperature']}) "
            " if the reason for truncation is repetition."
        )

    if not getattr(results, "cache_hit", False) and dspy.settings.usage_tracker and hasattr(results, "usage"):
        settings.usage_tracker.add_usage(self.model, dict(results.usage))
    return results

```
#### 
 copy(**kwargs)
Returns a copy of the language model with possibly updated parameters.
Source code in
dspy/clients/base_lm.py
```
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
```
```
def copy(self, **kwargs):
    """Returns a copy of the language model with possibly updated parameters."""

    import copy

    new_instance = copy.deepcopy(self)
    new_instance.history = []

    for key, value in kwargs.items():
        if hasattr(self, key):
            setattr(new_instance, key, value)
        if (key in self.kwargs) or (not hasattr(self, key)):
            new_instance.kwargs[key] = value

    return new_instance

```
#### 
 dump_state()
Source code in
dspy/clients/lm.py
```
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
```
```
def dump_state(self):
    state_keys = [
        "model",
        "model_type",
        "cache",
        "cache_in_memory",
        "num_retries",
        "finetuning_model",
        "launch_kwargs",
        "train_kwargs",
    ]
    return {key: getattr(self, key) for key in state_keys} | self.kwargs

```
#### 
 finetune(train_data: list[dict[str, Any]], train_data_format: TrainDataFormat | None, train_kwargs: dict[str, Any] | None = None) -> TrainingJob
Source code in
dspy/clients/lm.py
```
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
```
```
def finetune(
    self,
    train_data: list[dict[str, Any]],
    train_data_format: TrainDataFormat | None,
    train_kwargs: dict[str, Any] | None = None,
) -> TrainingJob:
    from dspy import settings as settings

    if not self.provider.finetunable:
        raise ValueError(
            f"Provider {self.provider} does not support fine-tuning, please specify your provider by explicitly "
            "setting `provider` when creating the `dspy.LM` instance. For example, "
            "`dspy.LM('openai/gpt-4.1-mini-2025-04-14', provider=dspy.OpenAIProvider())`."
        )

    def thread_function_wrapper():
        return self._run_finetune_job(job)

    thread = threading.Thread(target=thread_function_wrapper)
    train_kwargs = train_kwargs or self.train_kwargs
    model_to_finetune = self.finetuning_model or self.model
    job = self.provider.TrainingJob(
        thread=thread,
        model=model_to_finetune,
        train_data=train_data,
        train_data_format=train_data_format,
        train_kwargs=train_kwargs,
    )
    thread.start()

    return job

```
#### 
 forward(prompt=None, messages=None, **kwargs)
Source code in
dspy/clients/lm.py
```
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
def forward(self, prompt=None, messages=None, **kwargs):
    # Build the request.
    cache = kwargs.pop("cache", self.cache)
    enable_memory_cache = kwargs.pop("cache_in_memory", self.cache_in_memory)

    messages = messages or [{"role": "user", "content": prompt}]
    kwargs = {**self.kwargs, **kwargs}

    completion = litellm_completion if self.model_type == "chat" else litellm_text_completion
    completion, litellm_cache_args = self._get_cached_completion_fn(completion, cache, enable_memory_cache)

    results = completion(
        request=dict(model=self.model, messages=messages, **kwargs),
        num_retries=self.num_retries,
        cache=litellm_cache_args,
    )

    if any(c.finish_reason == "length" for c in results["choices"]):
        logger.warning(
            f"LM response was truncated due to exceeding max_tokens={self.kwargs['max_tokens']}. "
            "You can inspect the latest LM interactions with `dspy.inspect_history()`. "
            "To avoid truncation, consider passing a larger max_tokens when setting up dspy.LM. "
            f"You may also consider increasing the temperature (currently {self.kwargs['temperature']}) "
            " if the reason for truncation is repetition."
        )

    if not getattr(results, "cache_hit", False) and dspy.settings.usage_tracker and hasattr(results, "usage"):
        settings.usage_tracker.add_usage(self.model, dict(results.usage))
    return results

```
#### 
 infer_provider() -> Provider
Source code in
dspy/clients/lm.py
```
[243](#__codelineno-0-243)
[244](#__codelineno-0-244)
[245](#__codelineno-0-245)
[246](#__codelineno-0-246)
```
```
def infer_provider(self) -> Provider:
    if OpenAIProvider.is_provider_model(self.model):
        return OpenAIProvider()
    return Provider()

```
#### 
 inspect_history(n: int = 1)
Source code in
dspy/clients/base_lm.py
```
[137](#__codelineno-0-137)
[138](#__codelineno-0-138)
```
```
def inspect_history(self, n: int = 1):
    return pretty_print_history(self.history, n)

```
#### 
 kill(launch_kwargs: dict[str, Any] | None = None)
Source code in
dspy/clients/lm.py
```
[180](#__codelineno-0-180)
[181](#__codelineno-0-181)
```
```
def kill(self, launch_kwargs: dict[str, Any] | None = None):
    self.provider.kill(self, launch_kwargs)

```
#### 
 launch(launch_kwargs: dict[str, Any] | None = None)
Source code in
dspy/clients/lm.py
```
[177](#__codelineno-0-177)
[178](#__codelineno-0-178)
```
```
def launch(self, launch_kwargs: dict[str, Any] | None = None):
    self.provider.launch(self, launch_kwargs)

```
#### 
 reinforce(train_kwargs) -> ReinforceJob
Source code in
dspy/clients/lm.py
```
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
```
```
def reinforce(self, train_kwargs) -> ReinforceJob:
    # TODO(GRPO Team): Should we return an initialized job here?
    from dspy import settings as settings

    err = f"Provider {self.provider} does not implement the reinforcement learning interface."
    assert self.provider.reinforceable, err

    job = self.provider.ReinforceJob(lm=self, train_kwargs=train_kwargs)
    job.initialize()
    return job

```
#### 
 update_history(entry)
Source code in
dspy/clients/base_lm.py
```
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
```
```
def update_history(self, entry):
    if settings.disable_history:
        return

    # Global LM history
    if len(GLOBAL_HISTORY) >= MAX_HISTORY_SIZE:
        GLOBAL_HISTORY.pop(0)

    GLOBAL_HISTORY.append(entry)

    if settings.max_history_size == 0:
        return

    # dspy.LM.history
    if len(self.history) >= settings.max_history_size:
        self.history.pop(0)

    self.history.append(entry)

    # Per-module history
    caller_modules = settings.caller_modules or []
    for module in caller_modules:
        if len(module.history) >= settings.max_history_size:
            module.history.pop(0)
        module.history.append(entry)

```
:::