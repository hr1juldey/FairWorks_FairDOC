# Twostepadapter

**Source:** https://dspy.ai/api/adapters/TwoStepAdapter
**Fetched:** 2025-08-24 17:10:32
**Status:** Success

---

[

](https://github.com/stanfordnlp/dspy/blob/main/docs/docs/api/adapters/TwoStepAdapter.md)
# dspy.TwoStepAdapter
## 
 dspy.TwoStepAdapter(extraction_model: LM, **kwargs)
Bases: Adapter
A two-stage adapter that
1. Uses a simpler, more natural prompt for the main LM
2. Uses a smaller LM with chat adapter to extract structured data from the response of main LM
This adapter uses a common call logic defined in base Adapter class.
This class is particularly useful when interacting with reasoning models as the main LM since reasoning models
are known to struggle with structured outputs.
Example:
import dspy
lm = dspy.LM(model="openai/o3-mini", max_tokens=10000, temperature = 1.0)
adapter = dspy.TwoStepAdapter(dspy.LM("openai/gpt-4o-mini"))
dspy.configure(lm=lm, adapter=adapter)
program = dspy.ChainOfThought("question->answer")
result = program("What is the capital of France?")
print(result)
Source code in
dspy/adapters/two_step_adapter.py
```
[42](#__codelineno-0-42)
[43](#__codelineno-0-43)
[44](#__codelineno-0-44)
[45](#__codelineno-0-45)
[46](#__codelineno-0-46)
```
```
def __init__(self, extraction_model: LM, **kwargs):
    super().__init__(**kwargs)
    if not isinstance(extraction_model, LM):
        raise ValueError("extraction_model must be an instance of LM")
    self.extraction_model = extraction_model

```
### Functions
#### 
 __call__(lm: LM, lm_kwargs: dict[str, Any], signature: type[Signature], demos: list[dict[str, Any]], inputs: dict[str, Any]) -> list[dict[str, Any]]
Source code in
dspy/adapters/base.py
```
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
```
```
def __call__(
    self,
    lm: "LM",
    lm_kwargs: dict[str, Any],
    signature: type[Signature],
    demos: list[dict[str, Any]],
    inputs: dict[str, Any],
) -> list[dict[str, Any]]:
    processed_signature = self._call_preprocess(lm, lm_kwargs, signature, inputs)
    inputs = self.format(processed_signature, demos, inputs)

    outputs = lm(messages=inputs, **lm_kwargs)
    return self._call_postprocess(processed_signature, signature, outputs)

```
#### 
 acall(lm: LM, lm_kwargs: dict[str, Any], signature: type[Signature], demos: list[dict[str, Any]], inputs: dict[str, Any]) -> list[dict[str, Any]]

async
Source code in
dspy/adapters/two_step_adapter.py
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
```
```
async def acall(
    self,
    lm: "LM",
    lm_kwargs: dict[str, Any],
    signature: type[Signature],
    demos: list[dict[str, Any]],
    inputs: dict[str, Any],
) -> list[dict[str, Any]]:
    inputs = self.format(signature, demos, inputs)

    outputs = await lm.acall(messages=inputs, **lm_kwargs)
    # The signature is supposed to be "text -> {original output fields}"
    extractor_signature = self._create_extractor_signature(signature)

    values = []

    tool_call_output_field_name = self._get_tool_call_output_field_name(signature)
    for output in outputs:
        output_logprobs = None
        tool_calls = None
        text = output

        if isinstance(output, dict):
            text = output["text"]
            output_logprobs = output.get("logprobs")
            tool_calls = output.get("tool_calls")

        try:
            # Call the smaller LM to extract structured data from the raw completion text with ChatAdapter
            value = await ChatAdapter().acall(
                lm=self.extraction_model,
                lm_kwargs={},
                signature=extractor_signature,
                demos=[],
                inputs={"text": text},
            )
            value = value[0]

        except Exception as e:
            raise ValueError(f"Failed to parse response from the original completion: {output}") from e

        if tool_calls and tool_call_output_field_name:
            tool_calls = [
                {
                    "name": v["function"]["name"],
                    "args": json_repair.loads(v["function"]["arguments"]),
                }
                for v in tool_calls
            ]
            value[tool_call_output_field_name] = ToolCalls.from_dict_list(tool_calls)

        if output_logprobs is not None:
            value["logprobs"] = output_logprobs

        values.append(value)
    return values

```
#### 
 format(signature: type[Signature], demos: list[dict[str, Any]], inputs: dict[str, Any]) -> list[dict[str, Any]]
Format a prompt for the first stage with the main LM.
This no specific structure is required for the main LM, we customize the format method
instead of format_field_description or format_field_structure.
Parameters:
Name
Type
Description
Default
signature
type
[
[Signature](../../signatures/Signature/#dspy.Signature)
]
The signature of the original task
required
demos
list
[
dict
[
str
,
Any
]]
A list of demo examples
required
inputs
dict
[
str
,
Any
]
The current input
required
Returns:
Type
Description
list
[
dict
[
str
,
Any
]]
A list of messages to be passed to the main LM.
Source code in
dspy/adapters/two_step_adapter.py
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
```
```
def format(
    self, signature: type[Signature], demos: list[dict[str, Any]], inputs: dict[str, Any]
) -> list[dict[str, Any]]:
    """
    Format a prompt for the first stage with the main LM.
    This no specific structure is required for the main LM, we customize the format method
    instead of format_field_description or format_field_structure.

    Args:
        signature: The signature of the original task
        demos: A list of demo examples
        inputs: The current input

    Returns:
        A list of messages to be passed to the main LM.
    """
    messages = []

    # Create a task description for the main LM
    task_description = self.format_task_description(signature)
    messages.append({"role": "system", "content": task_description})

    messages.extend(self.format_demos(signature, demos))

    # Format the current input
    messages.append({"role": "user", "content": self.format_user_message_content(signature, inputs)})

    return messages

```
#### 
 format_assistant_message_content(signature: type[Signature], outputs: dict[str, Any], missing_field_message: str | None = None) -> str
Source code in
dspy/adapters/two_step_adapter.py
```
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
```
```
def format_assistant_message_content(
    self,
    signature: type[Signature],
    outputs: dict[str, Any],
    missing_field_message: str | None = None,
) -> str:
    parts = []

    for name in signature.output_fields.keys():
        if name in outputs:
            parts.append(f"{name}: {outputs.get(name, missing_field_message)}")

    return "\n\n".join(parts).strip()

```
#### 
 format_conversation_history(signature: type[Signature], history_field_name: str, inputs: dict[str, Any]) -> list[dict[str, Any]]
Format the conversation history.
This method formats the conversation history and the current input as multiturn messages.
Parameters:
Name
Type
Description
Default
signature
type
[
[Signature](../../signatures/Signature/#dspy.Signature)
]
The DSPy signature for which to format the conversation history.
required
history_field_name
str
The name of the history field in the signature.
required
inputs
dict
[
str
,
Any
]
The input arguments to the DSPy module.
required
Returns:
Type
Description
list
[
dict
[
str
,
Any
]]
A list of multiturn messages.
Source code in
dspy/adapters/base.py
```
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
```
```
def format_conversation_history(
    self,
    signature: type[Signature],
    history_field_name: str,
    inputs: dict[str, Any],
) -> list[dict[str, Any]]:
    """Format the conversation history.

    This method formats the conversation history and the current input as multiturn messages.

    Args:
        signature: The DSPy signature for which to format the conversation history.
        history_field_name: The name of the history field in the signature.
        inputs: The input arguments to the DSPy module.

    Returns:
        A list of multiturn messages.
    """
    conversation_history = inputs[history_field_name].messages if history_field_name in inputs else None

    if conversation_history is None:
        return []

    messages = []
    for message in conversation_history:
        messages.append(
            {
                "role": "user",
                "content": self.format_user_message_content(signature, message),
            }
        )
        messages.append(
            {
                "role": "assistant",
                "content": self.format_assistant_message_content(signature, message),
            }
        )

    # Remove the history field from the inputs
    del inputs[history_field_name]

    return messages

```
#### 
 format_demos(signature: type[Signature], demos: list[dict[str, Any]]) -> list[dict[str, Any]]
Format the few-shot examples.
This method formats the few-shot examples as multiturn messages.
Parameters:
Name
Type
Description
Default
signature
type
[
[Signature](../../signatures/Signature/#dspy.Signature)
]
The DSPy signature for which to format the few-shot examples.
required
demos
list
[
dict
[
str
,
Any
]]
A list of few-shot examples, each element is a dictionary with keys of the input and output fields of
the signature.
required
Returns:
Type
Description
list
[
dict
[
str
,
Any
]]
A list of multiturn messages.
Source code in
dspy/adapters/base.py
```
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
```
```
def format_demos(self, signature: type[Signature], demos: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Format the few-shot examples.

    This method formats the few-shot examples as multiturn messages.

    Args:
        signature: The DSPy signature for which to format the few-shot examples.
        demos: A list of few-shot examples, each element is a dictionary with keys of the input and output fields of
            the signature.

    Returns:
        A list of multiturn messages.
    """
    complete_demos = []
    incomplete_demos = []

    for demo in demos:
        # Check if all fields are present and not None
        is_complete = all(k in demo and demo[k] is not None for k in signature.fields)

        # Check if demo has at least one input and one output field
        has_input = any(k in demo for k in signature.input_fields)
        has_output = any(k in demo for k in signature.output_fields)

        if is_complete:
            complete_demos.append(demo)
        elif has_input and has_output:
            # We only keep incomplete demos that have at least one input and one output field
            incomplete_demos.append(demo)

    messages = []

    incomplete_demo_prefix = "This is an example of the task, though some input or output fields are not supplied."
    for demo in incomplete_demos:
        messages.append(
            {
                "role": "user",
                "content": self.format_user_message_content(signature, demo, prefix=incomplete_demo_prefix),
            }
        )
        messages.append(
            {
                "role": "assistant",
                "content": self.format_assistant_message_content(
                    signature, demo, missing_field_message="Not supplied for this particular example. "
                ),
            }
        )

    for demo in complete_demos:
        messages.append({"role": "user", "content": self.format_user_message_content(signature, demo)})
        messages.append(
            {
                "role": "assistant",
                "content": self.format_assistant_message_content(
                    signature, demo, missing_field_message="Not supplied for this conversation history message. "
                ),
            }
        )

    return messages

```
#### 
 format_field_description(signature: type[Signature]) -> str
Format the field description for the system message.
This method formats the field description for the system message. It should return a string that contains
the field description for the input fields and the output fields.
Parameters:
Name
Type
Description
Default
signature
type
[
[Signature](../../signatures/Signature/#dspy.Signature)
]
The DSPy signature for which to format the field description.
required
Returns:
Type
Description
str
A string that contains the field description for the input fields and the output fields.
Source code in
dspy/adapters/base.py
```
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
```
```
def format_field_description(self, signature: type[Signature]) -> str:
    """Format the field description for the system message.

    This method formats the field description for the system message. It should return a string that contains
    the field description for the input fields and the output fields.

    Args:
        signature: The DSPy signature for which to format the field description.

    Returns:
        A string that contains the field description for the input fields and the output fields.
    """
    raise NotImplementedError

```
#### 
 format_field_structure(signature: type[Signature]) -> str
Format the field structure for the system message.
This method formats the field structure for the system message. It should return a string that dictates the
format the input fields should be provided to the LM, and the format the output fields will be in the response.
Refer to the ChatAdapter and JsonAdapter for an example.
Parameters:
Name
Type
Description
Default
signature
type
[
[Signature](../../signatures/Signature/#dspy.Signature)
]
The DSPy signature for which to format the field structure.
required
Source code in
dspy/adapters/base.py
```
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
```
```
def format_field_structure(self, signature: type[Signature]) -> str:
    """Format the field structure for the system message.

    This method formats the field structure for the system message. It should return a string that dictates the
    format the input fields should be provided to the LM, and the format the output fields will be in the response.
    Refer to the ChatAdapter and JsonAdapter for an example.

    Args:
        signature: The DSPy signature for which to format the field structure.
    """
    raise NotImplementedError

```
#### 
 format_task_description(signature: Signature) -> str
Create a description of the task based on the signature
Source code in
dspy/adapters/two_step_adapter.py
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
```
```
def format_task_description(self, signature: Signature) -> str:
    """Create a description of the task based on the signature"""
    parts = []

    parts.append("You are a helpful assistant that can solve tasks based on user input.")
    parts.append("As input, you will be provided with:\n" + get_field_description_string(signature.input_fields))
    parts.append("Your outputs must contain:\n" + get_field_description_string(signature.output_fields))
    parts.append("You should lay out your outputs in detail so that your answer can be understood by another agent")

    if signature.instructions:
        parts.append(f"Specific instructions: {signature.instructions}")

    return "\n".join(parts)

```
#### 
 format_user_message_content(signature: type[Signature], inputs: dict[str, Any], prefix: str = '', suffix: str = '') -> str
Source code in
dspy/adapters/two_step_adapter.py
```
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
```
```
def format_user_message_content(
    self,
    signature: type[Signature],
    inputs: dict[str, Any],
    prefix: str = "",
    suffix: str = "",
) -> str:
    parts = [prefix]

    for name in signature.input_fields.keys():
        if name in inputs:
            parts.append(f"{name}: {inputs.get(name, '')}")

    parts.append(suffix)
    return "\n\n".join(parts).strip()

```
#### 
 parse(signature: Signature, completion: str) -> dict[str, Any]
Use a smaller LM (extraction_model) with chat adapter to extract structured data
from the raw completion text of the main LM.
Parameters:
Name
Type
Description
Default
signature
[Signature](../../signatures/Signature/#dspy.Signature)
The signature of the original task
required
completion
str
The completion from the main LM
required
Returns:
Type
Description
dict
[
str
,
Any
]
A dictionary containing the extracted structured data.
Source code in
dspy/adapters/two_step_adapter.py
```
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
```
```
def parse(self, signature: Signature, completion: str) -> dict[str, Any]:
    """
    Use a smaller LM (extraction_model) with chat adapter to extract structured data
    from the raw completion text of the main LM.

    Args:
        signature: The signature of the original task
        completion: The completion from the main LM

    Returns:
        A dictionary containing the extracted structured data.
    """
    # The signature is supposed to be "text -> {original output fields}"
    extractor_signature = self._create_extractor_signature(signature)

    try:
        # Call the smaller LM to extract structured data from the raw completion text with ChatAdapter
        parsed_result = ChatAdapter()(
            lm=self.extraction_model,
            lm_kwargs={},
            signature=extractor_signature,
            demos=[],
            inputs={"text": completion},
        )
        return parsed_result[0]

    except Exception as e:
        raise ValueError(f"Failed to parse response from the original completion: {completion}") from e

```
:::