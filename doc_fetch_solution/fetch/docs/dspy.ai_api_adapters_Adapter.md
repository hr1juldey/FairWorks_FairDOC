# Adapter

**Source:** https://dspy.ai/api/adapters/Adapter
**Fetched:** 2025-08-24 17:10:33
**Status:** Success

---

[

](https://github.com/stanfordnlp/dspy/blob/main/docs/docs/api/adapters/Adapter.md)
# dspy.Adapter
## 
 dspy.Adapter(callbacks: list[BaseCallback] | None = None, use_native_function_calling: bool = False)
Source code in
dspy/adapters/base.py
```
[20](#__codelineno-0-20)
[21](#__codelineno-0-21)
[22](#__codelineno-0-22)
```
```
def __init__(self, callbacks: list[BaseCallback] | None = None, use_native_function_calling: bool = False):
    self.callbacks = callbacks or []
    self.use_native_function_calling = use_native_function_calling

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
dspy/adapters/base.py
```
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
    processed_signature = self._call_preprocess(lm, lm_kwargs, signature, inputs)
    inputs = self.format(processed_signature, demos, inputs)

    outputs = await lm.acall(messages=inputs, **lm_kwargs)
    return self._call_postprocess(processed_signature, signature, outputs)

```
#### 
 format(signature: type[Signature], demos: list[dict[str, Any]], inputs: dict[str, Any]) -> list[dict[str, Any]]
Format the input messages for the LM call.
This method converts the DSPy structured input along with few-shot examples and conversation history into
multiturn messages as expected by the LM. For custom adapters, this method can be overridden to customize
the formatting of the input messages.
In general we recommend the messages to have the following structure:
[
    {"role": "system", "content": system_message},
    # Begin few-shot examples
    {"role": "user", "content": few_shot_example_1_input},
    {"role": "assistant", "content": few_shot_example_1_output},
    {"role": "user", "content": few_shot_example_2_input},
    {"role": "assistant", "content": few_shot_example_2_output},
    ...
    # End few-shot examples
    # Begin conversation history
    {"role": "user", "content": conversation_history_1_input},
    {"role": "assistant", "content": conversation_history_1_output},
    {"role": "user", "content": conversation_history_2_input},
    {"role": "assistant", "content": conversation_history_2_output},
    ...
    # End conversation history
    {"role": "user", "content": current_input},
]

And system message should contain the field description, field structure, and task description.
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
The DSPy signature for which to format the input messages.
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
A list of few-shot examples.
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
A list of multiturn messages as expected by the LM.
Source code in
dspy/adapters/base.py
```
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
```
```
def format(
    self,
    signature: type[Signature],
    demos: list[dict[str, Any]],
    inputs: dict[str, Any],
) -> list[dict[str, Any]]:
    """Format the input messages for the LM call.

    This method converts the DSPy structured input along with few-shot examples and conversation history into
    multiturn messages as expected by the LM. For custom adapters, this method can be overridden to customize
    the formatting of the input messages.

    In general we recommend the messages to have the following structure:
    ```
    [
        {"role": "system", "content": system_message},
        # Begin few-shot examples
        {"role": "user", "content": few_shot_example_1_input},
        {"role": "assistant", "content": few_shot_example_1_output},
        {"role": "user", "content": few_shot_example_2_input},
        {"role": "assistant", "content": few_shot_example_2_output},
        ...
        # End few-shot examples
        # Begin conversation history
        {"role": "user", "content": conversation_history_1_input},
        {"role": "assistant", "content": conversation_history_1_output},
        {"role": "user", "content": conversation_history_2_input},
        {"role": "assistant", "content": conversation_history_2_output},
        ...
        # End conversation history
        {"role": "user", "content": current_input},
    ]

    And system message should contain the field description, field structure, and task description.
    ```

    Args:
        signature: The DSPy signature for which to format the input messages.
        demos: A list of few-shot examples.
        inputs: The input arguments to the DSPy module.

    Returns:
        A list of multiturn messages as expected by the LM.
    """
    inputs_copy = dict(inputs)

    # If the signature and inputs have conversation history, we need to format the conversation history and
    # remove the history field from the signature.
    history_field_name = self._get_history_field_name(signature)
    if history_field_name:
        # In order to format the conversation history, we need to remove the history field from the signature.
        signature_without_history = signature.delete(history_field_name)
        conversation_history = self.format_conversation_history(
            signature_without_history,
            history_field_name,
            inputs_copy,
        )

    messages = []
    system_message = (
        f"{self.format_field_description(signature)}\n"
        f"{self.format_field_structure(signature)}\n"
        f"{self.format_task_description(signature)}"
    )
    messages.append({"role": "system", "content": system_message})
    messages.extend(self.format_demos(signature, demos))
    if history_field_name:
        # Conversation history and current input
        content = self.format_user_message_content(signature_without_history, inputs_copy, main_request=True)
        messages.extend(conversation_history)
        messages.append({"role": "user", "content": content})
    else:
        # Only current input
        content = self.format_user_message_content(signature, inputs_copy, main_request=True)
        messages.append({"role": "user", "content": content})

    messages = split_message_content_for_custom_types(messages)
    return messages

```
#### 
 format_assistant_message_content(signature: type[Signature], outputs: dict[str, Any], missing_field_message: str | None = None) -> str
Format the assistant message content.
This method formats the assistant message content, which can be used in formatting few-shot examples,
conversation history.
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
The DSPy signature for which to format the assistant message content.
required
outputs
dict
[
str
,
Any
]
The output fields to be formatted.
required
missing_field_message
str
| None
A message to be used when a field is missing.
None
Returns:
Type
Description
str
A string that contains the assistant message content.
Source code in
dspy/adapters/base.py
```
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
```
```
def format_assistant_message_content(
    self,
    signature: type[Signature],
    outputs: dict[str, Any],
    missing_field_message: str | None = None,
) -> str:
    """Format the assistant message content.

    This method formats the assistant message content, which can be used in formatting few-shot examples,
    conversation history.

    Args:
        signature: The DSPy signature for which to format the assistant message content.
        outputs: The output fields to be formatted.
        missing_field_message: A message to be used when a field is missing.

    Returns:
        A string that contains the assistant message content.
    """
    raise NotImplementedError

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
 format_task_description(signature: type[Signature]) -> str
Format the task description for the system message.
This method formats the task description for the system message. In most cases this is just a thin wrapper
over signature.instructions.
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
The DSPy signature of the DSpy module.
required
Returns:
Type
Description
str
A string that describes the task.
Source code in
dspy/adapters/base.py
```
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
```
```
def format_task_description(self, signature: type[Signature]) -> str:
    """Format the task description for the system message.

    This method formats the task description for the system message. In most cases this is just a thin wrapper
    over `signature.instructions`.

    Args:
        signature: The DSPy signature of the DSpy module.

    Returns:
        A string that describes the task.
    """
    raise NotImplementedError

```
#### 
 format_user_message_content(signature: type[Signature], inputs: dict[str, Any], prefix: str = '', suffix: str = '', main_request: bool = False) -> str
Format the user message content.
This method formats the user message content, which can be used in formatting few-shot examples, conversation
history, and the current input.
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
The DSPy signature for which to format the user message content.
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
prefix
str
A prefix to the user message content.
''
suffix
str
A suffix to the user message content.
''
Returns:
Type
Description
str
A string that contains the user message content.
Source code in
dspy/adapters/base.py
```
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
```
```
def format_user_message_content(
    self,
    signature: type[Signature],
    inputs: dict[str, Any],
    prefix: str = "",
    suffix: str = "",
    main_request: bool = False,
) -> str:
    """Format the user message content.

    This method formats the user message content, which can be used in formatting few-shot examples, conversation
    history, and the current input.

    Args:
        signature: The DSPy signature for which to format the user message content.
        inputs: The input arguments to the DSPy module.
        prefix: A prefix to the user message content.
        suffix: A suffix to the user message content.

    Returns:
        A string that contains the user message content.
    """
    raise NotImplementedError

```
#### 
 parse(signature: type[Signature], completion: str) -> dict[str, Any]
Parse the LM output into a dictionary of the output fields.
This method parses the LM output into a dictionary of the output fields.
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
The DSPy signature for which to parse the LM output.
required
completion
str
The LM output to be parsed.
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
A dictionary of the output fields.
Source code in
dspy/adapters/base.py
```
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
```
```
def parse(self, signature: type[Signature], completion: str) -> dict[str, Any]:
    """Parse the LM output into a dictionary of the output fields.

    This method parses the LM output into a dictionary of the output fields.

    Args:
        signature: The DSPy signature for which to parse the LM output.
        completion: The LM output to be parsed.

    Returns:
        A dictionary of the output fields.
    """
    raise NotImplementedError

```
:::