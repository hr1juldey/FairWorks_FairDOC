# Jsonadapter

**Source:** https://dspy.ai/api/adapters/JSONAdapter
**Fetched:** 2025-08-24 17:10:35
**Status:** Success

---

[

](https://github.com/stanfordnlp/dspy/blob/main/docs/docs/api/adapters/JSONAdapter.md)
# dspy.JSONAdapter
## 
 dspy.JSONAdapter(callbacks: list[BaseCallback] | None = None, use_native_function_calling: bool = True)
Bases: ChatAdapter
Source code in
dspy/adapters/json_adapter.py
```
[42](#__codelineno-0-42)
[43](#__codelineno-0-43)
[44](#__codelineno-0-44)
```
```
def __init__(self, callbacks: list[BaseCallback] | None = None, use_native_function_calling: bool = True):
    # JSONAdapter uses native function calling by default.
    super().__init__(callbacks=callbacks, use_native_function_calling=use_native_function_calling)

```
### Functions
#### 
 __call__(lm: LM, lm_kwargs: dict[str, Any], signature: type[Signature], demos: list[dict[str, Any]], inputs: dict[str, Any]) -> list[dict[str, Any]]
Source code in
dspy/adapters/json_adapter.py
```
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
```
```
def __call__(
    self,
    lm: LM,
    lm_kwargs: dict[str, Any],
    signature: type[Signature],
    demos: list[dict[str, Any]],
    inputs: dict[str, Any],
) -> list[dict[str, Any]]:
    result = self._json_adapter_call_common(lm, lm_kwargs, signature, demos, inputs, super().__call__)
    if result:
        return result

    try:
        structured_output_model = _get_structured_outputs_response_format(
            signature, self.use_native_function_calling
        )
        lm_kwargs["response_format"] = structured_output_model
        return super().__call__(lm, lm_kwargs, signature, demos, inputs)
    except Exception:
        logger.warning("Failed to use structured output format, falling back to JSON mode.")
        lm_kwargs["response_format"] = {"type": "json_object"}
        return super().__call__(lm, lm_kwargs, signature, demos, inputs)

```
#### 
 acall(lm: LM, lm_kwargs: dict[str, Any], signature: type[Signature], demos: list[dict[str, Any]], inputs: dict[str, Any]) -> list[dict[str, Any]]

async
Source code in
dspy/adapters/json_adapter.py
```
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
```
```
async def acall(
    self,
    lm: LM,
    lm_kwargs: dict[str, Any],
    signature: type[Signature],
    demos: list[dict[str, Any]],
    inputs: dict[str, Any],
) -> list[dict[str, Any]]:
    result = self._json_adapter_call_common(lm, lm_kwargs, signature, demos, inputs, super().acall)
    if result:
        return await result

    try:
        structured_output_model = _get_structured_outputs_response_format(signature)
        lm_kwargs["response_format"] = structured_output_model
        return await super().acall(lm, lm_kwargs, signature, demos, inputs)
    except Exception:
        logger.warning("Failed to use structured output format, falling back to JSON mode.")
        lm_kwargs["response_format"] = {"type": "json_object"}
        return await super().acall(lm, lm_kwargs, signature, demos, inputs)

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
 format_assistant_message_content(signature: type[Signature], outputs: dict[str, Any], missing_field_message=None) -> str
Source code in
dspy/adapters/json_adapter.py
```
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
```
```
def format_assistant_message_content(
    self,
    signature: type[Signature],
    outputs: dict[str, Any],
    missing_field_message=None,
) -> str:
    fields_with_values = {
        FieldInfoWithName(name=k, info=v): outputs.get(k, missing_field_message)
        for k, v in signature.output_fields.items()
    }
    return self.format_field_with_value(fields_with_values, role="assistant")

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
Source code in
dspy/adapters/chat_adapter.py
```
[69](#__codelineno-0-69)
[70](#__codelineno-0-70)
[71](#__codelineno-0-71)
[72](#__codelineno-0-72)
[73](#__codelineno-0-73)
```
```
def format_field_description(self, signature: type[Signature]) -> str:
    return (
        f"Your input fields are:\n{get_field_description_string(signature.input_fields)}\n"
        f"Your output fields are:\n{get_field_description_string(signature.output_fields)}"
    )

```
#### 
 format_field_structure(signature: type[Signature]) -> str
Source code in
dspy/adapters/json_adapter.py
```
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
```
```
def format_field_structure(self, signature: type[Signature]) -> str:
    parts = []
    parts.append("All interactions will be structured in the following way, with the appropriate values filled in.")

    def format_signature_fields_for_instructions(fields: dict[str, FieldInfo], role: str):
        return self.format_field_with_value(
            fields_with_values={
                FieldInfoWithName(name=field_name, info=field_info): translate_field_type(field_name, field_info)
                for field_name, field_info in fields.items()
            },
            role=role,
        )

    parts.append("Inputs will have the following structure:")
    parts.append(format_signature_fields_for_instructions(signature.input_fields, role="user"))
    parts.append("Outputs will be a JSON object with the following fields.")
    parts.append(format_signature_fields_for_instructions(signature.output_fields, role="assistant"))
    return "\n\n".join(parts).strip()

```
#### 
 format_field_with_value(fields_with_values: dict[FieldInfoWithName, Any], role: str = 'user') -> str
Formats the values of the specified fields according to the field's DSPy type (input or output),
annotation (e.g. str, int, etc.), and the type of the value itself. Joins the formatted values
into a single string, which is a multiline string if there are multiple fields.
Parameters:
Name
Type
Description
Default
fields_with_values
dict
[
FieldInfoWithName
,
Any
]
A dictionary mapping information about a field to its corresponding value.
required
Returns:
    The joined formatted values of the fields, represented as a string.
Source code in
dspy/adapters/json_adapter.py
```
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
```
```
def format_field_with_value(self, fields_with_values: dict[FieldInfoWithName, Any], role: str = "user") -> str:
    """
    Formats the values of the specified fields according to the field's DSPy type (input or output),
    annotation (e.g. str, int, etc.), and the type of the value itself. Joins the formatted values
    into a single string, which is a multiline string if there are multiple fields.

    Args:
        fields_with_values: A dictionary mapping information about a field to its corresponding value.
    Returns:
        The joined formatted values of the fields, represented as a string.
    """
    if role == "user":
        output = []
        for field, field_value in fields_with_values.items():
            formatted_field_value = format_field_value(field_info=field.info, value=field_value)
            output.append(f"[[ ## {field.name} ## ]]\n{formatted_field_value}")
        return "\n\n".join(output).strip()
    else:
        d = fields_with_values.items()
        d = {k.name: v for k, v in d}
        return json.dumps(serialize_for_json(d), indent=2)

```
#### 
 format_finetune_data(signature: type[Signature], demos: list[dict[str, Any]], inputs: dict[str, Any], outputs: dict[str, Any]) -> dict[str, list[Any]]
Source code in
dspy/adapters/json_adapter.py
```
[203](#__codelineno-0-203)
[204](#__codelineno-0-204)
[205](#__codelineno-0-205)
[206](#__codelineno-0-206)
[207](#__codelineno-0-207)
```
```
def format_finetune_data(
    self, signature: type[Signature], demos: list[dict[str, Any]], inputs: dict[str, Any], outputs: dict[str, Any]
) -> dict[str, list[Any]]:
    # TODO: implement format_finetune_data method in JSONAdapter
    raise NotImplementedError

```
#### 
 format_task_description(signature: type[Signature]) -> str
Source code in
dspy/adapters/chat_adapter.py
```
[ 97](#__codelineno-0-97)
[ 98](#__codelineno-0-98)
[ 99](#__codelineno-0-99)
[100](#__codelineno-0-100)
```
```
def format_task_description(self, signature: type[Signature]) -> str:
    instructions = textwrap.dedent(signature.instructions)
    objective = ("\n" + " " * 8).join([""] + instructions.splitlines())
    return f"In adhering to this structure, your objective is: {objective}"

```
#### 
 format_user_message_content(signature: type[Signature], inputs: dict[str, Any], prefix: str = '', suffix: str = '', main_request: bool = False) -> str
Source code in
dspy/adapters/chat_adapter.py
```
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
    messages = [prefix]
    for k, v in signature.input_fields.items():
        if k in inputs:
            value = inputs.get(k)
            formatted_field_value = format_field_value(field_info=v, value=value)
            messages.append(f"[[ ## {k} ## ]]\n{formatted_field_value}")

    if main_request:
        output_requirements = self.user_message_output_requirements(signature)
        if output_requirements is not None:
            messages.append(output_requirements)

    messages.append(suffix)
    return "\n\n".join(messages).strip()

```
#### 
 parse(signature: type[Signature], completion: str) -> dict[str, Any]
Source code in
dspy/adapters/json_adapter.py
```
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
```
```
def parse(self, signature: type[Signature], completion: str) -> dict[str, Any]:
    pattern = r"\{(?:[^{}]|(?R))*\}"
    match = regex.search(pattern, completion, regex.DOTALL)
    if match:
        completion = match.group(0)
    fields = json_repair.loads(completion)

    if not isinstance(fields, dict):
        raise AdapterParseError(
            adapter_name="JSONAdapter",
            signature=signature,
            lm_response=completion,
            message="LM response cannot be serialized to a JSON object.",
        )

    fields = {k: v for k, v in fields.items() if k in signature.output_fields}

    # Attempt to cast each value to type signature.output_fields[k].annotation.
    for k, v in fields.items():
        if k in signature.output_fields:
            fields[k] = parse_value(v, signature.output_fields[k].annotation)

    if fields.keys() != signature.output_fields.keys():
        raise AdapterParseError(
            adapter_name="JSONAdapter",
            signature=signature,
            lm_response=completion,
            parsed_result=fields,
        )

    return fields

```
#### 
 user_message_output_requirements(signature: type[Signature]) -> str
Source code in
dspy/adapters/json_adapter.py
```
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
def user_message_output_requirements(self, signature: type[Signature]) -> str:
    def type_info(v):
        return (
            f" (must be formatted as a valid Python {get_annotation_name(v.annotation)})"
            if v.annotation is not str
            else ""
        )

    message = "Respond with a JSON object in the following order of fields: "
    message += ", then ".join(f"`{f}`{type_info(v)}" for f, v in signature.output_fields.items())
    message += "."
    return message

```
:::