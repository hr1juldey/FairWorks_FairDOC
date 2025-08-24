# Signature

**Source:** https://dspy.ai/api/signatures/Signature
**Fetched:** 2025-08-24 17:10:35
**Status:** Success

---

[

](https://github.com/stanfordnlp/dspy/blob/main/docs/docs/api/signatures/Signature.md)
# dspy.Signature
## 
 dspy.Signature
Bases: BaseModel
### Functions
#### 
 append(name, field, type_=None) -> type[Signature]

classmethod
Source code in
dspy/signatures/signature.py
```
[280](#__codelineno-0-280)
[281](#__codelineno-0-281)
[282](#__codelineno-0-282)
```
```
@classmethod
def append(cls, name, field, type_=None) -> type["Signature"]:
    return cls.insert(-1, name, field, type_)

```
#### 
 delete(name) -> type[Signature]

classmethod
Source code in
dspy/signatures/signature.py
```
[284](#__codelineno-0-284)
[285](#__codelineno-0-285)
[286](#__codelineno-0-286)
[287](#__codelineno-0-287)
[288](#__codelineno-0-288)
[289](#__codelineno-0-289)
[290](#__codelineno-0-290)
```
```
@classmethod
def delete(cls, name) -> type["Signature"]:
    fields = dict(cls.fields)

    fields.pop(name, None)

    return Signature(fields, cls.instructions)

```
#### 
 dump_state()

classmethod
Source code in
dspy/signatures/signature.py
```
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
```
```
@classmethod
def dump_state(cls):
    state = {"instructions": cls.instructions, "fields": []}
    for field in cls.fields:
        state["fields"].append(
            {
                "prefix": cls.fields[field].json_schema_extra["prefix"],
                "description": cls.fields[field].json_schema_extra["desc"],
            }
        )

    return state

```
#### 
 equals(other) -> bool

classmethod
Compare the JSON schema of two Signature classes.
Source code in
dspy/signatures/signature.py
```
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
```
```
@classmethod
def equals(cls, other) -> bool:
    """Compare the JSON schema of two Signature classes."""
    if not isinstance(other, type) or not issubclass(other, BaseModel):
        return False
    if cls.instructions != other.instructions:
        return False
    for name in cls.fields.keys() | other.fields.keys():
        if name not in other.fields or name not in cls.fields:
            return False
        if cls.fields[name].json_schema_extra != other.fields[name].json_schema_extra:
            return False
    return True

```
#### 
 insert(index: int, name: str, field, type_: type | None = None) -> type[Signature]

classmethod
Source code in
dspy/signatures/signature.py
```
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
```
```
@classmethod
def insert(cls, index: int, name: str, field, type_: type | None = None) -> type["Signature"]:
    # It's possible to set the type as annotation=type in pydantic.Field(...)
    # But this may be annoying for users, so we allow them to pass the type
    if type_ is None:
        type_ = field.annotation
    if type_ is None:
        type_ = str

    input_fields = list(cls.input_fields.items())
    output_fields = list(cls.output_fields.items())

    # Choose the list to insert into based on the field type
    lst = input_fields if field.json_schema_extra["__dspy_field_type"] == "input" else output_fields
    # We support negative insert indices
    if index < 0:
        index += len(lst) + 1
    if index < 0 or index > len(lst):
        raise ValueError(
            f"Invalid index to insert: {index}, index must be in the range of [{len(lst) - 1}, {len(lst)}] for "
            f"{field.json_schema_extra['__dspy_field_type']} fields, but received: {index}.",
        )
    lst.insert(index, (name, (type_, field)))

    new_fields = dict(input_fields + output_fields)
    return Signature(new_fields, cls.instructions)

```
#### 
 load_state(state)

classmethod
Source code in
dspy/signatures/signature.py
```
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
```
```
@classmethod
def load_state(cls, state):
    signature_copy = Signature(deepcopy(cls.fields), cls.instructions)

    signature_copy.instructions = state["instructions"]
    for field, saved_field in zip(signature_copy.fields.values(), state["fields"], strict=False):
        field.json_schema_extra["prefix"] = saved_field["prefix"]
        field.json_schema_extra["desc"] = saved_field["description"]

    return signature_copy

```
#### 
 prepend(name, field, type_=None) -> type[Signature]

classmethod
Source code in
dspy/signatures/signature.py
```
[276](#__codelineno-0-276)
[277](#__codelineno-0-277)
[278](#__codelineno-0-278)
```
```
@classmethod
def prepend(cls, name, field, type_=None) -> type["Signature"]:
    return cls.insert(0, name, field, type_)

```
#### 
 with_instructions(instructions: str) -> type[Signature]

classmethod
Source code in
dspy/signatures/signature.py
```
[246](#__codelineno-0-246)
[247](#__codelineno-0-247)
[248](#__codelineno-0-248)
```
```
@classmethod
def with_instructions(cls, instructions: str) -> type["Signature"]:
    return Signature(cls.fields, instructions)

```
#### 
 with_updated_fields(name: str, type_: type | None = None, **kwargs: dict[str, Any]) -> type[Signature]

classmethod
Create a new Signature class with the updated field information.
Returns a new Signature class with the field, name, updated
with fields[name].json_schema_extra[key] = value.
Parameters:
Name
Type
Description
Default
name
str
The name of the field to update.
required
type_
type
| None
The new type of the field.
None
kwargs
dict
[
str
,
Any
]
The new values for the field.
{}
Returns:
Type
Description
type
[
[Signature](#dspy.Signature)
]
A new Signature class (not an instance) with the updated field information.
Source code in
dspy/signatures/signature.py
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
```
```
@classmethod
def with_updated_fields(cls, name: str, type_: type | None = None, **kwargs: dict[str, Any]) -> type["Signature"]:
    """Create a new Signature class with the updated field information.

    Returns a new Signature class with the field, name, updated
    with fields[name].json_schema_extra[key] = value.

    Args:
        name: The name of the field to update.
        type_: The new type of the field.
        kwargs: The new values for the field.

    Returns:
        A new Signature class (not an instance) with the updated field information.
    """
    fields_copy = deepcopy(cls.fields)
    # Update `fields_copy[name].json_schema_extra` with the new kwargs, on conflicts
    # we use the new value in kwargs.
    fields_copy[name].json_schema_extra = {
        **fields_copy[name].json_schema_extra,
        **kwargs,
    }
    if type_ is not None:
        fields_copy[name].annotation = type_
    return Signature(fields_copy, cls.instructions)

```
:::