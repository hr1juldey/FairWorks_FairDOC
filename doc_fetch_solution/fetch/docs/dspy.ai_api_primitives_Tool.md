# Tool

**Source:** https://dspy.ai/api/primitives/Tool
**Fetched:** 2025-08-24 17:10:36
**Status:** Success

---

[

](https://github.com/stanfordnlp/dspy/blob/main/docs/docs/api/primitives/Tool.md)
# dspy.Tool
## 
 dspy.Tool(func: Callable, name: str | None = None, desc: str | None = None, args: dict[str, Any] | None = None, arg_types: dict[str, Any] | None = None, arg_desc: dict[str, str] | None = None)
Bases: Type
Tool class.
This class is used to simplify the creation of tools for tool calling (function calling) in LLMs. Only supports
functions for now.
Initialize the Tool class.
Users can choose to specify the name, desc, args, and arg_types, or let the dspy.Tool
automatically infer the values from the function. For values that are specified by the user, automatic inference
will not be performed on them.
Parameters:
Name
Type
Description
Default
func
Callable
The actual function that is being wrapped by the tool.
required
name
Optional
[
str
]
The name of the tool. Defaults to None.
None
desc
Optional
[
str
]
The description of the tool. Defaults to None.
None
args
Optional
[
dict
[
str
,
Any
]]
The args and their schema of the tool, represented as a
dictionary from arg name to arg's json schema. Defaults to None.
None
arg_types
Optional
[
dict
[
str
,
Any
]]
The argument types of the tool, represented as a dictionary
from arg name to the type of the argument. Defaults to None.
None
arg_desc
Optional
[
dict
[
str
,
str
]]
Descriptions for each arg, represented as a
dictionary from arg name to description string. Defaults to None.
None
Example:
```
def foo(x: int, y: str = "hello"):
    return str(x) + y

tool = Tool(foo)
print(tool.args)
# Expected output: {'x': {'type': 'integer'}, 'y': {'type': 'string', 'default': 'hello'}}

```
Source code in
dspy/adapters/types/tool.py
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
```
```
def __init__(
    self,
    func: Callable,
    name: str | None = None,
    desc: str | None = None,
    args: dict[str, Any] | None = None,
    arg_types: dict[str, Any] | None = None,
    arg_desc: dict[str, str] | None = None,
):
    """Initialize the Tool class.

    Users can choose to specify the `name`, `desc`, `args`, and `arg_types`, or let the `dspy.Tool`
    automatically infer the values from the function. For values that are specified by the user, automatic inference
    will not be performed on them.

    Args:
        func (Callable): The actual function that is being wrapped by the tool.
        name (Optional[str], optional): The name of the tool. Defaults to None.
        desc (Optional[str], optional): The description of the tool. Defaults to None.
        args (Optional[dict[str, Any]], optional): The args and their schema of the tool, represented as a
            dictionary from arg name to arg's json schema. Defaults to None.
        arg_types (Optional[dict[str, Any]], optional): The argument types of the tool, represented as a dictionary
            from arg name to the type of the argument. Defaults to None.
        arg_desc (Optional[dict[str, str]], optional): Descriptions for each arg, represented as a
            dictionary from arg name to description string. Defaults to None.

    Example:

    ```python
    def foo(x: int, y: str = "hello"):
        return str(x) + y

    tool = Tool(foo)
    print(tool.args)
    # Expected output: {'x': {'type': 'integer'}, 'y': {'type': 'string', 'default': 'hello'}}
    ```
    """
    super().__init__(func=func, name=name, desc=desc, args=args, arg_types=arg_types, arg_desc=arg_desc)
    self._parse_function(func, arg_desc)

```
### Functions
#### 
 __call__(**kwargs)
Source code in
dspy/adapters/types/tool.py
```
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
```
```
@with_callbacks
def __call__(self, **kwargs):
    parsed_kwargs = self._validate_and_parse_args(**kwargs)
    result = self.func(**parsed_kwargs)
    if asyncio.iscoroutine(result):
        if settings.allow_tool_async_sync_conversion:
            return self._run_async_in_sync(result)
        else:
            raise ValueError(
                "You are calling `__call__` on an async tool, please use `acall` instead or set "
                "`allow_async=True` to run the async tool in sync mode."
            )
    return result

```
#### 
 acall(**kwargs)

async
Source code in
dspy/adapters/types/tool.py
```
[187](#__codelineno-0-187)
[188](#__codelineno-0-188)
[189](#__codelineno-0-189)
[190](#__codelineno-0-190)
[191](#__codelineno-0-191)
[192](#__codelineno-0-192)
[193](#__codelineno-0-193)
[194](#__codelineno-0-194)
[195](#__codelineno-0-195)
```
```
@with_callbacks
async def acall(self, **kwargs):
    parsed_kwargs = self._validate_and_parse_args(**kwargs)
    result = self.func(**parsed_kwargs)
    if asyncio.iscoroutine(result):
        return await result
    else:
        # We should allow calling a sync tool in the async path.
        return result

```
#### 
 description() -> str

classmethod
Description of the custom type
Source code in
dspy/adapters/types/base_type.py
```
[32](#__codelineno-0-32)
[33](#__codelineno-0-33)
[34](#__codelineno-0-34)
[35](#__codelineno-0-35)
```
```
@classmethod
def description(cls) -> str:
    """Description of the custom type"""
    return ""

```
#### 
 extract_custom_type_from_annotation(annotation)

classmethod
Extract all custom types from the annotation.
This is used to extract all custom types from the annotation of a field, while the annotation can
have arbitrary level of nesting. For example, we detect Tool is in list[dict[str, Tool]].
Source code in
dspy/adapters/types/base_type.py
```
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
```
```
@classmethod
def extract_custom_type_from_annotation(cls, annotation):
    """Extract all custom types from the annotation.

    This is used to extract all custom types from the annotation of a field, while the annotation can
    have arbitrary level of nesting. For example, we detect `Tool` is in `list[dict[str, Tool]]`.
    """
    # Direct match. Nested type like `list[dict[str, Event]]` passes `isinstance(annotation, type)` in python 3.10
    # while fails in python 3.11. To accommodate users using python 3.10, we need to capture the error and ignore it.
    try:
        if isinstance(annotation, type) and issubclass(annotation, cls):
            return [annotation]
    except TypeError:
        pass

    origin = get_origin(annotation)
    if origin is None:
        return []

    result = []
    # Recurse into all type args
    for arg in get_args(annotation):
        result.extend(cls.extract_custom_type_from_annotation(arg))

    return result

```
#### 
 format()
Source code in
dspy/adapters/types/tool.py
```
[148](#__codelineno-0-148)
[149](#__codelineno-0-149)
```
```
def format(self):
    return str(self)

```
#### 
 format_as_litellm_function_call()
Source code in
dspy/adapters/types/tool.py
```
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
```
```
def format_as_litellm_function_call(self):
    return {
        "type": "function",
        "function": {
            "name": self.name,
            "description": self.desc,
            "parameters": {
                "type": "object",
                "properties": self.args,
                "required": list(self.args.keys()),
            },
        },
    }

```
#### 
 from_langchain(tool: BaseTool) -> Tool

classmethod
Build a DSPy tool from a LangChain tool.
Parameters:
Name
Type
Description
Default
tool
BaseTool
The LangChain tool to convert.
required
Returns:
Type
Description
[Tool](#dspy.Tool)
A Tool object.
Example:
```
import asyncio
import dspy
from langchain.tools import tool as lc_tool

@lc_tool
def add(x: int, y: int):
    "Add two numbers together."
    return x + y

dspy_tool = dspy.Tool.from_langchain(add)

async def run_tool():
    return await dspy_tool.acall(x=1, y=2)

print(asyncio.run(run_tool()))
# 3

```
Source code in
dspy/adapters/types/tool.py
```
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
```
```
@classmethod
def from_langchain(cls, tool: "BaseTool") -> "Tool":
    """
    Build a DSPy tool from a LangChain tool.

    Args:
        tool: The LangChain tool to convert.

    Returns:
        A Tool object.

    Example:

    ```python
    import asyncio
    import dspy
    from langchain.tools import tool as lc_tool

    @lc_tool
    def add(x: int, y: int):
        "Add two numbers together."
        return x + y

    dspy_tool = dspy.Tool.from_langchain(add)

    async def run_tool():
        return await dspy_tool.acall(x=1, y=2)

    print(asyncio.run(run_tool()))
    # 3
    ```
    """
    from dspy.utils.langchain_tool import convert_langchain_tool

    return convert_langchain_tool(tool)

```
#### 
 from_mcp_tool(session: mcp.client.session.ClientSession, tool: mcp.types.Tool) -> Tool

classmethod
Build a DSPy tool from an MCP tool and a ClientSession.
Parameters:
Name
Type
Description
Default
session
ClientSession
The MCP session to use.
required
tool
Tool
The MCP tool to convert.
required
Returns:
Type
Description
[Tool](#dspy.Tool)
A Tool object.
Source code in
dspy/adapters/types/tool.py
```
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
```
```
@classmethod
def from_mcp_tool(cls, session: "mcp.client.session.ClientSession", tool: "mcp.types.Tool") -> "Tool":
    """
    Build a DSPy tool from an MCP tool and a ClientSession.

    Args:
        session: The MCP session to use.
        tool: The MCP tool to convert.

    Returns:
        A Tool object.
    """
    from dspy.utils.mcp import convert_mcp_tool

    return convert_mcp_tool(session, tool)

```
#### 
 serialize_model()
Source code in
dspy/adapters/types/base_type.py
```
[63](#__codelineno-0-63)
[64](#__codelineno-0-64)
[65](#__codelineno-0-65)
[66](#__codelineno-0-66)
[67](#__codelineno-0-67)
[68](#__codelineno-0-68)
```
```
@pydantic.model_serializer()
def serialize_model(self):
    formatted = self.format()
    if isinstance(formatted, list):
        return f"{CUSTOM_TYPE_START_IDENTIFIER}{formatted}{CUSTOM_TYPE_END_IDENTIFIER}"
    return formatted

```
:::