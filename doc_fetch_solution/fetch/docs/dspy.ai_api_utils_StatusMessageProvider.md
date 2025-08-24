# Statusmessageprovider

**Source:** https://dspy.ai/api/utils/StatusMessageProvider
**Fetched:** 2025-08-24 17:10:34
**Status:** Success

---

[

](https://github.com/stanfordnlp/dspy/blob/main/docs/docs/api/utils/StatusMessageProvider.md)
# dspy.streaming.StatusMessageProvider
## 
 dspy.streaming.StatusMessageProvider
Provides customizable status message streaming for DSPy programs.
This class serves as a base for creating custom status message providers. Users can subclass
and override its methods to define specific status messages for different stages of program execution,
each method must return a string.
Example:
class MyStatusMessageProvider(StatusMessageProvider):
    def lm_start_status_message(self, instance, inputs):
        return f"Calling LM with inputs {inputs}..."

    def module_end_status_message(self, outputs):
        return f"Module finished with output: {outputs}!"

program = dspy.streamify(dspy.Predict("q->a"), status_message_provider=MyStatusMessageProvider())
### Functions
#### 
 lm_end_status_message(outputs: Any)
Status message after a dspy.LM is called.
Source code in
dspy/streaming/messages.py
```
[93](#__codelineno-0-93)
[94](#__codelineno-0-94)
[95](#__codelineno-0-95)
```
```
def lm_end_status_message(self, outputs: Any):
    """Status message after a `dspy.LM` is called."""
    pass

```
#### 
 lm_start_status_message(instance: Any, inputs: dict[str, Any])
Status message before a dspy.LM is called.
Source code in
dspy/streaming/messages.py
```
[89](#__codelineno-0-89)
[90](#__codelineno-0-90)
[91](#__codelineno-0-91)
```
```
def lm_start_status_message(self, instance: Any, inputs: dict[str, Any]):
    """Status message before a `dspy.LM` is called."""
    pass

```
#### 
 module_end_status_message(outputs: Any)
Status message after a dspy.Module or dspy.Predict is called.
Source code in
dspy/streaming/messages.py
```
[85](#__codelineno-0-85)
[86](#__codelineno-0-86)
[87](#__codelineno-0-87)
```
```
def module_end_status_message(self, outputs: Any):
    """Status message after a `dspy.Module` or `dspy.Predict` is called."""
    pass

```
#### 
 module_start_status_message(instance: Any, inputs: dict[str, Any])
Status message before a dspy.Module or dspy.Predict is called.
Source code in
dspy/streaming/messages.py
```
[81](#__codelineno-0-81)
[82](#__codelineno-0-82)
[83](#__codelineno-0-83)
```
```
def module_start_status_message(self, instance: Any, inputs: dict[str, Any]):
    """Status message before a `dspy.Module` or `dspy.Predict` is called."""
    pass

```
#### 
 tool_end_status_message(outputs: Any)
Status message after a dspy.Tool is called.
Source code in
dspy/streaming/messages.py
```
[77](#__codelineno-0-77)
[78](#__codelineno-0-78)
[79](#__codelineno-0-79)
```
```
def tool_end_status_message(self, outputs: Any):
    """Status message after a `dspy.Tool` is called."""
    return "Tool calling finished! Querying the LLM with tool calling results..."

```
#### 
 tool_start_status_message(instance: Any, inputs: dict[str, Any])
Status message before a dspy.Tool is called.
Source code in
dspy/streaming/messages.py
```
[73](#__codelineno-0-73)
[74](#__codelineno-0-74)
[75](#__codelineno-0-75)
```
```
def tool_start_status_message(self, instance: Any, inputs: dict[str, Any]):
    """Status message before a `dspy.Tool` is called."""
    return f"Calling tool {instance.name}..."

```
:::