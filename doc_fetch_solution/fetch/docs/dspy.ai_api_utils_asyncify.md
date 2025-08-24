# Asyncify

**Source:** https://dspy.ai/api/utils/asyncify
**Fetched:** 2025-08-24 17:10:34
**Status:** Success

---

[

](https://github.com/stanfordnlp/dspy/blob/main/docs/docs/api/utils/asyncify.md)
# dspy.asyncify
## 
 dspy.asyncify(program: Module) -> Callable[[Any, Any], Awaitable[Any]]
Wraps a DSPy program so that it can be called asynchronously. This is useful for running a
program in parallel with another task (e.g., another DSPy program).
This implementation propagates the current thread's configuration context to the worker thread.
Parameters:
Name
Type
Description
Default
program
[Module](../../modules/Module/#dspy.Module)
The DSPy program to be wrapped for asynchronous execution.
required
Returns:
Type
Description
Callable
[[
Any
,
Any
],
Awaitable
[
Any
]]
An async function: An async function that, when awaited, runs the program in a worker thread.
The current thread's configuration context is inherited for each call.
Source code in
dspy/utils/asyncify.py
```
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
```
```
def asyncify(program: "Module") -> Callable[[Any, Any], Awaitable[Any]]:
    """
    Wraps a DSPy program so that it can be called asynchronously. This is useful for running a
    program in parallel with another task (e.g., another DSPy program).

    This implementation propagates the current thread's configuration context to the worker thread.

    Args:
        program: The DSPy program to be wrapped for asynchronous execution.

    Returns:
        An async function: An async function that, when awaited, runs the program in a worker thread.
            The current thread's configuration context is inherited for each call.
    """

    async def async_program(*args, **kwargs) -> Any:
        # Capture the current overrides at call-time.
        from dspy.dsp.utils.settings import thread_local_overrides

        parent_overrides = thread_local_overrides.get().copy()

        def wrapped_program(*a, **kw):
            from dspy.dsp.utils.settings import thread_local_overrides

            original_overrides = thread_local_overrides.get()
            token = thread_local_overrides.set({**original_overrides, **parent_overrides.copy()})
            try:
                return program(*a, **kw)
            finally:
                thread_local_overrides.reset(token)

        # Create a fresh asyncified callable each time, ensuring the latest context is used.
        call_async = asyncer.asyncify(wrapped_program, abandon_on_cancel=True, limiter=get_limiter())
        return await call_async(*args, **kwargs)

    return async_program

```
:::