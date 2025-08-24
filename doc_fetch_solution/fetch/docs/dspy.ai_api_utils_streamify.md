# Streamify

**Source:** https://dspy.ai/api/utils/streamify
**Fetched:** 2025-08-24 17:10:32
**Status:** Success

---

[

](https://github.com/stanfordnlp/dspy/blob/main/docs/docs/api/utils/streamify.md)
# dspy.streamify
## 
 dspy.streamify(program: Module, status_message_provider: StatusMessageProvider | None = None, stream_listeners: list[StreamListener] | None = None, include_final_prediction_in_output_stream: bool = True, is_async_program: bool = False, async_streaming: bool = True) -> Callable[[Any, Any], Awaitable[Any]]
Wrap a DSPy program so that it streams its outputs incrementally, rather than returning them
all at once. It also provides status messages to the user to indicate the progress of the program, and users
can implement their own status message provider to customize the status messages and what module to generate
status messages for.
Parameters:
Name
Type
Description
Default
program
[Module](../../modules/Module/#dspy.Module)
The DSPy program to wrap with streaming functionality.
required
status_message_provider
[StatusMessageProvider](../StatusMessageProvider/#dspy.streaming.StatusMessageProvider)
| None
A custom status message generator to use instead of the default one. Users can
implement their own status message generator to customize the status messages and what module to generate
status messages for.
None
stream_listeners
list
[
[StreamListener](../StreamListener/#dspy.streaming.StreamListener)
] | None
A list of stream listeners to capture the streaming output of specific fields of sub predicts
in the program. When provided, only the target fields in the target predict will be streamed to the user.
None
include_final_prediction_in_output_stream
bool
Whether to include the final prediction in the output stream, only
useful when stream_listeners is provided. If False, the final prediction will not be included in the
output stream. When the program hit cache, or no listeners captured anything, the final prediction will
still be included in the output stream even if this is False.
True
is_async_program
bool
Whether the program is async. If False, the program will be wrapped with asyncify,
otherwise the program will be called with acall.
False
async_streaming
bool
Whether to return an async generator or a sync generator. If False, the streaming will be
converted to a sync generator.
True
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
A function that takes the same arguments as the original program, but returns an async
generator that yields the program's outputs incrementally.
Example:
```
import asyncio
import dspy

dspy.settings.configure(lm=dspy.LM("openai/gpt-4o-mini"))
# Create the program and wrap it with streaming functionality
program = dspy.streamify(dspy.Predict("q->a"))

# Use the program with streaming output
async def use_streaming():
    output = program(q="Why did a chicken cross the kitchen?")
    return_value = None
    async for value in output:
        if isinstance(value, dspy.Prediction):
            return_value = value
        else:
            print(value)
    return return_value

output = asyncio.run(use_streaming())
print(output)

```
Example with custom status message provider:
import asyncio
import dspy

dspy.settings.configure(lm=dspy.LM("openai/gpt-4o-mini"))

class MyStatusMessageProvider(StatusMessageProvider):
    def module_start_status_message(self, instance, inputs):
        return f"Predicting..."

    def tool_end_status_message(self, outputs):
        return f"Tool calling finished with output: {outputs}!"

# Create the program and wrap it with streaming functionality
program = dspy.streamify(dspy.Predict("q->a"), status_message_provider=MyStatusMessageProvider())

# Use the program with streaming output
async def use_streaming():
    output = program(q="Why did a chicken cross the kitchen?")
    return_value = None
    async for value in output:
        if isinstance(value, dspy.Prediction):
            return_value = value
        else:
            print(value)
    return return_value

output = asyncio.run(use_streaming())
print(output)
Example with stream listeners:
```
import asyncio
import dspy

dspy.settings.configure(lm=dspy.LM("openai/gpt-4o-mini", cache=False))

# Create the program and wrap it with streaming functionality
predict = dspy.Predict("question->answer, reasoning")
stream_listeners = [
    dspy.streaming.StreamListener(signature_field_name="answer"),
    dspy.streaming.StreamListener(signature_field_name="reasoning"),
]
stream_predict = dspy.streamify(predict, stream_listeners=stream_listeners)

async def use_streaming():
    output = stream_predict(
        question="why did a chicken cross the kitchen?",
        include_final_prediction_in_output_stream=False,
    )
    return_value = None
    async for value in output:
        if isinstance(value, dspy.Prediction):
            return_value = value
        else:
            print(value)
    return return_value

output = asyncio.run(use_streaming())
print(output)

```
You should see the streaming chunks (in the format of dspy.streaming.StreamResponse) in the console output.
Source code in
dspy/streaming/streamify.py
```
[ 27](#__codelineno-0-27)
[ 28](#__codelineno-0-28)
[ 29](#__codelineno-0-29)
[ 30](#__codelineno-0-30)
[ 31](#__codelineno-0-31)
[ 32](#__codelineno-0-32)
[ 33](#__codelineno-0-33)
[ 34](#__codelineno-0-34)
[ 35](#__codelineno-0-35)
[ 36](#__codelineno-0-36)
[ 37](#__codelineno-0-37)
[ 38](#__codelineno-0-38)
[ 39](#__codelineno-0-39)
[ 40](#__codelineno-0-40)
[ 41](#__codelineno-0-41)
[ 42](#__codelineno-0-42)
[ 43](#__codelineno-0-43)
[ 44](#__codelineno-0-44)
[ 45](#__codelineno-0-45)
[ 46](#__codelineno-0-46)
[ 47](#__codelineno-0-47)
[ 48](#__codelineno-0-48)
[ 49](#__codelineno-0-49)
[ 50](#__codelineno-0-50)
[ 51](#__codelineno-0-51)
[ 52](#__codelineno-0-52)
[ 53](#__codelineno-0-53)
[ 54](#__codelineno-0-54)
[ 55](#__codelineno-0-55)
[ 56](#__codelineno-0-56)
[ 57](#__codelineno-0-57)
[ 58](#__codelineno-0-58)
[ 59](#__codelineno-0-59)
[ 60](#__codelineno-0-60)
[ 61](#__codelineno-0-61)
[ 62](#__codelineno-0-62)
[ 63](#__codelineno-0-63)
[ 64](#__codelineno-0-64)
[ 65](#__codelineno-0-65)
[ 66](#__codelineno-0-66)
[ 67](#__codelineno-0-67)
[ 68](#__codelineno-0-68)
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
```
```
def streamify(
    program: "Module",
    status_message_provider: StatusMessageProvider | None = None,
    stream_listeners: list[StreamListener] | None = None,
    include_final_prediction_in_output_stream: bool = True,
    is_async_program: bool = False,
    async_streaming: bool = True,
) -> Callable[[Any, Any], Awaitable[Any]]:
    """
    Wrap a DSPy program so that it streams its outputs incrementally, rather than returning them
    all at once. It also provides status messages to the user to indicate the progress of the program, and users
    can implement their own status message provider to customize the status messages and what module to generate
    status messages for.

    Args:
        program: The DSPy program to wrap with streaming functionality.
        status_message_provider: A custom status message generator to use instead of the default one. Users can
            implement their own status message generator to customize the status messages and what module to generate
            status messages for.
        stream_listeners: A list of stream listeners to capture the streaming output of specific fields of sub predicts
            in the program. When provided, only the target fields in the target predict will be streamed to the user.
        include_final_prediction_in_output_stream: Whether to include the final prediction in the output stream, only
            useful when `stream_listeners` is provided. If `False`, the final prediction will not be included in the
            output stream. When the program hit cache, or no listeners captured anything, the final prediction will
            still be included in the output stream even if this is `False`.
        is_async_program: Whether the program is async. If `False`, the program will be wrapped with `asyncify`,
            otherwise the program will be called with `acall`.
        async_streaming: Whether to return an async generator or a sync generator. If `False`, the streaming will be
            converted to a sync generator.

    Returns:
        A function that takes the same arguments as the original program, but returns an async
            generator that yields the program's outputs incrementally.

    Example:

    ```python
    import asyncio
    import dspy

    dspy.settings.configure(lm=dspy.LM("openai/gpt-4o-mini"))
    # Create the program and wrap it with streaming functionality
    program = dspy.streamify(dspy.Predict("q->a"))

    # Use the program with streaming output
    async def use_streaming():
        output = program(q="Why did a chicken cross the kitchen?")
        return_value = None
        async for value in output:
            if isinstance(value, dspy.Prediction):
                return_value = value
            else:
                print(value)
        return return_value

    output = asyncio.run(use_streaming())
    print(output)
    ```

    Example with custom status message provider:
    ```python
    import asyncio
    import dspy

    dspy.settings.configure(lm=dspy.LM("openai/gpt-4o-mini"))

    class MyStatusMessageProvider(StatusMessageProvider):
        def module_start_status_message(self, instance, inputs):
            return f"Predicting..."

        def tool_end_status_message(self, outputs):
            return f"Tool calling finished with output: {outputs}!"

    # Create the program and wrap it with streaming functionality
    program = dspy.streamify(dspy.Predict("q->a"), status_message_provider=MyStatusMessageProvider())

    # Use the program with streaming output
    async def use_streaming():
        output = program(q="Why did a chicken cross the kitchen?")
        return_value = None
        async for value in output:
            if isinstance(value, dspy.Prediction):
                return_value = value
            else:
                print(value)
        return return_value

    output = asyncio.run(use_streaming())
    print(output)
    ```

    Example with stream listeners:

    ```python
    import asyncio
    import dspy

    dspy.settings.configure(lm=dspy.LM("openai/gpt-4o-mini", cache=False))

    # Create the program and wrap it with streaming functionality
    predict = dspy.Predict("question->answer, reasoning")
    stream_listeners = [
        dspy.streaming.StreamListener(signature_field_name="answer"),
        dspy.streaming.StreamListener(signature_field_name="reasoning"),
    ]
    stream_predict = dspy.streamify(predict, stream_listeners=stream_listeners)

    async def use_streaming():
        output = stream_predict(
            question="why did a chicken cross the kitchen?",
            include_final_prediction_in_output_stream=False,
        )
        return_value = None
        async for value in output:
            if isinstance(value, dspy.Prediction):
                return_value = value
            else:
                print(value)
        return return_value

    output = asyncio.run(use_streaming())
    print(output)
    ```

    You should see the streaming chunks (in the format of `dspy.streaming.StreamResponse`) in the console output.
    """
    stream_listeners = stream_listeners or []
    if len(stream_listeners) > 0:
        predict_id_to_listener = find_predictor_for_stream_listeners(program, stream_listeners)
    else:
        predict_id_to_listener = {}

    if is_async_program:
        program = program.acall
    elif not iscoroutinefunction(program):
        program = asyncify(program)

    callbacks = settings.callbacks
    status_streaming_callback = StatusStreamingCallback(status_message_provider)
    if not any(isinstance(c, StatusStreamingCallback) for c in callbacks):
        callbacks.append(status_streaming_callback)

    async def generator(args, kwargs, stream: MemoryObjectSendStream):
        with settings.context(send_stream=stream, callbacks=callbacks, stream_listeners=stream_listeners):
            prediction = await program(*args, **kwargs)

        await stream.send(prediction)

    async def async_streamer(*args, **kwargs):
        send_stream, receive_stream = create_memory_object_stream(16)
        async with create_task_group() as tg, send_stream, receive_stream:
            tg.start_soon(generator, args, kwargs, send_stream)

            async for value in receive_stream:
                if isinstance(value, ModelResponseStream):
                    if len(predict_id_to_listener) == 0:
                        # No listeners are configured, yield the chunk directly for backwards compatibility.
                        yield value
                    else:
                        # We are receiving a chunk from the LM's response stream, delegate it to the listeners to
                        # determine if we should yield a value to the user.
                        output = None
                        for listener in predict_id_to_listener[value.predict_id]:
                            # There should be at most one listener provides a return value.
                            output = listener.receive(value) or output
                        if output:
                            yield output
                elif isinstance(value, StatusMessage):
                    yield value
                elif isinstance(value, Prediction):
                    if include_final_prediction_in_output_stream:
                        yield value
                    elif (
                        len(stream_listeners) == 0
                        or any(listener.cache_hit for listener in stream_listeners)
                        or not any(listener.stream_start for listener in stream_listeners)
                    ):
                        yield value
                    return
                else:
                    # This wildcard case allows for customized streaming behavior.
                    # It is useful when a users have a custom LM which returns stream chunks in a custom format.
                    # We let those chunks pass through to the user to handle them as needed.
                    yield value

    if async_streaming:
        return async_streamer
    else:

        def sync_streamer(*args, **kwargs):
            output = async_streamer(*args, **kwargs)
            return apply_sync_streaming(output)

        return sync_streamer

```
:::