# Streamlistener

**Source:** https://dspy.ai/api/utils/StreamListener
**Fetched:** 2025-08-24 17:10:37
**Status:** Success

---

[

](https://github.com/stanfordnlp/dspy/blob/main/docs/docs/api/utils/StreamListener.md)
# dspy.streaming.StreamListener
## 
 dspy.streaming.StreamListener(signature_field_name: str, predict: Any = None, predict_name: str | None = None, allow_reuse: bool = False)
Class that listens to the stream to capture the streeaming of a specific output field of a predictor.
Parameters:
Name
Type
Description
Default
signature_field_name
str
The name of the field to listen to.
required
predict
Any
The predictor to listen to. If None, when calling streamify() it will automatically look for
the predictor that has the signature_field_name in its signature.
None
predict_name
str
| None
The name of the predictor to listen to. If None, when calling streamify() it will
automatically look for the predictor that has the signature_field_name in its signature.
None
allow_reuse
bool
If True, the stream listener can be reused for multiple streams. Please note that this could
hurt the performance because the same stream chunk is sent to multiple listeners.
False
Source code in
dspy/streaming/streaming_listener.py
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
def __init__(
    self,
    signature_field_name: str,
    predict: Any = None,
    predict_name: str | None = None,
    allow_reuse: bool = False,
):
    """
    Args:
        signature_field_name: The name of the field to listen to.
        predict: The predictor to listen to. If None, when calling `streamify()` it will automatically look for
            the predictor that has the `signature_field_name` in its signature.
        predict_name: The name of the predictor to listen to. If None, when calling `streamify()` it will
            automatically look for the predictor that has the `signature_field_name` in its signature.
        allow_reuse: If True, the stream listener can be reused for multiple streams. Please note that this could
            hurt the performance because the same stream chunk is sent to multiple listeners.
    """
    self.signature_field_name = signature_field_name
    self.predict = predict
    self.predict_name = predict_name

    self.field_start_queue = []
    self.field_end_queue = Queue()
    self.stream_start = False
    self.stream_end = False
    self.cache_hit = False
    self.allow_reuse = allow_reuse

    self.adapter_identifiers = {
        "ChatAdapter": {
            "start_identifier": f"[[ ## {self.signature_field_name} ## ]]",
            "end_identifier": re.compile(r"\[\[ ## (\w+) ## \]\]"),
            "start_indicator": "[",
        },
        "JSONAdapter": {
            "start_identifier": f'"{self.signature_field_name}":',
            "end_identifier": re.compile(r"\w*\"(,|\s*})"),
            "start_indicator": '"',
        },
        "XMLAdapter": {
            "start_identifier": f"<{self.signature_field_name}>",
            "end_identifier": re.compile(rf"</{self.signature_field_name}>"),
            "start_indicator": "<",
        },
    }

```
### Functions
#### 
 flush() -> str
Flush all tokens in the field end queue.
This method is called to flush out the last a few tokens when the stream is ended. These tokens
are in the buffer because we don't directly yield the tokens received by the stream listener
with the purpose to not yield the end_identifier tokens, e.g., "[[ ## ... ## ]]" for ChatAdapter.
Source code in
dspy/streaming/streaming_listener.py
```
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
```
```
def flush(self) -> str:
    """Flush all tokens in the field end queue.

    This method is called to flush out the last a few tokens when the stream is ended. These tokens
    are in the buffer because we don't directly yield the tokens received by the stream listener
    with the purpose to not yield the end_identifier tokens, e.g., "[[ ## ... ## ]]" for ChatAdapter.
    """
    last_tokens = "".join(self.field_end_queue.queue)
    self.field_end_queue = Queue()
    if isinstance(settings.adapter, JSONAdapter):
        match = re.search(r'",|"\s*}', last_tokens)
        if match:
            boundary_index = match.start()
        else:
            boundary_index = len(last_tokens)
        return last_tokens[:boundary_index]
    elif isinstance(settings.adapter, XMLAdapter):
        boundary_index = last_tokens.find(f"</{self.signature_field_name}>")
        if boundary_index == -1:
            boundary_index = len(last_tokens)
        return last_tokens[:boundary_index]
    elif isinstance(settings.adapter, ChatAdapter) or settings.adapter is None:
        boundary_index = last_tokens.find("[[")
        return last_tokens[:boundary_index]
    else:
        raise ValueError(
            f"Unsupported adapter for streaming: {settings.adapter}, please use one of the following adapters: "
            f"{', '.join([a.__name__ for a in ADAPTER_SUPPORT_STREAMING])}"
        )

```
#### 
 receive(chunk: ModelResponseStream)
Source code in
dspy/streaming/streaming_listener.py
```
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
```
```
def receive(self, chunk: ModelResponseStream):
    adapter_name = settings.adapter.__class__.__name__ if settings.adapter else "ChatAdapter"
    if adapter_name not in self.adapter_identifiers:
        raise ValueError(
            f"Unsupported adapter for streaming: {adapter_name}, please use one of the following adapters: "
            f"{', '.join([a.__name__ for a in ADAPTER_SUPPORT_STREAMING])}"
        )
    start_identifier = self.adapter_identifiers[adapter_name]["start_identifier"]
    end_identifier = self.adapter_identifiers[adapter_name]["end_identifier"]
    start_indicator = self.adapter_identifiers[adapter_name]["start_indicator"]

    if self.stream_end:
        if self.allow_reuse:
            # Clear up the state for the next stream.
            self.stream_end = False
            self.cache_hit = False
            self.field_start_queue = []
            self.field_end_queue = Queue()
            self.stream_start = False
        else:
            return

    try:
        chunk_message = chunk.choices[0].delta.content
        if chunk_message is None:
            return
    except Exception:
        return

    if chunk_message and start_identifier in chunk_message:
        # If the cache is hit, the chunk_message could be the full response. When it happens we can
        # directly end the stream listening. In some models like gemini, each stream chunk can be multiple
        # tokens, so it's possible that response only has one chunk, we also fall back to this logic.
        message_after_start_identifier = chunk_message[
            chunk_message.find(start_identifier) + len(start_identifier) :
        ]
        if re.search(end_identifier, message_after_start_identifier):
            self.cache_hit = True
            self.stream_start = True
            self.stream_end = True
            return

    if len(self.field_start_queue) == 0 and not self.stream_start and start_indicator in chunk_message:
        # We look for the pattern of start_identifier, i.e., "[[ ## {self.signature_field_name} ## ]]" for
        # ChatAdapter to identify the start of the stream of our target field. Once the start_indicator, i.e., "[["
        # for ChatAdapter, is found, we start checking the next tokens
        self.field_start_queue.append(chunk_message)
        return

    if len(self.field_start_queue) > 0 and not self.stream_start:
        # We keep appending the tokens to the queue until we have a full identifier or the concanated
        # tokens no longer match our expected identifier.
        self.field_start_queue.append(chunk_message)
        concat_message = "".join(self.field_start_queue)

        if start_identifier in concat_message:
            # We have a full identifier, we can start the stream.
            self.stream_start = True
            self.field_start_queue = []
            # Keep the part after the start_identifier from the concat_message, we need to write it to the buffer.
            value_start_index = concat_message.find(start_identifier) + len(start_identifier)
            chunk_message = concat_message[value_start_index:].lstrip()
            if isinstance(settings.adapter, JSONAdapter) and chunk_message.startswith('"'):
                # For JSONAdapter, we need to remove the leading ". We cannot do this with the start_identifier
                # because there could be a few splitters between ':' and '"', e.g., '"name": "value"'.
                chunk_message = chunk_message[1:]

        elif self._buffered_message_end_with_start_identifier(concat_message.strip(), start_identifier):
            # If the buffered message ends with part of the start_identifier, we keep looking for the
            # start_identifier from the token stream.
            return
        else:
            # Doesn't match the expected identifier, reset the queue.
            self.field_start_queue = []
            return

    if self.stream_start:
        # The stream is started, we keep returning the token until we see the start of the next field.
        token = None
        self.field_end_queue.put(chunk_message)
        if self.field_end_queue.qsize() > 10:
            # We keep the last 10 tokens in the buffer to check if they form a valid identifier for end_identifier,
            # i.e., "[[ ## {next_field_name} ## ]]" for ChatAdapter to identify the end of the current field.
            # In most cases 10 tokens are enough to cover the end_identifier for all adapters.
            token = self.field_end_queue.get()
        concat_message = "".join(self.field_end_queue.queue).strip()
        if re.search(end_identifier, concat_message):
            # The next field is identified, we can end the stream and flush out all tokens in the buffer.
            self.stream_end = True
            last_token = self.flush()
            token = token + last_token if token else last_token
            token = token.rstrip()  # Remove the trailing \n\n

        if token:
            return StreamResponse(
                self.predict_name,
                self.signature_field_name,
                token,
                is_last_chunk=self.stream_end,
            )

```
:::