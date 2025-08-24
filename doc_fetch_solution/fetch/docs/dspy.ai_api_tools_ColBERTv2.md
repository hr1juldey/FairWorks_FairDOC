# Colbertv2

**Source:** https://dspy.ai/api/tools/ColBERTv2
**Fetched:** 2025-08-24 17:10:31
**Status:** Success

---

[

](https://github.com/stanfordnlp/dspy/blob/main/docs/docs/api/tools/ColBERTv2.md)
# dspy.ColBERTv2
## 
 dspy.ColBERTv2(url: str = 'http://0.0.0.0', port: str | int | None = None, post_requests: bool = False)
Wrapper for the ColBERTv2 Retrieval.
Source code in
dspy/dsp/colbertv2.py
```
[14](#__codelineno-0-14)
[15](#__codelineno-0-15)
[16](#__codelineno-0-16)
[17](#__codelineno-0-17)
[18](#__codelineno-0-18)
[19](#__codelineno-0-19)
[20](#__codelineno-0-20)
[21](#__codelineno-0-21)
```
```
def __init__(
    self,
    url: str = "http://0.0.0.0",
    port: str | int | None = None,
    post_requests: bool = False,
):
    self.post_requests = post_requests
    self.url = f"{url}:{port}" if port else url

```
### Functions
#### 
 __call__(query: str, k: int = 10, simplify: bool = False) -> list[str] | list[dotdict]
Source code in
dspy/dsp/colbertv2.py
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
```
```
def __call__(
    self,
    query: str,
    k: int = 10,
    simplify: bool = False,
) -> list[str] | list[dotdict]:
    if self.post_requests:
        topk: list[dict[str, Any]] = colbertv2_post_request(self.url, query, k)
    else:
        topk: list[dict[str, Any]] = colbertv2_get_request(self.url, query, k)

    if simplify:
        return [psg["long_text"] for psg in topk]

    return [dotdict(psg) for psg in topk]

```
:::