# Embeddings

**Source:** https://dspy.ai/api/tools/Embeddings
**Fetched:** 2025-08-24 17:10:33
**Status:** Success

---

[

](https://github.com/stanfordnlp/dspy/blob/main/docs/docs/api/tools/Embeddings.md)
# dspy.retrievers.Embeddings
## 
 dspy.Embeddings(corpus: list[str], embedder, k: int = 5, callbacks: list[Any] | None = None, cache: bool = False, brute_force_threshold: int = 20000, normalize: bool = True)
Source code in
dspy/retrievers/embeddings.py
```
[11](#__codelineno-0-11)
[12](#__codelineno-0-12)
[13](#__codelineno-0-13)
[14](#__codelineno-0-14)
[15](#__codelineno-0-15)
[16](#__codelineno-0-16)
[17](#__codelineno-0-17)
[18](#__codelineno-0-18)
[19](#__codelineno-0-19)
[20](#__codelineno-0-20)
[21](#__codelineno-0-21)
[22](#__codelineno-0-22)
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
```
```
def __init__(
    self,
    corpus: list[str],
    embedder,
    k: int = 5,
    callbacks: list[Any] | None = None,
    cache: bool = False,
    brute_force_threshold: int = 20_000,
    normalize: bool = True,
):
    assert cache is False, "Caching is not supported for embeddings-based retrievers"

    self.embedder = embedder
    self.k = k
    self.corpus = corpus
    self.normalize = normalize

    self.corpus_embeddings = self.embedder(self.corpus)
    self.corpus_embeddings = self._normalize(self.corpus_embeddings) if self.normalize else self.corpus_embeddings

    self.index = self._build_faiss() if len(corpus) >= brute_force_threshold else None
    self.search_fn = Unbatchify(self._batch_forward)

```
### Functions
#### 
 __call__(query: str)
Source code in
dspy/retrievers/embeddings.py
```
[34](#__codelineno-0-34)
[35](#__codelineno-0-35)
```
```
def __call__(self, query: str):
    return self.forward(query)

```
#### 
 forward(query: str)
Source code in
dspy/retrievers/embeddings.py
```
[37](#__codelineno-0-37)
[38](#__codelineno-0-38)
[39](#__codelineno-0-39)
[40](#__codelineno-0-40)
[41](#__codelineno-0-41)
```
```
def forward(self, query: str):
    import dspy

    passages, indices = self.search_fn(query)
    return dspy.Prediction(passages=passages, indices=indices)

```
:::