# Knn

**Source:** https://dspy.ai/api/optimizers/KNN
**Fetched:** 2025-08-24 17:10:33
**Status:** Success

---

[

](https://github.com/stanfordnlp/dspy/blob/main/docs/docs/api/optimizers/KNN.md)
# dspy.KNN
## 
 dspy.KNN(k: int, trainset: list[Example], vectorizer: Embedder)
A k-nearest neighbors retriever that finds similar examples from a training set.
Parameters:
Name
Type
Description
Default
k
int
Number of nearest neighbors to retrieve
required
trainset
list
[
[Example](../../primitives/Example/#dspy.Example)
]
List of training examples to search through
required
vectorizer
[Embedder](../../models/Embedder/#dspy.Embedder)
The Embedder to use for vectorization
required
Example
```
import dspy
from sentence_transformers import SentenceTransformer

# Create a training dataset with examples
trainset = [
    dspy.Example(input="hello", output="world"),
    # ... more examples ...
]

# Initialize KNN with a sentence transformer model
knn = KNN(
    k=3,
    trainset=trainset,
    vectorizer=dspy.Embedder(SentenceTransformer("all-MiniLM-L6-v2").encode)
)

# Find similar examples
similar_examples = knn(input="hello")

```
Source code in
dspy/predict/knn.py
```
[ 8](#__codelineno-0-8)
[ 9](#__codelineno-0-9)
[10](#__codelineno-0-10)
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
```
```
def __init__(self, k: int, trainset: list[Example], vectorizer: Embedder):
    """
    A k-nearest neighbors retriever that finds similar examples from a training set.

    Args:
        k: Number of nearest neighbors to retrieve
        trainset: List of training examples to search through
        vectorizer: The `Embedder` to use for vectorization

    Example:
        ```python
        import dspy
        from sentence_transformers import SentenceTransformer

        # Create a training dataset with examples
        trainset = [
            dspy.Example(input="hello", output="world"),
            # ... more examples ...
        ]

        # Initialize KNN with a sentence transformer model
        knn = KNN(
            k=3,
            trainset=trainset,
            vectorizer=dspy.Embedder(SentenceTransformer("all-MiniLM-L6-v2").encode)
        )

        # Find similar examples
        similar_examples = knn(input="hello")
        ```
    """
    self.k = k
    self.trainset = trainset
    self.embedding = vectorizer
    trainset_casted_to_vectorize = [
        " | ".join([f"{key}: {value}" for key, value in example.items() if key in example._input_keys])
        for example in self.trainset
    ]
    self.trainset_vectors = self.embedding(trainset_casted_to_vectorize).astype(np.float32)

```
### Functions
#### 
 __call__(**kwargs) -> list
Source code in
dspy/predict/knn.py
```
[48](#__codelineno-0-48)
[49](#__codelineno-0-49)
[50](#__codelineno-0-50)
[51](#__codelineno-0-51)
[52](#__codelineno-0-52)
```
```
def __call__(self, **kwargs) -> list:
    input_example_vector = self.embedding([" | ".join([f"{key}: {val}" for key, val in kwargs.items()])])
    scores = np.dot(self.trainset_vectors, input_example_vector.T).squeeze()
    nearest_samples_idxs = scores.argsort()[-self.k :][::-1]
    return [self.trainset[cur_idx] for cur_idx in nearest_samples_idxs]

```
:::