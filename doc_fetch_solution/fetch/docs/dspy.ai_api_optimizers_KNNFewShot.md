# Knnfewshot

**Source:** https://dspy.ai/api/optimizers/KNNFewShot
**Fetched:** 2025-08-24 17:10:32
**Status:** Success

---

[

](https://github.com/stanfordnlp/dspy/blob/main/docs/docs/api/optimizers/KNNFewShot.md)
# dspy.KNNFewShot
## 
 dspy.KNNFewShot(k: int, trainset: list[Example], vectorizer: Embedder, **few_shot_bootstrap_args: dict[str, Any])
Bases: Teleprompter
KNNFewShot is an optimizer that uses an in-memory KNN retriever to find the k nearest neighbors
in a trainset at test time. For each input example in a forward call, it identifies the k most
similar examples from the trainset and attaches them as demonstrations to the student module.
Parameters:
Name
Type
Description
Default
k
int
The number of nearest neighbors to attach to the student model.
required
trainset
list
[
[Example](../../primitives/Example/#dspy.Example)
]
The training set to use for few-shot prompting.
required
vectorizer
[Embedder](../../models/Embedder/#dspy.Embedder)
The Embedder to use for vectorization
required
**few_shot_bootstrap_args
dict
[
str
,
Any
]
Additional arguments for the BootstrapFewShot optimizer.
{}
Example
```
import dspy
from sentence_transformers import SentenceTransformer

# Define a QA module with chain of thought
qa = dspy.ChainOfThought("question -> answer")

# Create a training dataset with examples
trainset = [
    dspy.Example(question="What is the capital of France?", answer="Paris").with_inputs("question"),
    # ... more examples ...
]

# Initialize KNNFewShot with a sentence transformer model
knn_few_shot = KNNFewShot(
    k=3,
    trainset=trainset,
    vectorizer=dspy.Embedder(SentenceTransformer("all-MiniLM-L6-v2").encode)
)

# Compile the QA module with few-shot learning
compiled_qa = knn_few_shot.compile(qa)

# Use the compiled module
result = compiled_qa("What is the capital of Belgium?")

```
Source code in
dspy/teleprompt/knn_fewshot.py
```
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
[47](#__codelineno-0-47)
[48](#__codelineno-0-48)
[49](#__codelineno-0-49)
[50](#__codelineno-0-50)
[51](#__codelineno-0-51)
[52](#__codelineno-0-52)
[53](#__codelineno-0-53)
```
```
def __init__(self, k: int, trainset: list[Example], vectorizer: Embedder, **few_shot_bootstrap_args: dict[str, Any]):
    """
    KNNFewShot is an optimizer that uses an in-memory KNN retriever to find the k nearest neighbors
    in a trainset at test time. For each input example in a forward call, it identifies the k most
    similar examples from the trainset and attaches them as demonstrations to the student module.

    Args:
        k: The number of nearest neighbors to attach to the student model.
        trainset: The training set to use for few-shot prompting.
        vectorizer: The `Embedder` to use for vectorization
        **few_shot_bootstrap_args: Additional arguments for the `BootstrapFewShot` optimizer.

    Example:
        ```python
        import dspy
        from sentence_transformers import SentenceTransformer

        # Define a QA module with chain of thought
        qa = dspy.ChainOfThought("question -> answer")

        # Create a training dataset with examples
        trainset = [
            dspy.Example(question="What is the capital of France?", answer="Paris").with_inputs("question"),
            # ... more examples ...
        ]

        # Initialize KNNFewShot with a sentence transformer model
        knn_few_shot = KNNFewShot(
            k=3,
            trainset=trainset,
            vectorizer=dspy.Embedder(SentenceTransformer("all-MiniLM-L6-v2").encode)
        )

        # Compile the QA module with few-shot learning
        compiled_qa = knn_few_shot.compile(qa)

        # Use the compiled module
        result = compiled_qa("What is the capital of Belgium?")
        ```
    """
    self.KNN = KNN(k, trainset, vectorizer=vectorizer)
    self.few_shot_bootstrap_args = few_shot_bootstrap_args

```
### Functions
#### 
 compile(student, *, teacher=None)
Source code in
dspy/teleprompt/knn_fewshot.py
```
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
```
```
def compile(self, student, *, teacher=None):
    student_copy = student.reset_copy()

    def forward_pass(_, **kwargs):
        knn_trainset = self.KNN(**kwargs)
        few_shot_bootstrap = BootstrapFewShot(**self.few_shot_bootstrap_args)
        compiled_program = few_shot_bootstrap.compile(
            student,
            teacher=teacher,
            trainset=knn_trainset,
        )
        return compiled_program(**kwargs)

    student_copy.forward = types.MethodType(forward_pass, student_copy)
    return student_copy

```
#### 
 get_params() -> dict[str, Any]
Get the parameters of the teleprompter.
Returns:
Type
Description
dict
[
str
,
Any
]
The parameters of the teleprompter.
Source code in
dspy/teleprompt/teleprompt.py
```
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
def get_params(self) -> dict[str, Any]:
    """
    Get the parameters of the teleprompter.

    Returns:
        The parameters of the teleprompter.
    """
    return self.__dict__

```
:::