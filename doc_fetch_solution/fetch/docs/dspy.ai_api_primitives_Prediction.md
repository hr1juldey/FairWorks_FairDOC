# Prediction

**Source:** https://dspy.ai/api/primitives/Prediction
**Fetched:** 2025-08-24 17:10:32
**Status:** Success

---

[

](https://github.com/stanfordnlp/dspy/blob/main/docs/docs/api/primitives/Prediction.md)
# dspy.Prediction
## 
 dspy.Prediction(*args, **kwargs)
Bases: Example
A prediction object that contains the output of a DSPy module.
Prediction inherits from Example.
To allow feedback-augmented scores, Prediction supports comparison operations
(<, >, <=, >=) for Predictions with a score field. The comparison operations
compare the 'score' values as floats. For equality comparison, Predictions are equal
if their underlying data stores are equal (inherited from Example).
Arithmetic operations (+, /, etc.) are also supported for Predictions with a 'score'
field, operating on the score value.
Source code in
dspy/primitives/prediction.py
```
[18](#__codelineno-0-18)
[19](#__codelineno-0-19)
[20](#__codelineno-0-20)
[21](#__codelineno-0-21)
[22](#__codelineno-0-22)
[23](#__codelineno-0-23)
[24](#__codelineno-0-24)
[25](#__codelineno-0-25)
```
```
def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)

    del self._demos
    del self._input_keys

    self._completions = None
    self._lm_usage = None

```
### Functions
#### 
 copy(**kwargs)
Source code in
dspy/primitives/example.py
```
[98](#__codelineno-0-98)
[99](#__codelineno-0-99)
```
```
def copy(self, **kwargs):
    return type(self)(base=self, **kwargs)

```
#### 
 from_completions(list_or_dict, signature=None)

classmethod
Source code in
dspy/primitives/prediction.py
```
[33](#__codelineno-0-33)
[34](#__codelineno-0-34)
[35](#__codelineno-0-35)
[36](#__codelineno-0-36)
[37](#__codelineno-0-37)
[38](#__codelineno-0-38)
[39](#__codelineno-0-39)
```
```
@classmethod
def from_completions(cls, list_or_dict, signature=None):
    obj = cls()
    obj._completions = Completions(list_or_dict, signature=signature)
    obj._store = {k: v[0] for k, v in obj._completions.items()}

    return obj

```
#### 
 get(key, default=None)
Source code in
dspy/primitives/example.py
```
[70](#__codelineno-0-70)
[71](#__codelineno-0-71)
```
```
def get(self, key, default=None):
    return self._store.get(key, default)

```
#### 
 get_lm_usage()
Source code in
dspy/primitives/prediction.py
```
[27](#__codelineno-0-27)
[28](#__codelineno-0-28)
```
```
def get_lm_usage(self):
    return self._lm_usage

```
#### 
 inputs()
Source code in
dspy/primitives/example.py
```
[78](#__codelineno-0-78)
[79](#__codelineno-0-79)
[80](#__codelineno-0-80)
[81](#__codelineno-0-81)
[82](#__codelineno-0-82)
[83](#__codelineno-0-83)
[84](#__codelineno-0-84)
[85](#__codelineno-0-85)
[86](#__codelineno-0-86)
[87](#__codelineno-0-87)
```
```
def inputs(self):
    if self._input_keys is None:
        raise ValueError("Inputs have not been set for this example. Use `example.with_inputs()` to set them.")

    # return items that are in input_keys
    d = {key: self._store[key] for key in self._store if key in self._input_keys}
    # return type(self)(d)
    new_instance = type(self)(base=d)
    new_instance._input_keys = self._input_keys  # Preserve input_keys in new instance
    return new_instance

```
#### 
 items(include_dspy=False)
Source code in
dspy/primitives/example.py
```
[67](#__codelineno-0-67)
[68](#__codelineno-0-68)
```
```
def items(self, include_dspy=False):
    return [(k, v) for k, v in self._store.items() if not k.startswith("dspy_") or include_dspy]

```
#### 
 keys(include_dspy=False)
Source code in
dspy/primitives/example.py
```
[61](#__codelineno-0-61)
[62](#__codelineno-0-62)
```
```
def keys(self, include_dspy=False):
    return [k for k in self._store.keys() if not k.startswith("dspy_") or include_dspy]

```
#### 
 labels()
Source code in
dspy/primitives/example.py
```
[89](#__codelineno-0-89)
[90](#__codelineno-0-90)
[91](#__codelineno-0-91)
[92](#__codelineno-0-92)
[93](#__codelineno-0-93)
```
```
def labels(self):
    # return items that are NOT in input_keys
    input_keys = self.inputs().keys()
    d = {key: self._store[key] for key in self._store if key not in input_keys}
    return type(self)(d)

```
#### 
 set_lm_usage(value)
Source code in
dspy/primitives/prediction.py
```
[30](#__codelineno-0-30)
[31](#__codelineno-0-31)
```
```
def set_lm_usage(self, value):
    self._lm_usage = value

```
#### 
 toDict()
Source code in
dspy/primitives/example.py
```
[107](#__codelineno-0-107)
[108](#__codelineno-0-108)
```
```
def toDict(self):  # noqa: N802
    return self._store.copy()

```
#### 
 values(include_dspy=False)
Source code in
dspy/primitives/example.py
```
[64](#__codelineno-0-64)
[65](#__codelineno-0-65)
```
```
def values(self, include_dspy=False):
    return [v for k, v in self._store.items() if not k.startswith("dspy_") or include_dspy]

```
#### 
 with_inputs(*keys)
Source code in
dspy/primitives/example.py
```
[73](#__codelineno-0-73)
[74](#__codelineno-0-74)
[75](#__codelineno-0-75)
[76](#__codelineno-0-76)
```
```
def with_inputs(self, *keys):
    copied = self.copy()
    copied._input_keys = set(keys)
    return copied

```
#### 
 without(*keys)
Source code in
dspy/primitives/example.py
```
[101](#__codelineno-0-101)
[102](#__codelineno-0-102)
[103](#__codelineno-0-103)
[104](#__codelineno-0-104)
[105](#__codelineno-0-105)
```
```
def without(self, *keys):
    copied = self.copy()
    for key in keys:
        del copied[key]
    return copied

```
:::