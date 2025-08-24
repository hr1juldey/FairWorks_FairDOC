# Example

**Source:** https://dspy.ai/api/primitives/Example
**Fetched:** 2025-08-24 17:10:36
**Status:** Success

---

[

](https://github.com/stanfordnlp/dspy/blob/main/docs/docs/api/primitives/Example.md)
# dspy.Example
## 
 dspy.Example(base=None, **kwargs)
Source code in
dspy/primitives/example.py
```
[ 2](#__codelineno-0-2)
[ 3](#__codelineno-0-3)
[ 4](#__codelineno-0-4)
[ 5](#__codelineno-0-5)
[ 6](#__codelineno-0-6)
[ 7](#__codelineno-0-7)
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
```
```
def __init__(self, base=None, **kwargs):
    # Internal storage and other attributes
    self._store = {}
    self._demos = []
    self._input_keys = None

    # Initialize from a base Example if provided
    if base and isinstance(base, type(self)):
        self._store = base._store.copy()

    # Initialize from a dict if provided
    elif base and isinstance(base, dict):
        self._store = base.copy()

    # Update with provided kwargs
    self._store.update(kwargs)

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