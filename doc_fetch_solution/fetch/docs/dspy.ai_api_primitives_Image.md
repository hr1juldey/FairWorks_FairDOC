# Image

**Source:** https://dspy.ai/api/primitives/Image
**Fetched:** 2025-08-24 17:10:34
**Status:** Success

---

[

](https://github.com/stanfordnlp/dspy/blob/main/docs/docs/api/primitives/Image.md)
# dspy.Image
## 
 dspy.Image
Bases: Type
### Functions
#### 
 description() -> str

classmethod
Description of the custom type
Source code in
dspy/adapters/types/base_type.py
```
[32](#__codelineno-0-32)
[33](#__codelineno-0-33)
[34](#__codelineno-0-34)
[35](#__codelineno-0-35)
```
```
@classmethod
def description(cls) -> str:
    """Description of the custom type"""
    return ""

```
#### 
 extract_custom_type_from_annotation(annotation)

classmethod
Extract all custom types from the annotation.
This is used to extract all custom types from the annotation of a field, while the annotation can
have arbitrary level of nesting. For example, we detect Tool is in list[dict[str, Tool]].
Source code in
dspy/adapters/types/base_type.py
```
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
```
```
@classmethod
def extract_custom_type_from_annotation(cls, annotation):
    """Extract all custom types from the annotation.

    This is used to extract all custom types from the annotation of a field, while the annotation can
    have arbitrary level of nesting. For example, we detect `Tool` is in `list[dict[str, Tool]]`.
    """
    # Direct match. Nested type like `list[dict[str, Event]]` passes `isinstance(annotation, type)` in python 3.10
    # while fails in python 3.11. To accommodate users using python 3.10, we need to capture the error and ignore it.
    try:
        if isinstance(annotation, type) and issubclass(annotation, cls):
            return [annotation]
    except TypeError:
        pass

    origin = get_origin(annotation)
    if origin is None:
        return []

    result = []
    # Recurse into all type args
    for arg in get_args(annotation):
        result.extend(cls.extract_custom_type_from_annotation(arg))

    return result

```
#### 
 format() -> list[dict[str, Any]] | str
Source code in
dspy/adapters/types/image.py
```
[31](#__codelineno-0-31)
[32](#__codelineno-0-32)
[33](#__codelineno-0-33)
[34](#__codelineno-0-34)
[35](#__codelineno-0-35)
[36](#__codelineno-0-36)
```
```
def format(self) -> list[dict[str, Any]] | str:
    try:
        image_url = encode_image(self.url)
    except Exception as e:
        raise ValueError(f"Failed to format image for DSPy: {e}")
    return [{"type": "image_url", "image_url": {"url": image_url}}]

```
#### 
 from_PIL(pil_image)

classmethod
Source code in
dspy/adapters/types/image.py
```
[62](#__codelineno-0-62)
[63](#__codelineno-0-63)
[64](#__codelineno-0-64)
```
```
@classmethod
def from_PIL(cls, pil_image):  # noqa: N802
    return cls(url=encode_image(pil_image))

```
#### 
 from_file(file_path: str)

classmethod
Source code in
dspy/adapters/types/image.py
```
[58](#__codelineno-0-58)
[59](#__codelineno-0-59)
[60](#__codelineno-0-60)
```
```
@classmethod
def from_file(cls, file_path: str):
    return cls(url=encode_image(file_path))

```
#### 
 from_url(url: str, download: bool = False)

classmethod
Source code in
dspy/adapters/types/image.py
```
[54](#__codelineno-0-54)
[55](#__codelineno-0-55)
[56](#__codelineno-0-56)
```
```
@classmethod
def from_url(cls, url: str, download: bool = False):
    return cls(url=encode_image(url, download))

```
#### 
 serialize_model()
Source code in
dspy/adapters/types/base_type.py
```
[63](#__codelineno-0-63)
[64](#__codelineno-0-64)
[65](#__codelineno-0-65)
[66](#__codelineno-0-66)
[67](#__codelineno-0-67)
[68](#__codelineno-0-68)
```
```
@pydantic.model_serializer()
def serialize_model(self):
    formatted = self.format()
    if isinstance(formatted, list):
        return f"{CUSTOM_TYPE_START_IDENTIFIER}{formatted}{CUSTOM_TYPE_END_IDENTIFIER}"
    return formatted

```
#### 
 validate_input(values)

classmethod
Source code in
dspy/adapters/types/image.py
```
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
```
```
@pydantic.model_validator(mode="before")
@classmethod
def validate_input(cls, values):
    # Allow the model to accept either a URL string or a dictionary with a single 'url' key
    if isinstance(values, str):
        # if a string, assume it's the URL directly and wrap it in a dict
        return {"url": values}
    elif isinstance(values, dict) and set(values.keys()) == {"url"}:
        # if it's a dict, ensure it has only the 'url' key
        return values
    elif isinstance(values, cls):
        return values.model_dump()
    else:
        raise TypeError("Expected a string URL or a dictionary with a key 'url'.")

```
:::