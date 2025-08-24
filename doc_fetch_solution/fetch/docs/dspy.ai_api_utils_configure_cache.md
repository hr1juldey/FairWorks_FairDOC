# Cache

**Source:** https://dspy.ai/api/utils/configure/cache
**Fetched:** 2025-08-24 17:10:34
**Status:** Success

---

[

](https://github.com/stanfordnlp/dspy/blob/main/docs/docs/api/utils/configure_cache.md)
# dspy.configure_cache
## 
 dspy.configure_cache(enable_disk_cache: bool | None = True, enable_memory_cache: bool | None = True, disk_cache_dir: str | None = DISK_CACHE_DIR, disk_size_limit_bytes: int | None = DISK_CACHE_LIMIT, memory_max_entries: int | None = 1000000, enable_litellm_cache: bool = False)
Configure the cache for DSPy.
Parameters:
Name
Type
Description
Default
enable_disk_cache
bool
| None
Whether to enable on-disk cache.
True
enable_memory_cache
bool
| None
Whether to enable in-memory cache.
True
disk_cache_dir
str
| None
The directory to store the on-disk cache.
DISK_CACHE_DIR
disk_size_limit_bytes
int
| None
The size limit of the on-disk cache.
DISK_CACHE_LIMIT
memory_max_entries
int
| None
The maximum number of entries in the in-memory cache.
1000000
enable_litellm_cache
bool
Whether to enable LiteLLM cache.
False
Source code in
dspy/clients/__init__.py
```
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
[68](#__codelineno-0-68)
[69](#__codelineno-0-69)
[70](#__codelineno-0-70)
[71](#__codelineno-0-71)
[72](#__codelineno-0-72)
[73](#__codelineno-0-73)
[74](#__codelineno-0-74)
```
```
def configure_cache(
    enable_disk_cache: bool | None = True,
    enable_memory_cache: bool | None = True,
    disk_cache_dir: str | None = DISK_CACHE_DIR,
    disk_size_limit_bytes: int | None = DISK_CACHE_LIMIT,
    memory_max_entries: int | None = 1000000,
    enable_litellm_cache: bool = False,
):
    """Configure the cache for DSPy.

    Args:
        enable_disk_cache: Whether to enable on-disk cache.
        enable_memory_cache: Whether to enable in-memory cache.
        disk_cache_dir: The directory to store the on-disk cache.
        disk_size_limit_bytes: The size limit of the on-disk cache.
        memory_max_entries: The maximum number of entries in the in-memory cache.
        enable_litellm_cache: Whether to enable LiteLLM cache.
    """
    if enable_disk_cache and enable_litellm_cache:
        raise ValueError(
            "Cannot enable both LiteLLM and DSPy on-disk cache, please set at most one of `enable_disk_cache` or "
            "`enable_litellm_cache` to True."
        )

    if enable_litellm_cache:
        try:
            litellm.cache = LitellmCache(disk_cache_dir=DISK_CACHE_DIR, type="disk")

            if litellm.cache.cache.disk_cache.size_limit != DISK_CACHE_LIMIT:
                litellm.cache.cache.disk_cache.reset("size_limit", DISK_CACHE_LIMIT)
        except Exception as e:
            # It's possible that users don't have the write permissions to the cache directory.
            # In that case, we'll just disable the cache.
            logger.warning("Failed to initialize LiteLLM cache: %s", e)
            litellm.cache = None
    else:
        litellm.cache = None

    import dspy

    dspy.cache = Cache(
        enable_disk_cache,
        enable_memory_cache,
        disk_cache_dir,
        disk_size_limit_bytes,
        memory_max_entries,
    )

```
:::