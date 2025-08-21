# Additional Fixes for DSPy Medical Agent Coroutine Reuse Issues

## Problem: Coroutine Reuse in Medical Agent

### Issue
The error `RuntimeError: cannot reuse already awaited coroutine` indicates that the same coroutine object is being awaited multiple times.

### Root Cause
In the `EvaluationProgram.forward` method, a coroutine object is created once and then potentially used multiple times across different execution paths, which is not allowed in Python's async model.

### Location
File: `src/app2/services/dspy/evaluation_optimizer.py`
Lines: Around 130-200

### Fix
Ensure that a fresh coroutine is created each time it's needed, rather than reusing the same coroutine object.

**Find:**

```python
# Decide whether process_turn is sync or async
is_coro_fn = inspect.iscoroutinefunction(self.medical_agent.process_turn)

if not is_coro_fn:
    # synchronous handler -> call directly
    result = self.medical_agent.process_turn(
        symptoms=user_message,
        nice_context=protocols
    )
else:
    # coroutine handler
    coro = self.medical_agent.process_turn(
        symptoms=user_message,
        nice_context=protocols
    )

    if not in_running_loop:
        # No running loop in this thread: safe to use asyncio.run
        result = asyncio.run(coro)
    else:
        # There is a running loop. Try to schedule safely.
        # Best option: if the running loop is in another thread, use run_coroutine_threadsafe
        try:
            future = asyncio.run_coroutine_threadsafe(coro, loop)
            # block until done or timeout
            result = future.result(timeout=wait_timeout)
        except Exception as exc_threadsafe:
            # Could be that the loop is the same thread (cannot use run_coroutine_threadsafe),
            # or other scheduling issues. Fall back to scheduling a task and polling gently.
            try:
                # Schedule a task on the running loop
                future_task = asyncio.ensure_future(coro, loop=loop)
            except Exception:
                # As a last resort, re-raise the original exception to be caught by outer handler
                raise exc_threadsafe

            # Poll with a small sleep to avoid CPU spin (best-effort)
            start = time.time()
            while not future_task.done():
                if time.time() - start > wait_timeout:
                    raise FuturesTimeoutError("Timeout waiting for coroutine to complete")
                # yield thread to event loop / other threads
                time.sleep(0.001)

            # Get result or exception
            result = future_task.result()
```

**Replace with:**

```python
# Helper function to create a fresh coroutine each time
def _create_fresh_coroutine():
    """Create a fresh coroutine to avoid reuse issues"""
    return self.medical_agent.process_turn(
        symptoms=user_message,
        nice_context=protocols
    )

# Decide whether process_turn is sync or async
is_coro_fn = inspect.iscoroutinefunction(self.medical_agent.process_turn)

if not is_coro_fn:
    # synchronous handler -> call directly
    result = self.medical_agent.process_turn(
        symptoms=user_message,
        nice_context=protocols
    )
else:
    # coroutine handler - create fresh coroutine each time to avoid reuse
    if not in_running_loop:
        # No running loop in this thread: safe to use asyncio.run
        # Create fresh coroutine for asyncio.run
        coro = _create_fresh_coroutine()
        result = asyncio.run(coro)
    else:
        # There is a running loop. Try to schedule safely.
        # Create fresh coroutine for run_coroutine_threadsafe
        coro = _create_fresh_coroutine()
        try:
            future = asyncio.run_coroutine_threadsafe(coro, loop)
            # block until done or timeout
            result = future.result(timeout=wait_timeout)
        except Exception as exc_threadsafe:
            # Could be that the loop is the same thread (cannot use run_coroutine_threadsafe),
            # or other scheduling issues. Fall back to scheduling a task and polling gently.
            try:
                # Create fresh coroutine for ensure_future
                coro2 = _create_fresh_coroutine()
                # Schedule a task on the running loop
                future_task = asyncio.ensure_future(coro2, loop=loop)
            except Exception:
                # As a last resort, re-raise the original exception to be caught by outer handler
                raise exc_threadsafe

            # Poll with a small sleep to avoid CPU spin (best-effort)
            start = time.time()
            while not future_task.done():
                if time.time() - start > wait_timeout:
                    raise FuturesTimeoutError("Timeout waiting for coroutine to complete")
                # yield thread to event loop / other threads
                time.sleep(0.001)

            # Get result or exception
            result = future_task.result()
```

## Problem: MIPROv2 Parameter Issue

### Issue
The error `TypeError: MIPROv2.__init__() got an unexpected keyword argument 'num_trials'` indicates incorrect parameter usage.

### Location
File: `src/app2/services/dspy/evaluation_optimizer.py`
Lines: Around 250-300

### Fix
Remove the incorrect `num_trials` parameter from MIPROv2 initialization.

**Find:**
```python
elif optimizer_type == "mipro":
    optimizer = MIPROv2(
        metric=self._medical_accuracy_metric,
        auto=None,  # Must set to None to use custom parameters
        num_candidates=4,
        num_trials=12,  # ADD: Required parameter
        init_temperature=0.1
    )
```

**Replace with:**
```python
elif optimizer_type == "mipro":
    optimizer = MIPROv2(
        metric=self._medical_accuracy_metric,
        auto=None,  # Must set to None to use custom parameters
        num_candidates=4,
        init_temperature=0.1
    )
```

## Problem: COPRO Parameter Issue

### Issue
The error shows that COPRO is being called with an incorrect parameter.

### Fix
Remove the incorrect `num_trials` parameter from COPRO initialization.

**Find:**
```python
elif optimizer_type == "copro":
    optimizer = COPRO(
        metric=self._medical_accuracy_metric,
        breadth=3,
        depth=2,
        num_trials=10,  # ADD: Required parameter
    )
```

**Replace with:**
```python
elif optimizer_type == "copro":
    optimizer = COPRO(
        metric=self._medical_accuracy_metric,
        breadth=3,
        depth=2
    )
```