# Pythoninterpreter

**Source:** https://dspy.ai/api/tools/PythonInterpreter
**Fetched:** 2025-08-24 17:10:37
**Status:** Success

---

[

](https://github.com/stanfordnlp/dspy/blob/main/docs/docs/api/tools/PythonInterpreter.md)
# dspy.PythonInterpreter
## 
 dspy.PythonInterpreter(deno_command: list[str] | None = None, enable_read_paths: list[PathLike | str] | None = None, enable_write_paths: list[PathLike | str] | None = None, enable_env_vars: list[str] | None = None, enable_network_access: list[str] | None = None, sync_files: bool = True)
PythonInterpreter that runs code in a sandboxed environment using Deno and Pyodide.
Prerequisites:
- Deno (https://docs.deno.com/runtime/getting_started/installation/).
Example Usage:
code_string = "print('Hello'); 1 + 2"
with PythonInterpreter() as interp:
    output = interp(code_string) # If final statement is non-None, prints the numeric result, else prints captured output
Parameters:
Name
Type
Description
Default
deno_command
list
[
str
] | None
command list to launch Deno.
None
enable_read_paths
list
[
PathLike
|
str
] | None
Files or directories to allow reading from in the sandbox.
None
enable_write_paths
list
[
PathLike
|
str
] | None
Files or directories to allow writing to in the sandbox.
None
enable_env_vars
list
[
str
] | None
Environment variable names to allow in the sandbox.
None
enable_network_access
list
[
str
] | None
Domains or IPs to allow network access in the sandbox.
None
sync_files
bool
If set, syncs changes within the sandbox back to original files after execution.
True
Source code in
dspy/primitives/python_interpreter.py
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
[75](#__codelineno-0-75)
[76](#__codelineno-0-76)
[77](#__codelineno-0-77)
[78](#__codelineno-0-78)
```
```
def __init__(
    self,
    deno_command: list[str] | None = None,
    enable_read_paths: list[PathLike | str] | None = None,
    enable_write_paths: list[PathLike | str] | None = None,
    enable_env_vars: list[str] | None = None,
    enable_network_access: list[str] | None = None,
    sync_files: bool = True,
) -> None:
    """
    Args:
        deno_command: command list to launch Deno.
        enable_read_paths: Files or directories to allow reading from in the sandbox.
        enable_write_paths: Files or directories to allow writing to in the sandbox.
        enable_env_vars: Environment variable names to allow in the sandbox.
        enable_network_access: Domains or IPs to allow network access in the sandbox.
        sync_files: If set, syncs changes within the sandbox back to original files after execution.
    """
    if isinstance(deno_command, dict):
        deno_command = None  # no-op, just a guard in case someone passes a dict

    self.enable_read_paths = enable_read_paths or []
    self.enable_write_paths = enable_write_paths or []
    self.enable_env_vars = enable_env_vars or []
    self.enable_network_access = enable_network_access or []
    self.sync_files = sync_files
    # TODO later on add enable_run (--allow-run) by proxying subprocess.run through Deno.run() to fix 'emscripten does not support processes' error

    if deno_command:
        self.deno_command = list(deno_command)
    else:
        args = ["deno", "run", "--allow-read"]
        self._env_arg  = ""
        if self.enable_env_vars:
            user_vars = [str(v).strip() for v in self.enable_env_vars]
            args.append("--allow-env=" + ",".join(user_vars))
            self._env_arg = ",".join(user_vars)
        if self.enable_network_access:
            args.append(f"--allow-net={','.join(str(x) for x in self.enable_network_access)}")
        if self.enable_write_paths:
            args.append(f"--allow-write={','.join(str(x) for x in self.enable_write_paths)}")

        args.append(self._get_runner_path())

        # For runner.js to load in env vars
        if self._env_arg:
            args.append(self._env_arg)
        self.deno_command = args

    self.deno_process = None
    self._mounted_files = False

```
### Functions
#### 
 __call__(code: str, variables: dict[str, Any] | None = None) -> Any
Source code in
dspy/primitives/python_interpreter.py
```
[232](#__codelineno-0-232)
[233](#__codelineno-0-233)
[234](#__codelineno-0-234)
[235](#__codelineno-0-235)
[236](#__codelineno-0-236)
[237](#__codelineno-0-237)
```
```
def __call__(
    self,
    code: str,
    variables: dict[str, Any] | None = None,
) -> Any:
    return self.execute(code, variables)

```
#### 
 execute(code: str, variables: dict[str, Any] | None = None) -> Any
Source code in
dspy/primitives/python_interpreter.py
```
[168](#__codelineno-0-168)
[169](#__codelineno-0-169)
[170](#__codelineno-0-170)
[171](#__codelineno-0-171)
[172](#__codelineno-0-172)
[173](#__codelineno-0-173)
[174](#__codelineno-0-174)
[175](#__codelineno-0-175)
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
[205](#__codelineno-0-205)
[206](#__codelineno-0-206)
[207](#__codelineno-0-207)
[208](#__codelineno-0-208)
[209](#__codelineno-0-209)
[210](#__codelineno-0-210)
[211](#__codelineno-0-211)
[212](#__codelineno-0-212)
[213](#__codelineno-0-213)
[214](#__codelineno-0-214)
[215](#__codelineno-0-215)
[216](#__codelineno-0-216)
[217](#__codelineno-0-217)
[218](#__codelineno-0-218)
```
```
def execute(
    self,
    code: str,
    variables: dict[str, Any] | None = None,
) -> Any:
    variables = variables or {}
    code = self._inject_variables(code, variables)
    self._ensure_deno_process()
    self._mount_files()

    # Send the code as JSON
    input_data = json.dumps({"code": code})
    try:
        self.deno_process.stdin.write(input_data + "\n")
        self.deno_process.stdin.flush()
    except BrokenPipeError:
        # If the process died, restart and try again once
        self._ensure_deno_process()
        self.deno_process.stdin.write(input_data + "\n")
        self.deno_process.stdin.flush()

    # Read one JSON line from stdout
    output_line = self.deno_process.stdout.readline().strip()
    if not output_line:
        # Possibly the subprocess died or gave no output
        err_output = self.deno_process.stderr.read()
        raise InterpreterError(f"No output from Deno subprocess. Stderr: {err_output}")

    # Parse that line as JSON
    try:
        result = json.loads(output_line)
    except json.JSONDecodeError:
        # If not valid JSON, just return raw text
        result = {"output": output_line}

    # If we have an error, determine if it's a SyntaxError or other error using error.errorType.
    if "error" in result:
        error_msg = result["error"]
        error_type = result.get("errorType", "Sandbox Error")
        if error_type == "FinalAnswer":
            # The `FinalAnswer` trick to receive output from the sandbox interpreter,
            # just simply replace the output with the arguments.
            result["output"] = result.get("errorArgs", None)
        elif error_type == "SyntaxError":
            raise SyntaxError(f"Invalid Python syntax. message: {error_msg}")
        else:
            raise InterpreterError(f"{error_type}: {result.get('errorArgs') or error_msg}")

    # If there's no error or got `FinalAnswer`, return the "output" field
    self._sync_files()
    return result.get("output", None)

```
#### 
 shutdown() -> None
Source code in
dspy/primitives/python_interpreter.py
```
[239](#__codelineno-0-239)
[240](#__codelineno-0-240)
[241](#__codelineno-0-241)
[242](#__codelineno-0-242)
[243](#__codelineno-0-243)
[244](#__codelineno-0-244)
[245](#__codelineno-0-245)
[246](#__codelineno-0-246)
```
```
def shutdown(self) -> None:
    if self.deno_process and self.deno_process.poll() is None:
        shutdown_message = json.dumps({"shutdown": True}) + "\n"
        self.deno_process.stdin.write(shutdown_message)
        self.deno_process.stdin.flush()
        self.deno_process.stdin.close()
        self.deno_process.wait()
        self.deno_process = None

```
:::