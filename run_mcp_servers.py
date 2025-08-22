#!/usr/bin/env python3
"""
run_mcp_servers.py (venv-aware + JSON-RPC probe)

Starts MCP servers concurrently, attach logs in ./logs, and performs
a simple liveness check (any stdout/stderr within timeout). Also optionally
sends a conservative JSON-RPC probe per stdio server to confirm RPC responsiveness.

Usage:
  python3 run_mcp_servers.py [--venv PATH] [--detach] [--no-probe]
"""

import os
import sys
import json
import subprocess
import threading
import time
import argparse
from pathlib import Path
from queue import Queue, Empty

# ---------- Config (copied from your message) ----------
CONFIG = {
  "mcpServers": {
    "github.com/modelcontextprotocol/servers/tree/main/src/sequentialthinking": {
      "autoApprove": [
        "sequential_thinking",
        "sequentialthinking"
      ],
      "timeout": 600,
      "type": "stdio",
      "command": "npx",
      "args": [
        "-y",
        "@modelcontextprotocol/server-sequential-thinking"
      ],
      "env": {
        "DISABLE_THOUGHT_LOGGING": "true"
      }
    },
    "github.com/modelcontextprotocol/servers/tree/main/src/filesystem": {
      "autoApprove": [
        "list_allowed_directories",
        "list_directory",
        "read_file",
        "read_text_file",
        "read_media_file",
        "read_multiple_files",
        "list_directory_with_sizes",
        "directory_tree",
        "search_files",
        "get_file_info"
      ],
      "timeout": 600,
      "type": "stdio",
      "command": "npx",
      "args": [
        "-y",
        "@modelcontextprotocol/server-filesystem",
        "/home/riju279/Documents/Code/Fairdoc/FairWorks_FairDOC",
        "/home/riju279/Documents/Cline/MCP"
      ]
    },
    "github.com/pashpashpash/perplexity-mcp": {
      "autoApprove": [
        "check_deprecated_code"
      ],
      "disabled": False,
      "timeout": 600,
      "type": "stdio",
      "command": "node",
      "args": [
        "/home/riju279/Documents/Cline/MCP/perplexity-mcp/build/index.js"
      ],
      "env": {
        "PERPLEXITY_API_KEY": "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"
      }
    },
    "github.com/NightTrek/Ollama-mcp": {
      "autoApprove": [
        "list",
        "show",
        "chat_completion"
      ],
      "disabled": True,
      "timeout": 600,
      "type": "stdio",
      "command": "node",
      "args": [
        "/home/riju279/Documents/Cline/MCP/Ollama-mcp/build/index.js"
      ],
      "env": {
        "OLLAMA_HOST": "http://127.0.0.1:11434"
      }
    },
    "github.com/modelcontextprotocol/servers/tree/main/src/memory": {
      "autoApprove": [
        "create_relations",
        "read_graph",
        "create_entities",
        "add_observations",
        "delete_entities",
        "delete_observations",
        "delete_relations",
        "search_nodes",
        "open_nodes"
      ],
      "disabled": False,
      "timeout": 600,
      "type": "stdio",
      "command": "npx",
      "args": [
        "-y",
        "@modelcontextprotocol/server-memory"
      ],
      "env": {
        "MEMORY_FILE_PATH": "/home/riju279/Documents/Cline/MCP/memory-server/memory.json"
      }
    },
    "github.com/upstash/context7-mcp": {
      "autoApprove": [
        "resolve-library-id",
        "get-library-docs"
      ],
      "disabled": False,
      "timeout": 600,
      "type": "stdio",
      "command": "npx",
      "args": [
        "-y",
        "@upstash/context7-mcp"
      ]
    },
    "github.com/modelcontextprotocol/servers/tree/main/src/time": {
      "autoApprove": [
        "get_current_time",
        "convert_time"
      ],
      "disabled": False,
      "timeout": 600,
      "type": "stdio",
      "command": "uvx",
      "args": [
        "mcp-server-time"
      ]
    }
  }
}
# -------------------------------------------------------------------

LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

def safe_name(key: str) -> str:
    return key.replace("/", "_").replace(":", "_").replace(" ", "_").replace(".", "_")

def enqueue_output(pipe, queues, prefix):
    """
    Read lines from `pipe` and put them into every queue in `queues`.
    `queues` is a list of Queue instances.
    """
    try:
        for line in iter(pipe.readline, ""):
            if not line:
                break
            for q in queues:
                try:
                    q.put((prefix, line))
                except Exception:
                    pass
    except Exception:
        pass
    finally:
        try:
            pipe.close()
        except Exception:
            pass

def apply_venv_to_env(base_env: dict, venv_path: str) -> dict:
    env = base_env.copy()
    if not venv_path:
        return env
    venv_path = os.path.expanduser(venv_path)
    bin_dir = os.path.join(venv_path, "bin")
    if os.path.isdir(bin_dir):
        env["VIRTUAL_ENV"] = venv_path
        env["PATH"] = bin_dir + os.pathsep + env.get("PATH", "")
        env.pop("PYTHONHOME", None)
    else:
        print(f"Warning: venv bin not found at {bin_dir}; continuing without venv integration.")
    return env

def start_server(name, cfg, report, venv_path=None, detach=False):
    if cfg.get("disabled", False):
        report[name] = {"started": False, "reason": "disabled"}
        return None

    cmd = [cfg.get("command")] + cfg.get("args", [])
    base_env = os.environ.copy()
    env = apply_venv_to_env(base_env, venv_path)
    env.update({k: str(v) for k, v in cfg.get("env", {}).items()})

    sname = safe_name(name)
    out_path = LOG_DIR / f"{sname}.log"
    err_path = LOG_DIR / f"{sname}.err"

    try:
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
            text=True,
            bufsize=1
        )
    except FileNotFoundError:
        report[name] = {"started": False, "reason": f"command not found: {cmd[0]}", "cmd": cmd}
        return None
    except Exception as e:
        report[name] = {"started": False, "reason": f"failed to start: {e}", "cmd": cmd}
        return None

    # Two queues: q (writer) and probe_q (for RPC probe)
    q = Queue()
    probe_q = Queue()

    t_out = threading.Thread(target=enqueue_output, args=(proc.stdout, [q, probe_q], "OUT"), daemon=True)
    t_err = threading.Thread(target=enqueue_output, args=(proc.stderr, [q, probe_q], "ERR"), daemon=True)
    t_out.start()
    t_err.start()

    fout = open(out_path, "a", encoding="utf-8")
    ferr = open(err_path, "a", encoding="utf-8")

    def writer():
        while True:
            try:
                prefix, line = q.get(timeout=0.5)
                ts = time.strftime("%Y-%m-%d %H:%M:%S")
                entry = f"[{ts}] {prefix} {line}"
                if prefix == "OUT":
                    fout.write(entry)
                    fout.flush()
                else:
                    ferr.write(entry)
                    ferr.flush()
                print(f"{sname} | {prefix} | {line}", end="", flush=True)
            except Empty:
                if proc.poll() is not None:
                    break
                continue
            except Exception:
                break
        fout.close()
        ferr.close()

    writer_thread = threading.Thread(target=writer, daemon=True)
    writer_thread.start()

    report[name] = {"started": True, "pid": proc.pid, "log": str(out_path), "err_log": str(err_path)}
    return {"proc": proc, "writer_thread": writer_thread, "probe_q": probe_q, "report": report}

# Conservative probe policy:
# Map key substring -> (method, params) that are safe to call for that server
PROBE_POLICY = [
    ("filesystem", ("list_allowed_directories", {})),
    ("time", ("get_current_time", {})),
    ("memory", ("search_nodes", {"query": ""})),
    ("sequentialthinking", ("sequential_thinking", {"input": "ping"})),
    ("perplexity", ("check_deprecated_code", {})),
    ("context7", ("get-library-docs", {"id": ""})),
    # fallback if none match
]

def build_probe_for(name, default_id):
    lname = name.lower()
    for key, (method, params) in PROBE_POLICY:
        if key in lname:
            return {"jsonrpc": "2.0", "id": default_id, "method": method, "params": params}
    # fallback generic probe
    return {"jsonrpc": "2.0", "id": default_id, "method": "__mcp_probe__", "params": {}}

def probe_server(handle, timeout_seconds, probe_id, name):
    """
    Write a JSON-RPC request to proc.stdin and wait up to timeout_seconds for a response line
    on handle['probe_q'] that mentions the probe id or contains 'result'/'error'.
    Returns (rpc_responded: bool, matched_line_or_none)
    """
    proc = handle['proc']
    if proc.stdin is None:
        return False, None
    probe_payload = build_probe_for(name, probe_id)
    try:
        # write JSON-RPC request and flush
        s = json.dumps(probe_payload)
        proc.stdin.write(s + "\n")
        proc.stdin.flush()
    except Exception as e:
        return False, f"failed-to-write:{e}"

    # wait for response
    end_t = time.time() + timeout_seconds
    matched_line = None
    while time.time() < end_t:
        try:
            prefix, line = handle['probe_q'].get(timeout=0.25)
        except Empty:
            # check if process exited
            if proc.poll() is not None:
                break
            continue
        # check for id or generic json-rpc tokens
        if f'"id": {probe_id}' in line or f'"id":{probe_id}' in line or '"result"' in line or '"error"' in line:
            matched_line = line.strip()
            return True, matched_line
        # some servers might echo or wrap; also accept any json-like line
        if "jsonrpc" in line.lower():
            matched_line = line.strip()
            return True, matched_line
    return False, None

def monitor_startup(process_handles, timeouts):
    """Wait up to timeout for each process to produce any stdout/stderr (basic liveness)."""
    results = {}
    for name, meta in process_handles.items():
        cfg_timeout = timeouts.get(name, 10)
        # check logs presence quickly
        out_path = Path(meta['report'][name]['log'])
        err_path = Path(meta['report'][name]['err_log'])
        end_t = time.time() + min(cfg_timeout, 5)
        responsive = False
        while time.time() < end_t:
            proc = meta['handle']['proc']
            if proc.poll() is not None:
                responsive = (out_path.exists() and out_path.stat().st_size > 0) or (err_path.exists() and err_path.stat().st_size > 0)
                break
            if (out_path.exists() and out_path.stat().st_size > 0) or (err_path.exists() and err_path.stat().st_size > 0):
                responsive = True
                break
            time.sleep(0.2)
        results[name] = {"responsive": responsive, "pid": meta['report'][name].get("pid"), "started": meta['report'][name].get("started")}
    return results

def run_rpc_probes(process_handles, timeouts):
    """For each stdio server, send a conservative JSON-RPC request and wait for answer."""
    probe_results = {}
    next_id = 1000
    for name, meta in process_handles.items():
        proc = meta['handle']['proc']
        # only probe stdio processes that started
        if not meta['report'].get(name, {}).get("started"):
            probe_results[name] = {"rpc_responded": False, "reason": "not_started_or_disabled"}
            continue
        # give short per-server probe timeout (min of configured timeout and 8s)
        per_to = min(8, timeouts.get(name, 8))
        handle = {"proc": proc, "probe_q": meta['handle']['probe_q']}
        rpc_ok, matched = probe_server(handle, per_to, next_id, name)
        probe_results[name] = {"rpc_responded": rpc_ok, "matched": matched}
        next_id += 1
    return probe_results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--detach", action="store_true", help="Start servers and exit (do not tail logs).")
    parser.add_argument("--venv", type=str,
                        default="/home/riju279/Documents/Code/Fairdoc/FairWorks_FairDOC/fairdoc_ai_triage/.venv",
                        help="Path to virtualenv to activate for child processes (default: your project .venv).")
    parser.add_argument("--no-probe", action="store_true", help="Skip JSON-RPC probes.")
    args = parser.parse_args()

    venv_path = args.venv
    if venv_path and not Path(venv_path).exists():
        print(f"Warning: specified venv path does not exist: {venv_path}")

    config = CONFIG.get("mcpServers", {})
    report = {}
    processes = {}
    timeouts = {}

    print("Starting MCP servers... (use --detach to background)\n")

    for name, cfg in config.items():
        timeouts[name] = cfg.get("timeout", 10)
        handle = start_server(name, cfg, report, venv_path=venv_path, detach=args.detach)
        if handle:
            # store handle and report reference
            processes[name] = {"handle": handle, "report": report}

    if args.detach:
        print("\nDetached mode: started processes (if any). Summary:")
        for k, v in report.items():
            print(f"- {safe_name(k)}: {v}")
        return

    # Quick startup checks (stdout/stderr)
    monitor_handles = {}
    for name, handle_obj in processes.items():
        monitor_handles[name] = {"handle": handle_obj["handle"], "report": report}
    simple_map = {name: {"handle": monitor_handles[name]["handle"], "report": report} for name in monitor_handles}
    monitored = monitor_startup({name: {"handle": simple_map[name]["handle"], "report": report} for name in simple_map}, timeouts)

    print("\n--- Startup summary ---")
    for name, res in monitored.items():
        sname = safe_name(name)
        started = res.get("started")
        pid = res.get("pid")
        resp = res.get("responsive")
        status = "RESPONSIVE" if resp else "NO OUTPUT YET"
        print(f"{sname:60} PID={pid:6}  started={started}  => {status}")

    # Run optional JSON-RPC probes
    if not args.no_probe:
        print("\nRunning conservative JSON-RPC probes (this will send a single safe request to each stdio server)...")
        # prepare process_handles in expected shape: name -> {'handle': handle, 'report': report}
        proc_handles_for_probe = {}
        for name, info in processes.items():
            proc_handles_for_probe[name] = {"handle": info["handle"], "report": report}
        # adapt to expected structure for run_rpc_probes: name -> {'handle': <handle>, 'report': report}
        # run probes
        probe_results = run_rpc_probes({name: {"handle": info["handle"], "report": report} for name, info in processes.items()}, timeouts)
        print("\n--- RPC probe results ---")
        for name, r in probe_results.items():
            sname = safe_name(name)
            ok = r.get("rpc_responded")
            matched = r.get("matched")
            print(f"{sname:60} RPC_RESPONDED={ok}  matched={matched}")
    else:
        print("\nSkipping JSON-RPC probes (--no-probe).")

    print("\nTailing logs. Press Ctrl+C to stop and kill children.\n")

    try:
        while True:
            alive = any(info['handle']['proc'].poll() is None for info in processes.values())
            if not alive:
                print("All processes exited.")
                break
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping: terminating child processes...")
    finally:
        for name, info in processes.items():
            proc = info['handle']['proc']
            if proc.poll() is None:
                try:
                    proc.terminate()
                    try:
                        proc.wait(timeout=3)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                except Exception as e:
                    print(f"Failed to stop {safe_name(name)}: {e}")
        print("All done. Logs are in ./logs/")

if __name__ == "__main__":
    main()
