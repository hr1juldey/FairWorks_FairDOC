#!/usr/bin/env python3
"""
Universal Fairdoc AI Control Tool (run.py)

- Start: V1 server, V2 server, or V1+V2 combined (on /api/v1 + /api/v2)
- List and run: Unit, Integration, or E2E tests—one, many, or all
- Interactive CLI (single-file)
"""

import subprocess
import sys
import os
from pathlib import Path

REPO_ROOT = Path(__file__).parent.resolve()
SRC = REPO_ROOT / "src"
TESTS = SRC / "tests"
APP_MAIN = "src.app.main:app"
APP_V2 = "src.app2.main_v2:app"

TEST_PATHS = {
    "unit": TESTS / "unit",
    "integration": TESTS / "integration",
    "e2e": TESTS / "e2e"
}

def discover_tests():
    """Auto-detect pytest targets in all test folders"""
    all_tests = []
    for ttype, path in TEST_PATHS.items():
        if not path.exists():
            continue
        for f in sorted(path.glob("test_*.py")):
            all_tests.append({
                "type": ttype,
                "name": f.name,
                "abs": str(f.resolve()),
                "rel": str(f.relative_to(REPO_ROOT))
            })
    return all_tests

def launch_uvicorn(module_str, port=8000, reload=True):
    args = [
        sys.executable, "-m", "uvicorn", module_str,
        "--host", "0.0.0.0", "--port", str(port)
    ]
    if reload: args.append("--reload")
    return subprocess.Popen(args)

def start_server():
    print("\n=== APP SERVER LAUNCH OPTIONS ===")
    print("[1] Run V1 only  (/api/v1, main.py)")
    print("[2] Run V2 only  (/api/v2, main_v2.py)")
    print("[3] Run V1 + V2  (V2 mounted in V1 at /api/v2)")
    print("[b] Back")
    ch = input("Select: ").strip().lower()
    if ch == "1":
        print("Starting V1 API (port 8000)...")
        proc = launch_uvicorn(APP_MAIN, 8000, reload=True)
        _wait(proc)
    elif ch == "2":
        print("Starting V2 API (port 8000)...")
        proc = launch_uvicorn(APP_V2, 8000, reload=True)
        _wait(proc)
    elif ch == "3":
        print("Starting V1 + V2, with V2 mounted inside V1 (port 8000)...")
        proc = launch_uvicorn(APP_MAIN, 8000, reload=True)
        _wait(proc)
    else:
        print("Back to main menu.")

def _wait(proc):
    try:
        print("Server running. Press Ctrl+C to stop.")
        proc.wait()
    except KeyboardInterrupt:
        print("Stopping server...")
        proc.terminate()
        proc.wait()

def list_and_run_tests():
    tests = discover_tests()
    groups = {k: [] for k in TEST_PATHS}
    for t in tests:
        groups[t["type"]].append(t)
    menu = []
    idx = 1
    print("\n=== AVAILABLE TESTS ===")
    for g in ["unit", "integration", "e2e"]:
        sub = groups[g]
        if sub:
            print(f"  {g.upper()} tests:")
            for t in sub:
                print(f"   [{idx}] {t['rel']}")
                menu.append(t)
                idx += 1
    print("\n  [a] Run ALL tests (unit+integration+e2e)")
    print("  [b] Back to main menu")
    sel = input("Enter test number(s) (comma/space separated), 'a' for all, 'b' to back: ").strip().lower()
    if sel == "b":
        return
    run_online = input("Run tests against live localhost servers? [y/N]: ").strip().lower() == "y"

    # build pytest command(s)
    test_cmds = []
    if sel == "a":
        for g in ["unit", "integration", "e2e"]:
            for t in groups[g]:
                test_cmds.append(t)
    else:
        indices = []
        for s in sel.replace(',', ' ').split():
            try:
                i = int(s) - 1
                if 0 <= i < len(menu): indices.append(i)
            except ValueError: pass
        for i in indices:
            test_cmds.append(menu[i])

    if not test_cmds:
        print("No tests selected.")
        return

    for t in test_cmds:
        print(f"\n==== Running {t['rel']} ({t['type']}) {'[LIVE MODE]' if run_online else '[OFFLINE MODE]'} ====")
        cmd = [sys.executable, "-m", "pytest", t["abs"], "-v", "--tb=short"]
        if t['type'] == "e2e" and run_online:
            # Many e2e tests auto-detect live endpoints via env/config
            os.environ["FAIRDOC_AI_LIVE"] = "true"
        else:
            os.environ["FAIRDOC_AI_LIVE"] = "false"
        try:
            subprocess.run(cmd, check=False)
        except KeyboardInterrupt:
            print("Test interrupted. Continuing to next/exit...")

def main():
    print("=" * 60)
    print("  Fairdoc AI Triage System: Universal Launcher & Test Tool")
    print("=" * 60)
    while True:
        print("\nOPTIONS:")
        print("  [1] Start API server (V1/V2/Both)")
        print("  [2] List/Run tests (unit/integration/e2e)")
        print("  [q] Quit")
        cmd = input("Select: ").strip().lower()
        if cmd == "1": start_server()
        elif cmd == "2": list_and_run_tests()
        elif cmd == "q": print("Goodbye!"); break
        else: print("Invalid input.")

if __name__ == "__main__":
    os.chdir(str(REPO_ROOT))
    main()
