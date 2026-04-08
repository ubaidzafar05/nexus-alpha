#!/usr/bin/env python3
"""Wait for the paper-eval process to finish, then run benchmark-learning-targets and
trigger a retrain attempt via the CLI. This is a convenience helper to automate
post-eval actions.

Run as: python infra/self_healing/post_eval_watcher.py --pid-file /tmp/nexus-paper-8h.pid
"""
from __future__ import annotations

import argparse
import os
import subprocess
import time
from pathlib import Path


def wait_for_pid(pidfile: Path, poll: int = 10):
    if not pidfile.exists():
        print("pidfile missing; exiting")
        return
    pid = None
    try:
        pid = int(pidfile.read_text().strip())
    except Exception:
        print("invalid pidfile")
        return
    print(f"waiting for pid {pid} to exit")
    while True:
        try:
            os.kill(pid, 0)
            time.sleep(poll)
        except ProcessLookupError:
            print("process ended")
            return
        except Exception as e:
            print("error checking pid", e)
            time.sleep(poll)


def main(pidfile: str):
    p = Path(pidfile)
    wait_for_pid(p)
    # Run benchmark-learning-targets
    cmd = ["python3", "-m", "nexus_alpha.cli", "benchmark-learning-targets", "--min-trades", "30"]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd)
    # Trigger retrain watcher once (run it quickly to let it attempt a retrain)
    cmd2 = ["python3", "-m", "nexus_alpha.cli", "run-retrain-watcher", "--interval", "3600"]
    print("Starting retrain watcher (background):", " ".join(cmd2))
    subprocess.Popen(cmd2)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("--pid-file", default="/tmp/nexus-paper-8h.pid")
    args = ap.parse_args()
    main(args.pid_file)
