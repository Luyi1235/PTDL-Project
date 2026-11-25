"""Run an isolated solver with file-backed logs and bounded process-tree cleanup."""

import json
import os
from pathlib import Path
import signal
import subprocess
import tempfile
import time

import 重算VSPAERO_Ew_DOE并训练代理 as original


def stop_tree(process):
    if process.poll() is not None:
        return
    if os.name == "nt":
        result = subprocess.run(
            ["taskkill", "/PID", str(process.pid), "/T", "/F"],
            capture_output=True, timeout=20, check=False)
        if result.returncode and process.poll() is None:
            raise RuntimeError("Failed to stop timed-out solver tree")
    else:
        os.killpg(process.pid, signal.SIGKILL)
    process.wait(timeout=20)


def run_process(command, request, folder, timeout):
    folder = Path(folder)
    stdout_path, stderr_path = folder / "worker.stdout.log", folder / "worker.stderr.log"
    with stdout_path.open("w", encoding="utf8") as stdout, stderr_path.open("w", encoding="utf8") as stderr:
        with tempfile.TemporaryDirectory(prefix="ptdl_controlled_") as startup:
            process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=stdout, stderr=stderr,
                                       cwd=startup, text=True, encoding="utf8",
                                       start_new_session=os.name != "nt")
            try:
                process.communicate(json.dumps(request), timeout=timeout)
            except subprocess.TimeoutExpired:
                stop_tree(process)
                raise
    log = stdout_path.read_text(encoding="utf8", errors="replace")
    error = stderr_path.read_text(encoding="utf8", errors="replace")
    (folder / "worker.log").write_text(log + "\n--- STDERR ---\n" + error, encoding="utf8")
    if process.returncode:
        raise RuntimeError(f"Solver worker exited {process.returncode}: {error[-1200:]}")
    return log


def run_case(python_exe, design, sample, case_dir, timeout_s, selection_role,
             selection_basis, oswald_label_bounds=(0., float("inf")),
             wake_iterations=30, mesh_level="medium"):
    folder = Path(case_dir)
    folder.mkdir(parents=True, exist_ok=False)
    request = dict(design=design, output_dir=str(folder), speed_mps=35.99708,
                   altitude_m=3000., wake_iterations=wake_iterations, mesh_level=mesh_level)
    started = time.monotonic()
    try:
        log = run_process([str(python_exe), str(original.WORKER)], request, folder, timeout_s)
        result = original._extract_worker_result(log)
        row = original._record_from_result(sample, design, result, folder, log,
                                            selection_role, selection_basis, oswald_label_bounds)
        row["solver_wall_time_s"] = time.monotonic()-started
        return row
    except Exception as error:
        (folder / "case_error.txt").write_text(repr(error), encoding="utf8")
        raise
