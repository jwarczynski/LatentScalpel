#!/usr/bin/env python3
"""Remote GPU cluster MCP server.

Runs on the remote machine (e.g. HPC login node) and exposes tools for:
- Running shell commands (sync, quick scripts)
- Submitting Slurm jobs via sbatch
- Checking Slurm job status
- Reading/tailing log files and result files
- Listing directory contents

Kiro connects to this over SSH stdio transport:
  ssh athena "cd /path/to/GenieSAE && uv run python remote_mcp_server.py"
"""

from __future__ import annotations

import asyncio
import os
import re
import subprocess
import time
from pathlib import Path

from mcp.server.fastmcp import FastMCP

WORKDIR = Path(os.environ.get("GENIESAE_WORKDIR", ".")).resolve()

mcp = FastMCP(name="GenieSAE Remote GPU")


def _run(cmd: str, timeout: int = 120, cwd: Path | None = None) -> tuple[int, str, str]:
    """Run a shell command and return (returncode, stdout, stderr)."""
    result = subprocess.run(
        cmd, shell=True, capture_output=True, text=True,
        timeout=timeout, cwd=cwd or WORKDIR,
    )
    return result.returncode, result.stdout, result.stderr


@mcp.tool()
async def run_command(command: str, timeout: int = 120) -> str:
    """Run a shell command on the remote machine and return combined output.

    Use for quick commands: uv run python scripts/..., cat, ls, squeue, etc.
    For long-running GPU work, use submit_slurm_job instead.

    Args:
        command: Shell command to execute.
        timeout: Max seconds to wait (default 120).
    """
    try:
        rc, stdout, stderr = _run(command, timeout=timeout)
    except subprocess.TimeoutExpired:
        return f"[TIMEOUT after {timeout}s] Command: {command}"

    output = ""
    if stdout:
        output += stdout
    if stderr:
        output += f"\n[stderr]\n{stderr}" if stderr.strip() else ""
    output += f"\n[exit code: {rc}]"
    return output.strip()


@mcp.tool()
async def submit_slurm_job(
    command: str,
    job_name: str = "geniesae",
    partition: str = "plgrid-gpu-a100",
    gpus: int = 1,
    cpus: int = 4,
    mem_gb: int = 64,
    time_minutes: int = 600,
    output_log: str | None = None,
) -> str:
    """Submit a Slurm job via sbatch and return the job ID.

    Args:
        command: The command to run inside the job (e.g. "uv run python main.py train-sae configs/train_sae.yaml").
        job_name: Slurm job name.
        partition: Slurm partition.
        gpus: Number of GPUs.
        cpus: CPUs per task.
        mem_gb: Memory in GB.
        time_minutes: Wall time limit in minutes.
        output_log: Path for stdout/stderr log. Defaults to ./logs/slurm-{job_name}-%j.out.
    """
    if output_log is None:
        log_dir = WORKDIR / "logs"
        log_dir.mkdir(exist_ok=True)
        output_log = str(log_dir / f"slurm-{job_name}-%j.out")

    hours = time_minutes // 60
    mins = time_minutes % 60

    sbatch_script = f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --partition={partition}
#SBATCH --gres=gpu:{gpus}
#SBATCH --cpus-per-task={cpus}
#SBATCH --mem={mem_gb}G
#SBATCH --time={hours:02d}:{mins:02d}:00
#SBATCH --output={output_log}

cd {WORKDIR}
{command}
"""
    # Write temp script and submit
    script_path = WORKDIR / "logs" / f".sbatch_{job_name}.sh"
    script_path.parent.mkdir(exist_ok=True)
    script_path.write_text(sbatch_script)

    rc, stdout, stderr = _run(f"sbatch {script_path}")
    if rc != 0:
        return f"[sbatch FAILED]\n{stderr}\n{stdout}"

    # Parse job ID from "Submitted batch job 12345"
    job_id = stdout.strip().split()[-1] if stdout.strip() else "unknown"
    return f"Job submitted: {job_id}\nLog: {output_log.replace('%j', job_id)}\nScript: {script_path}"


@mcp.tool()
async def submit_exca_job(
    pipeline_command: str,
    config_path: str,
    extra_args: str = "",
) -> str:
    """Submit a job using exca's --submit mechanism (main.py <command> <config> --submit).

    This uses the infra settings from the YAML config for Slurm resources.

    Args:
        pipeline_command: The main.py subcommand (e.g. "train-sae", "evaluate", "interpret-features").
        config_path: Path to the YAML config file.
        extra_args: Additional CLI args (e.g. "--layers 0 1 2 3" or "--features 0 1 2").
    """
    cmd = f"uv run python main.py {pipeline_command} {config_path} --submit"
    if extra_args:
        cmd += f" {extra_args}"

    try:
        rc, stdout, stderr = _run(cmd, timeout=60)
    except subprocess.TimeoutExpired:
        return f"[TIMEOUT] exca submit took >60s. Command: {cmd}"

    output = ""
    if stdout:
        output += stdout
    if stderr:
        output += f"\n[stderr]\n{stderr}" if stderr.strip() else ""
    output += f"\n[exit code: {rc}]"
    return output.strip()


@mcp.tool()
async def job_status(job_id: str | None = None) -> str:
    """Check Slurm job status.

    Args:
        job_id: Specific job ID to check. If None, shows all your recent jobs.
    """
    if job_id:
        cmd = f"sacct -j {job_id} --format=JobID,JobName,State,ExitCode,Elapsed,Start,End,NodeList --noheader"
    else:
        cmd = "squeue --me --format='%.10i %.20j %.8T %.10M %.6D %R' --noheader"

    rc, stdout, stderr = _run(cmd, timeout=30)
    if not stdout.strip() and job_id:
        # squeue might not show completed jobs, try sacct
        rc, stdout, stderr = _run(
            f"sacct -j {job_id} --format=JobID,JobName,State,ExitCode,Elapsed --noheader",
            timeout=30,
        )
    return (stdout + stderr).strip() or "No jobs found."


@mcp.tool()
async def read_file(path: str, tail: int | None = None) -> str:
    """Read a file on the remote machine.

    Args:
        path: File path (relative to workdir or absolute).
        tail: If set, only return the last N lines.
    """
    p = Path(path) if Path(path).is_absolute() else WORKDIR / path
    if not p.exists():
        return f"File not found: {p}"
    if not p.is_file():
        return f"Not a file: {p}"

    try:
        content = p.read_text(errors="replace")
    except Exception as e:
        return f"Error reading {p}: {e}"

    if tail:
        lines = content.splitlines()
        content = "\n".join(lines[-tail:])

    # Cap output to avoid overwhelming the transport
    max_chars = 50_000
    if len(content) > max_chars:
        content = f"[truncated to last {max_chars} chars]\n" + content[-max_chars:]

    return content


@mcp.tool()
async def tail_log(job_id: str, lines: int = 100) -> str:
    """Tail the Slurm log file for a given job ID.

    Searches both ./logs/ and experiments/cache/ for matching log files.

    Args:
        job_id: The Slurm job ID.
        lines: Number of lines from the end to show.
    """
    matches: list[Path] = []

    # Check ./logs/
    log_dir = WORKDIR / "logs"
    if log_dir.exists():
        matches.extend(log_dir.glob(f"*{job_id}*"))

    # Check exca cache
    cache_dir = WORKDIR / "experiments" / "cache"
    if cache_dir.exists():
        matches.extend(cache_dir.rglob(f"*{job_id}*_log.out"))
        matches.extend(cache_dir.rglob(f"*{job_id}*_log.err"))

    if not matches:
        return f"No log file found for job {job_id}"

    results = []
    for log_file in sorted(set(matches)):
        if log_file.is_file():
            rc, stdout, stderr = _run(f"tail -n {lines} {log_file}", timeout=10)
            results.append(f"=== {log_file.name} ===\n{stdout}")

    return "\n".join(results).strip() or f"No readable log files for job {job_id}"


@mcp.tool()
async def list_dir(path: str = ".", depth: int = 1) -> str:
    """List directory contents on the remote machine.

    Args:
        path: Directory path (relative to workdir or absolute).
        depth: Max depth for recursive listing (default 1).
    """
    p = Path(path) if Path(path).is_absolute() else WORKDIR / path
    if not p.exists():
        return f"Path not found: {p}"
    if not p.is_dir():
        return f"Not a directory: {p}"

    rc, stdout, stderr = _run(f"find {p} -maxdepth {depth} -type f -o -type d | sort | head -200", timeout=10)
    return stdout.strip() or "Empty directory."


@mcp.tool()
async def gpu_info() -> str:
    """Check GPU availability and status via nvidia-smi (if on a compute node)."""
    rc, stdout, stderr = _run("nvidia-smi", timeout=10)
    if rc != 0:
        return f"nvidia-smi not available (likely on login node).\n{stderr.strip()}"
    return stdout.strip()

# Stage name -> exca cache subfolder mapping
_STAGE_ALIASES: dict[str, str] = {
    "training": "training",
    "train": "training",
    "train-sae": "training",
    "evaluation": "evaluation",
    "evaluate": "evaluation",
    "eval": "evaluation",
    "interpretation": "interpretation",
    "interpret": "interpretation",
    "interpret-features": "interpretation",
    "top-examples": "top_examples",
    "top_examples": "top_examples",
    "find-top-examples": "top_examples",
    "collection": "collection_val",
    "collection_val": "collection_val",
    "collection_test": "collection_test",
}


def _resolve_cache_dir(stage: str) -> Path | None:
    """Resolve a stage name to its exca cache directory."""
    folder_name = _STAGE_ALIASES.get(stage, stage)
    cache_dir = WORKDIR / "experiments" / "cache" / folder_name
    if cache_dir.is_dir():
        return cache_dir
    # Try direct match
    cache_dir = WORKDIR / "experiments" / "cache" / stage
    if cache_dir.is_dir():
        return cache_dir
    return None


def _find_exca_logs(cache_dir: Path, job_id: str | None = None) -> list[Path]:
    """Find exca log files in a cache directory, optionally filtered by job ID."""
    pattern = f"*{job_id}*log.*" if job_id else "*_log.*"
    logs = sorted(cache_dir.rglob(pattern))
    return logs


@mcp.tool()
async def exca_logs(
    stage: str,
    job_id: str | None = None,
    stream: str = "out",
    tail_lines: int = 80,
) -> str:
    """Find and read exca job logs for a pipeline stage.

    Searches experiments/cache/<stage>/ for log files. Returns the latest
    job's log by default, or a specific job if job_id is given.

    Args:
        stage: Pipeline stage name (training, evaluation, interpretation,
               top-examples, collection_val, collection_test) or alias
               (train, eval, interpret, etc.)
        job_id: Specific Slurm job ID. If None, shows the latest job.
        stream: "out" for stdout, "err" for stderr, "both" for both.
        tail_lines: Number of lines from the end to show (default 80).
    """
    cache_dir = _resolve_cache_dir(stage)
    if cache_dir is None:
        available = [
            d.name for d in (WORKDIR / "experiments" / "cache").iterdir()
            if d.is_dir()
        ]
        return f"Stage '{stage}' not found. Available: {available}"

    # Find all log files
    all_logs = sorted(cache_dir.rglob("*_log.out")) + sorted(cache_dir.rglob("*_log.err"))
    if not all_logs:
        return f"No log files found in {cache_dir}"

    # Extract unique job IDs and find the latest
    job_ids: dict[str, list[Path]] = {}
    for log_path in all_logs:
        # Pattern: .../logs/plgjentker/<job_id>/<job_id>_0_log.{out,err}
        match = re.search(r"/(\d+)/\d+_\d+_log\.", str(log_path))
        if match:
            jid = match.group(1)
            job_ids.setdefault(jid, []).append(log_path)

    if not job_ids:
        return f"Could not parse job IDs from logs in {cache_dir}"

    if job_id:
        if job_id not in job_ids:
            return f"Job {job_id} not found. Available jobs: {sorted(job_ids.keys())}"
        target_id = job_id
    else:
        # Latest = highest job ID number
        target_id = max(job_ids.keys(), key=int)

    # Filter by stream type
    logs_to_read = []
    for log_path in job_ids[target_id]:
        name = log_path.name
        if stream == "both":
            logs_to_read.append(log_path)
        elif stream == "out" and name.endswith("_log.out"):
            logs_to_read.append(log_path)
        elif stream == "err" and name.endswith("_log.err"):
            logs_to_read.append(log_path)

    if not logs_to_read:
        return f"No {stream} logs for job {target_id}"

    # Also get job status
    rc, sacct_out, _ = _run(
        f"sacct -j {target_id} --format=JobID,State,ExitCode,Elapsed --noheader --parsable2",
        timeout=10,
    )
    status_line = sacct_out.strip().split("\n")[0] if sacct_out.strip() else "status unknown"

    results = [f"Job {target_id} | {status_line}"]
    for log_path in sorted(logs_to_read):
        try:
            content = log_path.read_text(errors="replace")
            lines = content.splitlines()
            tail = "\n".join(lines[-tail_lines:])
            results.append(f"\n=== {log_path.name} (last {min(tail_lines, len(lines))}/{len(lines)} lines) ===\n{tail}")
        except Exception as e:
            results.append(f"\n=== {log_path.name} === ERROR: {e}")

    return "\n".join(results)


@mcp.tool()
async def exca_jobs(stage: str | None = None) -> str:
    """List all exca jobs across pipeline stages with their status.

    Shows job IDs, stages, and current Slurm status for quick overview.

    Args:
        stage: Optional stage filter. If None, shows all stages.
    """
    cache_root = WORKDIR / "experiments" / "cache"
    if not cache_root.is_dir():
        return "No experiments/cache directory found."

    stages = [stage] if stage else None
    if stages:
        dirs = []
        for s in stages:
            d = _resolve_cache_dir(s)
            if d:
                dirs.append((s, d))
        if not dirs:
            return f"Stage '{stage}' not found."
    else:
        dirs = [(d.name, d) for d in sorted(cache_root.iterdir()) if d.is_dir()]

    # Collect all job IDs per stage
    all_job_ids: list[tuple[str, str, str]] = []  # (stage, job_id, log_path)
    for stage_name, cache_dir in dirs:
        for log_path in cache_dir.rglob("*_log.out"):
            match = re.search(r"/(\d+)/\d+_\d+_log\.", str(log_path))
            if match:
                jid = match.group(1)
                all_job_ids.append((stage_name, jid, str(log_path)))

    if not all_job_ids:
        return "No exca jobs found."

    # Deduplicate
    seen = set()
    unique_jobs = []
    for stage_name, jid, lp in all_job_ids:
        key = (stage_name, jid)
        if key not in seen:
            seen.add(key)
            unique_jobs.append((stage_name, jid))

    # Sort by job ID descending (newest first)
    unique_jobs.sort(key=lambda x: int(x[1]), reverse=True)

    # Batch sacct query for all job IDs
    job_id_list = ",".join(jid for _, jid in unique_jobs)
    rc, sacct_out, _ = _run(
        f"sacct -j {job_id_list} --format=JobID,JobName,State,ExitCode,Elapsed --noheader --parsable2",
        timeout=30,
    )

    # Parse sacct output into a dict
    status_map: dict[str, str] = {}
    if sacct_out.strip():
        for line in sacct_out.strip().split("\n"):
            parts = line.split("|")
            if parts and not "." in parts[0]:  # skip sub-steps like 12345.batch
                jid = parts[0].strip()
                state = parts[2].strip() if len(parts) > 2 else "?"
                elapsed = parts[4].strip() if len(parts) > 4 else "?"
                status_map[jid] = f"{state} ({elapsed})"

    lines = [f"{'Stage':<20} {'Job ID':<12} {'Status'}"]
    lines.append("-" * 60)
    for stage_name, jid in unique_jobs:
        status = status_map.get(jid, "unknown")
        lines.append(f"{stage_name:<20} {jid:<12} {status}")

    return "\n".join(lines)


@mcp.tool()
async def wait_for_job(
    job_id: str,
    poll_interval: int = 15,
    timeout: int = 300,
    tail_lines: int = 40,
) -> str:
    """Wait for a SHORT Slurm job to finish, then return status and log tail.

    Only use for jobs expected to complete within a few minutes (quick evals,
    test scripts, etc.). For long training jobs, use job_status + exca_logs
    to check periodically instead.

    Default timeout is 5 minutes. If the job hasn't finished by then, returns
    the current status so you can continue working.

    Args:
        job_id: Slurm job ID to wait for.
        poll_interval: Seconds between status checks (default 15).
        timeout: Max seconds to wait (default 300 = 5 min). Keep this short.
        tail_lines: Lines of log to return when done (default 40).
    """
    terminal_states = {"COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "NODE_FAIL", "PREEMPTED", "OUT_OF_MEMORY"}
    start = time.time()

    final_state = "UNKNOWN"
    while time.time() - start < timeout:
        rc, stdout, _ = _run(
            f"sacct -j {job_id} --format=State --noheader --parsable2",
            timeout=10,
        )
        states = [s.strip() for s in stdout.strip().split("\n") if s.strip() and "." not in s.strip().split("|")[0]]
        if states:
            current = states[0].split("|")[0].strip() if "|" in states[0] else states[0].strip()
            if current in terminal_states:
                final_state = current
                break

        await asyncio.sleep(poll_interval)

    # Get full status
    rc, sacct_out, _ = _run(
        f"sacct -j {job_id} --format=JobID,State,ExitCode,Elapsed,MaxRSS --noheader --parsable2",
        timeout=10,
    )

    if final_state == "UNKNOWN":
        elapsed = int(time.time() - start)
        return (
            f"Job {job_id} still running after {elapsed}s (timeout={timeout}s).\n"
            f"{sacct_out.strip()}\n\n"
            f"Use exca_logs or job_status to check on it later."
        )

    # Try to find and tail the log
    log_content = ""
    cache_root = WORKDIR / "experiments" / "cache"
    if cache_root.is_dir():
        log_files = list(cache_root.rglob(f"*{job_id}*_log.out"))
        if log_files:
            try:
                content = log_files[0].read_text(errors="replace")
                lines = content.splitlines()
                log_content = f"\n\n=== Log tail ({log_files[0].name}) ===\n" + "\n".join(lines[-tail_lines:])
            except Exception:
                pass

    return f"Job {job_id}: {final_state}\n{sacct_out.strip()}{log_content}"



if __name__ == "__main__":
    mcp.run()
