"""Job completion notifications via ntfy.sh.

Provides a decorator that wraps exca `@infra.apply` methods to send
push notifications when a Slurm job completes or crashes.

Setup:
    1. Install the ntfy app on your phone (iOS/Android)
    2. Subscribe to your topic (e.g. "geniesae-jobs-yourname")
    3. Set NTFY_TOPIC env var or pass topic to the decorator

Environment variables:
    NTFY_TOPIC:   ntfy.sh topic name (required)
    NTFY_SERVER:  ntfy server URL (default: https://ntfy.sh)
    NTFY_ENABLED: set to "0" or "false" to disable notifications
"""

from __future__ import annotations

import functools
import logging
import os
import sys
import time
import traceback
import urllib.request
import urllib.error
import json
from typing import Any, Callable

logger = logging.getLogger("geniesae.notify")

_NTFY_SERVER = os.environ.get("NTFY_SERVER", "https://ntfy.sh")


def _get_ntfy_topic() -> str:
    return os.environ.get("NTFY_TOPIC", "geniesae-hpc-jentker")


def _is_enabled() -> bool:
    val = os.environ.get("NTFY_ENABLED", "1").lower()
    return val not in ("0", "false", "no")


def _get_slurm_info() -> dict[str, str]:
    """Grab useful Slurm env vars if running inside a job."""
    return {
        k: os.environ.get(k, "")
        for k in ("SLURM_JOB_ID", "SLURM_JOB_NAME", "SLURM_NODELIST")
        if os.environ.get(k)
    }


def _get_elapsed_str(seconds: float) -> str:
    """Format seconds into human-readable duration."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h > 0:
        return f"{h}h{m:02d}m{s:02d}s"
    if m > 0:
        return f"{m}m{s:02d}s"
    return f"{s}s"


def _read_log_tail(lines: int = 8) -> str:
    """Try to read the last N lines of the submitit stdout log."""
    try:
        import submitit
        env = submitit.JobEnvironment()
        log_path = env.paths.stdout
        if log_path and os.path.exists(log_path):
            with open(log_path) as f:
                all_lines = f.readlines()
            return "".join(all_lines[-lines:]).strip()
    except Exception:
        pass
    return ""


def _send_ntfy(
    topic: str,
    title: str,
    message: str,
    tags: str = "",
    priority: str = "default",
) -> None:
    """Send a notification via ntfy.sh (no dependencies, just urllib)."""
    if not topic:
        logger.debug("No ntfy topic configured, skipping notification")
        return

    url = f"{_NTFY_SERVER}/{topic}"
    headers: dict[str, str] = {
        "Priority": priority,
    }
    if tags:
        headers["Tags"] = tags

    # Strip non-ASCII from title (emojis break HTTP headers).
    # ntfy renders Tags as emojis in the notification instead.
    safe_title = title.encode("ascii", errors="ignore").decode("ascii").strip()
    if safe_title:
        headers["Title"] = safe_title

    # Truncate message to avoid hitting ntfy limits (4096 bytes)
    msg_bytes = message.encode("utf-8")
    if len(msg_bytes) > 3800:
        message = message[:3800] + "\n..."

    try:
        req = urllib.request.Request(
            url,
            data=message.encode("utf-8"),
            headers=headers,
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            logger.debug("ntfy response: %d", resp.status)
    except Exception as e:
        # Never let notification failure break the actual job
        logger.warning("Failed to send ntfy notification: %s", e)


def notify_on_completion(
    stage_name: str | None = None,
    topic: str | None = None,
):
    """Decorator for exca apply() methods that sends notifications on completion.

    Use this to wrap the *original* method BEFORE @infra.apply, so that
    the notification runs inside the Slurm job.

    Usage:
        class MyConfig(BaseModel):
            infra: exca.TaskInfra = exca.TaskInfra(version="1")

            @infra.apply
            @notify_on_completion("train-sae")
            def apply(self) -> str:
                ...

    Args:
        stage_name: Human-readable stage name for the notification title.
                    Defaults to the class name.
        topic: ntfy topic override. Defaults to NTFY_TOPIC env var.
    """
    def decorator(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
            if not _is_enabled():
                return fn(self, *args, **kwargs)

            ntfy_topic = topic or _get_ntfy_topic()
            name = stage_name or self.__class__.__name__
            slurm = _get_slurm_info()
            job_id = slurm.get("SLURM_JOB_ID", "local")

            start_time = time.time()
            try:
                result = fn(self, *args, **kwargs)
                elapsed = time.time() - start_time

                # Build success message
                title = f"✅ {name} COMPLETED ({_get_elapsed_str(elapsed)})"
                parts = [f"Job: {job_id}"]
                if slurm.get("SLURM_NODELIST"):
                    parts.append(f"Node: {slurm['SLURM_NODELIST']}")
                parts.append(f"Duration: {_get_elapsed_str(elapsed)}")

                # Include return value summary
                if isinstance(result, str):
                    parts.append(f"Result: {result}")
                elif isinstance(result, dict):
                    # For eval results, show key metrics
                    for key in ("baseline_loss", "patched_loss", "num_features_interpreted"):
                        if key in result:
                            parts.append(f"{key}: {result[key]}")

                log_tail = _read_log_tail(5)
                if log_tail:
                    parts.append(f"\nLog:\n{log_tail}")

                _send_ntfy(
                    ntfy_topic,
                    title=title,
                    message="\n".join(parts),
                    tags="white_check_mark",
                    priority="default",
                )
                return result

            except Exception as e:
                elapsed = time.time() - start_time

                # Build failure message
                title = f"❌ {name} FAILED ({_get_elapsed_str(elapsed)})"
                parts = [f"Job: {job_id}"]
                if slurm.get("SLURM_NODELIST"):
                    parts.append(f"Node: {slurm['SLURM_NODELIST']}")
                parts.append(f"Duration: {_get_elapsed_str(elapsed)}")
                parts.append(f"\nError: {type(e).__name__}: {e}")

                # Last few lines of traceback
                tb = traceback.format_exc()
                tb_lines = tb.strip().split("\n")
                parts.append("\nTraceback (last 6 lines):")
                parts.append("\n".join(tb_lines[-6:]))

                _send_ntfy(
                    ntfy_topic,
                    title=title,
                    message="\n".join(parts),
                    tags="x",
                    priority="high",
                )
                raise  # Re-raise so exca/submitit sees the failure

        return wrapper
    return decorator
