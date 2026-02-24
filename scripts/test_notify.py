"""Dummy job to test ntfy notifications. No GPU required."""

from __future__ import annotations

import os
import time

import exca
from pydantic import BaseModel, Field

# Set topic for testing if not already set
if not os.environ.get("NTFY_TOPIC"):
    os.environ["NTFY_TOPIC"] = "geniesae-hpc-jentker"

from geniesae.notify import notify_on_completion


class NotifyTestConfig(BaseModel):
    """Minimal config that sleeps and returns, triggering a notification."""

    sleep_seconds: int = Field(default=5, gt=0)
    should_fail: bool = False

    infra: exca.TaskInfra = exca.TaskInfra(version="1")

    @infra.apply
    @notify_on_completion("notify-test")
    def apply(self) -> str:
        print(f"[NotifyTest] Starting dummy job, sleeping {self.sleep_seconds}s...")
        time.sleep(self.sleep_seconds)
        if self.should_fail:
            raise RuntimeError("Intentional test failure!")
        print("[NotifyTest] Done!")
        return "success"
