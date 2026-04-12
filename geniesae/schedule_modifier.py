"""Schedule modification logic for denoising schedule experiments.

Supports modifying both Genie (discrete) and PLAID (continuous) diffusion
schedules for temporal feature analysis experiments.
"""

from __future__ import annotations

import numpy as np

from geniesae.genie_model import DiffusionHelper


class ScheduleModifier:
    """Utilities for modifying diffusion schedules and comparing timestep positions."""

    VALID_MODIFICATION_TYPES = {"compress_early", "extend_late", "reduce_total", "custom"}

    @staticmethod
    def modify_genie_schedule(
        diffusion_helper: DiffusionHelper,
        modification_type: str,
        params: dict,
    ) -> tuple[DiffusionHelper, list[int]]:
        """Create a modified DiffusionHelper and return the new timestep sequence.

        Args:
            diffusion_helper: Original DiffusionHelper instance.
            modification_type: One of ``compress_early``, ``extend_late``,
                ``reduce_total``, or ``custom``.
            params: Modification-specific parameters.

        Returns:
            Tuple of (new_diffusion_helper, timestep_sequence) where
            timestep_sequence is a descending list of integer timesteps.

        Raises:
            ValueError: If modification_type is invalid or params are out of range.
        """
        ScheduleModifier._validate_modification_type(modification_type)
        total = diffusion_helper.num_timesteps

        if modification_type == "compress_early":
            speedup_steps = params.get("speedup_steps")
            if speedup_steps is None:
                raise ValueError("compress_early requires 'speedup_steps' parameter")
            if speedup_steps >= total:
                raise ValueError(
                    f"speedup_steps ({speedup_steps}) must be less than "
                    f"total timesteps ({total})"
                )
            if speedup_steps <= 0:
                raise ValueError("speedup_steps must be positive")
            # Compress the early (high-noise) phase by skipping every other step
            # in the first speedup_steps, keep the rest unchanged.
            # Genie timesteps go from num_timesteps-1 (max noise) down to 0 (clean).
            early_steps = list(range(total - 1, total - 1 - speedup_steps, -2))
            late_steps = list(range(total - 1 - speedup_steps, -1, -1))
            timesteps = early_steps + late_steps
            # The DiffusionHelper itself stays the same (same beta schedule),
            # we just change which timesteps we visit.
            new_helper = DiffusionHelper(
                num_timesteps=total, schedule_name="sqrt"
            )
            return new_helper, timesteps

        elif modification_type == "extend_late":
            extra_steps = params.get("extra_steps")
            if extra_steps is None:
                raise ValueError("extend_late requires 'extra_steps' parameter")
            if extra_steps <= 0:
                raise ValueError("extra_steps must be positive")
            # Extend the late (low-noise) phase by creating a new schedule
            # with more total steps, giving finer resolution near clean end.
            new_total = total + extra_steps
            new_helper = DiffusionHelper(
                num_timesteps=new_total, schedule_name="sqrt"
            )
            timesteps = list(range(new_total - 1, -1, -1))
            return new_helper, timesteps

        elif modification_type == "reduce_total":
            new_total = params.get("new_total")
            if new_total is None:
                raise ValueError("reduce_total requires 'new_total' parameter")
            if new_total >= total:
                raise ValueError(
                    f"new_total ({new_total}) must be less than "
                    f"original total ({total})"
                )
            if new_total <= 0:
                raise ValueError("new_total must be positive")
            new_helper = DiffusionHelper(
                num_timesteps=new_total, schedule_name="sqrt"
            )
            timesteps = list(range(new_total - 1, -1, -1))
            return new_helper, timesteps

        else:  # custom
            custom_timesteps = params.get("timesteps")
            if custom_timesteps is None:
                raise ValueError("custom requires 'timesteps' parameter")
            if len(custom_timesteps) == 0:
                raise ValueError("custom timesteps must be non-empty")
            ScheduleModifier._validate_monotonic_descending(
                custom_timesteps, max_val=total - 1, min_val=0
            )
            new_helper = DiffusionHelper(
                num_timesteps=total, schedule_name="sqrt"
            )
            return new_helper, list(custom_timesteps)

    @staticmethod
    def modify_plaid_schedule(
        sampling_timesteps: int,
        modification_type: str,
        params: dict,
    ) -> list[float]:
        """Return modified continuous timestep sequence for PLAID.

        PLAID uses continuous t in [0, 1]. The standard schedule is
        ``np.linspace(1, 0, sampling_timesteps + 1)`` (from max noise to clean).

        Args:
            sampling_timesteps: Original number of sampling timesteps.
            modification_type: One of ``compress_early``, ``extend_late``,
                ``reduce_total``, or ``custom``.
            params: Modification-specific parameters.

        Returns:
            List of floats in [0.0, 1.0] representing the modified schedule,
            ordered from 1.0 (max noise) toward 0.0 (clean).

        Raises:
            ValueError: If modification_type is invalid or params are out of range.
        """
        ScheduleModifier._validate_modification_type(modification_type)

        if modification_type == "compress_early":
            speedup_steps = params.get("speedup_steps")
            if speedup_steps is None:
                raise ValueError("compress_early requires 'speedup_steps' parameter")
            if speedup_steps >= sampling_timesteps:
                raise ValueError(
                    f"speedup_steps ({speedup_steps}) must be less than "
                    f"total sampling timesteps ({sampling_timesteps})"
                )
            if speedup_steps <= 0:
                raise ValueError("speedup_steps must be positive")
            # Compress the early (high-noise) phase: use fewer steps in [1.0, midpoint]
            # and keep normal density in [midpoint, 0.0].
            standard = np.linspace(1.0, 0.0, sampling_timesteps + 1)
            # Take every other step from the first speedup_steps
            early = standard[:speedup_steps:2].tolist()
            late = standard[speedup_steps:].tolist()
            return early + late

        elif modification_type == "extend_late":
            extra_steps = params.get("extra_steps")
            if extra_steps is None:
                raise ValueError("extend_late requires 'extra_steps' parameter")
            if extra_steps <= 0:
                raise ValueError("extra_steps must be positive")
            # More steps overall, giving finer resolution near clean end.
            new_total = sampling_timesteps + extra_steps
            return np.linspace(1.0, 0.0, new_total + 1).tolist()

        elif modification_type == "reduce_total":
            new_total = params.get("new_total")
            if new_total is None:
                raise ValueError("reduce_total requires 'new_total' parameter")
            if new_total >= sampling_timesteps:
                raise ValueError(
                    f"new_total ({new_total}) must be less than "
                    f"original total ({sampling_timesteps})"
                )
            if new_total <= 0:
                raise ValueError("new_total must be positive")
            return np.linspace(1.0, 0.0, new_total + 1).tolist()

        else:  # custom
            custom_timesteps = params.get("timesteps")
            if custom_timesteps is None:
                raise ValueError("custom requires 'timesteps' parameter")
            if len(custom_timesteps) == 0:
                raise ValueError("custom timesteps must be non-empty")
            ScheduleModifier._validate_monotonic_descending_float(
                custom_timesteps, max_val=1.0, min_val=0.0
            )
            return [float(t) for t in custom_timesteps]

    @staticmethod
    def compute_relative_positions(
        original_timesteps: list,
        modified_timesteps: list,
    ) -> list[float]:
        """Map modified timesteps to relative positions [0, 1] matching original.

        For each modified timestep, computes its relative position within the
        span of the original schedule. Position 0.0 corresponds to the first
        element of the original sequence and 1.0 to the last.

        Args:
            original_timesteps: The original timestep sequence.
            modified_timesteps: The modified timestep sequence.

        Returns:
            List of floats in [0.0, 1.0] with the same length as
            modified_timesteps.
        """
        if len(modified_timesteps) == 0:
            return []
        if len(original_timesteps) == 0:
            return [0.0] * len(modified_timesteps)

        orig = [float(t) for t in original_timesteps]
        orig_min = min(orig)
        orig_max = max(orig)
        span = orig_max - orig_min

        if span == 0:
            return [0.5] * len(modified_timesteps)

        positions = []
        for t in modified_timesteps:
            pos = (float(t) - orig_min) / span
            # Clamp to [0, 1]
            pos = max(0.0, min(1.0, pos))
            positions.append(pos)
        return positions

    # ------------------------------------------------------------------
    # Validation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_modification_type(modification_type: str) -> None:
        if modification_type not in ScheduleModifier.VALID_MODIFICATION_TYPES:
            raise ValueError(
                f"Invalid modification_type '{modification_type}'. "
                f"Must be one of {sorted(ScheduleModifier.VALID_MODIFICATION_TYPES)}"
            )

    @staticmethod
    def _validate_monotonic_descending(
        timesteps: list, max_val: int, min_val: int
    ) -> None:
        """Validate integer timesteps are monotonically descending and in range."""
        for i, t in enumerate(timesteps):
            if not isinstance(t, (int, np.integer)):
                raise ValueError(f"Timestep at index {i} must be an integer, got {type(t)}")
            if t < min_val or t > max_val:
                raise ValueError(
                    f"Timestep {t} at index {i} is out of range [{min_val}, {max_val}]"
                )
            if i > 0 and t >= timesteps[i - 1]:
                raise ValueError(
                    f"Timesteps must be monotonically descending: "
                    f"timesteps[{i - 1}]={timesteps[i - 1]} >= timesteps[{i}]={t}"
                )

    @staticmethod
    def _validate_monotonic_descending_float(
        timesteps: list, max_val: float, min_val: float
    ) -> None:
        """Validate float timesteps are monotonically descending and in range."""
        for i, t in enumerate(timesteps):
            val = float(t)
            if val < min_val or val > max_val:
                raise ValueError(
                    f"Timestep {val} at index {i} is out of range [{min_val}, {max_val}]"
                )
            if i > 0 and val >= float(timesteps[i - 1]):
                raise ValueError(
                    f"Timesteps must be monotonically descending: "
                    f"timesteps[{i - 1}]={timesteps[i - 1]} >= timesteps[{i}]={val}"
                )
