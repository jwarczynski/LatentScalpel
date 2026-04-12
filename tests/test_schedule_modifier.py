"""Property-based and unit tests for geniesae.schedule_modifier."""

from __future__ import annotations

from hypothesis import given, settings, strategies as st

from geniesae.genie_model import DiffusionHelper
from geniesae.schedule_modifier import ScheduleModifier


# ---------------------------------------------------------------------------
# Hypothesis strategies
# ---------------------------------------------------------------------------

# Genie total timesteps — keep small-ish to avoid slow DiffusionHelper init
_genie_total_st = st.integers(min_value=10, max_value=200)

# PLAID sampling timesteps
_plaid_total_st = st.integers(min_value=10, max_value=200)


@st.composite
def genie_modification_st(draw: st.DrawFn) -> tuple[int, str, dict]:
    """Generate a valid (total_timesteps, modification_type, params) for Genie.

    Returns a tuple that can be fed directly to
    ``ScheduleModifier.modify_genie_schedule``.
    """
    total = draw(_genie_total_st)
    mod_type = draw(st.sampled_from(["compress_early", "extend_late", "reduce_total", "custom"]))

    if mod_type == "compress_early":
        # speedup_steps must be in [1, total-1]
        speedup = draw(st.integers(min_value=1, max_value=total - 1))
        params = {"speedup_steps": speedup}

    elif mod_type == "extend_late":
        extra = draw(st.integers(min_value=1, max_value=100))
        params = {"extra_steps": extra}

    elif mod_type == "reduce_total":
        # new_total must be in [1, total-1]
        new_total = draw(st.integers(min_value=1, max_value=total - 1))
        params = {"new_total": new_total}

    else:  # custom
        # Generate a non-empty strictly descending list of ints in [0, total-1]
        n = draw(st.integers(min_value=1, max_value=min(total, 50)))
        values = sorted(
            draw(
                st.lists(
                    st.integers(min_value=0, max_value=total - 1),
                    min_size=n,
                    max_size=n,
                    unique=True,
                )
            ),
            reverse=True,
        )
        params = {"timesteps": values}

    return total, mod_type, params


@st.composite
def plaid_modification_st(draw: st.DrawFn) -> tuple[int, str, dict]:
    """Generate a valid (sampling_timesteps, modification_type, params) for PLAID.

    Returns a tuple that can be fed directly to
    ``ScheduleModifier.modify_plaid_schedule``.
    """
    total = draw(_plaid_total_st)
    mod_type = draw(st.sampled_from(["compress_early", "extend_late", "reduce_total", "custom"]))

    if mod_type == "compress_early":
        speedup = draw(st.integers(min_value=1, max_value=total - 1))
        params = {"speedup_steps": speedup}

    elif mod_type == "extend_late":
        extra = draw(st.integers(min_value=1, max_value=100))
        params = {"extra_steps": extra}

    elif mod_type == "reduce_total":
        new_total = draw(st.integers(min_value=1, max_value=total - 1))
        params = {"new_total": new_total}

    else:  # custom
        # Generate a non-empty strictly descending list of floats in [0.0, 1.0]
        n = draw(st.integers(min_value=1, max_value=50))
        values = sorted(
            draw(
                st.lists(
                    st.floats(min_value=0.0, max_value=1.0,
                              allow_nan=False, allow_infinity=False),
                    min_size=n,
                    max_size=n,
                    unique=True,
                )
            ),
            reverse=True,
        )
        # Ensure strictly descending (unique=True + sort handles this, but
        # floats can have duplicates at extreme precision — filter)
        deduped: list[float] = []
        for v in values:
            if not deduped or v < deduped[-1]:
                deduped.append(v)
        if not deduped:
            deduped = [0.5]
        params = {"timesteps": deduped}

    return total, mod_type, params


# ---------------------------------------------------------------------------
# Property 9: Schedule modification produces valid timesteps
# ---------------------------------------------------------------------------


class TestScheduleModificationValidTimesteps:
    """Property 9: Schedule modification produces valid timesteps.

    For any valid modification type (compress_early, extend_late,
    reduce_total, custom) with valid parameters, ScheduleModifier shall
    produce a non-empty, monotonic timestep sequence where all values are
    within the valid range for the model type (0 to diffusion_steps-1 for
    Genie, 0.0 to 1.0 for PLAID).

    **Validates: Requirements 5.2, 8.3**
    """

    @given(data=genie_modification_st())
    @settings(max_examples=200, deadline=None)
    def test_genie_schedule_modification_produces_valid_timesteps(
        self, data: tuple[int, str, dict]
    ) -> None:
        # Feature: temporal-feature-analysis, Property 9: Schedule modification produces valid timesteps
        total, mod_type, params = data
        helper = DiffusionHelper(num_timesteps=total, schedule_name="sqrt")

        new_helper, timesteps = ScheduleModifier.modify_genie_schedule(
            helper, mod_type, params
        )

        # Non-empty
        assert len(timesteps) > 0, "Timestep sequence must be non-empty"

        # All values within valid range [0, new_helper.num_timesteps - 1]
        max_val = new_helper.num_timesteps - 1
        for i, t in enumerate(timesteps):
            assert isinstance(t, int), (
                f"Timestep at index {i} should be int, got {type(t)}"
            )
            assert 0 <= t <= max_val, (
                f"Timestep {t} at index {i} out of range [0, {max_val}]"
            )

        # Monotonically descending (each element strictly less than previous)
        for i in range(1, len(timesteps)):
            assert timesteps[i] <= timesteps[i - 1], (
                f"Timesteps not monotonically descending: "
                f"timesteps[{i - 1}]={timesteps[i - 1]}, timesteps[{i}]={timesteps[i]}"
            )

    @given(data=plaid_modification_st())
    @settings(max_examples=200, deadline=None)
    def test_plaid_schedule_modification_produces_valid_timesteps(
        self, data: tuple[int, str, dict]
    ) -> None:
        # Feature: temporal-feature-analysis, Property 9: Schedule modification produces valid timesteps
        total, mod_type, params = data

        timesteps = ScheduleModifier.modify_plaid_schedule(
            total, mod_type, params
        )

        # Non-empty
        assert len(timesteps) > 0, "Timestep sequence must be non-empty"

        # All values within valid range [0.0, 1.0]
        for i, t in enumerate(timesteps):
            assert isinstance(t, float), (
                f"Timestep at index {i} should be float, got {type(t)}"
            )
            assert 0.0 <= t <= 1.0, (
                f"Timestep {t} at index {i} out of range [0.0, 1.0]"
            )

        # Monotonically descending (non-increasing)
        for i in range(1, len(timesteps)):
            assert timesteps[i] <= timesteps[i - 1], (
                f"Timesteps not monotonically descending: "
                f"timesteps[{i - 1}]={timesteps[i - 1]}, timesteps[{i}]={timesteps[i]}"
            )


# ---------------------------------------------------------------------------
# Hypothesis strategies for Property 10
# ---------------------------------------------------------------------------

# Generate timestep sequences that can be either int (Genie) or float (PLAID).
# The sequences must be monotonic (ascending or descending) to represent valid
# schedules.

@st.composite
def _monotonic_timestep_seq(draw: st.DrawFn, *, min_size: int = 1) -> list:
    """Generate a monotonic timestep sequence (int or float).

    Randomly chooses between integer (Genie-style) and float (PLAID-style)
    timesteps, and between ascending and descending order.
    """
    use_float = draw(st.booleans())
    descending = draw(st.booleans())
    n = draw(st.integers(min_value=min_size, max_value=50))

    if use_float:
        values = sorted(
            draw(
                st.lists(
                    st.floats(min_value=0.0, max_value=1.0,
                              allow_nan=False, allow_infinity=False),
                    min_size=n,
                    max_size=n,
                )
            )
        )
    else:
        values = sorted(
            draw(
                st.lists(
                    st.integers(min_value=0, max_value=2000),
                    min_size=n,
                    max_size=n,
                )
            )
        )

    if descending:
        values = list(reversed(values))
    return values


# ---------------------------------------------------------------------------
# Property 10: Relative position matching for schedule comparison
# ---------------------------------------------------------------------------


class TestRelativePositionMatching:
    """Property 10: Relative position matching for schedule comparison.

    For any original timestep sequence and modified timestep sequence,
    ``ScheduleModifier.compute_relative_positions()`` shall return a list of
    the same length as the modified sequence, where each value is in
    [0.0, 1.0], and the values are monotonically ordered in the same
    direction as the input.

    **Validates: Requirements 5.3**
    """

    @given(
        original=_monotonic_timestep_seq(min_size=2),
        modified=_monotonic_timestep_seq(min_size=1),
    )
    @settings(max_examples=200, deadline=None)
    def test_relative_positions_length_range_and_monotonicity(
        self, original: list, modified: list,
    ) -> None:
        # Feature: temporal-feature-analysis, Property 10: Relative position matching for schedule comparison
        positions = ScheduleModifier.compute_relative_positions(original, modified)

        # Same length as modified sequence
        assert len(positions) == len(modified), (
            f"Expected {len(modified)} positions, got {len(positions)}"
        )

        # All values in [0.0, 1.0]
        for i, p in enumerate(positions):
            assert isinstance(p, float), (
                f"Position at index {i} should be float, got {type(p)}"
            )
            assert 0.0 <= p <= 1.0, (
                f"Position {p} at index {i} out of range [0.0, 1.0]"
            )

        # Monotonically ordered in the same direction as modified input.
        # Determine direction of modified sequence.
        if len(modified) >= 2:
            # Check if modified is non-decreasing or non-increasing
            is_non_decreasing = all(
                float(modified[i]) <= float(modified[i + 1])
                for i in range(len(modified) - 1)
            )
            is_non_increasing = all(
                float(modified[i]) >= float(modified[i + 1])
                for i in range(len(modified) - 1)
            )

            if is_non_decreasing:
                for i in range(len(positions) - 1):
                    assert positions[i] <= positions[i + 1], (
                        f"Positions not non-decreasing: "
                        f"positions[{i}]={positions[i]} > positions[{i + 1}]={positions[i + 1]}"
                    )
            elif is_non_increasing:
                for i in range(len(positions) - 1):
                    assert positions[i] >= positions[i + 1], (
                        f"Positions not non-increasing: "
                        f"positions[{i}]={positions[i]} < positions[{i + 1}]={positions[i + 1]}"
                    )

