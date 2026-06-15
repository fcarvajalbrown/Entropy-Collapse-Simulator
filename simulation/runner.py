"""
simulation/runner.py
====================
Orchestrates the full progressive collapse simulation loop.

Each step:
  1. Solve equilibrium (Ku = F) → EnergyState
  2. Compute entropy metrics → EntropyRecord
  3. Check for collapse detection → halt if detected
  4. Check for member failures → mark failed members
  5. Repeat until collapse, full failure, or max steps reached

Energy redistribution after a member fails is NOT modelled with a separate
phenomenological law. When a member is marked failed, it is excluded from the
global stiffness assembly, so the next equilibrium solve automatically routes
its load through the surviving members. This re-analysis IS the physically
exact alternate-load-path redistribution prescribed by progressive-collapse
guidance (e.g. UFC 4-023-03), and it requires no tuning coefficients.

Inputs:  FrameData (from any frame in structure/frames/)
Outputs: SimulationResult (consumed by visualization/)
"""

from core.models import FrameData, SimulationResult, EnergyState, EntropyRecord
from solver.equilibrium import solve
from solver.failure import check_and_apply_failures, all_failed
from entropy.metrics import compute as compute_entropy
from entropy.localization import detect_collapse_zscore, detect_collapse_threshold
from entropy.robustness import is_stable


def run(
    frame: FrameData,
    max_steps: int = 100,
    collapse_method: str = "zscore",
    collapse_threshold: float = -0.5,
    collapse_zscore: float = 3.0,
    load_factor_start: float = 1.0,
    load_factor_step: float = 0.0,
) -> SimulationResult:
    """
    Execute the full progressive collapse simulation for a given frame.

    Args:
        frame: Fully defined structural frame (nodes, members, loads).
        max_steps: Maximum number of load/failure steps before stopping.
        collapse_method: Detection strategy — "zscore" or "threshold".
        collapse_threshold: dS threshold for threshold-based detection.
        collapse_zscore: Z-score cutoff for zscore-based detection.
        load_factor_start: Initial load multiplier applied to all loads.
                           Default 1.0 = design load. Use e.g. 0.5 to start at
                           half load and ramp up.
        load_factor_step: Increment added to the load factor each step.
                          Default 0.0 = static loading (no incremental ramp).
                          Set e.g. 0.1 to increase load by 10% per step and
                          drive progressive failures under real material limits.

    Returns:
        SimulationResult with full energy and entropy history.
    """
    energy_history: list[EnergyState] = []
    entropy_history: list[EntropyRecord] = []
    failed_sequence: list[int] = []
    previous_entropy = 0.0

    for step in range(max_steps):

        load_factor = load_factor_start + step * load_factor_step

        # --- Step 0: Stability gate ---
        # If accumulated failures have turned the frame into a mechanism, the
        # equilibrium solve would return meaningless displacements. Treat this
        # as collapse and stop rather than feed garbage downstream.
        if not is_stable(frame):
            return SimulationResult(
                frame_name=frame.name,
                energy_history=energy_history,
                entropy_history=entropy_history,
                collapse_detected=True,
                collapse_step=step,
                failed_sequence=failed_sequence,
            )

        # --- Step 1: Solve equilibrium ---
        energy_state = solve(frame, step, load_factor=load_factor)
        energy_history.append(energy_state)

        # --- Step 2: Compute entropy ---
        entropy_record = compute_entropy(energy_state, previous_entropy)
        entropy_history.append(entropy_record)
        previous_entropy = entropy_record.entropy

        # --- Step 3: Check for collapse ---
        collapsed, collapse_step = _detect(
            entropy_history, collapse_method, collapse_threshold, collapse_zscore
        )
        if collapsed:
            return SimulationResult(
                frame_name=frame.name,
                energy_history=energy_history,
                entropy_history=entropy_history,
                collapse_detected=True,
                collapse_step=collapse_step,
                failed_sequence=failed_sequence
            )

        # --- Step 4: Check member failures ---
        newly_failed = check_and_apply_failures(frame, energy_state, load_factor=load_factor)
        failed_sequence.extend(newly_failed)

        if all_failed(frame):
            return SimulationResult(
                frame_name=frame.name,
                energy_history=energy_history,
                entropy_history=entropy_history,
                collapse_detected=True,
                collapse_step=step,
                failed_sequence=failed_sequence
            )

        # No explicit redistribution step: any member marked failed above is
        # dropped from the stiffness assembly, so the next iteration's solve()
        # re-routes its load through the surviving members automatically. This
        # re-analysis is the exact alternate-load-path redistribution.

    # Max steps reached without collapse
    return SimulationResult(
        frame_name=frame.name,
        energy_history=energy_history,
        entropy_history=entropy_history,
        collapse_detected=False,
        collapse_step=None,
        failed_sequence=failed_sequence
    )


def _detect(
    entropy_history: list[EntropyRecord],
    method: str,
    threshold: float,
    zscore: float
) -> tuple[bool, int | None]:
    """
    Dispatch to the selected collapse detection strategy.

    Args:
        entropy_history: Full entropy record history so far.
        method: "zscore" or "threshold".
        threshold: Used if method is "threshold".
        zscore: Used if method is "zscore".

    Returns:
        (collapsed, step) from the chosen detector.

    Raises:
        ValueError: If method is not recognized.
    """
    if method == "zscore":
        return detect_collapse_zscore(entropy_history, z_threshold=zscore)
    elif method == "threshold":
        return detect_collapse_threshold(entropy_history, threshold=threshold)
    else:
        raise ValueError(f"Unknown collapse detection method: '{method}'. Use 'zscore' or 'threshold'.")