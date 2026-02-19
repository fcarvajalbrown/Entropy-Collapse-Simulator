"""
simulation/runner.py
====================
Orchestrates the full progressive collapse simulation loop.

Each step:
  1. Solve equilibrium (Ku = F) → EnergyState
  2. Compute entropy metrics → EntropyRecord
  3. Check for collapse detection → halt if detected
  4. Check for member failures → mark failed members
  5. Redistribute energy among surviving members
  6. Repeat until collapse, full failure, or max steps reached

Inputs:  FrameData (from any frame in structure/frames/)
Outputs: SimulationResult (consumed by visualization/)
"""

from core.models import FrameData, SimulationResult, EnergyState, EntropyRecord
from solver.equilibrium import solve
from solver.failure import check_and_apply_failures, all_failed
from solver.redistribution import redistribute
from entropy.metrics import compute as compute_entropy
from entropy.localization import detect_collapse_zscore, detect_collapse_threshold


def run(
    frame: FrameData,
    max_steps: int = 100,
    redistribution_dt: float = 1.0,
    collapse_method: str = "zscore",
    collapse_threshold: float = -0.5,
    collapse_zscore: float = 3.0,
) -> SimulationResult:
    """
    Execute the full progressive collapse simulation for a given frame.

    Args:
        frame: Fully defined structural frame (nodes, members, loads).
        max_steps: Maximum number of load/failure steps before stopping.
        redistribution_dt: Time step for the energy redistribution ODE.
        collapse_method: Detection strategy — "zscore" or "threshold".
        collapse_threshold: dS threshold for threshold-based detection.
        collapse_zscore: Z-score cutoff for zscore-based detection.

    Returns:
        SimulationResult with full energy and entropy history.
    """
    energy_history: list[EnergyState] = []
    entropy_history: list[EntropyRecord] = []
    failed_sequence: list[int] = []
    previous_entropy = 0.0

    for step in range(max_steps):

        # --- Step 1: Solve equilibrium ---
        energy_state = solve(frame, step)
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
        newly_failed = check_and_apply_failures(frame, energy_state)
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

        # --- Step 5: Redistribute energy if failures occurred ---
        if newly_failed:
            energy_state = redistribute(frame, energy_state, redistribution_dt)
            energy_history[-1] = energy_state  # Replace with post-redistribution state

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
        raise ValueError(f"Unknown collapse detection method: '{method}'. Use 'zscore' or 'threshold'.")"""
simulation/runner.py
====================
Orchestrates the full progressive collapse simulation loop.

Each step:
  1. Solve equilibrium (Ku = F) → EnergyState
  2. Compute entropy metrics → EntropyRecord
  3. Check for collapse detection → halt if detected
  4. Check for member failures → mark failed members
  5. Redistribute energy among surviving members
  6. Repeat until collapse, full failure, or max steps reached

Inputs:  FrameData (from any frame in structure/frames/)
Outputs: SimulationResult (consumed by visualization/)
"""

from core.models import FrameData, SimulationResult, EnergyState, EntropyRecord
from solver.equilibrium import solve
from solver.failure import check_and_apply_failures, all_failed
from solver.redistribution import redistribute
from entropy.metrics import compute as compute_entropy
from entropy.localization import detect_collapse_zscore, detect_collapse_threshold


def run(
    frame: FrameData,
    max_steps: int = 100,
    redistribution_dt: float = 1.0,
    collapse_method: str = "zscore",
    collapse_threshold: float = -0.5,
    collapse_zscore: float = 3.0,
) -> SimulationResult:
    """
    Execute the full progressive collapse simulation for a given frame.

    Args:
        frame: Fully defined structural frame (nodes, members, loads).
        max_steps: Maximum number of load/failure steps before stopping.
        redistribution_dt: Time step for the energy redistribution ODE.
        collapse_method: Detection strategy — "zscore" or "threshold".
        collapse_threshold: dS threshold for threshold-based detection.
        collapse_zscore: Z-score cutoff for zscore-based detection.

    Returns:
        SimulationResult with full energy and entropy history.
    """
    energy_history: list[EnergyState] = []
    entropy_history: list[EntropyRecord] = []
    failed_sequence: list[int] = []
    previous_entropy = 0.0

    for step in range(max_steps):

        # --- Step 1: Solve equilibrium ---
        energy_state = solve(frame, step)
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
        newly_failed = check_and_apply_failures(frame, energy_state)
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

        # --- Step 5: Redistribute energy if failures occurred ---
        if newly_failed:
            energy_state = redistribute(frame, energy_state, redistribution_dt)
            energy_history[-1] = energy_state  # Replace with post-redistribution state

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