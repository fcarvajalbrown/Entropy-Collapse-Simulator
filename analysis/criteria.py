"""
analysis/criteria.py
====================
Head-to-head comparison of four progressive-collapse criteria.

For a fair comparison every criterion is evaluated on the *same* incremental
analysis. At each load step the frame is solved, members that exceed their
strength are removed (alternate-load-path re-analysis), and four scalar
indicators are recorded:

    1. Displacement criterion  — peak nodal translation exceeds a drift limit.
    2. DCR criterion           — peak demand-capacity ratio reaches yielding.
    3. Energy criterion        — total strain energy exceeds a multiple of the
                                 intact (first-step) value (energy blow-up).
    4. Entropy criterion       — the step-to-step drop in normalized strain-
                                 energy entropy is a statistical outlier
                                 (calibration-free z-score), i.e. energy has
                                 suddenly localized.

The point of the comparison is methodological: the displacement, DCR and
energy criteria each need a structure-specific threshold (a drift limit, an
acceptance DCR, an energy multiple), whereas the entropy criterion adapts to
the structure's own history and needs no manual calibration. The driver
reports, for each criterion, the first load step at which it fires so the
detection ordering can be tabulated and plotted.

References for the comparison framing: GSA (2003) and UFC 4-023-03
(displacement / DCR acceptance); Feng et al. (2024) compare displacement-,
resistance- and energy-based criteria for RC frames under column removal —
this module adds the information-theoretic (entropy) criterion they omit.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from core.models import FrameData
from solver.equilibrium import solve_full
from solver.failure import check_and_apply_failures, all_failed, _combined_stress
from entropy.metrics import compute as compute_entropy, normalized_entropy
from entropy.robustness import is_stable


@dataclass
class StepMetrics:
    """All four indicators recorded at a single load step."""
    step: int
    load_factor: float
    peak_displacement: float     # max |translation| over free DOFs (m)
    peak_dcr: float              # max sigma_max / sigma_y over active members
    total_energy: float          # sum of member strain energies (J)
    entropy_norm: float          # S / ln(N) over surviving members
    delta_entropy: float         # change in (un-normalized) entropy vs prev step
    n_failed: int                # cumulative failed members


@dataclass
class CriteriaComparison:
    """
    Result of comparing the four criteria over one incremental analysis.

    first_trigger maps a criterion name to the first step index at which it
    fired (or None if it never fired within the run).
    """
    frame_name: str
    history: List[StepMetrics]
    first_trigger: Dict[str, Optional[int]]
    failed_sequence: List[int]


# ---------------------------------------------------------------------------
# Metric extraction
# ---------------------------------------------------------------------------

def _peak_translation(u: np.ndarray, frame: FrameData) -> float:
    """Largest absolute translational displacement over all free DOFs (m)."""
    fixed = {(n.id, d) for n in frame.nodes for d in n.fixed_dofs}
    peak = 0.0
    for node in frame.nodes:
        for dof in (0, 1):  # ux, uy (in-plane translations)
            if (node.id, dof) in fixed:
                continue
            peak = max(peak, abs(u[node.id * 6 + dof]))
    return peak


def _peak_dcr(u: np.ndarray, frame: FrameData) -> float:
    """Largest demand-capacity ratio sigma_max/sigma_y over active members."""
    peak = 0.0
    for member in frame.members:
        if member.failed:
            continue
        dcr = _combined_stress(member, u, frame) / member.sigma_y
        peak = max(peak, dcr)
    return peak


def run_with_metrics(
    frame: FrameData,
    max_steps: int = 100,
    load_factor_start: float = 1.0,
    load_factor_step: float = 0.1,
) -> tuple[List[StepMetrics], List[int]]:
    """
    Run an incremental-load progressive analysis, recording all four
    indicators at each step. Failed members are removed and the load is
    redistributed by re-analysis (no phenomenological law).

    Args:
        frame: Intact frame (deep-copied internally; caller's frame untouched).
        max_steps: Maximum number of load steps.
        load_factor_start: Initial load multiplier.
        load_factor_step: Load multiplier increment per step.

    Returns:
        (history, failed_sequence)
    """
    work = copy.deepcopy(frame)
    history: List[StepMetrics] = []
    failed_sequence: List[int] = []
    prev_entropy = 0.0

    for step in range(max_steps):
        load_factor = load_factor_start + step * load_factor_step

        # Stop if accumulated failures have formed a mechanism (see runner.py).
        if not is_stable(work):
            break

        u, energy_state = solve_full(work, step, load_factor=load_factor)
        record = compute_entropy(energy_state, prev_entropy)
        n_active = sum(1 for ms in energy_state.member_states if not ms.failed)

        history.append(StepMetrics(
            step=step,
            load_factor=load_factor,
            peak_displacement=_peak_translation(u, work),
            peak_dcr=_peak_dcr(u, work),
            total_energy=energy_state.total_energy,
            entropy_norm=normalized_entropy(record, n_active),
            delta_entropy=record.delta_entropy,
            n_failed=len(failed_sequence),
        ))
        prev_entropy = record.entropy

        newly_failed = check_and_apply_failures(work, energy_state, load_factor=load_factor)
        failed_sequence.extend(newly_failed)

        if all_failed(work):
            break

    return history, failed_sequence


# ---------------------------------------------------------------------------
# Criterion evaluators (first-crossing detection)
# ---------------------------------------------------------------------------

def _first_where(values: List[float], predicate) -> Optional[int]:
    """Index of the first value satisfying predicate, or None."""
    for i, v in enumerate(values):
        if predicate(v):
            return i
    return None


def _entropy_trigger(history: List[StepMetrics],
                     z_threshold: float = 3.0,
                     min_history: int = 5) -> Optional[int]:
    """
    Causal z-score outlier test on the entropy drop dS. A step fires when its
    dS sits more than z_threshold standard deviations below the mean of the
    strictly preceding steps. For a linear-elastic frame the entropy is scale
    invariant, so the baseline is flat until the first failure; in that case a
    clearly negative drop is treated as maximally significant. No
    structure-specific threshold is required.
    """
    deltas = np.array([m.delta_entropy for m in history], dtype=float)
    if deltas.size:
        deltas[0] = 0.0   # first step has no predecessor
    DROP_FLOOR = 1e-3

    for i in range(min_history, len(history)):
        baseline = deltas[:i]
        std = baseline.std()
        if std <= DROP_FLOOR:
            if deltas[i] < -DROP_FLOOR:
                return i
            continue
        if (deltas[i] - baseline.mean()) / std < -z_threshold:
            return i
    return None


def compare(
    frame: FrameData,
    max_steps: int = 100,
    load_factor_start: float = 1.0,
    load_factor_step: float = 0.1,
    drift_limit: float = 0.05,
    dcr_limit: float = 1.0,
    energy_factor: float = 3.0,
    entropy_z: float = 3.0,
) -> CriteriaComparison:
    """
    Run the incremental analysis and report the first triggering step for each
    of the four criteria.

    Args:
        frame: Intact frame to analyse.
        max_steps, load_factor_start, load_factor_step: incremental loading.
        drift_limit: Peak-translation limit (m) for the displacement criterion.
        dcr_limit: DCR at which the DCR criterion fires (1.0 = first yield).
        energy_factor: Multiple of the intact total energy that fires the
                       energy criterion.
        entropy_z: Z-score cutoff for the entropy criterion.

    Returns:
        CriteriaComparison with the per-step history and first-trigger map.
    """
    history, failed_sequence = run_with_metrics(
        frame, max_steps, load_factor_start, load_factor_step
    )

    disp = [m.peak_displacement for m in history]
    dcr = [m.peak_dcr for m in history]
    # Strain energy scales with the square of the load factor for a fixed
    # topology, so compare the COMPLIANCE U/lambda^2 (energy at unit load).
    # It is constant for the intact elastic frame and jumps only when members
    # soften or are removed -- a genuine collapse signal rather than a load
    # artefact.
    compliance = [m.total_energy / (m.load_factor ** 2) for m in history]
    intact_compliance = compliance[0] if compliance else 0.0

    first_trigger: Dict[str, Optional[int]] = {
        "displacement": _first_where(disp, lambda v: v > drift_limit),
        "dcr": _first_where(dcr, lambda v: v >= dcr_limit),
        "energy": _first_where(
            compliance,
            lambda v: intact_compliance > 0 and v > energy_factor * intact_compliance
        ),
        "entropy": _entropy_trigger(history, z_threshold=entropy_z),
    }

    return CriteriaComparison(
        frame_name=frame.name,
        history=history,
        first_trigger=first_trigger,
        failed_sequence=failed_sequence,
    )
