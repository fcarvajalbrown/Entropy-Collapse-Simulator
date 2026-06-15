"""
entropy/localization.py
=======================
Detects structural collapse by analyzing the entropy history.

Collapse is signaled by a sharp negative spike in dS/dt — the entropy
drops suddenly as energy concentrates into a shrinking set of members.

Two detection strategies are provided:
  - Threshold-based: dS < -threshold (simple, fast)
  - Z-score-based:   dS deviates beyond N standard deviations from the
                     rolling mean (adaptive, more robust for research use)

Consumed by simulation/runner.py each step to decide whether to halt.
"""

import numpy as np
from core.models import EntropyRecord


def detect_collapse_threshold(
    history: list[EntropyRecord],
    threshold: float = -0.5
) -> tuple[bool, int | None]:
    """
    Detect collapse when dS/dt drops below a fixed negative threshold.

    Simple and interpretable. Suitable when the frame and loading are
    well understood and a calibrated threshold can be chosen.

    Args:
        history: Full entropy record history up to the current step.
        threshold: Negative dS value below which collapse is declared.
                   Default -0.5 (half a nat drop per step).

    Returns:
        (collapsed, step): collapsed is True if detected, step is the
        index where it occurred, or (False, None) if not yet detected.
    """
    for record in history:
        if record.delta_entropy < threshold:
            return True, record.step
    return False, None


def detect_collapse_zscore(
    history: list[EntropyRecord],
    z_threshold: float = 3.0,
    min_history: int = 5
) -> tuple[bool, int | None]:
    """
    Detect collapse when dS/dt drops more than z_threshold standard deviations
    below the mean of the *preceding* dS values.

    The baseline statistics (mean, std) for each candidate step are computed
    from the points that came strictly before it. This makes the test causal —
    a step is judged only against history available up to that point — and
    avoids the common pitfall of inflating the baseline variance with the very
    spike one is trying to detect. The method is adaptive: it needs no
    per-frame threshold calibration.

    Args:
        history: Full entropy record history up to the current step.
        z_threshold: Number of standard deviations below the prior mean that
                     flags collapse. Default 3.0 (extreme negative outlier).
        min_history: Minimum number of preceding points required before the
                     test activates. Prevents false positives in early steps.

    Returns:
        (collapsed, step): collapsed is True if detected, step is the
        index where it occurred, or (False, None) if not yet detected.
    """
    deltas = np.array([r.delta_entropy for r in history], dtype=float)
    if deltas.size:
        deltas[0] = 0.0   # first step has no predecessor; its dS is an artifact

    # Numerical floor separating a genuine entropy drop (~0.1-1 nats at a
    # failure) from floating-point noise (~1e-16) on a flat history.
    DROP_FLOOR = 1e-3

    for i in range(min_history, len(history)):
        baseline = deltas[:i]            # strictly preceding points
        mean = baseline.mean()
        std = baseline.std()
        if std <= DROP_FLOOR:
            # Flat baseline: a clearly negative drop is maximally significant.
            if deltas[i] < -DROP_FLOOR:
                return True, history[i].step
            continue
        z = (deltas[i] - mean) / std
        if z < -z_threshold:
            return True, history[i].step

    return False, None


def most_localized_members(
    record: EntropyRecord,
    top_n: int = 3
) -> list[tuple[int, float]]:
    """
    Return the top_n members with the highest normalized energy fraction p_i.

    Identifies which members are accumulating the most energy — the likely
    failure candidates in the next step.

    Args:
        record: Current entropy record with energy distribution.
        top_n: Number of highest-energy members to return.

    Returns:
        List of (member_id, p_i) tuples sorted by p_i descending.
    """
    sorted_dist = sorted(record.energy_distribution, key=lambda x: x[1], reverse=True)
    return sorted_dist[:top_n]


def localization_index(record: EntropyRecord) -> float:
    """
    Compute a scalar localization index from the energy distribution.

    Defined as the Gini coefficient of the p_i distribution.
    0.0 = perfectly uniform (maximum entropy, safe).
    1.0 = all energy in one member (maximum localization, critical).

    Args:
        record: Current entropy record.

    Returns:
        Gini coefficient in [0, 1].
    """
    if not record.energy_distribution:
        return 0.0

    values = np.array([p for _, p in record.energy_distribution], dtype=float)
    values = np.sort(values)
    n = len(values)

    if n == 0 or values.sum() == 0:
        return 0.0

    # Standard Gini formula
    index = np.arange(1, n + 1)
    return float((2 * np.sum(index * values)) / (n * values.sum()) - (n + 1) / n)