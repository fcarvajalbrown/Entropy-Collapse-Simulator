"""
entropy/metrics.py
==================
Computes structural entropy from an EnergyState.

Structural entropy is defined as:
    S = -sum( p_i * ln(p_i) )

where p_i = U_i / sum(U) is the normalized energy fraction of member i.

High S → energy is evenly distributed (structure is stable).
Low S  → energy is concentrated in few members (localization, danger).
dS/dt → rate of change; a sharp negative spike signals imminent collapse.

This is the novel contribution of the project. All other modules exist
to feed clean EnergyState data into this one.
"""

import numpy as np
from core.models import EnergyState, EntropyRecord


def compute(energy_state: EnergyState, previous_entropy: float = 0.0) -> EntropyRecord:
    """
    Compute structural entropy and its rate of change for one time step.

    Args:
        energy_state: Current energy distribution across all members.
        previous_entropy: S value from the previous step, used to compute dS/dt.
                          Pass 0.0 for the first step.

    Returns:
        EntropyRecord with S, dS, and normalized energy distribution.
    """
    active = [(ms.member_id, ms.strain_energy)
              for ms in energy_state.member_states if not ms.failed]

    if not active:
        return EntropyRecord(
            step=energy_state.step,
            entropy=0.0,
            delta_entropy=0.0,
            energy_distribution=[]
        )

    ids, energies = zip(*active)
    energies = np.array(energies, dtype=float)

    total = energies.sum()

    if total == 0.0:
        # No energy in system — fully unloaded or pre-collapse
        distribution = [(mid, 0.0) for mid in ids]
        return EntropyRecord(
            step=energy_state.step,
            entropy=0.0,
            delta_entropy=0.0 - previous_entropy,
            energy_distribution=distribution
        )

    p = energies / total
    entropy = _shannon_entropy(p)
    delta_entropy = entropy - previous_entropy
    distribution = list(zip(ids, p.tolist()))

    return EntropyRecord(
        step=energy_state.step,
        entropy=entropy,
        delta_entropy=delta_entropy,
        energy_distribution=distribution
    )


def _shannon_entropy(p: np.ndarray) -> float:
    """
    Compute Shannon entropy H = -sum(p_i * ln(p_i)) over a probability vector.

    Zero-probability terms are excluded (0 * ln(0) = 0 by convention).

    Args:
        p: Normalized probability vector (must sum to 1.0).

    Returns:
        Entropy value in nats.
    """
    nonzero = p[p > 0]
    return float(-np.sum(nonzero * np.log(nonzero)))


def max_entropy(n_active_members: int) -> float:
    """
    Compute the theoretical maximum entropy for a given number of active members.

    Maximum entropy occurs when energy is perfectly uniform: p_i = 1/n for all i.
    S_max = ln(n)

    Useful for normalizing entropy to a [0, 1] scale in plots.

    Args:
        n_active_members: Number of non-failed members.

    Returns:
        S_max = ln(n), or 0.0 if n <= 1.
    """
    if n_active_members <= 1:
        return 0.0
    return float(np.log(n_active_members))


def normalized_entropy(record: EntropyRecord, n_active_members: int) -> float:
    """
    Return entropy normalized to [0, 1] relative to the theoretical maximum.

    Args:
        record: Entropy record for the current step.
        n_active_members: Number of non-failed members at this step.

    Returns:
        S / S_max, or 0.0 if S_max is zero.
    """
    s_max = max_entropy(n_active_members)
    if s_max == 0.0:
        return 0.0
    return record.entropy / s_max