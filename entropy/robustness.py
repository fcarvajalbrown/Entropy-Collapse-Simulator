"""
entropy/robustness.py
=====================
Entropy Robustness Index (R_S) for structural frames.

This module is the central methodological contribution of the project. It
turns the Shannon entropy of the strain-energy distribution into a
calibration-free, dimensionless robustness measure by combining it with the
standardized *alternate-load-path* (ALP) procedure of progressive-collapse
practice (e.g. UFC 4-023-03): each primary member is notionally removed, the
frame is re-analysed, and the entropy of the surviving members is recorded.

Definitions
-----------
Let the intact frame under the design load have normalized strain-energy
entropy

    H0 = S0 / ln(N0)                       (in [0, 1])

where S0 = -sum p_i ln p_i, p_i = U_i / sum(U), and N0 is the number of
members. For each notionally removed member k:

    H_k = S_k / ln(N_k)                     (normalized entropy after removal)
    dH_k = H0 - H_k                         (entropy drop / criticality of k)

If removing k turns the frame into a mechanism (singular reduced stiffness),
the structure cannot redistribute the load and we set H_k = 0 (maximal
localization). The **Entropy Robustness Index** is the mean post-removal
normalized entropy over the removed set R:

    R_S = (1 / |R|) * sum_k H_k             (in [0, 1])

R_S -> 1 : losing any single member barely changes how evenly energy is
           shared (high redundancy, robust).
R_S -> 0 : losing a member funnels energy into few members or collapses the
           frame (low redundancy, fragile).

The index is dimensionless and needs no per-structure threshold calibration,
which is its advantage over displacement- or DCR-based acceptance limits.

The worst single removal (max dH_k) identifies the most critical member; the
ordering of members by dH_k is an entropy-based importance ranking that can be
compared against established member-importance measures.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from core.models import FrameData
from solver.equilibrium import solve
from structure.stiffness import assemble_global_stiffness
from entropy.metrics import compute as compute_entropy, normalized_entropy


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------

@dataclass
class RemovalResult:
    """
    Outcome of notionally removing a single member and re-analysing.

    Attributes:
        member_id (int): The member that was removed.
        entropy_norm (float): Normalized entropy H_k of the surviving members
                              after removal (0.0 if removal causes a mechanism).
        entropy_drop (float): dH_k = H0 - H_k. Larger = more critical.
        stable (bool): False if removing the member left the frame kinematically
                       unstable (a mechanism); True otherwise.
        max_energy_fraction (float): Largest surviving p_i after removal
                                     (peak localization), or 1.0 if unstable.
    """
    member_id: int
    entropy_norm: float
    entropy_drop: float
    stable: bool
    max_energy_fraction: float


@dataclass
class RobustnessReport:
    """
    Full Entropy Robustness Index assessment of a frame.

    Attributes:
        frame_name (str): Name of the analysed frame.
        intact_entropy_norm (float): Baseline normalized entropy H0.
        robustness_index (float): R_S = mean post-removal normalized entropy
                                  over the whole removal set, with H_k = 0 for
                                  removals that form a mechanism.
        robustness_index_stable_only (float): mean H_k over the removals that
                                  remain stable (mechanisms excluded). For a
                                  structure with no single-loss mechanisms this
                                  equals robustness_index.
        mechanism_fraction (float): fraction of removals that form a mechanism.
        worst_case_entropy_norm (float): min_k H_k (most damaging single loss).
        critical_member (Optional[int]): Member whose removal maximises dH_k.
        removals (List[RemovalResult]): Per-member removal outcomes.
        unstable_members (List[int]): Members whose removal causes a mechanism.
    """
    frame_name: str
    intact_entropy_norm: float
    robustness_index: float
    robustness_index_stable_only: float
    mechanism_fraction: float
    worst_case_entropy_norm: float
    critical_member: Optional[int]
    removals: List[RemovalResult]
    unstable_members: List[int]

    def ranking(self) -> List[Tuple[int, float]]:
        """Return (member_id, dH_k) pairs sorted by criticality (descending)."""
        return sorted(
            ((r.member_id, r.entropy_drop) for r in self.removals),
            key=lambda x: x[1],
            reverse=True,
        )


# ---------------------------------------------------------------------------
# Stability test
# ---------------------------------------------------------------------------

def _free_dof_indices(frame: FrameData) -> List[int]:
    """Return global indices of all unconstrained degrees of freedom."""
    free: List[int] = []
    fixed = {(node.id, dof) for node in frame.nodes for dof in node.fixed_dofs}
    for node in frame.nodes:
        for dof in range(6):
            if (node.id, dof) not in fixed:
                free.append(node.id * 6 + dof)
    return free


def is_stable(frame: FrameData, tol: float = 1e-9) -> bool:
    """
    Test whether the current frame (with its failed members excluded) is
    kinematically stable, i.e. the stiffness matrix restricted to the free
    degrees of freedom is positive definite.

    A member removal that drops the smallest eigenvalue of the free-DOF
    stiffness to ~0 indicates a mechanism: the frame can deform at no energy
    cost and cannot carry the load through an alternate path.

    Args:
        frame: Frame whose stability is tested (failed members are skipped in
               assembly, so mark members failed before calling).
        tol: Relative eigenvalue tolerance (smallest/largest).

    Returns:
        True if stable (positive definite free-DOF stiffness), else False.
    """
    free = _free_dof_indices(frame)
    if not free:
        return True

    K = assemble_global_stiffness(frame)
    Kff = K[np.ix_(free, free)]

    # A free DOF with an all-zero row has no stiffness -> unconstrained.
    row_sums = np.abs(Kff).sum(axis=1)
    if np.any(row_sums <= 0.0):
        return False

    eigvals = np.linalg.eigvalsh(0.5 * (Kff + Kff.T))
    largest = np.abs(eigvals).max()
    if largest == 0.0:
        return False
    return bool(eigvals.min() > tol * largest)


# ---------------------------------------------------------------------------
# Single-member notional removal
# ---------------------------------------------------------------------------

def removal_entropy(
    frame: FrameData,
    member_id: int,
    load_factor: float = 1.0,
) -> RemovalResult:
    """
    Notionally remove one member, re-analyse via the alternate-load-path
    method, and report the resulting normalized entropy.

    The frame is deep-copied so the caller's frame is never mutated.

    Args:
        frame: Intact frame.
        member_id: Member to remove.
        load_factor: Load multiplier for the re-analysis (default 1.0).

    Returns:
        RemovalResult for this member.
    """
    work = copy.deepcopy(frame)
    target = next(m for m in work.members if m.id == member_id)
    target.failed = True

    # Baseline (intact) normalized entropy for the drop calculation.
    h0 = _intact_entropy_norm(frame, load_factor)

    if not is_stable(work):
        return RemovalResult(
            member_id=member_id,
            entropy_norm=0.0,
            entropy_drop=h0,           # full drop: load cannot be redistributed
            stable=False,
            max_energy_fraction=1.0,
        )

    energy_state = solve(work, step=0, load_factor=load_factor)
    record = compute_entropy(energy_state, previous_entropy=0.0)
    n_active = sum(1 for ms in energy_state.member_states if not ms.failed)
    h_k = normalized_entropy(record, n_active)

    p_values = [p for _, p in record.energy_distribution]
    p_max = max(p_values) if p_values else 1.0

    return RemovalResult(
        member_id=member_id,
        entropy_norm=h_k,
        entropy_drop=h0 - h_k,
        stable=True,
        max_energy_fraction=p_max,
    )


def _intact_entropy_norm(frame: FrameData, load_factor: float = 1.0) -> float:
    """Normalized strain-energy entropy H0 of the intact frame."""
    energy_state = solve(frame, step=0, load_factor=load_factor)
    record = compute_entropy(energy_state, previous_entropy=0.0)
    n_active = sum(1 for ms in energy_state.member_states if not ms.failed)
    return normalized_entropy(record, n_active)


# ---------------------------------------------------------------------------
# Full robustness assessment
# ---------------------------------------------------------------------------

def analyze(
    frame: FrameData,
    members_to_remove: Optional[List[int]] = None,
    load_factor: float = 1.0,
) -> RobustnessReport:
    """
    Compute the Entropy Robustness Index R_S of a frame via single-member
    notional removal over a set of primary members.

    Args:
        frame: Intact frame to assess (not mutated).
        members_to_remove: Member IDs to notionally remove. Defaults to every
                           non-failed member in the frame (full census).
        load_factor: Load multiplier for all re-analyses.

    Returns:
        RobustnessReport with R_S, the worst case, the critical member, and
        the per-member removal results.
    """
    if members_to_remove is None:
        members_to_remove = [m.id for m in frame.members if not m.failed]

    h0 = _intact_entropy_norm(frame, load_factor)

    removals = [removal_entropy(frame, mid, load_factor) for mid in members_to_remove]

    if removals:
        r_s = float(np.mean([r.entropy_norm for r in removals]))
        stable = [r for r in removals if r.stable]
        r_s_stable = float(np.mean([r.entropy_norm for r in stable])) if stable else 0.0
        mech_fraction = 1.0 - len(stable) / len(removals)
        worst = min(removals, key=lambda r: r.entropy_norm)
        critical = max(removals, key=lambda r: r.entropy_drop)
        worst_case = worst.entropy_norm
        critical_member = critical.member_id
    else:
        r_s = h0
        r_s_stable = h0
        mech_fraction = 0.0
        worst_case = h0
        critical_member = None

    unstable = [r.member_id for r in removals if not r.stable]

    return RobustnessReport(
        frame_name=frame.name,
        intact_entropy_norm=h0,
        robustness_index=r_s,
        robustness_index_stable_only=r_s_stable,
        mechanism_fraction=mech_fraction,
        worst_case_entropy_norm=worst_case,
        critical_member=critical_member,
        removals=removals,
        unstable_members=unstable,
    )


# ---------------------------------------------------------------------------
# Progressive (sequential) removal trajectory
# ---------------------------------------------------------------------------

def sequential_trajectory(
    frame: FrameData,
    removal_order: Optional[List[int]] = None,
    load_factor: float = 1.0,
) -> List[Tuple[int, float, bool]]:
    """
    Track the normalized entropy as members are removed one after another.

    If no order is given, a greedy "worst-first" order is used: at each stage
    the member whose removal most reduces entropy is removed next. This traces
    the fastest entropy-collapse path and complements the single-removal index.

    Args:
        frame: Intact frame (not mutated; a working copy is used).
        removal_order: Explicit member-id removal order, or None for greedy.
        load_factor: Load multiplier for all re-analyses.

    Returns:
        List of (removed_member_id, normalized_entropy_after_removal, stable)
        in removal order. A stable=False entry marks the step at which the
        structure becomes a mechanism; the trajectory stops there.
    """
    work = copy.deepcopy(frame)
    trajectory: List[Tuple[int, float, bool]] = []

    remaining = [m.id for m in work.members if not m.failed]
    explicit = list(removal_order) if removal_order is not None else None

    while remaining:
        if explicit is not None:
            if not explicit:
                break
            nxt = explicit.pop(0)
        else:
            # Greedy: pick the still-present member with the largest drop.
            candidates = [
                (mid, removal_entropy(work, mid, load_factor))
                for mid in remaining
            ]
            nxt = max(candidates, key=lambda c: c[1].entropy_drop)[0]

        target = next(m for m in work.members if m.id == nxt)
        target.failed = True
        remaining.remove(nxt)

        if not is_stable(work):
            trajectory.append((nxt, 0.0, False))
            break

        energy_state = solve(work, step=0, load_factor=load_factor)
        record = compute_entropy(energy_state, previous_entropy=0.0)
        n_active = sum(1 for ms in energy_state.member_states if not ms.failed)
        trajectory.append((nxt, normalized_entropy(record, n_active), True))

    return trajectory
