"""
analysis/importance.py
======================
Validate the entropy-based member-criticality ranking against an established,
removal-based member-importance measure, and quantify their agreement.

A standard way to rank members by importance is the increase in structural
compliance (total elastic strain energy at fixed load) caused by removing the
member: a member whose loss greatly softens the structure is important. For
member k,

    I_k = (U_k - U_0) / U_0

where U_0 is the intact total strain energy and U_k the total after removing k
and re-analysing. A removal that forms a mechanism has I_k = infinity.

The entropy criticality is the entropy drop dH_k = H_0 - H_k from
entropy/robustness.py. The two answer different questions: I_k measures how much
the loss softens the structure (magnitude), while dH_k measures how much the
loss concentrates the remaining strain energy (distribution shape). This module
computes both, ranks members by each, and reports the Spearman rank correlation
so the paper can state precisely where the entropy ranking agrees with, and
where it departs from, the compliance-based one.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from core.models import FrameData
from solver.equilibrium import solve
from entropy.robustness import analyze as analyze_robustness, is_stable


@dataclass
class ImportanceComparison:
    frame_name: str
    entropy_drop: Dict[int, float]          # dH_k per member
    compliance_importance: Dict[int, float]  # I_k per member (inf for mechanism)
    spearman_rho: float                      # rank correlation of dH_k vs I_k
    entropy_rank: List[int]                  # member ids, most critical first
    compliance_rank: List[int]


def _intact_energy(frame: FrameData, load_factor: float) -> float:
    return solve(frame, step=0, load_factor=load_factor).total_energy


def compliance_importance(
    frame: FrameData,
    members_to_remove: Optional[List[int]] = None,
    load_factor: float = 1.0,
) -> Dict[int, float]:
    """
    Removal-based compliance importance I_k = (U_k - U_0)/U_0 for each member.

    Mechanisms (removals that destabilise the frame) get I_k = +inf.
    """
    if members_to_remove is None:
        members_to_remove = [m.id for m in frame.members if not m.failed]

    u0 = _intact_energy(frame, load_factor)
    out: Dict[int, float] = {}
    for mid in members_to_remove:
        work = copy.deepcopy(frame)
        next(m for m in work.members if m.id == mid).failed = True
        if not is_stable(work):
            out[mid] = math.inf
            continue
        uk = solve(work, step=0, load_factor=load_factor).total_energy
        out[mid] = (uk - u0) / u0 if u0 > 0 else 0.0
    return out


def _rankdata(values: np.ndarray) -> np.ndarray:
    """Average-rank of each value (ties share the mean rank), ascending."""
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    ranks[order] = np.arange(1, len(values) + 1, dtype=float)
    # Resolve ties to average ranks.
    sv = values[order]
    i = 0
    while i < len(sv):
        j = i
        while j + 1 < len(sv) and sv[j + 1] == sv[i]:
            j += 1
        if j > i:
            avg = np.mean(ranks[order[i:j + 1]])
            ranks[order[i:j + 1]] = avg
        i = j + 1
    return ranks


def spearman_rho(x: List[float], y: List[float]) -> float:
    """
    Spearman rank correlation between x and y. Infinities are kept (they rank
    as the largest values), which is the intended behaviour: a mechanism is the
    most important / most critical removal under both measures.
    """
    xa = np.array(x, dtype=float)
    ya = np.array(y, dtype=float)
    # Map +inf to a value just above the finite maximum so it ranks highest.
    for arr in (xa, ya):
        finite = arr[np.isfinite(arr)]
        hi = finite.max() if finite.size else 0.0
        arr[np.isinf(arr)] = hi + 1.0
    rx = _rankdata(xa)
    ry = _rankdata(ya)
    rx -= rx.mean()
    ry -= ry.mean()
    denom = math.sqrt((rx @ rx) * (ry @ ry))
    return float(rx @ ry / denom) if denom > 0 else 0.0


def compare(frame: FrameData, load_factor: float = 1.0) -> ImportanceComparison:
    """
    Compute dH_k and I_k for every member and their Spearman correlation.
    """
    rep = analyze_robustness(frame, load_factor=load_factor)
    dH = {r.member_id: r.entropy_drop for r in rep.removals}
    I = compliance_importance(frame, list(dH.keys()), load_factor)

    ids = list(dH.keys())
    rho = spearman_rho([dH[i] for i in ids], [I[i] for i in ids])

    entropy_rank = sorted(ids, key=lambda i: dH[i], reverse=True)
    compliance_rank = sorted(ids, key=lambda i: I[i], reverse=True)

    return ImportanceComparison(
        frame_name=frame.name,
        entropy_drop=dH,
        compliance_importance=I,
        spearman_rho=rho,
        entropy_rank=entropy_rank,
        compliance_rank=compliance_rank,
    )
