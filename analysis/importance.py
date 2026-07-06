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
from solver.equilibrium import solve, solve_full
from solver.failure import _combined_stress
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


# ===========================================================================
# Code-style alternate-load-path (ALP) agreement study
# ===========================================================================
#
# Progressive-collapse guidance (GSA 2003; UFC 4-023-03) assesses robustness by
# the alternate load path method: notionally remove a primary column, re-analyse,
# and check whether the surviving structure carries the load. The engineering
# severity of losing a column is naturally measured by the worst demand-capacity
# ratio (DCR) that appears in the survivors -- a removal that leaves a member
# grossly overstressed (or forms a mechanism) is critical. That check is what the
# expensive nonlinear ALP analysis ultimately refines.
#
# The claim under test: the calibration-free entropy criticality dH_k (obtained
# from a SINGLE linear solve, no per-structure threshold) ranks the columns in
# the same order as this code-style DCR severity, so R_S is a cheap triage that
# tells the engineer which columns to spend the expensive nonlinear ALP effort on.
#
# Because the model is linear, the post-removal max DCR scales linearly with the
# load factor, so the *ranking* of columns by severity is independent of the load
# level -- as is dH_k (scale-invariant). The agreement is therefore a structural
# property, not an artefact of the chosen load.


@dataclass
class ColumnALPComparison:
    frame_name: str
    column_ids: List[int]                 # members classified as columns
    entropy_drop: Dict[int, float]        # dH_k per column
    alp_severity: Dict[int, float]        # post-removal max DCR (inf = mechanism)
    spearman_rho: float                   # rank correlation of dH_k vs ALP severity
    entropy_rank: List[int]               # columns, most critical first (dH_k)
    alp_rank: List[int]                   # columns, most severe first (max DCR)
    topk_overlap: Dict[int, int]          # k -> shared members in both top-k sets
    load_factor: float


def column_member_ids(frame: FrameData) -> List[int]:
    """
    Ids of the vertical (column) members, classified purely by geometry: a
    column connects two nodes at the same x with different y. This keeps the
    ALP set frame-agnostic (no reliance on member-id ordering) and excludes
    beams (same y) and any diagonal.
    """
    xy = {n.id: (n.x, n.y) for n in frame.nodes}
    cols: List[int] = []
    for m in frame.members:
        (x1, y1), (x2, y2) = xy[m.node_start], xy[m.node_end]
        if abs(x1 - x2) < 1e-9 and abs(y1 - y2) > 1e-9:
            cols.append(m.id)
    return cols


def _peak_dcr_surviving(u: np.ndarray, frame: FrameData) -> float:
    """Largest DCR = sigma_max/sigma_y over the non-failed (surviving) members."""
    peak = 0.0
    for m in frame.members:
        if m.failed:
            continue
        peak = max(peak, _combined_stress(m, u, frame) / m.sigma_y)
    return peak


def alp_column_severity(
    frame: FrameData,
    column_ids: Optional[List[int]] = None,
    load_factor: float = 1.0,
) -> Dict[int, float]:
    """
    Code-style ALP severity of each column: notionally remove the column,
    re-analyse at the design load, and return the worst surviving DCR. A removal
    that forms a mechanism returns +inf (the alternate path fails outright).
    """
    if column_ids is None:
        column_ids = column_member_ids(frame)

    out: Dict[int, float] = {}
    for cid in column_ids:
        work = copy.deepcopy(frame)
        next(m for m in work.members if m.id == cid).failed = True
        if not is_stable(work):
            out[cid] = math.inf
            continue
        u, _ = solve_full(work, step=0, load_factor=load_factor)
        out[cid] = _peak_dcr_surviving(u, work)
    return out


def compare_column_alp(
    frame: FrameData,
    load_factor: float = 1.0,
) -> ColumnALPComparison:
    """
    Compare the entropy column-criticality ranking (dH_k) against the code-style
    single-column-removal ALP severity (post-removal max DCR) on every column,
    and report their Spearman rank correlation and top-k overlap.
    """
    cols = column_member_ids(frame)
    rep = analyze_robustness(frame, load_factor=load_factor)
    dH_all = {r.member_id: r.entropy_drop for r in rep.removals}
    cols = [c for c in cols if c in dH_all]          # keep analysed columns
    dH = {c: dH_all[c] for c in cols}
    sev = alp_column_severity(frame, cols, load_factor)

    rho = spearman_rho([dH[c] for c in cols], [sev[c] for c in cols])
    entropy_rank = sorted(cols, key=lambda c: dH[c], reverse=True)
    alp_rank = sorted(cols, key=lambda c: sev[c], reverse=True)

    topk_overlap: Dict[int, int] = {}
    for k in (1, 3, 5):
        if k <= len(cols):
            topk_overlap[k] = len(set(entropy_rank[:k]) & set(alp_rank[:k]))

    return ColumnALPComparison(
        frame_name=frame.name,
        column_ids=cols,
        entropy_drop=dH,
        alp_severity=sev,
        spearman_rho=rho,
        entropy_rank=entropy_rank,
        alp_rank=alp_rank,
        topk_overlap=topk_overlap,
        load_factor=load_factor,
    )
