"""
analysis/parametric.py
======================
Parametric studies that test two claims about the Entropy Robustness Index:

1. R_S tracks a ground-truth redundancy measure. Using the truss bridge with a
   controllable number of counter-diagonals, the degree of static
   indeterminacy (DSI) is known exactly (DSI = n_counter for this geometry).
   redundancy_sweep() shows R_S rising and the single-loss mechanism fraction
   falling as DSI increases.

2. The calibration-free advantage is real. threshold_sensitivity() shows that
   the displacement and DCR criteria report different collapse loads when their
   thresholds are changed, whereas R_S and the entropy criterion return one
   fixed value with no threshold to choose.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from structure.frames import frame_pratt_bridge, frame_building_2d
from entropy import robustness as rb
from analysis import criteria as C
from analysis import importance as imp


# Truss bookkeeping for the bridge geometry, derived from the determinate
# (no counter-diagonal) build so it cannot desynchronise from the frame.
# DSI = (members + reactions) - 2*nodes.
_det = frame_pratt_bridge.build(n_counter=0)
_N_NODES = len(_det.nodes)
_BASE_MEMBERS = len(_det.members)
_REACTIONS = 3


@dataclass
class RedundancyPoint:
    n_counter: int
    dsi: int                 # degree of static indeterminacy
    n_members: int
    robustness_index: float          # R_S including mechanisms
    robustness_index_stable: float   # R_S over stable removals only
    mechanism_fraction: float
    worst_case_entropy: float


def redundancy_sweep(load_factor: float = 1.0) -> List[RedundancyPoint]:
    """
    Sweep the number of counter-diagonals 0..6 and report R_S vs DSI.
    """
    points: List[RedundancyPoint] = []
    for n in range(0, 7):
        frame = frame_pratt_bridge.build(n_counter=n)
        dsi = (_BASE_MEMBERS + n + _REACTIONS) - 2 * _N_NODES
        rep = rb.analyze(frame, load_factor=load_factor)
        points.append(RedundancyPoint(
            n_counter=n,
            dsi=dsi,
            n_members=len(frame.members),
            robustness_index=rep.robustness_index,
            robustness_index_stable=rep.robustness_index_stable_only,
            mechanism_fraction=rep.mechanism_fraction,
            worst_case_entropy=rep.worst_case_entropy_norm,
        ))
    return points


@dataclass
class ThresholdRow:
    criterion: str
    threshold_label: str
    threshold_value: float
    trigger_load_factor: float | None


def threshold_sensitivity(
    max_steps: int = 40,
    load_factor_step: float = 0.3,
) -> List[ThresholdRow]:
    """
    On the fully redundant truss bridge, vary each calibrated threshold and
    record the collapse load factor it reports. The entropy criterion is
    included once: it has no threshold, so it returns a single value.
    """
    frame = frame_pratt_bridge.build()
    rows: List[ThresholdRow] = []

    def trigger_lf(cmp, key):
        step = cmp.first_trigger[key]
        return cmp.history[step].load_factor if step is not None else None

    # Displacement: vary the drift limit.
    for dl in (0.02, 0.05, 0.10, 0.20):
        cmp = C.compare(frame, max_steps=max_steps, load_factor_step=load_factor_step,
                        drift_limit=dl)
        rows.append(ThresholdRow("displacement", "drift_limit", dl, trigger_lf(cmp, "displacement")))

    # DCR: vary the acceptance ratio.
    for dcr in (1.0, 1.5, 2.0):
        cmp = C.compare(frame, max_steps=max_steps, load_factor_step=load_factor_step,
                        dcr_limit=dcr)
        rows.append(ThresholdRow("dcr", "dcr_limit", dcr, trigger_lf(cmp, "dcr")))

    # Energy: vary the compliance multiple.
    for ef in (2.0, 3.0, 5.0):
        cmp = C.compare(frame, max_steps=max_steps, load_factor_step=load_factor_step,
                        energy_factor=ef)
        rows.append(ThresholdRow("energy", "energy_factor", ef, trigger_lf(cmp, "energy")))

    # Entropy: no threshold; one value.
    cmp = C.compare(frame, max_steps=max_steps, load_factor_step=load_factor_step)
    rows.append(ThresholdRow("entropy", "(none)", float("nan"), trigger_lf(cmp, "entropy")))

    return rows


@dataclass
class EnsembleRow:
    label: str
    n_members: int
    spearman_rho: float


def importance_ensemble() -> List[EnsembleRow]:
    """
    Compute the Spearman correlation between the entropy criticality dH_k and
    the compliance importance I_k over a small ensemble of frames of varied
    topology and size. This tests whether the non-reducibility of dH_k to
    compliance importance is a property of the measure (low/variable rho across
    many frames) rather than of a single example.

    The ensemble mixes moment frames of several bay/story counts with redundant
    trusses (enough counter-diagonals that removals are not all mechanisms, so
    the rank correlation is well defined).
    """
    rows: List[EnsembleRow] = []
    for nb, ns in [(1, 2), (1, 3), (2, 3), (2, 4), (3, 3), (3, 5), (3, 6), (4, 4)]:
        f = frame_building_2d.build(n_bays=nb, n_stories=ns)
        rows.append(EnsembleRow(f"moment {nb}x{ns}", len(f.members),
                                imp.compare(f).spearman_rho))
    for nc in (4, 5, 6):
        f = frame_pratt_bridge.build(n_counter=nc)
        rows.append(EnsembleRow(f"truss cd={nc}", len(f.members),
                                imp.compare(f).spearman_rho))
    return rows


def step_size_sensitivity(load_factor_steps=(0.05, 0.10, 0.15, 0.30)) -> Dict[float, Dict[str, float | None]]:
    """
    Show how the first-trigger LOAD FACTOR of each criterion depends on the
    load-step size, for the redundant truss bridge. This exposes that the
    "entropy fires at the same load as DCR" result is a one-step-lag effect:
    the DCR-to-entropy gap shrinks as the step shrinks.
    """
    frame = frame_pratt_bridge.build()
    out: Dict[float, Dict[str, float | None]] = {}
    for step in load_factor_steps:
        n_steps = int(20 / step) + 5
        cmp = C.compare(frame, max_steps=n_steps, load_factor_step=step)
        out[step] = {
            k: (cmp.history[v].load_factor if v is not None else None)
            for k, v in cmp.first_trigger.items()
        }
    return out
