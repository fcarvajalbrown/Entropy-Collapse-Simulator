"""
analysis/design_variants.py
===========================
Design-decision study: does R_S distinguish two design variants of the same
steel building by their alternate-load-path redundancy, and does the entropy
critical-member ranking point at the members that drive the difference?

The variant axis is a genuine steel-design decision: the column-base condition
(moment/fixed bases vs pinned bases). Fixed bases give more alternate load paths
under column loss; pinned bases are cheaper but less redundant. This is the axis
R_S is built to detect: by the determinacy bound (THEORY.md 5.3), R_S falls when
a design starts to admit single-member-loss mechanisms.

Important scope note (honest): R_S measures load-path redundancy, i.e. how evenly
strain energy is shared and whether single losses are survivable. It is NOT a
strength/stiffness measure and deliberately does not reward mere member
stiffening (stiffening concentrates energy and can lower R_S; adding a stiff
brace to an already-redundant frame can too). This study therefore compares
designs on the redundancy axis, where R_S is meaningful, not on stiffening.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from core.models import FrameData
from entropy.robustness import analyze as analyze_robustness
from analysis.importance import column_member_ids


@dataclass
class DesignVariant:
    label: str
    robustness_index: float           # R_S
    mechanism_fraction: float          # share of single removals that collapse
    worst_case_entropy_norm: float     # min H_k (0 => a single loss is fatal)
    critical_member: int               # argmax dH_k
    mechanism_columns: List[int]       # columns whose single loss forms a mechanism
    column_rank: List[int]             # columns by dH_k, most critical first


@dataclass
class DesignVariantStudy:
    variants: List[DesignVariant]      # in the order supplied
    more_robust: str                   # label of the highest-R_S variant
    delta_rs: float                    # R_S(most robust) - R_S(least robust)


def evaluate_variant(frame: FrameData, label: str) -> DesignVariant:
    """Compute the redundancy summary and column criticality for one design."""
    rep = analyze_robustness(frame)
    cols = set(column_member_ids(frame))
    dH = {r.member_id: r.entropy_drop for r in rep.removals}
    mech_columns = sorted(c for c in rep.unstable_members if c in cols)
    column_rank = sorted((c for c in cols if c in dH), key=lambda c: dH[c], reverse=True)
    return DesignVariant(
        label=label,
        robustness_index=rep.robustness_index,
        mechanism_fraction=rep.mechanism_fraction,
        worst_case_entropy_norm=rep.worst_case_entropy_norm,
        critical_member=rep.critical_member,
        mechanism_columns=mech_columns,
        column_rank=column_rank,
    )


def compare_designs(named_frames: Dict[str, FrameData]) -> DesignVariantStudy:
    """
    Evaluate each named design variant and report which is most robust by R_S.

    Args:
        named_frames: mapping label -> FrameData, one entry per design variant.

    Returns:
        DesignVariantStudy with per-variant summaries (input order preserved),
        the most-robust label, and the R_S spread across variants.
    """
    variants = [evaluate_variant(f, label) for label, f in named_frames.items()]
    ordered = sorted(variants, key=lambda v: v.robustness_index)
    delta = ordered[-1].robustness_index - ordered[0].robustness_index
    return DesignVariantStudy(
        variants=variants,
        more_robust=ordered[-1].label,
        delta_rs=delta,
    )


def vogel_base_fixity_study() -> DesignVariantStudy:
    """
    The paper's design-decision demo: the Vogel building with fixed (moment)
    bases vs pinned bases. R_S is expected to rank the fixed-base design as more
    robust and to flag, in the pinned-base design, the base columns whose loss
    forms a mechanism -- which the entropy criticality ranking should also place
    at the top.
    """
    from structure.frames import frame_vogel_six_storey as fv
    return compare_designs({
        "Fixed bases (as-built)": fv.build(base="fixed"),
        "Pinned bases": fv.build(base="pinned"),
    })
