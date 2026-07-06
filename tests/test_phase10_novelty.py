"""
tests/test_phase10_novelty.py
=============================
Phase 10: Novelty studies (analysis/importance.py, analysis/parametric.py).

Checks:
  - Spearman rho is well-defined and in [-1, 1]; entropy and compliance
    rankings are genuinely different (dH_k is not a re-encoding of compliance)
  - R_S rises monotonically with the degree of static indeterminacy, and the
    single-loss mechanism fraction falls to zero at full redundancy
  - The entropy criterion never fires before DCR (one-step-lag property), at
    every load-step size
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from structure.frames import frame_building_2d, frame_vogel_six_storey
from analysis import importance as imp
from analysis import parametric as par


def test_spearman_bounded_and_rankings_differ():
    c = imp.compare(frame_building_2d.build())
    assert -1.0 <= c.spearman_rho <= 1.0
    # The two rankings should not be identical: entropy criticality measures
    # distribution shape, compliance importance measures magnitude.
    assert c.entropy_rank != c.compliance_rank
    print(f"  PASS: Spearman rho = {c.spearman_rho:+.3f}, rankings differ")


def test_rs_monotonic_in_redundancy():
    pts = par.redundancy_sweep()
    dsi = [p.dsi for p in pts]
    rs = [p.robustness_index for p in pts]
    assert dsi == sorted(dsi) and dsi[0] == 0
    # R_S is non-decreasing in DSI and strictly increases overall.
    assert all(b >= a - 1e-9 for a, b in zip(rs, rs[1:]))
    assert rs[-1] > rs[0]
    # Determinate truss: every loss is a mechanism; full X-bracing: none.
    assert pts[0].mechanism_fraction == 1.0
    assert pts[-1].mechanism_fraction == 0.0
    print(f"  PASS: R_S rises {rs[0]:.2f}->{rs[-1]:.2f} as DSI 0->{dsi[-1]}, "
          f"mech {pts[0].mechanism_fraction:.0%}->{pts[-1].mechanism_fraction:.0%}")


def test_importance_ensemble_not_monotone():
    """
    Across an ensemble of frames the dH_k vs I_k rank correlation must stay
    well below 1 and vary in sign, confirming dH_k is not a monotone
    re-encoding of compliance importance in general (not just on one frame).
    """
    rows = par.importance_ensemble()
    rhos = [r.spearman_rho for r in rows]
    assert len(rhos) >= 8
    assert all(-1.0 <= r <= 1.0 for r in rhos)
    assert max(rhos) < 0.95               # never a near-perfect re-encoding
    assert any(r < 0 for r in rhos)       # sign varies across topologies
    print(f"  PASS: ensemble rho in [{min(rhos):+.2f}, {max(rhos):+.2f}], "
          f"{sum(1 for r in rhos if r < 0)}/{len(rhos)} negative")


def test_column_alp_agreement_vogel():
    """
    On the Vogel benchmark building the calibration-free entropy column ranking
    (dH_k) strongly agrees with the code-style single-column-removal ALP severity
    (post-removal max DCR): high Spearman rho, same most-critical column, and the
    ranking is load-invariant (linear model). This substantiates the screening
    claim -- R_S triage points at the columns worth expensive nonlinear ALP.
    """
    c = imp.compare_column_alp(frame_vogel_six_storey.build())
    assert len(c.column_ids) == 18
    assert -1.0 <= c.spearman_rho <= 1.0
    assert abs(c.spearman_rho - 0.9092) < 1e-2      # strong agreement (anchor)
    assert c.entropy_rank[0] == c.alp_rank[0]        # same worst column
    assert c.topk_overlap[5] >= 4                    # top-5 mostly shared
    # Ranking is independent of load level (both measures are scale-invariant).
    c2 = imp.compare_column_alp(frame_vogel_six_storey.build(), load_factor=0.4)
    assert c2.alp_rank == c.alp_rank
    print(f"  PASS: Vogel column dH_k vs code-ALP DCR rho = {c.spearman_rho:+.3f}, "
          f"top-1 agree, top-5 overlap {c.topk_overlap[5]}/5, load-invariant")


def test_entropy_lags_dcr_by_construction():
    ss = par.step_size_sensitivity()
    for step, trig in ss.items():
        dcr, ent = trig["dcr"], trig["entropy"]
        if dcr is not None and ent is not None:
            assert ent >= dcr - 1e-9, f"entropy fired before DCR at step {step}"
    print("  PASS: entropy never precedes DCR across all load-step sizes")


if __name__ == "__main__":
    print("=== Phase 10: Novelty Studies ===")
    test_spearman_bounded_and_rankings_differ()
    test_rs_monotonic_in_redundancy()
    test_importance_ensemble_not_monotone()
    test_column_alp_agreement_vogel()
    test_entropy_lags_dcr_by_construction()
    print("All Phase 10 tests passed.\n")
