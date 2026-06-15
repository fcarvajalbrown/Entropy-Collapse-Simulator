"""
tests/test_phase9_criteria.py
==============================
Phase 9: Four-criteria comparison (analysis/criteria.py).

Checks:
  - run_with_metrics records one StepMetrics per step with finite indicators
  - Entropy is scale-invariant before the first failure (flat dS)
  - The entropy criterion fires no earlier than the first member failure
  - compare() returns a first-trigger entry for all four criteria
  - The energy (compliance) criterion does not fire on the intact ramp alone
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from structure.frames import frame_pratt_bridge, frame_2d_simple
from analysis import criteria as C


def test_metrics_recorded_per_step():
    """Each step yields finite metrics."""
    hist, _ = C.run_with_metrics(frame_pratt_bridge.build(), max_steps=10,
                                 load_factor_step=0.15)
    assert len(hist) == 10
    for m in hist:
        assert m.peak_displacement >= 0.0
        assert m.peak_dcr >= 0.0
        assert 0.0 <= m.entropy_norm <= 1.0 + 1e-9
    print(f"  PASS: {len(hist)} step-metric records, all finite")


def test_entropy_flat_before_failure():
    """With no failures, entropy is scale invariant (|dS| ~ 0 after step 0)."""
    hist, seq = C.run_with_metrics(frame_pratt_bridge.build(), max_steps=8,
                                   load_factor_start=1.0, load_factor_step=0.1)
    assert seq == [], "no members should fail at low load in this short run"
    for m in hist[1:]:
        assert abs(m.delta_entropy) < 1e-6, f"dS not flat: {m.delta_entropy}"
    print("  PASS: entropy flat (scale invariant) before any failure")


def test_entropy_fires_no_earlier_than_first_failure():
    """The entropy criterion cannot fire before the first failure step."""
    cmp = C.compare(frame_pratt_bridge.build(), max_steps=40,
                    load_factor_step=0.3)
    ent = cmp.first_trigger["entropy"]
    assert ent is not None, "entropy criterion should fire once collapse begins"
    first_fail_step = next(
        (m.step for m in cmp.history if m.n_failed > 0), None
    )
    assert first_fail_step is not None
    assert ent >= first_fail_step
    print(f"  PASS: entropy fires at step {ent} (first failure at {first_fail_step})")


def test_golden_trigger_load_factors():
    """Regression anchor for the truss-bridge criteria first-trigger loads."""
    cmp = C.compare(frame_pratt_bridge.build(), max_steps=40, load_factor_step=0.3)
    lf = {k: (cmp.history[v].load_factor if v is not None else None)
          for k, v in cmp.first_trigger.items()}
    assert abs(lf["displacement"] - 4.6) < 1e-6
    assert abs(lf["dcr"] - 8.5) < 1e-6
    assert abs(lf["energy"] - 8.8) < 1e-6
    assert abs(lf["entropy"] - 8.8) < 1e-6
    print(f"  PASS: golden trigger load factors {lf}")


def test_compare_reports_all_criteria():
    """compare() returns a key for each of the four criteria."""
    cmp = C.compare(frame_pratt_bridge.build(), max_steps=40,
                    load_factor_step=0.15)
    assert set(cmp.first_trigger) == {"displacement", "dcr", "energy", "entropy"}
    print(f"  PASS: triggers = {cmp.first_trigger}")


def test_energy_criterion_ignores_pure_load_ramp():
    """
    Compliance (U/lambda^2) is constant for the intact frame, so the energy
    criterion must NOT fire while no member has failed.
    """
    cmp = C.compare(frame_pratt_bridge.build(), max_steps=8,
                    load_factor_start=1.0, load_factor_step=0.1)
    assert cmp.failed_sequence == []
    assert cmp.first_trigger["energy"] is None
    print("  PASS: energy criterion does not fire on the intact load ramp")


if __name__ == "__main__":
    print("=== Phase 9: Four-Criteria Comparison ===")
    test_metrics_recorded_per_step()
    test_entropy_flat_before_failure()
    test_entropy_fires_no_earlier_than_first_failure()
    test_golden_trigger_load_factors()
    test_compare_reports_all_criteria()
    test_energy_criterion_ignores_pure_load_ramp()
    print("All Phase 9 tests passed.\n")
