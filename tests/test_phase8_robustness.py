"""
tests/test_phase8_robustness.py
================================
Phase 8: Entropy Robustness Index (entropy/robustness.py).

Checks:
  - Intact frames are detected as stable
  - A non-redundant truss scores R_S = 0 (any loss collapses it)
  - A redundant moment frame scores 0 < R_S <= 1 with no unstable single loss
  - R_S, H0 and per-removal entropies all lie in [0, 1]
  - The criticality ranking is ordered by entropy drop
  - A single notional removal does not mutate the caller's frame
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from structure.frames import frame_2d_simple, frame_building_2d, frame_pratt_bridge
from entropy import robustness as rb


def test_intact_frames_stable():
    """Every intact benchmark frame is kinematically stable."""
    for mod in (frame_2d_simple, frame_building_2d, frame_pratt_bridge):
        assert rb.is_stable(mod.build()), f"{mod.__name__} intact frame unstable"
    print("  PASS: all intact frames stable")


def test_non_redundant_truss_zero_robustness():
    """The 2-member truss collapses on any single removal: R_S = 0."""
    rep = rb.analyze(frame_2d_simple.build())
    assert rep.robustness_index == 0.0
    assert set(rep.unstable_members) == {0, 1}
    print(f"  PASS: non-redundant truss R_S = {rep.robustness_index:.3f}")


def test_redundant_frame_positive_robustness():
    """The moment frame is redundant: 0 < R_S <= 1 and no single loss is fatal."""
    rep = rb.analyze(frame_building_2d.build())
    assert 0.0 < rep.robustness_index <= 1.0
    assert rep.unstable_members == []
    print(f"  PASS: moment-frame R_S = {rep.robustness_index:.3f}, "
          f"no unstable single removals")


def test_indices_in_unit_interval():
    """H0, R_S and every per-removal normalized entropy are within [0, 1]."""
    rep = rb.analyze(frame_pratt_bridge.build())
    assert 0.0 <= rep.intact_entropy_norm <= 1.0
    assert 0.0 <= rep.robustness_index <= 1.0
    for r in rep.removals:
        assert 0.0 <= r.entropy_norm <= 1.0 + 1e-9
    print("  PASS: all entropy indices in [0, 1]")


def test_ranking_sorted_by_drop():
    """ranking() returns members ordered by descending entropy drop."""
    rep = rb.analyze(frame_pratt_bridge.build())
    drops = [d for _, d in rep.ranking()]
    assert drops == sorted(drops, reverse=True)
    assert rep.critical_member == rep.ranking()[0][0]
    print(f"  PASS: critical member = {rep.critical_member}")


def test_golden_values():
    """
    Regression anchors for the headline numbers, so a silent change (e.g. a
    section-property edit) cannot keep the suite green while moving every
    reported value.
    """
    b = rb.analyze(frame_building_2d.build())
    assert abs(b.intact_entropy_norm - 0.7124) < 1e-3
    assert abs(b.robustness_index - 0.7194) < 1e-3
    t = rb.analyze(frame_pratt_bridge.build())
    assert abs(t.intact_entropy_norm - 0.7906) < 1e-3
    assert abs(t.robustness_index - 0.7861) < 1e-3
    print("  PASS: golden R_S/H0 values match (moment frame & truss)")


def test_removal_does_not_mutate_caller():
    """removal_entropy works on a copy; caller's frame keeps all members."""
    frame = frame_building_2d.build()
    rb.removal_entropy(frame, member_id=0)
    assert all(not m.failed for m in frame.members), "caller frame was mutated"
    print("  PASS: notional removal does not mutate caller frame")


if __name__ == "__main__":
    print("=== Phase 8: Entropy Robustness Index ===")
    test_intact_frames_stable()
    test_non_redundant_truss_zero_robustness()
    test_redundant_frame_positive_robustness()
    test_indices_in_unit_interval()
    test_ranking_sorted_by_drop()
    test_golden_values()
    test_removal_does_not_mutate_caller()
    print("All Phase 8 tests passed.\n")
