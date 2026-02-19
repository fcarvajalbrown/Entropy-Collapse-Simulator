"""
tests/test_phase4_failure.py
=============================
Phase 4: Verify member failure detection and energy redistribution.

Checks:
  - A member with very low sigma_y fails immediately under any load
  - check_and_apply_failures returns the correct member ID
  - Energy redistribution conserves total energy approximately
  - all_failed() correctly detects total collapse
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from structure.frames import frame_2d_simple
from solver.equilibrium import solve
from solver.failure import check_and_apply_failures, all_failed
from solver.redistribution import redistribute


def test_member_fails_under_low_capacity():
    """A member with sigma_y = 1 Pa fails immediately under any load."""
    frame = frame_2d_simple.build()
    frame.members[0].sigma_y = 1.0  # 1 Pa — fails under any real force

    es = solve(frame, step=0)
    newly_failed = check_and_apply_failures(frame, es)

    assert 0 in newly_failed, f"Expected member 0 to fail, got: {newly_failed}"
    assert frame.members[0].failed == True
    print(f"  PASS: Member 0 failed as expected (sigma_y=1 Pa, force={es.member_states[0].axial_force:.2f} N)")


def test_failure_marks_member_in_frame():
    """After failure, frame.members[0].failed is True."""
    frame = frame_2d_simple.build()
    frame.members[0].sigma_y = 1.0
    es = solve(frame, step=0)
    check_and_apply_failures(frame, es)
    assert frame.members[0].failed == True
    print("  PASS: member.failed flag set correctly")


def test_redistribution_conserves_energy():
    """
    After redistribution, total energy changes by less than 5%.
    (Not strictly conserved due to ODE discretization, but should be close.)
    """
    frame = frame_2d_simple.build()
    frame.members[0].sigma_y = 1.0
    es = solve(frame, step=0)
    check_and_apply_failures(frame, es)

    energy_before = es.total_energy
    es_after = redistribute(frame, es, dt=1.0)
    energy_after = es_after.total_energy

    if energy_before > 0:
        relative_change = abs(energy_after - energy_before) / energy_before
        assert relative_change < 0.05, \
            f"Energy changed by {relative_change*100:.1f}% after redistribution"
    print(f"  PASS: Energy before={energy_before:.4f}, after={energy_after:.4f} J")


def test_all_failed_false_initially():
    """all_failed() returns False when no members have failed."""
    frame = frame_2d_simple.build()
    assert all_failed(frame) == False
    print("  PASS: all_failed() = False initially")


def test_all_failed_true_when_all_marked():
    """all_failed() returns True when all members are manually failed."""
    frame = frame_2d_simple.build()
    for m in frame.members:
        m.failed = True
    assert all_failed(frame) == True
    print("  PASS: all_failed() = True when all failed")


if __name__ == "__main__":
    print("=== Phase 4: Failure & Redistribution ===")
    test_member_fails_under_low_capacity()
    test_failure_marks_member_in_frame()
    test_redistribution_conserves_energy()
    test_all_failed_false_initially()
    test_all_failed_true_when_all_marked()
    print("All Phase 4 tests passed.\n")