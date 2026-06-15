"""
tests/test_phase4_failure.py
=============================
Phase 4: Verify member failure detection and alternate-load-path re-analysis.

Checks:
  - A member with very low sigma_y fails immediately under any load
  - check_and_apply_failures returns the correct member ID
  - Removing a failed member reroutes load via re-analysis (no separate
    phenomenological redistribution law)
  - all_failed() correctly detects total collapse
"""

import sys
import dataclasses
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from structure.frames import frame_building_2d
from structure.frames import frame_2d_simple
from solver.equilibrium import solve
from solver.failure import check_and_apply_failures, all_failed


def test_member_fails_under_low_capacity():
    """A member with sigma_y = 1 Pa fails immediately under any load."""
    frame = frame_2d_simple.build()
    frame.members[0].material = dataclasses.replace(frame.members[0].material, sigma_y=1.0)  # 1 Pa — fails under any real force

    es = solve(frame, step=0)
    newly_failed = check_and_apply_failures(frame, es)

    assert 0 in newly_failed, f"Expected member 0 to fail, got: {newly_failed}"
    assert frame.members[0].failed == True
    print(f"  PASS: Member 0 failed as expected (sigma_y=1 Pa, force={es.member_states[0].axial_force:.2f} N)")


def test_failure_marks_member_in_frame():
    """After failure, frame.members[0].failed is True."""
    frame = frame_2d_simple.build()
    frame.members[0].material = dataclasses.replace(frame.members[0].material, sigma_y=1.0)
    es = solve(frame, step=0)
    check_and_apply_failures(frame, es)
    assert frame.members[0].failed == True
    print("  PASS: member.failed flag set correctly")


def test_member_removal_reroutes_load():
    """
    Alternate-load-path re-analysis: failing one member of a redundant frame
    and re-solving must (a) keep the system solvable, (b) leave the failed
    member with zero strain energy, and (c) change the surviving members'
    energy distribution (load has been rerouted). No separate redistribution
    law is used — exclusion from the stiffness assembly does the work.
    """
    frame = frame_building_2d.build()

    es_before = solve(frame, step=0)
    energy_before = {ms.member_id: ms.strain_energy for ms in es_before.member_states}

    # Force the most-loaded member to fail, then re-analyse.
    target = max(es_before.member_states, key=lambda ms: ms.strain_energy)
    frame.members[target.member_id].failed = True

    es_after = solve(frame, step=1)
    energy_after = {ms.member_id: ms.strain_energy for ms in es_after.member_states}

    assert energy_after[target.member_id] == 0.0, \
        "Failed member must carry zero strain energy after removal"

    survivors_changed = any(
        abs(energy_after[mid] - energy_before[mid]) > 1e-9
        for mid in energy_after if mid != target.member_id
    )
    assert survivors_changed, "Surviving members must absorb the rerouted load"
    print(f"  PASS: load rerouted after removing member {target.member_id} "
          f"(survivor energy redistributed by re-analysis)")


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
    print("=== Phase 4: Failure & Alternate-Load-Path Re-analysis ===")
    test_member_fails_under_low_capacity()
    test_failure_marks_member_in_frame()
    test_member_removal_reroutes_load()
    test_all_failed_false_initially()
    test_all_failed_true_when_all_marked()
    print("All Phase 4 tests passed.\n")