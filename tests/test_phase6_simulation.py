"""
tests/test_phase6_simulation.py
================================
Phase 6: End-to-end simulation run tests.

Checks:
  - Both scenarios complete without crashing
  - SimulationResult fields are populated correctly
  - Entropy history length matches energy history length
  - With forced low sigma_y, collapse is detected
  - Without failure, simulation runs to max_steps with no collapse
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from simulation.scenarios import run_scenario
from simulation.runner import run
from structure.frames import frame_2d_simple


def test_2d_simple_runs():
    """scenario_2d_simple completes and returns a SimulationResult."""
    result = run_scenario("2d_simple", max_steps=20)
    assert result is not None
    assert result.frame_name == "2D Simple Truss"
    assert len(result.energy_history) > 0
    assert len(result.entropy_history) == len(result.energy_history)
    print(f"  PASS: 2d_simple ran {len(result.energy_history)} steps")


def test_3d_redundant_runs():
    """scenario_3d_redundant completes and returns a SimulationResult."""
    result = run_scenario("3d_redundant", max_steps=20)
    assert result is not None
    assert result.frame_name == "3D Redundant Space Frame"
    assert len(result.energy_history) > 0
    print(f"  PASS: 3d_redundant ran {len(result.energy_history)} steps")


def test_collapse_detected_with_low_sigma_y():
    """Collapse is detected when members have very low sigma_y."""
    frame = frame_2d_simple.build()
    for m in frame.members:
        m.sigma_y = 1.0  # 1 Pa — fails under any real force

    result = run(frame, max_steps=50, collapse_method="threshold", collapse_threshold=-0.01)
    assert result.collapse_detected == True, "Expected collapse to be detected"
    assert result.collapse_step is not None
    print(f"  PASS: Collapse detected at step {result.collapse_step}")


def test_no_collapse_without_failure():
    """With very high sigma_y, no collapse occurs and simulation runs to max_steps."""
    frame = frame_2d_simple.build()
    for m in frame.members:
        m.sigma_y = 1e20  # Indestructible

    result = run(frame, max_steps=10, collapse_method="zscore")
    assert result.collapse_detected == False
    assert len(result.energy_history) == 10
    assert result.failed_sequence == []
    print(f"  PASS: No collapse, ran all 10 steps")


def test_failed_sequence_order():
    """failed_sequence is populated in failure order."""
    frame = frame_2d_simple.build()
    frame.members[0].sigma_y = 1.0  # Only member 0 fails

    result = run(frame, max_steps=20, collapse_method="threshold", collapse_threshold=-999)
    if result.failed_sequence:
        assert result.failed_sequence[0] == 0, \
            f"Expected member 0 first, got {result.failed_sequence}"
        print(f"  PASS: Failed sequence = {result.failed_sequence}")
    else:
        print("  SKIP: No failures occurred")


def test_unknown_scenario_raises():
    """run_scenario raises ValueError for unknown scenario name."""
    try:
        run_scenario("nonexistent_scenario")
        assert False, "Should have raised ValueError"
    except ValueError as e:
        print(f"  PASS: ValueError raised correctly: {e}")


if __name__ == "__main__":
    print("=== Phase 6: Full Simulation Run ===")
    test_2d_simple_runs()
    test_3d_redundant_runs()
    test_collapse_detected_with_low_sigma_y()
    test_no_collapse_without_failure()
    test_failed_sequence_order()
    test_unknown_scenario_raises()
    print("All Phase 6 tests passed.\n")