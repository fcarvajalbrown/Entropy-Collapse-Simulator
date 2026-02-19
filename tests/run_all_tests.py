"""
tests/run_all_tests.py
=======================
Runs all test phases in order and reports a summary.
Execute from the project root:

    python tests/run_all_tests.py
"""

import subprocess
import sys
import os

TESTS = [
    ("Phase 1 — Models",          "tests/test_phase1_models.py"),
    ("Phase 2 — Stiffness",        "tests/test_phase2_stiffness.py"),
    ("Phase 3 — Solver",           "tests/test_phase3_solver.py"),
    ("Phase 4 — Failure",          "tests/test_phase4_failure.py"),
    ("Phase 5 — Entropy",          "tests/test_phase5_entropy.py"),
    ("Phase 6 — Simulation",       "tests/test_phase6_simulation.py"),
    ("Phase 7 — Visualization",    "tests/test_phase7_visualization.py"),
]


def run_all():
    """Run each test file as a subprocess and report pass/fail per phase."""
    passed, failed = [], []

    for label, path in TESTS:
        print(f"\n{'='*50}")
        print(f"Running: {label}")
        print('='*50)
        result = subprocess.run(
            [sys.executable, path],
            capture_output=False
        )
        if result.returncode == 0:
            passed.append(label)
        else:
            failed.append(label)

    print(f"\n{'='*50}")
    print("SUMMARY")
    print('='*50)
    print(f"  Passed: {len(passed)}/{len(TESTS)}")
    for p in passed:
        print(f"    PASS  {p}")
    if failed:
        print(f"  Failed: {len(failed)}/{len(TESTS)}")
        for f in failed:
            print(f"    FAIL  {f}")
        sys.exit(1)
    else:
        print("\nAll tests passed.")


if __name__ == "__main__":
    run_all()