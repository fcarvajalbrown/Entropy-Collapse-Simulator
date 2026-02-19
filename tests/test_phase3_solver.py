"""
tests/test_phase3_solver.py
============================
Phase 3: Verify the equilibrium solver produces physically correct results.

Checks:
  - solve() returns an EnergyState without crashing
  - Midspan node deflects downward (negative uy) under downward load
  - All active member strain energies are non-negative
  - Total energy equals sum of member energies
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from structure.frames import frame_2d_simple, frame_3d_redundant
from structure.stiffness import assemble_global_stiffness, apply_boundary_conditions
from solver.equilibrium import solve, _build_load_vector, _solve_system


def test_solve_returns_energy_state():
    """solve() completes without error and returns EnergyState."""
    frame = frame_2d_simple.build()
    es = solve(frame, step=0)
    assert es is not None
    assert es.step == 0
    assert len(es.member_states) == len(frame.members)
    print("  PASS: solve() returns EnergyState")


def test_midspan_deflects_downward():
    """Midspan node (Node 1) has negative uy displacement under downward load."""
    frame = frame_2d_simple.build()
    K = assemble_global_stiffness(frame)
    K = apply_boundary_conditions(K, frame)
    F = _build_load_vector(frame)
    u = _solve_system(K, F)

    # Node 1 uy = DOF index (1 * 6 + 1) = 7
    uy_midspan = u[1 * 6 + 1]
    assert uy_midspan < 0, f"Expected downward deflection, got uy={uy_midspan:.6f}"
    print(f"  PASS: Midspan uy = {uy_midspan:.6e} m (downward)")


def test_strain_energies_non_negative():
    """All active member strain energies are >= 0."""
    frame = frame_2d_simple.build()
    es = solve(frame, step=0)
    for ms in es.member_states:
        assert ms.strain_energy >= 0, f"Member {ms.member_id} has negative energy: {ms.strain_energy}"
    print("  PASS: All strain energies non-negative")


def test_total_energy_consistent():
    """EnergyState.total_energy equals sum of member strain energies."""
    frame = frame_2d_simple.build()
    es = solve(frame, step=0)
    computed_total = sum(ms.strain_energy for ms in es.member_states)
    assert abs(es.total_energy - computed_total) < 1e-10, \
        f"Total mismatch: {es.total_energy} vs {computed_total}"
    print(f"  PASS: Total energy consistent ({es.total_energy:.4f} J)")


def test_solve_3d():
    """solve() works on the 3D redundant frame without crashing."""
    frame = frame_3d_redundant.build()
    es = solve(frame, step=0)
    assert len(es.member_states) == 8
    assert es.total_energy >= 0
    print(f"  PASS: 3D frame solved (total energy = {es.total_energy:.4f} J)")


if __name__ == "__main__":
    print("=== Phase 3: Equilibrium Solver ===")
    test_solve_returns_energy_state()
    test_midspan_deflects_downward()
    test_strain_energies_non_negative()
    test_total_energy_consistent()
    test_solve_3d()
    print("All Phase 3 tests passed.\n")