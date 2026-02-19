"""
tests/test_phase2_stiffness.py
===============================
Phase 2: Verify stiffness matrix assembly.

Checks:
  - K has the correct shape (n_nodes * 6)
  - K is symmetric
  - Boundary conditions zero out the correct rows/cols
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from structure.frames import frame_2d_simple, frame_3d_redundant
from structure.stiffness import assemble_global_stiffness, apply_boundary_conditions


def test_k_shape_2d():
    """K has shape (n_nodes*6, n_nodes*6) for 2D simple frame."""
    frame = frame_2d_simple.build()
    K = assemble_global_stiffness(frame)
    expected = len(frame.nodes) * 6
    assert K.shape == (expected, expected), f"Expected {expected}x{expected}, got {K.shape}"
    print(f"  PASS: K shape correct ({expected}x{expected})")


def test_k_symmetry_2d():
    """K is symmetric for 2D simple frame."""
    frame = frame_2d_simple.build()
    K = assemble_global_stiffness(frame)
    assert np.allclose(K, K.T, atol=1e-6), "K is not symmetric"
    print("  PASS: K is symmetric")


def test_k_shape_3d():
    """K has correct shape for 3D redundant frame."""
    frame = frame_3d_redundant.build()
    K = assemble_global_stiffness(frame)
    expected = len(frame.nodes) * 6
    assert K.shape == (expected, expected)
    print(f"  PASS: K shape correct 3D ({expected}x{expected})")


def test_k_symmetry_3d():
    """K is symmetric for 3D redundant frame."""
    frame = frame_3d_redundant.build()
    K = assemble_global_stiffness(frame)
    assert np.allclose(K, K.T, atol=1e-6), "K is not symmetric for 3D frame"
    print("  PASS: K is symmetric (3D)")


def test_boundary_conditions_2d():
    """Fixed DOF rows and cols are zeroed after BC application."""
    frame = frame_2d_simple.build()
    K = assemble_global_stiffness(frame)
    K = apply_boundary_conditions(K, frame)

    for node in frame.nodes:
        for dof in node.fixed_dofs:
            gdof = node.id * 6 + dof
            row_sum = np.sum(np.abs(K[gdof, :])) - K[gdof, gdof]
            col_sum = np.sum(np.abs(K[:, gdof])) - K[gdof, gdof]
            assert row_sum < 1e-10, f"Row {gdof} not zeroed: sum={row_sum}"
            assert col_sum < 1e-10, f"Col {gdof} not zeroed: sum={col_sum}"
            assert K[gdof, gdof] == 1.0, f"Diagonal {gdof} not set to 1"

    print("  PASS: Boundary conditions applied correctly")


if __name__ == "__main__":
    print("=== Phase 2: Stiffness Assembly ===")
    test_k_shape_2d()
    test_k_symmetry_2d()
    test_k_shape_3d()
    test_k_symmetry_3d()
    test_boundary_conditions_2d()
    print("All Phase 2 tests passed.\n")