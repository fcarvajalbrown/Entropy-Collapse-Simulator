"""
tests/test_phase7_visualization.py
====================================
Phase 7: Smoke tests for visualization — no window is opened,
figures are saved to a temp directory and then deleted.

Checks:
  - plot_frame() runs without error and produces a file
  - plot_collapse_sequence() runs without error
  - plot_entropy() runs without error and produces a file
"""

import sys
import dataclasses
import os
import tempfile
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend — no window, safe for testing

from structure.frames import frame_2d_simple
from solver.equilibrium import solve
from entropy.metrics import compute
from simulation.runner import run
from visualization.graph_view import plot_frame, plot_collapse_sequence
from visualization.entropy_plot import plot_entropy


def test_plot_frame_saves_file():
    """plot_frame() saves a PNG without crashing."""
    frame = frame_2d_simple.build()
    es = solve(frame, step=0)
    er = compute(es, previous_entropy=0.0)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "frame_test.png")
        fig = plot_frame(frame, es, er, step=0, show=False, save_path=path)
        assert os.path.exists(path), "Frame plot file was not created"
        assert fig is not None
    print("  PASS: plot_frame() saved file successfully")


def test_plot_collapse_sequence_saves_file():
    """plot_collapse_sequence() saves a PNG without crashing."""
    frame = frame_2d_simple.build()
    failed_sequence = [0, 1]  # Simulate both members failed

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "collapse_test.png")
        fig = plot_collapse_sequence(frame, failed_sequence, show=False, save_path=path)
        assert os.path.exists(path)
        assert fig is not None
    print("  PASS: plot_collapse_sequence() saved file successfully")


def test_plot_entropy_saves_file():
    """plot_entropy() saves a PNG without crashing."""
    frame = frame_2d_simple.build()
    for m in frame.members:
        m.material = dataclasses.replace(m.material, sigma_y=1e20)  # No failures, clean entropy curve

    result = run(frame, max_steps=10, collapse_method="zscore")

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "entropy_test.png")
        fig = plot_entropy(result, show=False, save_path=path)
        assert os.path.exists(path)
        assert fig is not None
    print("  PASS: plot_entropy() saved file successfully")


def test_plot_entropy_with_collapse():
    """plot_entropy() handles collapse_detected=True without crashing."""
    frame = frame_2d_simple.build()
    for m in frame.members:
        m.material = dataclasses.replace(m.material, sigma_y=1.0)

    result = run(frame, max_steps=20, collapse_method="threshold", collapse_threshold=-0.01)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "entropy_collapse_test.png")
        fig = plot_entropy(result, show=False, save_path=path)
        assert os.path.exists(path)
    print("  PASS: plot_entropy() with collapse rendered correctly")


if __name__ == "__main__":
    print("=== Phase 7: Visualization ===")
    test_plot_frame_saves_file()
    test_plot_collapse_sequence_saves_file()
    test_plot_entropy_saves_file()
    test_plot_entropy_with_collapse()
    print("All Phase 7 tests passed.\n")