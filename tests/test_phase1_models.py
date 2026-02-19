"""
tests/test_phase1_models.py
============================
Phase 1: Verify all dataclasses instantiate correctly and
that both frame build() functions return valid FrameData.
No solver logic is tested here — pure data contract validation.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.models import Node, Member, Load, FrameData, EnergyState, MemberState, EntropyRecord, SimulationResult
from structure.frames import frame_2d_simple, frame_3d_redundant


def test_node_instantiation():
    """Node creates correctly with and without fixed_dofs."""
    n = Node(id=0, x=0.0, y=0.0, z=0.0, fixed_dofs=[0, 1])
    assert n.id == 0
    assert n.fixed_dofs == [0, 1]
    n_free = Node(id=1, x=5.0, y=0.0)
    assert n_free.fixed_dofs == []
    print("  PASS: Node instantiation")


def test_member_instantiation():
    """Member creates correctly with default failed=False."""
    m = Member(id=0, node_start=0, node_end=1, E=200e9, A=0.01, I=1e-4)
    assert m.failed == False
    print("  PASS: Member instantiation")


def test_load_instantiation():
    """Load creates correctly."""
    l = Load(node_id=1, dof=1, magnitude=-50000.0)
    assert l.magnitude == -50000.0
    print("  PASS: Load instantiation")


def test_frame_2d_simple_build():
    """frame_2d_simple.build() returns a valid FrameData."""
    frame = frame_2d_simple.build()
    assert isinstance(frame, FrameData)
    assert len(frame.nodes) == 3
    assert len(frame.members) == 2
    assert len(frame.loads) == 1
    assert frame.loads[0].magnitude == -50_000.0
    print("  PASS: frame_2d_simple builds correctly")


def test_frame_3d_redundant_build():
    """frame_3d_redundant.build() returns a valid FrameData."""
    frame = frame_3d_redundant.build()
    assert isinstance(frame, FrameData)
    assert len(frame.nodes) == 5
    assert len(frame.members) == 8
    assert len(frame.loads) == 1
    print("  PASS: frame_3d_redundant builds correctly")


def test_energy_state():
    """EnergyState and MemberState instantiate correctly."""
    ms = MemberState(member_id=0, strain_energy=100.0, axial_force=5000.0, deformation=0.001)
    es = EnergyState(step=0, total_energy=100.0, member_states=[ms])
    assert es.total_energy == 100.0
    print("  PASS: EnergyState instantiation")


def test_entropy_record():
    """EntropyRecord instantiates correctly."""
    r = EntropyRecord(step=0, entropy=1.2, delta_entropy=-0.1, energy_distribution=[(0, 0.6), (1, 0.4)])
    assert r.entropy == 1.2
    print("  PASS: EntropyRecord instantiation")


if __name__ == "__main__":
    print("=== Phase 1: Models ===")
    test_node_instantiation()
    test_member_instantiation()
    test_load_instantiation()
    test_frame_2d_simple_build()
    test_frame_3d_redundant_build()
    test_energy_state()
    test_entropy_record()
    print("All Phase 1 tests passed.\n")