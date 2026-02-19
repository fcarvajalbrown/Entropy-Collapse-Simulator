"""
structure/frames/frame_3d_redundant.py
=======================================
Defines a 3D redundant space frame with multiple load paths.

Geometry — square base with a central apex node:

    Base nodes (z=0):
        Node 0: (0, 0, 0) — pinned
        Node 1: (5, 0, 0) — pinned
        Node 2: (5, 5, 0) — pinned
        Node 3: (0, 5, 0) — pinned

    Apex node:
        Node 4: (2.5, 2.5, 4) — free

Members:
    0-3: Diagonal legs (base corners to apex)
    4-7: Base chords (square perimeter)

Load: -200 kN downward at apex (DOF 2 = uz).
Material: S355 Steel (higher grade for 3D frame)
"""

from core.models import FrameData, Node, Member, Load, STEEL_S355
import dataclasses


# S355 with larger section for 3D frame
STEEL_S355_LARGE = dataclasses.replace(STEEL_S355, A=0.02, I=2e-4)


def build() -> FrameData:
    """
    Construct and return the 3D redundant space frame FrameData.

    Returns:
        FrameData with 5 nodes, 8 members, pinned base, apex load.
    """
    return FrameData(
        name="3D Redundant Space Frame",
        nodes=_define_nodes(),
        members=_define_members(),
        loads=_define_loads()
    )


def _define_nodes() -> list[Node]:
    """
    Define 4 pinned base corners and 1 free apex node.

    Returns:
        List of 5 Node objects.
    """
    return [
        Node(id=0, x=0.0, y=0.0, z=0.0, fixed_dofs=[0, 1, 2]),
        Node(id=1, x=5.0, y=0.0, z=0.0, fixed_dofs=[0, 1, 2]),
        Node(id=2, x=5.0, y=5.0, z=0.0, fixed_dofs=[0, 1, 2]),
        Node(id=3, x=0.0, y=5.0, z=0.0, fixed_dofs=[0, 1, 2]),
        Node(id=4, x=2.5, y=2.5, z=4.0, fixed_dofs=[]),
    ]


def _define_members() -> list[Member]:
    """
    Define 4 diagonal legs and 4 base chord members.

    Returns:
        List of 8 Member objects using S355 large section.
    """
    return [
        # Diagonal legs
        Member(id=0, node_start=0, node_end=4, material=STEEL_S355_LARGE),
        Member(id=1, node_start=1, node_end=4, material=STEEL_S355_LARGE),
        Member(id=2, node_start=2, node_end=4, material=STEEL_S355_LARGE),
        Member(id=3, node_start=3, node_end=4, material=STEEL_S355_LARGE),
        # Base chords
        Member(id=4, node_start=0, node_end=1, material=STEEL_S355_LARGE),
        Member(id=5, node_start=1, node_end=2, material=STEEL_S355_LARGE),
        Member(id=6, node_start=2, node_end=3, material=STEEL_S355_LARGE),
        Member(id=7, node_start=3, node_end=0, material=STEEL_S355_LARGE),
    ]


def _define_loads() -> list[Load]:
    """
    Apply a 200 kN downward load at the apex node.

    DOF 2 = uz (vertical in 3D).

    Returns:
        List with one Load object.
    """
    return [
        Load(node_id=4, dof=2, magnitude=-200_000.0)
    ]