"""
structure/frames/frame_3d_redundant.py
=======================================
Defines a 3D redundant space frame with multiple load paths.

Geometry — a square base with a central apex node above it:

         Node 4 (apex)
        /|\ 
       / | \
      /  |  \
    (0)-(1)-(2)-(3)  ← base square, z = 0
     all 4 corners pinned

    Base nodes:
        Node 0: (0, 0, 0) — pinned
        Node 1: (5, 0, 0) — pinned
        Node 2: (5, 5, 0) — pinned
        Node 3: (0, 5, 0) — pinned

    Apex node:
        Node 4: (2.5, 2.5, 4) — free

Members (8 total):
    0: Node 0 → Node 4  (diagonal leg)
    1: Node 1 → Node 4  (diagonal leg)
    2: Node 2 → Node 4  (diagonal leg)
    3: Node 3 → Node 4  (diagonal leg)
    4: Node 0 → Node 1  (base chord)
    5: Node 1 → Node 2  (base chord)
    6: Node 2 → Node 3  (base chord)
    7: Node 3 → Node 0  (base chord)

Load: Downward force at apex (Node 4, DOF 2 = uz).

Redundancy: 4 diagonal legs share the apex load — when one fails,
the remaining three redistribute energy. This produces a multi-step
entropy curve ideal for validating the zscore collapse detector.

Material: Steel (E = 200 GPa, A = 0.02 m², I = 2e-4 m⁴)
"""

from core.models import FrameData, Node, Member, Load


# ---------------------------------------------------------------------------
# Material & section constants
# ---------------------------------------------------------------------------

E_STEEL = 200e9
A_SECTION = 0.02   # Slightly larger than 2D frame — carries more load
I_SECTION = 2e-4


def build() -> FrameData:
    """
    Construct and return the 3D redundant space frame FrameData.

    Returns:
        FrameData with 5 nodes, 8 members, pinned base supports,
        and a downward point load at the apex node.
    """
    nodes = _define_nodes()
    members = _define_members()
    loads = _define_loads()

    return FrameData(
        name="3D Redundant Space Frame",
        nodes=nodes,
        members=members,
        loads=loads
    )


def _define_nodes() -> list[Node]:
    """
    Define 4 pinned base nodes and 1 free apex node.

    Base nodes are fully pinned in x, y, z (DOFs 0, 1, 2).
    Apex node is free to displace in all directions.

    Returns:
        List of 5 Node objects.
    """
    return [
        Node(id=0, x=0.0, y=0.0, z=0.0, fixed_dofs=[0, 1, 2]),
        Node(id=1, x=5.0, y=0.0, z=0.0, fixed_dofs=[0, 1, 2]),
        Node(id=2, x=5.0, y=5.0, z=0.0, fixed_dofs=[0, 1, 2]),
        Node(id=3, x=0.0, y=5.0, z=0.0, fixed_dofs=[0, 1, 2]),
        Node(id=4, x=2.5, y=2.5, z=4.0, fixed_dofs=[]),     # Apex — free
    ]


def _define_members() -> list[Member]:
    """
    Define 4 diagonal legs from base corners to apex,
    and 4 base chord members forming the square perimeter.

    Returns:
        List of 8 Member objects with steel properties.
    """
    return [
        # Diagonal legs (carry vertical load, primary collapse path)
        Member(id=0, node_start=0, node_end=4, E=E_STEEL, A=A_SECTION, I=I_SECTION),
        Member(id=1, node_start=1, node_end=4, E=E_STEEL, A=A_SECTION, I=I_SECTION),
        Member(id=2, node_start=2, node_end=4, E=E_STEEL, A=A_SECTION, I=I_SECTION),
        Member(id=3, node_start=3, node_end=4, E=E_STEEL, A=A_SECTION, I=I_SECTION),

        # Base chords (provide lateral stability, secondary redistribution path)
        Member(id=4, node_start=0, node_end=1, E=E_STEEL, A=A_SECTION, I=I_SECTION),
        Member(id=5, node_start=1, node_end=2, E=E_STEEL, A=A_SECTION, I=I_SECTION),
        Member(id=6, node_start=2, node_end=3, E=E_STEEL, A=A_SECTION, I=I_SECTION),
        Member(id=7, node_start=3, node_end=0, E=E_STEEL, A=A_SECTION, I=I_SECTION),
    ]


def _define_loads() -> list[Load]:
    """
    Apply a downward point load at the apex node (Node 4).

    DOF 2 = uz (vertical in 3D).
    Magnitude = -200,000 N — larger than the 2D case to
    drive progressive failure across the redundant legs.

    Returns:
        List with one Load object.
    """
    return [
        Load(node_id=4, dof=2, magnitude=-200_000.0)
    ]