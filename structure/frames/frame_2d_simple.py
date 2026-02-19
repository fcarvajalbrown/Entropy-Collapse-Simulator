"""
structure/frames/frame_2d_simple.py
====================================
Defines a simple 2D three-member truss for baseline validation.

Geometry (all z = 0.0):

    Node 0 -------- Node 1 -------- Node 2
    (0, 0)          (5, 0)          (10, 0)
    pinned                           pinned
                      |
                  F = -50 kN (downward, DOF 1 = uy)

Members:
    0 → connects Node 0 to Node 1 (left span)
    1 → connects Node 1 to Node 2 (right span)

Material: Steel (E = 200 GPa, A = 0.01 m², I = 1e-4 m⁴)

This frame is intentionally simple enough to validate results by hand.
Both members will fail sequentially under the central load, producing
a clean two-step entropy drop for calibrating the collapse detector.
"""

from core.models import FrameData, Node, Member, Load


# ---------------------------------------------------------------------------
# Material & section constants
# ---------------------------------------------------------------------------

E_STEEL = 200e9   # Young's modulus, Pa
A_SECTION = 0.01  # Cross-sectional area, m²
I_SECTION = 1e-4  # Second moment of area, m⁴


def build() -> FrameData:
    """
    Construct and return the simple 2D truss FrameData.

    Returns:
        FrameData with 3 nodes, 2 members, boundary conditions,
        and a single downward point load at the central node.
    """
    nodes = _define_nodes()
    members = _define_members()
    loads = _define_loads()

    return FrameData(
        name="2D Simple Truss",
        nodes=nodes,
        members=members,
        loads=loads
    )


def _define_nodes() -> list[Node]:
    """
    Define the three nodes of the truss.

    Node 0: left support  — pinned (ux, uy fixed → DOFs 0, 1)
    Node 1: midspan joint — free
    Node 2: right support — pinned (ux, uy fixed → DOFs 0, 1)

    Returns:
        List of 3 Node objects.
    """
    return [
        Node(id=0, x=0.0,  y=0.0, z=0.0, fixed_dofs=[0, 1]),
        Node(id=1, x=5.0,  y=0.0, z=0.0, fixed_dofs=[]),
        Node(id=2, x=10.0, y=0.0, z=0.0, fixed_dofs=[0, 1]),
    ]


def _define_members() -> list[Member]:
    """
    Define two horizontal members spanning between the three nodes.

    Member 0: Node 0 → Node 1 (left span, 5 m)
    Member 1: Node 1 → Node 2 (right span, 5 m)

    Returns:
        List of 2 Member objects with steel properties.
    """
    return [
        Member(id=0, node_start=0, node_end=1, E=E_STEEL, A=A_SECTION, I=I_SECTION),
        Member(id=1, node_start=1, node_end=2, E=E_STEEL, A=A_SECTION, I=I_SECTION),
    ]


def _define_loads() -> list[Load]:
    """
    Apply a single downward point load at Node 1 (midspan).

    DOF 1 = uy (vertical displacement).
    Magnitude = -50,000 N (negative = downward).

    Returns:
        List with one Load object.
    """
    return [
        Load(node_id=1, dof=1, magnitude=-50_000.0)
    ]