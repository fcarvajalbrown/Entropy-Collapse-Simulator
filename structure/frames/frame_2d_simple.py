"""
structure/frames/frame_2d_simple.py
====================================
Defines a simple 2D two-span beam for baseline validation.

This is a beam, not a truss: the members are rigid-jointed Euler-Bernoulli
frame elements and the transverse mid-span load is carried entirely by
bending. It is the degenerate, non-redundant reference case.

Geometry (all z = 0.0):

    Node 0 -------- Node 1 -------- Node 2
    (0, 0)          (5, 0)          (10, 0)
    pinned                           pinned
                      |
                  F = -50 kN (downward, DOF 1 = uy)

Members:
    0 → connects Node 0 to Node 1 (left span)
    1 → connects Node 1 to Node 2 (right span)

Material: S275 Steel (E=200 GPa, A=0.01 m², I=1e-4 m⁴, sigma_y=275 MPa)
"""

from core.models import FrameData, Node, Member, Load, STEEL_S275


def build() -> FrameData:
    """
    Construct and return the two-span beam FrameData.

    Returns:
        FrameData with 3 nodes, 2 members, boundary conditions,
        and a single downward point load at the central node.
    """
    return FrameData(
        name="Two-span beam",
        nodes=_define_nodes(),
        members=_define_members(),
        loads=_define_loads()
    )


def _define_nodes() -> list[Node]:
    """
    Define three nodes: two pinned supports and one free midspan joint.

    Returns:
        List of 3 Node objects.
    """
    # Planar (XY) analysis: constrain out-of-plane DOFs (2=uz, 3=rx, 4=ry)
    # at every node so the global stiffness matrix is non-singular. The
    # in-plane rotation (5=rz) is left free to capture bending.
    PLANAR_DOFS = [2, 3, 4]
    return [
        Node(id=0, x=0.0,  y=0.0, z=0.0, fixed_dofs=[0, 1] + PLANAR_DOFS),
        Node(id=1, x=5.0,  y=0.0, z=0.0, fixed_dofs=PLANAR_DOFS),
        Node(id=2, x=10.0, y=0.0, z=0.0, fixed_dofs=[0, 1] + PLANAR_DOFS),
    ]


def _define_members() -> list[Member]:
    """
    Define two horizontal members using S275 steel.

    Returns:
        List of 2 Member objects.
    """
    return [
        Member(id=0, node_start=0, node_end=1, material=STEEL_S275),
        Member(id=1, node_start=1, node_end=2, material=STEEL_S275),
    ]


def _define_loads() -> list[Load]:
    """
    Apply a 50 kN downward point load at Node 1 (midspan).

    DOF 1 = uy. Magnitude negative = downward.

    Returns:
        List with one Load object.
    """
    return [
        Load(node_id=1, dof=1, magnitude=-50_000.0)
    ]