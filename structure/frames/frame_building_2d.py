"""
structure/frames/frame_building_2d.py
======================================
Planar (2D) multi-story, multi-bay steel moment frame.

This frame replaces the earlier 3D pyramid scenario. The finite-element
formulation in structure/stiffness.py is a planar Euler-Bernoulli frame
element (axial + in-plane bending + in-plane rotation); a moment frame is its
natural, fully valid application and is directly relevant to building
progressive-collapse studies via notional column removal.

Geometry — 2 bays x 3 stories (in the global XY plane, z = 0):

    L3:  9----10----11        y = 10.5 m
         |     |     |
    L2:  6-----7-----8        y = 7.0 m
         |     |     |
    L1:  3-----4-----5        y = 3.5 m
         |     |     |
    L0:  0-----1-----2        y = 0.0 m   (fully fixed bases)
        x=0   x=6   x=12

Node id = level * 3 + column (column in {0,1,2}, level in {0,1,2,3}).

Members:
    Columns (vertical), ids  0-8 : 3 column lines x 3 stories
    Beams   (horizontal), ids 9-14: 3 elevated levels x 2 bays

Supports:
    Base nodes 0,1,2 are fully fixed (encastre): fixed_dofs = all 6 DOFs.
    All other nodes constrain only the out-of-plane DOFs (2=uz, 3=rx, 4=ry);
    in-plane ux (0), uy (1) and rz (5) are free.

Loads:
    Gravity applied as downward vertical point loads (DOF 1 = uy) at every
    elevated beam-column joint, representing tributary floor load.

Materials:
    Columns : heavier wide-flange section (HEB-class), S355.
    Beams   : lighter section (IPE-class), S355.
"""

import dataclasses
from core.models import FrameData, Node, Member, Load, STEEL_S355


# ---------------------------------------------------------------------------
# Geometry constants
# ---------------------------------------------------------------------------

BAY_WIDTH   = 6.0    # meters between column lines
STORY_HEIGHT = 3.5   # meters between floors
N_COLS  = 3          # column lines (2 bays)
N_LEVELS = 4         # ground + 3 elevated floors

# Out-of-plane DOFs constrained at every node for planar analysis.
PLANAR_DOFS = [2, 3, 4]

# ---------------------------------------------------------------------------
# Section / material definitions (explicit extreme-fibre distance c)
# ---------------------------------------------------------------------------

# Columns — HEB300-class (d ~ 300 mm)
COLUMN_MAT = dataclasses.replace(
    STEEL_S355, name="S355 Column", A=0.0149, I=2.517e-4, c=0.150
)

# Beams — IPE400-class (d ~ 400 mm)
BEAM_MAT = dataclasses.replace(
    STEEL_S355, name="S355 Beam", A=0.00845, I=2.313e-4, c=0.200
)

# Tributary gravity load per elevated joint (downward, N).
JOINT_LOAD = -80_000.0


def build(n_bays: int = 2, n_stories: int = 3) -> FrameData:
    """
    Construct a planar steel moment frame of arbitrary size.

    Args:
        n_bays: Number of bays (column lines = n_bays + 1).
        n_stories: Number of elevated floors (levels = n_stories + 1).

    The default (2 bays, 3 stories) is the standard building case; larger
    values (e.g. 3 bays, 6 stories) give the realistic scalability case study.

    Returns:
        FrameData with fixed column bases and gravity joint loads at every
        elevated beam-column connection.
    """
    n_cols = n_bays + 1
    n_levels = n_stories + 1

    def node_id(level, col):
        return level * n_cols + col

    nodes = []
    for level in range(n_levels):
        for col in range(n_cols):
            fixed = [0, 1, 2, 3, 4, 5] if level == 0 else list(PLANAR_DOFS)
            nodes.append(Node(id=node_id(level, col),
                              x=col * BAY_WIDTH, y=level * STORY_HEIGHT, z=0.0,
                              fixed_dofs=fixed))

    members = []
    mid = 0
    for level in range(n_levels - 1):           # columns
        for col in range(n_cols):
            members.append(Member(id=mid, node_start=node_id(level, col),
                                  node_end=node_id(level + 1, col), material=COLUMN_MAT))
            mid += 1
    for level in range(1, n_levels):            # beams
        for col in range(n_cols - 1):
            members.append(Member(id=mid, node_start=node_id(level, col),
                                  node_end=node_id(level, col + 1), material=BEAM_MAT))
            mid += 1

    loads = [Load(node_id=node_id(level, col), dof=1, magnitude=JOINT_LOAD)
             for level in range(1, n_levels) for col in range(n_cols)]

    name = (f"2D Moment Frame ({n_bays}-bay, {n_stories}-story)")
    return FrameData(name=name, nodes=nodes, members=members, loads=loads)
