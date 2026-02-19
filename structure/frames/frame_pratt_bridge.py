"""
structure/frames/frame_pratt_bridge.py
=======================================
Defines a Pratt truss bridge frame for progressive collapse simulation.

Geometry — 6-panel Pratt truss (2D, z=0):

    Top chord:    T0---T1---T2---T3---T4---T5---T6
                  |  / |  / |  / |  / |  / |  / |
    Bot chord:    B0---B1---B2---B3---B4---B5---B6

    Panel width  : 5.0 m
    Truss height : 4.0 m
    Total span   : 30.0 m

Node numbering:
    Bottom chord: nodes 0–6  (y = 0.0)
    Top chord:    nodes 7–13 (y = 4.0)

    B0=0, B1=1, B2=2, B3=3, B4=4, B5=5, B6=6
    T0=7, T1=8, T2=9, T3=10, T4=11, T5=12, T6=13

Member layout (Pratt pattern — diagonals carry tension under gravity):
    Bottom chords : B0-B1, B1-B2, B2-B3, B3-B4, B4-B5, B5-B6  (members 0–5)
    Top chords    : T0-T1, T1-T2, T2-T3, T3-T4, T4-T5, T5-T6  (members 6–11)
    Verticals     : B0-T0, B1-T1, B2-T2, B3-T3, B4-T4, B5-T5, B6-T6 (members 12–18)
    Diagonals     : B1-T0, B2-T1, B3-T2, B4-T3, B5-T4, B6-T5  (members 19–24)
                    (Pratt: diagonals slope toward center from bottom)

Supports:
    B0 (node 0): pinned  — fixed_dofs [0, 1]
    B6 (node 6): roller  — fixed_dofs [1] (free to slide horizontally)

Load:
    Distributed traffic load as point loads at each bottom chord node B1–B5.
    Magnitude: -100 kN per node (downward, DOF 1 = uy).
    End nodes B0, B6 carry half load: -50 kN each.

Materials:
    Bottom chord  : high-strength S355 (tension members, larger section)
    Top chord     : S355 standard (compression members)
    Verticals     : S275 standard
    Diagonals     : S355 standard (primary load-carrying members)
"""

import dataclasses
from core.models import FrameData, Node, Member, Load, Material, STEEL_S275, STEEL_S355


# ---------------------------------------------------------------------------
# Bridge geometry constants
# ---------------------------------------------------------------------------

PANEL_WIDTH  = 5.0   # meters per panel
TRUSS_HEIGHT = 4.0   # meters
N_PANELS     = 6     # number of panels
N_NODES_CHORD = N_PANELS + 1  # 7 nodes per chord

# ---------------------------------------------------------------------------
# Material definitions
# ---------------------------------------------------------------------------

# Section properties use realistic I-section proportions for a 30m bridge.
# Keeping EA/EI ratios well-conditioned to avoid numerical overflow.
# Reference: typical W-section steel members for medium-span bridges.

# Bottom chord — tension dominant (W360x122 equivalent)
BOTTOM_CHORD_MAT = dataclasses.replace(
    STEEL_S355, name="S355 Bottom Chord", A=0.0155, I=3.65e-4
)

# Top chord — compression dominant (W310x97 equivalent)
TOP_CHORD_MAT = dataclasses.replace(
    STEEL_S355, name="S355 Top Chord", A=0.0123, I=2.22e-4
)

# Verticals — lighter section (W200x52 equivalent)
VERTICAL_MAT = dataclasses.replace(
    STEEL_S275, name="S275 Vertical", A=0.0066, I=5.27e-5
)

# Diagonals — primary load path (W250x89 equivalent)
DIAGONAL_MAT = dataclasses.replace(
    STEEL_S355, name="S355 Diagonal", A=0.0114, I=1.42e-4
)


def build() -> FrameData:
    """
    Construct and return the Pratt bridge FrameData.

    Returns:
        FrameData with 14 nodes, 25 members, pinned/roller supports,
        and distributed point loads along the bottom chord.
    """
    return FrameData(
        name="Pratt Truss Bridge (6-panel, 30m span)",
        nodes=_define_nodes(),
        members=_define_members(),
        loads=_define_loads()
    )


def _define_nodes() -> list[Node]:
    """
    Define 7 bottom chord nodes and 7 top chord nodes.

    Bottom chord at y=0, top chord at y=TRUSS_HEIGHT.
    Supports at B0 (pinned) and B6 (roller).

    Returns:
        List of 14 Node objects.
    """
    nodes = []

    # Out-of-plane and rotational DOFs are fixed for all nodes (planar analysis).
    # DOFs 2=uz, 3=rx, 4=ry are constrained everywhere.
    # In-plane DOF 5=rz is left free to allow bending in the XY plane.
    PLANAR_DOFS = [2, 3, 4]

    # Bottom chord: nodes 0–6
    for i in range(N_NODES_CHORD):
        x = i * PANEL_WIDTH
        if i == 0:
            fixed_dofs = [0, 1] + PLANAR_DOFS   # Pinned left support
        elif i == N_PANELS:
            fixed_dofs = [1] + PLANAR_DOFS       # Roller right support
        else:
            fixed_dofs = PLANAR_DOFS             # Free in-plane only
        nodes.append(Node(id=i, x=x, y=0.0, z=0.0, fixed_dofs=fixed_dofs))

    # Top chord: nodes 7–13
    for i in range(N_NODES_CHORD):
        x = i * PANEL_WIDTH
        node_id = N_NODES_CHORD + i  # 7–13
        nodes.append(Node(id=node_id, x=x, y=TRUSS_HEIGHT, z=0.0, fixed_dofs=PLANAR_DOFS))

    return nodes


def _define_members() -> list[Member]:
    """
    Define all 25 members: bottom chords, top chords, verticals, diagonals.

    Member ID layout:
        0–5:   Bottom chords (B0-B1 through B5-B6)
        6–11:  Top chords    (T0-T1 through T5-T6)
        12–18: Verticals     (B0-T0 through B6-T6)
        19–24: Diagonals     (B1-T0 through B6-T5, Pratt pattern)

    Returns:
        List of 25 Member objects.
    """
    members = []
    mid = 0  # member id counter

    # Bottom chords
    for i in range(N_PANELS):
        members.append(Member(id=mid, node_start=i, node_end=i+1, material=BOTTOM_CHORD_MAT))
        mid += 1

    # Top chords (node offset = N_NODES_CHORD = 7)
    for i in range(N_PANELS):
        members.append(Member(
            id=mid,
            node_start=N_NODES_CHORD + i,
            node_end=N_NODES_CHORD + i + 1,
            material=TOP_CHORD_MAT
        ))
        mid += 1

    # Verticals (connecting bottom node i to top node i)
    for i in range(N_NODES_CHORD):
        members.append(Member(
            id=mid,
            node_start=i,
            node_end=N_NODES_CHORD + i,
            material=VERTICAL_MAT
        ))
        mid += 1

    # Diagonals — Pratt pattern: slope from bottom-right to top-left
    # Each diagonal connects bottom node (i+1) to top node (i)
    for i in range(N_PANELS):
        members.append(Member(
            id=mid,
            node_start=i + 1,           # Bottom node (right side of panel)
            node_end=N_NODES_CHORD + i, # Top node (left side of panel)
            material=DIAGONAL_MAT
        ))
        mid += 1

    return members


def _define_loads() -> list[Load]:
    """
    Apply distributed traffic load as point loads at bottom chord nodes.

    Interior nodes (B1–B5): -100 kN each (DOF 1 = uy, downward).
    End nodes (B0, B6):     -50 kN each (half load, tributary area).

    Total load on bridge: 2 * 50 + 5 * 100 = 600 kN.

    Returns:
        List of 7 Load objects.
    """
    loads = []
    for i in range(N_NODES_CHORD):
        if i == 0 or i == N_PANELS:
            magnitude = -50_000.0   # Half load at supports
        else:
            magnitude = -100_000.0  # Full panel load at interior nodes
        loads.append(Load(node_id=i, dof=1, magnitude=magnitude))
    return loads