"""
structure/frames/frame_pratt_bridge.py
=======================================
Defines a redundant pin-jointed truss bridge for the robustness case study.

All members are TRUSS (axial-only) bars and every joint is a pin: rz is
restrained at every node along with the out-of-plane DOFs. The geometry is a
6-panel Pratt truss to which a counter-diagonal is added in every panel
(X-bracing), so the truss is statically indeterminate (redundant). A plain
single-diagonal Pratt truss of this size is statically determinate and would
collapse on the loss of any member; the counter-diagonals give it genuine
alternate load paths, which is what makes R_S meaningful for a truss.

Geometry — 6-panel X-braced truss (2D, z=0):

    Top chord:    T0---T1---T2---T3---T4---T5---T6
                  | X  | X  | X  | X  | X  | X  |
    Bot chord:    B0---B1---B2---B3---B4---B5---B6

    Panel width  : 5.0 m
    Truss height : 4.0 m
    Total span   : 30.0 m

Node numbering:
    Bottom chord: nodes 0-6  (y = 0.0)
    Top chord:    nodes 7-13 (y = 4.0)

Member layout:
    Bottom chords : B0-B1 ... B5-B6        (6)
    Top chords    : T0-T1 ... T5-T6        (6)
    Verticals     : B0-T0 ... B6-T6        (7)
    Diagonals     : B(i+1)-T(i)            (6, Pratt direction)
    Counter-diag. : B(i)-T(i+1)            (6, X-bracing -> redundancy)
    Total                                  (31)

Supports:
    B0 (node 0): pinned  — fixed_dofs [0, 1]
    B6 (node 6): roller  — fixed_dofs [1] (free to slide horizontally)

Static check (truss): nodes n=14 -> 2n=28; members m=31, reactions r=3,
m + r = 34 > 28, so the truss is indeterminate to degree 6 (redundant).

Load:
    Distributed traffic load as point loads at the bottom chord nodes.
    Interior nodes -100 kN; end nodes -50 kN (DOF 1 = uy, downward).
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

# Section properties follow standard wide-flange members; c is half the
# nominal section depth (extreme-fibre distance) for the bending-stress term.

# Bottom chord — tension dominant (W360x122 equivalent, d ≈ 363 mm)
BOTTOM_CHORD_MAT = dataclasses.replace(
    STEEL_S355, name="S355 Bottom Chord", A=0.0155, I=3.65e-4, c=0.1815
)

# Top chord — compression dominant (W310x97 equivalent, d ≈ 308 mm)
TOP_CHORD_MAT = dataclasses.replace(
    STEEL_S355, name="S355 Top Chord", A=0.0123, I=2.22e-4, c=0.154
)

# Verticals — lighter section (W200x52 equivalent, d ≈ 206 mm)
VERTICAL_MAT = dataclasses.replace(
    STEEL_S275, name="S275 Vertical", A=0.0066, I=5.27e-5, c=0.103
)

# Diagonals — primary load path (W250x89 equivalent, d ≈ 260 mm)
DIAGONAL_MAT = dataclasses.replace(
    STEEL_S355, name="S355 Diagonal", A=0.0114, I=1.42e-4, c=0.130
)


def build(n_counter: int = N_PANELS) -> FrameData:
    """
    Construct and return the truss bridge FrameData.

    Args:
        n_counter: Number of counter-diagonals (0..6) to include, applied to
                   the leftmost panels. 0 gives the statically determinate
                   single-diagonal Pratt truss (degree of static indeterminacy
                   = 0); each counter-diagonal adds one redundancy, so the
                   degree of static indeterminacy equals n_counter. The default
                   (6) is the fully X-braced redundant truss. This parameter is
                   used by the redundancy parametric study.

    Returns:
        FrameData with 14 nodes, (25 + n_counter) members, pinned/roller
        supports, and distributed point loads along the bottom chord.
    """
    name = ("Redundant truss bridge (6-panel, 30 m, X-braced)"
            if n_counter == N_PANELS
            else f"Truss bridge (6-panel, {n_counter} counter-diagonals)")
    return FrameData(
        name=name,
        nodes=_define_nodes(),
        members=_define_members(n_counter),
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

    # Pin-jointed truss: out-of-plane DOFs (2=uz, 3=rx, 4=ry) AND the in-plane
    # rotation (5=rz) are restrained at every node, since bars carry no moment
    # and provide no rotational stiffness. Each node then has 2 active DOFs
    # (ux, uy), the classical truss bookkeeping.
    PIN_DOFS = [2, 3, 4, 5]

    # Bottom chord: nodes 0-6
    for i in range(N_NODES_CHORD):
        x = i * PANEL_WIDTH
        if i == 0:
            fixed_dofs = [0, 1] + PIN_DOFS   # Pinned left support
        elif i == N_PANELS:
            fixed_dofs = [1] + PIN_DOFS       # Roller right support
        else:
            fixed_dofs = PIN_DOFS             # Free in-plane translations only
        nodes.append(Node(id=i, x=x, y=0.0, z=0.0, fixed_dofs=fixed_dofs))

    # Top chord: nodes 7-13
    for i in range(N_NODES_CHORD):
        x = i * PANEL_WIDTH
        node_id = N_NODES_CHORD + i  # 7-13
        nodes.append(Node(id=node_id, x=x, y=TRUSS_HEIGHT, z=0.0, fixed_dofs=PIN_DOFS))

    return nodes


def _define_members(n_counter: int = N_PANELS) -> list[Member]:
    """
    Define the truss members: chords, verticals, diagonals, counter-diagonals.

    All members are TRUSS (axial-only) bars. n_counter (0..6) sets how many
    counter-diagonals are added.

    Member ID layout:
        0-5:   Bottom chords  (B0-B1 through B5-B6)
        6-11:  Top chords     (T0-T1 through T5-T6)
        12-18: Verticals      (B0-T0 through B6-T6)
        19-24: Diagonals      (B(i+1)-T(i), Pratt direction)
        25-30: Counter-diags  (B(i)-T(i+1), X-bracing for redundancy)

    Returns:
        List of 31 Member objects.
    """
    members = []
    mid = 0  # member id counter

    # Bottom chords
    for i in range(N_PANELS):
        members.append(Member(id=mid, node_start=i, node_end=i+1,
                              material=BOTTOM_CHORD_MAT, kind="truss"))
        mid += 1

    # Top chords (node offset = N_NODES_CHORD = 7)
    for i in range(N_PANELS):
        members.append(Member(
            id=mid,
            node_start=N_NODES_CHORD + i,
            node_end=N_NODES_CHORD + i + 1,
            material=TOP_CHORD_MAT,
            kind="truss",
        ))
        mid += 1

    # Verticals (connecting bottom node i to top node i)
    for i in range(N_NODES_CHORD):
        members.append(Member(
            id=mid,
            node_start=i,
            node_end=N_NODES_CHORD + i,
            material=VERTICAL_MAT,
            kind="truss",
        ))
        mid += 1

    # Diagonals — Pratt direction: bottom node (i+1) to top node (i)
    for i in range(N_PANELS):
        members.append(Member(
            id=mid,
            node_start=i + 1,
            node_end=N_NODES_CHORD + i,
            material=DIAGONAL_MAT,
            kind="truss",
        ))
        mid += 1

    # Counter-diagonals — opposite direction: bottom node (i) to top node (i+1).
    # These X-brace a panel and add one redundancy each. Only the first
    # n_counter panels receive a counter-diagonal.
    for i in range(n_counter):
        members.append(Member(
            id=mid,
            node_start=i,
            node_end=N_NODES_CHORD + i + 1,
            material=DIAGONAL_MAT,
            kind="truss",
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