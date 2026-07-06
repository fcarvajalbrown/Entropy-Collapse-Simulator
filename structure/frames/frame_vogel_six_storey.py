"""
structure/frames/frame_vogel_six_storey.py
===========================================
Vogel (1985) six-storey, two-bay planar steel calibration frame.

This is a published, widely recognised benchmark building frame -- the
"European calibration frame" introduced by Vogel to calibrate second-order
inelastic frame analyses -- included in the Ziemian & Ziemian (2021) steel
benchmark collection cited in THEORY.md. It replaces the toy-scale scenarios
with a realistic, provenance-backed steel moment frame for the robustness
study.

Provenance and honest scope note
--------------------------------
Geometry, member sections and the design load pattern are taken verbatim from
Vogel's frame. Vogel's own results are second-order *inelastic* collapse loads;
this tool is first-order *linear-elastic* and does NOT reproduce them and does
not claim to. The value here is a recognisable, real steel-building geometry on
which the calibration-free Entropy Robustness Index R_S is exercised, not a
match to Vogel's nonlinear numbers. Consistent with the repo invariants, the
model stays planar, linear-elastic and quasi-static.

Geometry -- 2 bays x 6.0 m, 6 storeys x 3.75 m (global XY plane, z = 0):

    L6 (roof):  18---19---20        y = 22.50 m
                |    |    |
    L5:         15---16---17        y = 18.75 m
                |    |    |
    L4:         12---13---14        y = 15.00 m
                |    |    |
    L3:          9---10---11        y = 11.25 m
                |    |    |
    L2:          6----7----8        y =  7.50 m
                |    |    |
    L1:          3----4----5        y =  3.75 m
                |    |    |
    L0 (bases):  0----1----2        y =  0.00 m   (fully fixed)
               x=0   x=6  x=12

Node id = level * 3 + column, column in {0 (left), 1 (mid), 2 (right)}.

Members (columns first, then beams):
    Columns (18): ids  0-17, 3 column lines x 6 storeys
    Beams   (12): ids 18-29, 6 elevated levels x 2 bays

Sections (Vogel 1985; strong-axis I-section properties, c = h/2):
    Beams by level:  L1 IPE400, L2 IPE360, L3 IPE330, L4 IPE300,
                     L5 IPE300, L6 IPE240.
    Exterior columns (lines 0 and 2): storeys 1-4 HEB220, storeys 5-6 HEB160.
    Interior column  (line 1):        storeys 1-3 HEB260, storey 4 HEB240,
                                      storeys 5-6 HEB200.
    Section A and I verified against Eurocode-3 / EN 10025 section tables.

Supports and connections:
    Base nodes 0,1,2 fully fixed (encastre). All other nodes constrain only the
    out-of-plane DOFs [2,3,4]; in-plane ux, uy, rz are free. Rigid (moment)
    beam-to-column connections throughout -- every member is a frame element.

Loads (Vogel design combination):
    Gravity: uniformly distributed beam loads, 31.7 kN/m on the roof beams and
    49.1 kN/m on every floor beam. The solver takes nodal loads only, so each
    beam UDL is represented by its work-equivalent tributary vertical joint
    loads (w*L/2 at each beam end): exterior joints receive w*3, interior
    joints w*6 (m). This is the same nodal-lumping idealisation used by
    frame_building_2d and keeps every beam/column a single element (clean
    member-removal semantics for R_S); beam span bending under gravity is not
    represented.
    Lateral: Vogel's notional horizontal loads at the windward (left) column
    line, F1 = 10.23 kN at the roof and F2 = 20.44 kN at each of the five lower
    floors, acting in +X.

Material: European structural steel S235, E = 205 GPa, fy = 235 MPa.
"""

import dataclasses
from core.models import FrameData, Node, Member, Load, Material


# ---------------------------------------------------------------------------
# Geometry constants
# ---------------------------------------------------------------------------

BAY_WIDTH    = 6.0    # meters between column lines
STORY_HEIGHT = 3.75   # meters between floors
N_COLS       = 3      # column lines (2 bays)
N_STORIES    = 6      # elevated floors
N_LEVELS     = N_STORIES + 1  # 7 node levels (ground + 6 floors)

# Out-of-plane DOFs constrained at every node for planar analysis.
PLANAR_DOFS = [2, 3, 4]

# ---------------------------------------------------------------------------
# Section / material definitions (European S235, explicit extreme-fibre c = h/2)
# Section area A (m^2) and second moment of area I (m^4) verified against
# Eurocode-3 (IPE) and EN 10025 (HEB) section property tables.
# ---------------------------------------------------------------------------

STEEL_S235 = Material(name="S235 Steel", E=205e9, A=0.0, I=0.0,
                      sigma_y=235e6, rho=7850.0, c=0.0)


def _section(name: str, A: float, I: float, c: float) -> Material:
    """Return an S235 material carrying one rolled section's A, I and c."""
    return dataclasses.replace(STEEL_S235, name=name, A=A, I=I, c=c)


# Beams -- IPE (c = h/2)
IPE240 = _section("IPE 240", 3.912e-3, 3.892e-5, 0.120)
IPE300 = _section("IPE 300", 5.381e-3, 8.356e-5, 0.150)
IPE330 = _section("IPE 330", 6.261e-3, 1.177e-4, 0.165)
IPE360 = _section("IPE 360", 7.273e-3, 1.627e-4, 0.180)
IPE400 = _section("IPE 400", 8.446e-3, 2.313e-4, 0.200)

# Columns -- HEB (c = h/2)
HEB160 = _section("HEB 160", 5.425e-3, 2.492e-5, 0.080)
HEB200 = _section("HEB 200", 7.808e-3, 5.696e-5, 0.100)
HEB220 = _section("HEB 220", 9.104e-3, 8.091e-5, 0.110)
HEB240 = _section("HEB 240", 1.060e-2, 1.126e-4, 0.120)
HEB260 = _section("HEB 260", 1.180e-2, 1.492e-4, 0.130)

# Beam section by elevated level (index 1..6; index 0 unused / ground).
BEAM_BY_LEVEL = {1: IPE400, 2: IPE360, 3: IPE330, 4: IPE300, 5: IPE300, 6: IPE240}

# Distributed gravity load on the beams (N/m, downward magnitude).
ROOF_UDL  = 31.7e3   # roof beams
FLOOR_UDL = 49.1e3   # all lower floor beams

# Vogel notional horizontal loads at the windward (left) line (N, +X).
H_ROOF  = 10.23e3    # F1, roof level
H_FLOOR = 20.44e3    # F2, each of the five lower floors


def _column_section(col: int, storey: int) -> Material:
    """
    Section for the column in ``col`` (0 left, 1 mid, 2 right) at ``storey``
    (1 = ground storey .. 6 = top storey), per Vogel's assignment.
    """
    if col == 1:  # interior line
        if storey <= 3:
            return HEB260
        if storey == 4:
            return HEB240
        return HEB200          # storeys 5-6
    # exterior lines (0 and 2), symmetric
    if storey <= 4:
        return HEB220
    return HEB160              # storeys 5-6


def build(base: str = "fixed") -> FrameData:
    """
    Construct the Vogel (1985) six-storey two-bay steel calibration frame.

    Args:
        base: Column-base condition, a genuine steel-design decision used by the
              design-variant robustness study (analysis/design_variants.py).
              "fixed" (default, the as-published Vogel frame) fully encastres the
              base nodes (moment bases). "pinned" restrains only translation at
              the bases (free in-plane rotation), a less redundant design whose
              lower alternate-load-path redundancy R_S is expected to detect.

    Returns:
        FrameData with 21 nodes, 30 members (18 columns, 12 beams), the selected
        base condition, tributary gravity joint loads and Vogel's notional
        horizontal loads at the windward column line.
    """
    if base not in ("fixed", "pinned"):
        raise ValueError(f"base must be 'fixed' or 'pinned', got {base!r}")
    # Base DOFs: fixed = encastre (all 6); pinned = translations + out-of-plane,
    # in-plane rotation (rz = 5) free.
    base_fixed = [0, 1, 2, 3, 4, 5] if base == "fixed" else [0, 1, 2, 3, 4]

    def node_id(level, col):
        return level * N_COLS + col

    # --- Nodes ---
    nodes = []
    for level in range(N_LEVELS):
        for col in range(N_COLS):
            fixed = list(base_fixed) if level == 0 else list(PLANAR_DOFS)
            nodes.append(Node(id=node_id(level, col),
                              x=col * BAY_WIDTH, y=level * STORY_HEIGHT, z=0.0,
                              fixed_dofs=fixed))

    # --- Members: columns then beams ---
    members = []
    mid = 0
    for level in range(N_LEVELS - 1):          # columns (storey = level + 1)
        storey = level + 1
        for col in range(N_COLS):
            members.append(Member(id=mid,
                                  node_start=node_id(level, col),
                                  node_end=node_id(level + 1, col),
                                  material=_column_section(col, storey)))
            mid += 1
    for level in range(1, N_LEVELS):           # beams
        beam_mat = BEAM_BY_LEVEL[level]
        for col in range(N_COLS - 1):
            members.append(Member(id=mid,
                                  node_start=node_id(level, col),
                                  node_end=node_id(level, col + 1),
                                  material=beam_mat))
            mid += 1

    # --- Loads ---
    loads = []
    for level in range(1, N_LEVELS):
        udl = ROOF_UDL if level == N_STORIES else FLOOR_UDL
        for col in range(N_COLS):
            # Tributary vertical joint load: exterior joints carry w*L/2 from
            # one beam, the interior joint carries w*L/2 from each of two beams.
            n_beams = 1 if col in (0, N_COLS - 1) else 2
            magnitude = -udl * (BAY_WIDTH / 2.0) * n_beams
            loads.append(Load(node_id=node_id(level, col), dof=1, magnitude=magnitude))
        # Vogel notional horizontal load at the windward (left) column line.
        h = H_ROOF if level == N_STORIES else H_FLOOR
        loads.append(Load(node_id=node_id(level, 0), dof=0, magnitude=h))

    name = "Vogel six-storey two-bay steel frame (1985)"
    if base != "fixed":
        name += f" [{base}-base variant]"
    return FrameData(name=name, nodes=nodes, members=members, loads=loads)
