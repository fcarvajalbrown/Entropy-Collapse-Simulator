"""
core/models.py
==============
Shared data contracts for the Entropy-Based Progressive Collapse Simulator.

Every module in this project communicates exclusively through these dataclasses.
No module should import internal classes from another module — only from here.
This ensures full decoupling and makes the system easily expandable.

Author: Felipe
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional


# ---------------------------------------------------------------------------
# Material
# ---------------------------------------------------------------------------

@dataclass
class Material:
    """
    Material and cross-section properties for a structural member.

    Centralizes all physical properties so members reference a single
    material object rather than carrying loose floats. This ensures
    scientific rigor — changing a material property updates all members
    that reference it.

    Attributes:
        name (str): Human-readable material name (e.g. "S275 Steel").
        E (float): Young's modulus in Pa.
        A (float): Cross-sectional area in m².
        I (float): Second moment of area in m⁴ (strong axis bending).
        sigma_y (float): Yield stress in Pa. Used for failure criterion.
        rho (float): Density in kg/m³. Reserved for dynamic analysis.
    """
    name: str
    E: float
    A: float
    I: float
    sigma_y: float
    rho: float = 7850.0  # Default: structural steel density


# ---------------------------------------------------------------------------
# Predefined common materials for convenience
# ---------------------------------------------------------------------------

STEEL_S275 = Material(
    name="S275 Steel",
    E=200e9,
    A=0.01,
    I=1e-4,
    sigma_y=275e6,
    rho=7850.0
)

STEEL_S355 = Material(
    name="S355 Steel",
    E=200e9,
    A=0.01,
    I=1e-4,
    sigma_y=355e6,
    rho=7850.0
)


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

@dataclass
class Node:
    """
    Represents a structural joint in the frame.

    Attributes:
        id (int): Unique identifier for the node.
        x (float): X-coordinate in meters.
        y (float): Y-coordinate in meters.
        z (float): Z-coordinate in meters. Use 0.0 for 2D frames.
        fixed_dofs (List[int]): Degrees of freedom that are restrained.
                                Convention: [0=ux, 1=uy, 2=uz, 3=rx, 4=ry, 5=rz].
                                Example: [0, 1, 2] = pinned in 3D.
    """
    id: int
    x: float
    y: float
    z: float = 0.0
    fixed_dofs: List[int] = field(default_factory=list)


@dataclass
class Member:
    """
    Represents a structural member (beam or column) connecting two nodes.

    Material and section properties are held in a Material object,
    not as loose floats. This ensures all physical properties are
    explicitly defined and scientifically traceable.

    Attributes:
        id (int): Unique identifier for the member.
        node_start (int): ID of the start node.
        node_end (int): ID of the end node.
        material (Material): Material and cross-section properties.
        failed (bool): Whether this member has failed. Set by failure module.
    """
    id: int
    node_start: int
    node_end: int
    material: Material
    failed: bool = False

    # Convenience properties so solver code stays readable
    @property
    def E(self) -> float:
        """Young's modulus from material."""
        return self.material.E

    @property
    def A(self) -> float:
        """Cross-sectional area from material."""
        return self.material.A

    @property
    def I(self) -> float:
        """Second moment of area from material."""
        return self.material.I

    @property
    def sigma_y(self) -> float:
        """Yield stress from material."""
        return self.material.sigma_y


@dataclass
class FrameData:
    """
    Complete definition of a structural frame, as exported by every frame module.

    This is the single input contract for the stiffness assembler and solver.
    Every file in structure/frames/ must return an instance of this class
    from its build() function.

    Attributes:
        name (str): Human-readable name for the frame (used in plots/reports).
        nodes (List[Node]): All nodes in the frame.
        members (List[Member]): All members in the frame.
        loads (List[Load]): Applied external loads.
    """
    name: str
    nodes: List[Node]
    members: List[Member]
    loads: List["Load"]


@dataclass
class Load:
    """
    Represents an external force or moment applied at a node.

    Attributes:
        node_id (int): ID of the node where the load is applied.
        dof (int): Degree of freedom the load acts on.
                   Convention: [0=ux, 1=uy, 2=uz, 3=rx, 4=ry, 5=rz].
        magnitude (float): Load magnitude in Newtons (forces) or N·m (moments).
    """
    node_id: int
    dof: int
    magnitude: float


# ---------------------------------------------------------------------------
# Solver State
# ---------------------------------------------------------------------------

@dataclass
class MemberState:
    """
    Snapshot of a single member's physical state at one time step.

    Produced by the solver and consumed by the entropy module.

    Attributes:
        member_id (int): Corresponds to Member.id.
        strain_energy (float): Elastic strain energy stored in the member (Joules).
        axial_force (float): Total internal force magnitude in Newtons.
                             Includes axial, shear, and moment resultants.
        deformation (float): Axial deformation in meters.
        failed (bool): Whether this member has failed at this time step.
    """
    member_id: int
    strain_energy: float
    axial_force: float
    deformation: float
    failed: bool = False


@dataclass
class EnergyState:
    """
    Full energy snapshot of the structure at one time step.

    Produced by the solver and passed to the entropy module each step.

    Attributes:
        step (int): Simulation time step index.
        total_energy (float): Sum of all member strain energies (Joules).
        member_states (List[MemberState]): Per-member energy and force data.
    """
    step: int
    total_energy: float
    member_states: List[MemberState]


# ---------------------------------------------------------------------------
# Entropy Metrics
# ---------------------------------------------------------------------------

@dataclass
class EntropyRecord:
    """
    Entropy metrics computed for a single time step.

    Produced by entropy/metrics.py and consumed by entropy/localization.py
    and visualization/entropy_plot.py.

    Attributes:
        step (int): Simulation time step index.
        entropy (float): Structural entropy S = -sum(p_i * ln(p_i)).
                         High entropy = distributed energy (safe).
                         Low entropy = localized energy (dangerous).
        delta_entropy (float): Change in entropy from previous step (dS/dt proxy).
                               A large negative spike indicates imminent collapse.
        energy_distribution (List[Tuple[int, float]]): List of (member_id, p_i)
                             pairs showing normalized energy per member.
    """
    step: int
    entropy: float
    delta_entropy: float
    energy_distribution: List[Tuple[int, float]]


# ---------------------------------------------------------------------------
# Simulation Output
# ---------------------------------------------------------------------------

@dataclass
class SimulationResult:
    """
    Complete output of a simulation run, passed to the visualization layer.

    Attributes:
        frame_name (str): Name of the frame that was simulated.
        energy_history (List[EnergyState]): Energy snapshots for every step.
        entropy_history (List[EntropyRecord]): Entropy metrics for every step.
        collapse_detected (bool): Whether collapse was detected during the run.
        collapse_step (Optional[int]): Step index at which collapse was detected,
                                       or None if no collapse occurred.
        failed_sequence (List[int]): Ordered list of member IDs that failed,
                                     in the order they were removed.
    """
    frame_name: str
    energy_history: List[EnergyState]
    entropy_history: List[EntropyRecord]
    collapse_detected: bool
    collapse_step: Optional[int]
    failed_sequence: List[int]