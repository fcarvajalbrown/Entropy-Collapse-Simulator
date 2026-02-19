"""
solver/equilibrium.py
=====================
Solves the static equilibrium equation Ku = F and computes
per-member strain energies using the full internal force vector.

Strain energy is computed as:
    U_i = 0.5 * u_local^T * k_local * u_local

This captures both axial and bending contributions correctly.

MemberState.axial_force stores the true axial force (f_local[0]),
not the full force magnitude. This keeps the failure criterion
(axial_force >= sigma_y * A) physically meaningful.
"""

import numpy as np
from core.models import FrameData, EnergyState, MemberState
from structure.stiffness import (
    assemble_global_stiffness,
    apply_boundary_conditions,
    _local_stiffness,
    _transformation_matrix,
    _get_node,
    _member_length
)


def solve(frame: FrameData, step: int, load_factor: float = 1.0) -> EnergyState:
    """
    Solve Ku = F for the current frame state and return an EnergyState.

    Args:
        frame: Current frame definition (members may be partially failed).
        step: Current simulation step index.
        load_factor: Scalar multiplier applied to all loads before solving.
                     Used by the runner for incremental loading (default 1.0,
                     which preserves backward-compatible behaviour).

    Returns:
        EnergyState with per-member strain energies and forces.
    """
    K = assemble_global_stiffness(frame)
    K = apply_boundary_conditions(K, frame)
    F = _build_load_vector(frame, load_factor=load_factor)
    F = apply_boundary_conditions_to_force(F, frame)
    u = _solve_system(K, F)

    member_states = [
        _compute_member_state(member, u, frame)
        for member in frame.members
    ]

    total_energy = sum(ms.strain_energy for ms in member_states)

    return EnergyState(step=step, total_energy=total_energy, member_states=member_states)


def _build_load_vector(frame: FrameData, load_factor: float = 1.0) -> np.ndarray:
    """
    Assemble the global force vector F from all applied loads.

    Args:
        frame: Frame definition containing load list.
        load_factor: Scalar multiplier applied to all load magnitudes.
                     Default 1.0 preserves backward-compatible behaviour.

    Returns:
        F (np.ndarray): Force vector of shape (n_dof,).
    """
    n_dof = len(frame.nodes) * 6
    F = np.zeros(n_dof)
    for load in frame.loads:
        global_dof = load.node_id * 6 + load.dof
        F[global_dof] += load.magnitude * load_factor
    return F


def _solve_system(K: np.ndarray, F: np.ndarray) -> np.ndarray:
    """
    Solve the linear system Ku = F.
    Falls back to least-squares if K is singular.

    Args:
        K: Global stiffness matrix.
        F: Global force vector.

    Returns:
        u (np.ndarray): Displacement vector of shape (n_dof,).
    """
    try:
        u = np.linalg.solve(K, F)
    except np.linalg.LinAlgError:
        u = np.linalg.lstsq(K, F, rcond=None)[0]
    return u


def apply_boundary_conditions_to_force(F: np.ndarray, frame: FrameData) -> np.ndarray:
    """
    Zero out force vector entries at constrained DOFs.

    When apply_boundary_conditions sets K[dof,dof]=1 and K[dof,:]=0,
    the system solves as u[dof] = F[dof]. If F[dof] is non-zero
    (e.g. a load applied directly at a support), the result is a
    spurious displacement equal to the load magnitude in meters.

    This function ensures F[dof]=0 at all fixed DOFs so the solver
    returns u[dof]=0 as expected for a constrained degree of freedom.

    Args:
        F: Global force vector (modified in-place).
        frame: Frame with node boundary conditions.

    Returns:
        F (np.ndarray): Modified force vector.
    """
    for node in frame.nodes:
        for dof in node.fixed_dofs:
            F[node.id * 6 + dof] = 0.0
    return F


def _compute_member_state(member, u: np.ndarray, frame: FrameData) -> MemberState:
    """
    Compute strain energy and axial force for a single member.

    Procedure:
        1. Extract 12 global DOFs for the member's two nodes
        2. Transform to local coordinates: u_local = T @ u_global
        3. Compute internal forces: f_local = k_local @ u_local
        4. Strain energy: U = 0.5 * u_local^T @ f_local
        5. Axial force: f_local[0] (local x-direction)

    Axial force is stored in MemberState.axial_force. The failure criterion
    (sigma_y * A) is compared against abs(axial_force) — compression members
    have negative axial force, tension members positive.

    Strain energy is always non-negative; any negative result from numerical
    noise is clamped to zero.

    Args:
        member: Member to evaluate.
        u: Global displacement vector.
        frame: Frame geometry.

    Returns:
        MemberState with strain energy, axial force, deformation, failed flag.
    """
    if member.failed:
        return MemberState(
            member_id=member.id,
            strain_energy=0.0,
            axial_force=0.0,
            deformation=0.0,
            failed=True
        )

    # Extract 12 global DOFs (6 per node)
    i = member.node_start * 6
    j = member.node_end   * 6
    u_global = np.concatenate([u[i:i+6], u[j:j+6]])

    # Transform to local coordinates
    T       = _transformation_matrix(member, frame)
    k_local = _local_stiffness(member, frame)
    u_local = T @ u_global

    # Internal force vector in local coordinates
    f_local = k_local @ u_local

    # Strain energy from full deformation (axial + bending)
    strain_energy = float(0.5 * u_local @ f_local)

    # Axial force: local DOF 0 (positive = tension, negative = compression)
    axial_force = float(f_local[0])

    # Axial deformation along member axis
    n_start = _get_node(frame, member.node_start)
    n_end   = _get_node(frame, member.node_end)
    L       = _member_length(member, frame)
    axis    = np.array([
        n_end.x - n_start.x,
        n_end.y - n_start.y,
        n_end.z - n_start.z
    ]) / L
    deformation = float(np.dot(u_global[3:6] - u_global[0:3], axis))

    return MemberState(
        member_id=member.id,
        strain_energy=max(strain_energy, 0.0),
        axial_force=axial_force,
        deformation=deformation,
        failed=False
    )