"""
solver/failure.py
=================
Evaluates member failure criteria and marks failed members in the frame.

Failure criterion: maximum combined stress exceeds yield stress.

    sigma_max = abs(N/A) + abs(M_max * c / I)

where:
    N     = axial force (f_local[0])
    M_max = max bending moment at either end (max of f_local[5], f_local[11])
    c     = I/S ≈ sqrt(I/A) — distance from neutral axis to extreme fiber
    A, I  = from member material

This correctly handles:
    - Axial-dominated members (columns, diagonals)
    - Bending-dominated members (beams under transverse load)
    - Combined axial + bending (general case)

Consumed by simulation/runner.py after each equilibrium solve.
"""

import numpy as np
from core.models import FrameData, EnergyState
from structure.stiffness import _local_stiffness, _transformation_matrix
from structure.stiffness import assemble_global_stiffness, apply_boundary_conditions
from solver.equilibrium import _build_load_vector, apply_boundary_conditions_to_force, _solve_system


def check_and_apply_failures(frame: FrameData, energy_state: EnergyState) -> list[int]:
    """
    Check all members for failure and mark them in the frame.

    Re-solves for the displacement vector to extract full internal force
    vectors, then checks combined axial + bending stress against sigma_y.

    Args:
        frame: Frame definition (modified in-place — failed flags updated).
        energy_state: Current energy state (used to skip already-failed members).

    Returns:
        List of member IDs that newly failed this step.
    """
    # Re-solve to get displacement vector
    K = assemble_global_stiffness(frame)
    K = apply_boundary_conditions(K, frame)
    F = _build_load_vector(frame)
    F = apply_boundary_conditions_to_force(F, frame)
    u = _solve_system(K, F)

    newly_failed = []

    for ms in energy_state.member_states:
        if ms.failed:
            continue

        member = _get_member(frame, ms.member_id)
        sigma_max = _combined_stress(member, u, frame)

        if sigma_max >= member.sigma_y:
            member.failed = True
            newly_failed.append(member.id)

    return newly_failed


def _combined_stress(member, u: np.ndarray, frame: FrameData) -> float:
    """
    Compute maximum combined axial + bending stress in a member.

    Extracts the full internal force vector in local coordinates and
    computes:
        sigma_axial  = abs(N) / A
        sigma_bending = abs(M_max) * c / I
        sigma_max    = sigma_axial + sigma_bending

    where c = sqrt(I/A) approximates the distance from neutral axis
    to extreme fiber for a compact section.

    Args:
        member: Member with material properties.
        u: Global displacement vector.
        frame: Frame geometry.

    Returns:
        Maximum combined stress in Pa.
    """
    i = member.node_start * 6
    j = member.node_end   * 6
    u_global = np.concatenate([u[i:i+6], u[j:j+6]])

    T       = _transformation_matrix(member, frame)
    k_local = _local_stiffness(member, frame)
    u_local = T @ u_global
    f_local = k_local @ u_local

    A, I = member.A, member.I
    c = np.sqrt(I / A)  # Approximate extreme fiber distance

    # Axial stress from local DOF 0
    sigma_axial = abs(f_local[0]) / A

    # Bending moment at start (DOF 5) and end (DOF 11)
    M_max = max(abs(f_local[5]), abs(f_local[11]))
    sigma_bending = M_max * c / I

    return sigma_axial + sigma_bending


def _get_member(frame: FrameData, member_id: int):
    """
    Retrieve a member by ID from the frame.

    Raises:
        ValueError: If member_id is not found.
    """
    for member in frame.members:
        if member.id == member_id:
            return member
    raise ValueError(f"Member {member_id} not found in frame.")


def all_failed(frame: FrameData) -> bool:
    """
    Check if all members in the frame have failed.

    Args:
        frame: Current frame state.

    Returns:
        True if every member is marked as failed.
    """
    return all(m.failed for m in frame.members)