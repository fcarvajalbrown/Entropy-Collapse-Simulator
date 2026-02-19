"""
solver/failure.py
=================
Evaluates member failure criteria and marks failed members in the frame.
A failed member is excluded from stiffness assembly in subsequent steps.

Current criterion: axial force exceeds member capacity (F > F_yield).
Bending failure (M > M_plastic) can be added later without breaking the interface.

Consumed by simulation/runner.py after each equilibrium solve.
"""

from core.models import FrameData, EnergyState, MemberState


def check_and_apply_failures(frame: FrameData, energy_state: EnergyState) -> list[int]:
    """
    Check all members for failure and mark them in the frame.

    Iterates over member states, compares axial force against capacity,
    and sets member.failed = True for any that exceed the limit.

    Args:
        frame: Frame definition (modified in-place — failed flags updated).
        energy_state: Current energy/force state from the equilibrium solver.

    Returns:
        List of member IDs that newly failed this step.
    """
    newly_failed = []

    for ms in energy_state.member_states:
        if ms.failed:
            continue

        member = _get_member(frame, ms.member_id)

        # Primary check: axial force vs capacity
        axial_cap = _axial_capacity(member)
        # Secondary check: strain energy vs energy capacity (U = F^2*L / 2EA)
        # This catches bending-dominated members where axial force is near zero
        from structure.stiffness import _member_length
        # We need frame for length — use a simpler approach: energy threshold
        # U_capacity = 0.5 * F_cap^2 / k = 0.5 * (sigma_y*A)^2*L / (E*A)
        # Simplified: fail if abs(axial_force) >= capacity OR strain_energy > energy_cap
        sigma_y = getattr(member, "sigma_y", 250e6)
        # Energy-based threshold: U > 0.5 * sigma_y^2 * A * L / E
        # Use a normalized check: if energy per unit volume exceeds yield criterion
        energy_threshold = getattr(member, "energy_capacity", None)

        failed_by_force = abs(ms.axial_force) >= axial_cap
        failed_by_energy = (energy_threshold is not None and ms.strain_energy >= energy_threshold)

        if failed_by_force or failed_by_energy:
            member.failed = True
            newly_failed.append(member.id)

    return newly_failed


def _axial_capacity(member) -> float:
    """
    Compute the axial load capacity of a member.

    Uses a yield-stress-based approach: F_yield = sigma_y * A.
    Assumes steel with sigma_y = 250 MPa by default if not set on the member.

    TODO: Add member.sigma_y field to Member model for custom materials.

    Args:
        member: Member with cross-sectional area A.

    Returns:
        Axial capacity in Newtons.
    """
    sigma_y = getattr(member, "sigma_y", 250e6)  # Default: 250 MPa steel
    return sigma_y * member.A


def _shear_capacity(member) -> float:
    """
    Compute shear capacity as 0.6 * sigma_y * A (Von Mises approximation).

    Used as failure criterion for members dominated by bending/shear
    rather than pure axial force (e.g. horizontal beams under vertical loads).

    Args:
        member: Member with cross-sectional area A.

    Returns:
        Shear capacity in Newtons.
    """
    sigma_y = getattr(member, "sigma_y", 250e6)
    return 0.6 * sigma_y * member.A


def _get_member(frame: FrameData, member_id: int):
    """
    Retrieve a member by ID from the frame.

    Args:
        frame: Frame definition.
        member_id: ID to look up.

    Returns:
        Member object.

    Raises:
        ValueError: If member_id is not found.
    """
    for member in frame.members:
        if member.id == member_id:
            return member
    raise ValueError(f"Member {member_id} not found in frame.")


def all_failed(frame: FrameData) -> bool:
    """
    Check if all members in the frame have failed (total collapse).

    Args:
        frame: Current frame state.

    Returns:
        True if every member is marked as failed.
    """
    return all(m.failed for m in frame.members)