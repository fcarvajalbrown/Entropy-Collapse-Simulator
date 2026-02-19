"""
tests/test_phase5_entropy.py
=============================
Phase 5: Verify entropy computation is mathematically correct.

Checks:
  - S ≈ 0 when all energy is in one member (maximum localization)
  - S ≈ ln(n) when energy is perfectly uniform (maximum entropy)
  - delta_entropy has correct sign between steps
  - normalized_entropy returns value in [0, 1]
  - Gini = 0 for uniform, Gini → 1 for concentrated distribution
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import math
import numpy as np
from core.models import EnergyState, MemberState
from entropy.metrics import compute, max_entropy, normalized_entropy
from entropy.localization import localization_index, most_localized_members


def _make_energy_state(energies: list[float], step: int = 0) -> EnergyState:
    """Helper: build an EnergyState from a list of per-member energies."""
    member_states = [
        MemberState(member_id=i, strain_energy=e, axial_force=0.0, deformation=0.0)
        for i, e in enumerate(energies)
    ]
    return EnergyState(step=step, total_energy=sum(energies), member_states=member_states)


def test_entropy_zero_when_all_in_one_member():
    """S ≈ 0 when all energy is concentrated in one member."""
    es = _make_energy_state([1000.0, 0.0, 0.0, 0.0])
    record = compute(es, previous_entropy=0.0)
    assert record.entropy < 0.01, f"Expected S≈0, got {record.entropy}"
    print(f"  PASS: S = {record.entropy:.6f} (concentrated energy)")


def test_entropy_max_when_uniform():
    """S ≈ ln(n) when energy is perfectly uniform across n members."""
    n = 4
    es = _make_energy_state([250.0] * n)
    record = compute(es, previous_entropy=0.0)
    expected = math.log(n)
    assert abs(record.entropy - expected) < 1e-6, \
        f"Expected S={expected:.4f}, got {record.entropy:.4f}"
    print(f"  PASS: S = {record.entropy:.6f} ≈ ln({n}) = {expected:.6f}")


def test_delta_entropy_negative_when_energy_concentrates():
    """dS is negative when energy becomes more concentrated over steps."""
    es_uniform = _make_energy_state([250.0, 250.0, 250.0, 250.0], step=0)
    record_uniform = compute(es_uniform, previous_entropy=0.0)

    es_concentrated = _make_energy_state([900.0, 50.0, 25.0, 25.0], step=1)
    record_concentrated = compute(es_concentrated, previous_entropy=record_uniform.entropy)

    assert record_concentrated.delta_entropy < 0, \
        f"Expected negative dS, got {record_concentrated.delta_entropy}"
    print(f"  PASS: dS = {record_concentrated.delta_entropy:.4f} (negative as expected)")


def test_normalized_entropy_in_range():
    """normalized_entropy returns value in [0, 1]."""
    es = _make_energy_state([400.0, 300.0, 200.0, 100.0])
    record = compute(es)
    norm = normalized_entropy(record, n_active_members=4)
    assert 0.0 <= norm <= 1.0, f"normalized_entropy out of range: {norm}"
    print(f"  PASS: Normalized entropy = {norm:.4f}")


def test_gini_zero_for_uniform():
    """Gini index ≈ 0 for uniform energy distribution."""
    record = compute(_make_energy_state([250.0, 250.0, 250.0, 250.0]))
    gini = localization_index(record)
    assert gini < 0.05, f"Expected Gini≈0, got {gini:.4f}"
    print(f"  PASS: Gini = {gini:.4f} (uniform)")


def test_gini_high_for_concentrated():
    """Gini index is high for concentrated energy."""
    record = compute(_make_energy_state([990.0, 5.0, 3.0, 2.0]))
    gini = localization_index(record)
    assert gini > 0.5, f"Expected high Gini, got {gini:.4f}"
    print(f"  PASS: Gini = {gini:.4f} (concentrated)")


def test_most_localized_members():
    """most_localized_members returns top N by p_i descending."""
    record = compute(_make_energy_state([600.0, 250.0, 100.0, 50.0]))
    top = most_localized_members(record, top_n=2)
    assert top[0][0] == 0, f"Expected member 0 first, got {top[0][0]}"
    assert top[0][1] > top[1][1], "Top member should have higher p_i"
    print(f"  PASS: most_localized = {top}")


if __name__ == "__main__":
    print("=== Phase 5: Entropy Metrics ===")
    test_entropy_zero_when_all_in_one_member()
    test_entropy_max_when_uniform()
    test_delta_entropy_negative_when_energy_concentrates()
    test_normalized_entropy_in_range()
    test_gini_zero_for_uniform()
    test_gini_high_for_concentrated()
    test_most_localized_members()
    print("All Phase 5 tests passed.\n")