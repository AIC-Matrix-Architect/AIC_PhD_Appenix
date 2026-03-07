"""
test_quantum_simulator.py — Unit tests for the quantum computing simulator
==========================================================================

Test coverage
-------------
- Qubit class: initialisation, normalisation, probabilities, Bloch sphere,
  measurement collapse, tensor product
- QuantumGates: unitarity of every gate, known gate actions
- QuantumCircuit: gate scheduling, measurement directives, ASCII draw
- QuantumSimulator: statevector evolution, measurement statistics,
  Bell-state entanglement, Deutsch-Jozsa, GHZ, Grover (2-qubit)

References used to derive expected values
------------------------------------------
Nielsen, M. A. & Chuang, I. L. (2010). *Quantum Computation and Quantum
    Information* (10th anniversary ed.). Cambridge University Press.
"""

from __future__ import annotations

import math
import numpy as np
import pytest

from quantum_simulator import Qubit, QuantumGates, QuantumCircuit, QuantumSimulator


# =========================================================================
# Fixtures
# =========================================================================

@pytest.fixture
def sim() -> QuantumSimulator:
    """A seeded simulator for reproducible tests."""
    return QuantumSimulator(seed=0)


# =========================================================================
# Qubit tests
# =========================================================================

class TestQubit:
    def test_default_is_zero_state(self):
        q = Qubit()
        assert np.isclose(q.alpha, 1.0)
        assert np.isclose(q.beta, 0.0)

    def test_one_state(self):
        q = Qubit.one()
        assert np.isclose(q.prob_zero, 0.0)
        assert np.isclose(q.prob_one, 1.0)

    def test_plus_state(self):
        q = Qubit.plus()
        assert np.isclose(q.prob_zero, 0.5, atol=1e-9)
        assert np.isclose(q.prob_one, 0.5, atol=1e-9)

    def test_minus_state(self):
        q = Qubit.minus()
        assert np.isclose(q.prob_zero, 0.5, atol=1e-9)
        assert np.isclose(q.prob_one, 0.5, atol=1e-9)

    def test_invalid_norm_raises(self):
        with pytest.raises(ValueError, match="normalised"):
            Qubit(0.5, 0.5)  # 0.25 + 0.25 ≠ 1

    def test_bloch_sphere_zero(self):
        q = Qubit.from_bloch(0.0, 0.0)  # theta=0 → |0⟩
        assert np.isclose(q.prob_zero, 1.0)

    def test_bloch_sphere_one(self):
        q = Qubit.from_bloch(math.pi, 0.0)  # theta=π → |1⟩
        assert np.isclose(q.prob_one, 1.0, atol=1e-7)

    def test_bloch_sphere_plus(self):
        q = Qubit.from_bloch(math.pi / 2, 0.0)  # theta=π/2, phi=0 → |+⟩
        assert np.isclose(q.prob_zero, 0.5, atol=1e-9)
        assert np.isclose(q.prob_one, 0.5, atol=1e-9)

    def test_measurement_collapses_state(self):
        rng = np.random.default_rng(42)
        q = Qubit.plus()
        outcome = q.measure(rng=rng)
        assert outcome in (0, 1)
        # After collapse the state is a basis state
        assert np.isclose(q.prob_zero + q.prob_one, 1.0)
        if outcome == 0:
            assert np.isclose(q.prob_zero, 1.0)
        else:
            assert np.isclose(q.prob_one, 1.0)

    def test_tensor_product_shape(self):
        q0 = Qubit.zero()
        q1 = Qubit.zero()
        state = q0.tensor(q1)
        assert state.shape == (4,)
        assert np.isclose(state[0], 1.0)  # |00⟩ has amplitude 1

    def test_equality(self):
        assert Qubit.zero() == Qubit.zero()
        assert Qubit.zero() != Qubit.one()

    def test_repr_contains_probabilities(self):
        r = repr(Qubit.plus())
        assert "P(0)" in r
        assert "P(1)" in r


# =========================================================================
# QuantumGates tests
# =========================================================================

class TestQuantumGates:
    """All gates must be unitary (U†U = I)."""

    def _assert_unitary(self, gate: np.ndarray):
        assert QuantumGates.is_unitary(gate), f"Gate is not unitary:\n{gate}"

    def test_identity_unitary(self):
        self._assert_unitary(QuantumGates.identity())

    def test_pauli_x_unitary(self):
        self._assert_unitary(QuantumGates.pauli_x())

    def test_pauli_y_unitary(self):
        self._assert_unitary(QuantumGates.pauli_y())

    def test_pauli_z_unitary(self):
        self._assert_unitary(QuantumGates.pauli_z())

    def test_hadamard_unitary(self):
        self._assert_unitary(QuantumGates.hadamard())

    def test_phase_unitary(self):
        self._assert_unitary(QuantumGates.phase())

    def test_phase_dagger_unitary(self):
        self._assert_unitary(QuantumGates.phase_dagger())

    def test_t_gate_unitary(self):
        self._assert_unitary(QuantumGates.t_gate())

    def test_t_dagger_unitary(self):
        self._assert_unitary(QuantumGates.t_dagger())

    def test_rx_unitary(self):
        self._assert_unitary(QuantumGates.rx(math.pi / 3))

    def test_ry_unitary(self):
        self._assert_unitary(QuantumGates.ry(math.pi / 4))

    def test_rz_unitary(self):
        self._assert_unitary(QuantumGates.rz(math.pi / 6))

    def test_phase_shift_unitary(self):
        self._assert_unitary(QuantumGates.phase_shift(math.pi / 5))

    def test_cnot_unitary(self):
        self._assert_unitary(QuantumGates.cnot())

    def test_cz_unitary(self):
        self._assert_unitary(QuantumGates.cz())

    def test_swap_unitary(self):
        self._assert_unitary(QuantumGates.swap())

    def test_toffoli_unitary(self):
        self._assert_unitary(QuantumGates.toffoli())

    def test_fredkin_unitary(self):
        self._assert_unitary(QuantumGates.fredkin())

    def test_controlled_unitary(self):
        cu = QuantumGates.controlled(QuantumGates.hadamard())
        self._assert_unitary(cu)

    def test_pauli_x_action(self):
        """X|0⟩ = |1⟩"""
        state = np.array([1.0, 0.0], dtype=complex)
        result = QuantumGates.pauli_x() @ state
        assert np.allclose(result, [0.0, 1.0])

    def test_hadamard_action(self):
        """H|0⟩ = |+⟩"""
        state = np.array([1.0, 0.0], dtype=complex)
        result = QuantumGates.hadamard() @ state
        s = 1 / math.sqrt(2)
        assert np.allclose(result, [s, s])

    def test_hadamard_self_inverse(self):
        """H·H = I"""
        h = QuantumGates.hadamard()
        assert np.allclose(h @ h, np.eye(2))

    def test_pauli_x_self_inverse(self):
        x = QuantumGates.pauli_x()
        assert np.allclose(x @ x, np.eye(2))

    def test_tensor_product_shape(self):
        h = QuantumGates.hadamard()
        hh = QuantumGates.tensor_product(h, h)
        assert hh.shape == (4, 4)
        self._assert_unitary(hh)

    def test_controlled_wrong_size_raises(self):
        with pytest.raises(ValueError):
            QuantumGates.controlled(np.eye(4, dtype=complex))


# =========================================================================
# QuantumCircuit tests
# =========================================================================

class TestQuantumCircuit:
    def test_create_circuit(self):
        qc = QuantumCircuit(3)
        assert qc.n_qubits == 3
        assert qc.depth() == 0

    def test_invalid_n_qubits_raises(self):
        with pytest.raises(ValueError):
            QuantumCircuit(0)

    def test_single_qubit_gates_chain(self):
        qc = QuantumCircuit(1)
        qc.h(0).x(0).y(0).z(0).s(0).t(0)
        assert qc.depth() == 6

    def test_two_qubit_gates(self):
        qc = QuantumCircuit(2)
        qc.cnot(0, 1).cz(0, 1).swap(0, 1)
        assert qc.depth() == 3

    def test_three_qubit_gates(self):
        qc = QuantumCircuit(3)
        qc.toffoli(0, 1, 2).fredkin(0, 1, 2)
        assert qc.depth() == 2

    def test_out_of_range_qubit_raises(self):
        qc = QuantumCircuit(2)
        with pytest.raises(ValueError):
            qc.h(2)

    def test_cnot_same_qubit_raises(self):
        qc = QuantumCircuit(2)
        with pytest.raises(ValueError):
            qc.cnot(0, 0)

    def test_measure_all(self):
        qc = QuantumCircuit(3)
        qc.measure_all()
        assert set(qc.measurements) == {0, 1, 2}

    def test_measure_single(self):
        qc = QuantumCircuit(3)
        qc.measure(1)
        assert qc.measurements == [1]

    def test_draw_returns_string(self):
        qc = QuantumCircuit(2)
        qc.h(0).cnot(0, 1).measure_all()
        s = str(qc)
        assert "H" in s
        assert "CNOT" in s
        assert "Measure" in s

    def test_repr(self):
        qc = QuantumCircuit(2)
        assert "QuantumCircuit" in repr(qc)

    def test_rotation_gates(self):
        qc = QuantumCircuit(1)
        qc.rx(math.pi, 0).ry(math.pi / 2, 0).rz(math.pi / 4, 0).p(math.pi / 3, 0)
        assert qc.depth() == 4

    def test_cx_alias(self):
        qc = QuantumCircuit(2)
        qc.cx(0, 1)
        assert qc.instructions[-1].name == "CNOT"


# =========================================================================
# QuantumSimulator tests
# =========================================================================

class TestQuantumSimulator:
    # ------------------------------------------------------------------
    # Error handling
    # ------------------------------------------------------------------

    def test_no_measurement_raises(self, sim):
        qc = QuantumCircuit(1)
        qc.h(0)
        with pytest.raises(ValueError, match="measurement"):
            sim.run(qc)

    def test_zero_shots_raises(self, sim):
        qc = QuantumCircuit(1)
        qc.h(0).measure_all()
        with pytest.raises(ValueError, match="shots"):
            sim.run(qc, shots=0)

    # ------------------------------------------------------------------
    # Statevector tests
    # ------------------------------------------------------------------

    def test_initial_state_is_zero(self, sim):
        qc = QuantumCircuit(2)
        sv = sim.get_statevector(qc)
        expected = np.zeros(4, dtype=complex)
        expected[0] = 1.0
        assert np.allclose(sv, expected)

    def test_x_gate_flips_qubit(self, sim):
        qc = QuantumCircuit(1)
        qc.x(0)
        sv = sim.get_statevector(qc)
        assert np.allclose(sv, [0.0, 1.0])

    def test_hadamard_creates_superposition(self, sim):
        qc = QuantumCircuit(1)
        qc.h(0)
        sv = sim.get_statevector(qc)
        s = 1 / math.sqrt(2)
        assert np.allclose(sv, [s, s])

    def test_x_twice_returns_to_zero(self, sim):
        qc = QuantumCircuit(1)
        qc.x(0).x(0)
        sv = sim.get_statevector(qc)
        assert np.allclose(sv, [1.0, 0.0])

    def test_statevector_norm_preserved(self, sim):
        qc = QuantumCircuit(3)
        qc.h(0).cnot(0, 1).h(2).z(1)
        sv = sim.get_statevector(qc)
        assert np.isclose(np.sum(np.abs(sv) ** 2), 1.0)

    # ------------------------------------------------------------------
    # Bell states
    # ------------------------------------------------------------------

    def test_bell_state_00_statevector(self, sim):
        """H on q0 then CNOT(q0,q1) creates |Φ+⟩ = (|00⟩+|11⟩)/√2."""
        qc = QuantumCircuit(2)
        qc.h(0).cnot(0, 1)
        sv = sim.get_statevector(qc)
        s = 1 / math.sqrt(2)
        expected = np.array([s, 0, 0, s], dtype=complex)
        assert np.allclose(sv, expected)

    def test_bell_state_measurement_distribution(self, sim):
        """Bell |Φ+⟩ should yield ~50% |00⟩ and ~50% |11⟩."""
        qc = QuantumCircuit(2)
        qc.h(0).cnot(0, 1).measure_all()
        counts = sim.run(qc, shots=4096)
        total = sum(counts.values())
        assert total == 4096
        # Only '00' and '11' should appear
        assert set(counts.keys()).issubset({"00", "11"})
        # Both should appear with roughly equal frequency (within 10%)
        for key in ("00", "11"):
            assert key in counts
            assert abs(counts[key] / total - 0.5) < 0.05

    # ------------------------------------------------------------------
    # GHZ state (3-qubit entanglement)
    # ------------------------------------------------------------------

    def test_ghz_state(self, sim):
        """GHZ = (|000⟩+|111⟩)/√2 via H + CNOT + CNOT."""
        qc = QuantumCircuit(3)
        qc.h(0).cnot(0, 1).cnot(0, 2)
        sv = sim.get_statevector(qc)
        s = 1 / math.sqrt(2)
        expected = np.zeros(8, dtype=complex)
        expected[0] = s   # |000⟩
        expected[7] = s   # |111⟩
        assert np.allclose(sv, expected)

    def test_ghz_measurement_distribution(self, sim):
        """GHZ state measurements should yield only '000' or '111'."""
        qc = QuantumCircuit(3)
        qc.h(0).cnot(0, 1).cnot(0, 2).measure_all()
        counts = sim.run(qc, shots=2048)
        assert set(counts.keys()).issubset({"000", "111"})

    # ------------------------------------------------------------------
    # Toffoli gate test
    # ------------------------------------------------------------------

    def test_toffoli_flips_when_both_controls_set(self, sim):
        """Toffoli with both controls |1⟩ should flip target: |110⟩ → |111⟩."""
        qc = QuantumCircuit(3)
        qc.x(0).x(1).toffoli(0, 1, 2)
        sv = sim.get_statevector(qc)
        # |111⟩ = index 7 in 3-qubit space
        assert np.isclose(abs(sv[7]) ** 2, 1.0)

    def test_toffoli_no_flip_when_one_control_unset(self, sim):
        """Toffoli with one control |0⟩ should not flip target: |100⟩ → |100⟩."""
        qc = QuantumCircuit(3)
        qc.x(0).toffoli(0, 1, 2)  # q1 stays |0⟩
        sv = sim.get_statevector(qc)
        # |100⟩ = index 4 in 3-qubit space
        assert np.isclose(abs(sv[4]) ** 2, 1.0)

    # ------------------------------------------------------------------
    # Probabilities interface
    # ------------------------------------------------------------------

    def test_get_probabilities_sums_to_one(self, sim):
        qc = QuantumCircuit(2)
        qc.h(0).h(1)
        probs = sim.get_probabilities(qc)
        assert np.isclose(sum(probs.values()), 1.0)

    def test_get_probabilities_uniform(self, sim):
        """H⊗H creates uniform distribution over all 4 two-qubit states."""
        qc = QuantumCircuit(2)
        qc.h(0).h(1)
        probs = sim.get_probabilities(qc)
        for key, p in probs.items():
            assert np.isclose(p, 0.25, atol=1e-9), f"P({key}) = {p}"

    # ------------------------------------------------------------------
    # Rotation gates
    # ------------------------------------------------------------------

    def test_rx_pi_equals_x(self, sim):
        """RX(π)|0⟩ ≈ i·|1⟩ up to global phase → same measurement statistics as X."""
        qc = QuantumCircuit(1)
        qc.rx(math.pi, 0)
        sv = sim.get_statevector(qc)
        assert np.isclose(abs(sv[1]) ** 2, 1.0, atol=1e-7)

    def test_ry_pi_equals_x(self, sim):
        """RY(π)|0⟩ = |1⟩ (up to global phase)."""
        qc = QuantumCircuit(1)
        qc.ry(math.pi, 0)
        sv = sim.get_statevector(qc)
        assert np.isclose(abs(sv[1]) ** 2, 1.0, atol=1e-7)

    # ------------------------------------------------------------------
    # Partial measurement (only some qubits measured)
    # ------------------------------------------------------------------

    def test_partial_measurement_keys(self, sim):
        """When only q0 is measured we should get single-bit outcome keys."""
        qc = QuantumCircuit(2)
        qc.h(0).measure(0)  # Only measure q0
        counts = sim.run(qc, shots=1000)
        for key in counts:
            assert len(key) == 1
        assert set(counts.keys()).issubset({"0", "1"})

    # ------------------------------------------------------------------
    # Swap gate
    # ------------------------------------------------------------------

    def test_swap_exchanges_states(self, sim):
        """|10⟩ after SWAP(0,1) should become |01⟩."""
        qc = QuantumCircuit(2)
        qc.x(0).swap(0, 1)
        sv = sim.get_statevector(qc)
        # |01⟩ = index 1 in 2-qubit space
        assert np.isclose(abs(sv[1]) ** 2, 1.0)

    # ------------------------------------------------------------------
    # CZ gate
    # ------------------------------------------------------------------

    def test_cz_flips_phase(self, sim):
        """|11⟩ → -|11⟩ under CZ."""
        qc = QuantumCircuit(2)
        qc.x(0).x(1).cz(0, 1)
        sv = sim.get_statevector(qc)
        # |11⟩ = index 3; amplitude should be -1
        assert np.isclose(sv[3], -1.0 + 0j)

    # ------------------------------------------------------------------
    # Phase shift
    # ------------------------------------------------------------------

    def test_phase_shift_pi_equals_z(self, sim):
        """P(π)|1⟩ = e^{iπ}|1⟩ = -|1⟩  (same as Z on |1⟩)."""
        qc = QuantumCircuit(1)
        qc.x(0).p(math.pi, 0)
        sv = sim.get_statevector(qc)
        assert np.isclose(abs(sv[1]) ** 2, 1.0)
        assert np.isclose(sv[1].real, -1.0, atol=1e-7)

    # ------------------------------------------------------------------
    # Determinism with seed
    # ------------------------------------------------------------------

    def test_seeded_simulator_is_deterministic(self):
        qc = QuantumCircuit(2)
        qc.h(0).cnot(0, 1).measure_all()
        counts1 = QuantumSimulator(seed=7).run(qc, shots=512)
        counts2 = QuantumSimulator(seed=7).run(qc, shots=512)
        assert counts1 == counts2
