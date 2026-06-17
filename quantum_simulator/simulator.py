"""
simulator.py — Statevector quantum simulator
=============================================

``QuantumSimulator`` executes a ``QuantumCircuit`` by maintaining a complex
state vector of length 2ⁿ and applying gate matrices via tensor-product
expansion.

Simulation algorithm
--------------------
1. Initialise the state vector |ψ₀⟩ = |0…0⟩ (length 2ⁿ, first element = 1).
2. For each gate instruction with target qubit(s) *q₁ … qₖ*:
   a. Build the full 2ⁿ × 2ⁿ operator by embedding the gate using
      identity-tensor-product expansion (Identity on all other qubits).
   b. Update the state: |ψ⟩ ← U_full |ψ⟩.
3. At measurement, sample outcomes using the Born-rule probabilities
   |⟨x|ψ⟩|² for each basis state |x⟩.

Theoretical background
-----------------------
Nielsen, M. A. & Chuang, I. L. (2010). *Quantum Computation and Quantum
    Information* (10th anniversary ed.). Cambridge University Press.
    Chapter 4 (quantum circuit model).

Steiger, D. S., Häner, T., & Troyer, M. (2018). ProjectQ: an open source
    software framework for quantum computing.  *Quantum*, 2, 49.
    https://doi.org/10.22331/q-2018-01-31-49

IBM Quantum Documentation — Simulators overview:
    https://docs.quantum.ibm.com/guides/simulators
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional

import numpy as np
from numpy.typing import NDArray

from .circuit import QuantumCircuit, Instruction
from .gates import QuantumGates


class QuantumSimulator:
    """
    Statevector-based quantum simulator.

    Parameters
    ----------
    seed : int, optional
        Seed for the pseudo-random-number generator used during measurement
        sampling.  Pass an integer for reproducible results.

    Examples
    --------
    Bell-state preparation and measurement::

        from quantum_simulator import QuantumCircuit, QuantumSimulator

        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cnot(0, 1)
        qc.measure_all()

        sim = QuantumSimulator(seed=42)
        counts = sim.run(qc, shots=1024)
        print(counts)   # {'00': ~512, '11': ~512}
    """

    def __init__(self, seed: Optional[int] = None) -> None:
        self._rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def run(
        self, circuit: QuantumCircuit, shots: int = 1024
    ) -> Dict[str, int]:
        """
        Execute *circuit* for *shots* repeated measurements and return a
        dictionary mapping binary-string outcomes to their observed counts.

        Parameters
        ----------
        circuit : QuantumCircuit
            The circuit to simulate.  Must have at least one qubit scheduled
            for measurement (via :meth:`~QuantumCircuit.measure` or
            :meth:`~QuantumCircuit.measure_all`).
        shots : int, optional
            Number of independent measurement repetitions.  Default 1024.

        Returns
        -------
        dict[str, int]
            Measurement outcome counts, e.g. ``{'00': 512, '11': 512}``.

        Raises
        ------
        ValueError
            If *circuit* has no measurement directives, or *shots* < 1.
        """
        if shots < 1:
            raise ValueError(f"shots must be ≥ 1, got {shots}.")
        if not circuit.measurements:
            raise ValueError(
                "Circuit has no measurement directives.  "
                "Call circuit.measure(qubit) or circuit.measure_all() first."
            )

        statevector = self._get_statevector(circuit)
        counts = self._sample(statevector, circuit, shots)
        return counts

    def get_statevector(
        self, circuit: QuantumCircuit
    ) -> NDArray[np.complex128]:
        """
        Return the final state vector |ψ⟩ after applying all gates in
        *circuit* (without any measurement collapse).

        Parameters
        ----------
        circuit : QuantumCircuit

        Returns
        -------
        numpy.ndarray, shape (2**n,), dtype complex128
        """
        return self._get_statevector(circuit)

    def get_probabilities(
        self, circuit: QuantumCircuit
    ) -> Dict[str, float]:
        """
        Return a dict mapping each basis state to its probability.

        Parameters
        ----------
        circuit : QuantumCircuit

        Returns
        -------
        dict[str, float]
            e.g. ``{'00': 0.5, '11': 0.5}`` for a Bell state.
        """
        sv = self._get_statevector(circuit)
        n = circuit.n_qubits
        probs: Dict[str, float] = {}
        for i, amplitude in enumerate(sv):
            p = float(abs(amplitude) ** 2)
            if p > 0.0:
                label = format(i, f"0{n}b")
                probs[label] = p
        return probs

    # ------------------------------------------------------------------
    # Internal: state-vector evolution
    # ------------------------------------------------------------------

    def _get_statevector(
        self, circuit: QuantumCircuit
    ) -> NDArray[np.complex128]:
        """Initialise |0…0⟩ and evolve under all circuit instructions."""
        n = circuit.n_qubits
        dim = 1 << n  # 2**n
        state = np.zeros(dim, dtype=np.complex128)
        state[0] = 1.0 + 0j  # |0...0⟩

        for instr in circuit.instructions:
            state = self._apply_instruction(state, instr, n)

        return state

    def _apply_instruction(
        self,
        state: NDArray[np.complex128],
        instr: Instruction,
        n: int,
    ) -> NDArray[np.complex128]:
        """
        Apply a gate instruction to the state vector.

        For single-qubit gates we use the efficient tensor-expansion approach:
        build the full 2ⁿ × 2ⁿ unitary by inserting Identity matrices for
        non-targeted qubits.

        For two- and three-qubit gates we check whether the target qubits are
        *adjacent* (in the ordering used by the circuit).  If they are, the
        gate matrix is embedded directly; if not, SWAP gates are inserted
        implicitly.
        """
        k = len(instr.qubits)
        if k == 1:
            return self._apply_single(state, instr.gate, instr.qubits[0], n)
        elif k == 2:
            return self._apply_two(
                state, instr.gate, instr.qubits[0], instr.qubits[1], n
            )
        elif k == 3:
            return self._apply_three(
                state,
                instr.gate,
                instr.qubits[0],
                instr.qubits[1],
                instr.qubits[2],
                n,
            )
        else:
            raise NotImplementedError(
                f"Gates acting on more than 3 qubits are not supported "
                f"(got {k} qubits)."
            )

    def _apply_single(
        self,
        state: NDArray[np.complex128],
        gate: NDArray[np.complex128],
        qubit: int,
        n: int,
    ) -> NDArray[np.complex128]:
        """
        Apply a single-qubit gate to *qubit* in an *n*-qubit state vector.

        We reshape the state vector into a tensor of shape (2, 2, …, 2),
        contract the gate along the target axis, then flatten back.
        """
        # Reshape to (2, 2, ..., 2)  with n axes
        psi = state.reshape([2] * n)
        # Contract: result_qubit_axis = sum_j gate[i,j] * psi[...,j,...]
        psi = np.tensordot(gate, psi, axes=[[1], [qubit]])
        # tensordot puts the contracted index (qubit) at position 0;
        # move it back to the correct axis.
        psi = np.moveaxis(psi, 0, qubit)
        return psi.reshape(-1)

    def _apply_two(
        self,
        state: NDArray[np.complex128],
        gate: NDArray[np.complex128],  # shape (4,4)
        q0: int,
        q1: int,
        n: int,
    ) -> NDArray[np.complex128]:
        """
        Apply a two-qubit gate to qubits *q0* (control) and *q1* (target).

        The gate matrix acts on the subspace spanned by (q0, q1) with basis
        ordering |00⟩, |01⟩, |10⟩, |11⟩ where q0 is the more significant bit.
        """
        # Reshape gate from (4,4) to (2,2,2,2): [out_q0, out_q1, in_q0, in_q1]
        g = gate.reshape(2, 2, 2, 2)
        psi = state.reshape([2] * n)
        # Contract over input indices [q0, q1]
        psi = np.tensordot(g, psi, axes=[[2, 3], [q0, q1]])
        # After tensordot the shape is (2, 2, ...) with the two new output
        # axes at positions 0 and 1; move them back to q0 and q1.
        psi = np.moveaxis(psi, [0, 1], [q0, q1])
        return psi.reshape(-1)

    def _apply_three(
        self,
        state: NDArray[np.complex128],
        gate: NDArray[np.complex128],  # shape (8,8)
        q0: int,
        q1: int,
        q2: int,
        n: int,
    ) -> NDArray[np.complex128]:
        """Apply a three-qubit gate to qubits *q0*, *q1*, *q2*."""
        g = gate.reshape(2, 2, 2, 2, 2, 2)
        psi = state.reshape([2] * n)
        psi = np.tensordot(g, psi, axes=[[3, 4, 5], [q0, q1, q2]])
        psi = np.moveaxis(psi, [0, 1, 2], [q0, q1, q2])
        return psi.reshape(-1)

    # ------------------------------------------------------------------
    # Internal: measurement sampling
    # ------------------------------------------------------------------

    def _sample(
        self,
        state: NDArray[np.complex128],
        circuit: QuantumCircuit,
        shots: int,
    ) -> Dict[str, int]:
        """
        Draw *shots* samples from the probability distribution |⟨x|ψ⟩|².

        Only qubits in *circuit.measurements* are included in the outcome
        strings; remaining qubits are traced out (marginalised) correctly.
        """
        n = circuit.n_qubits
        measured_qubits = sorted(circuit.measurements)
        dim = 1 << n

        # Compute per-basis-state probabilities
        probs = np.abs(state) ** 2
        # Numerical normalisation guard
        probs /= probs.sum()

        # Draw sample indices
        indices = self._rng.choice(dim, size=shots, p=probs)

        # Convert each index to a binary string over the measured qubits only
        counts: Dict[str, int] = {}
        for idx in indices:
            full_bits = format(int(idx), f"0{n}b")
            # Select bits corresponding to the measured qubits
            key = "".join(full_bits[q] for q in measured_qubits)
            counts[key] = counts.get(key, 0) + 1

        return counts
