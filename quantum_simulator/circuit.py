"""
circuit.py — Multi-qubit quantum circuit
==========================================

A ``QuantumCircuit`` holds an ordered list of *instructions* (gate + target
qubit indices) that are later executed by ``QuantumSimulator``.  This design
mirrors the layered architecture recommended in:

    Javadi-Abhari, A. et al. (2024). Quantum computing with Qiskit.
    arXiv:2405.08810.  https://arxiv.org/abs/2405.08810

    IBM Quantum Documentation — "QuantumCircuit":
    https://quantum.cloud.ibm.com/docs/en/api/qiskit/qiskit.circuit.QuantumCircuit

Usage
-----
>>> from quantum_simulator import QuantumCircuit
>>> qc = QuantumCircuit(2)       # two-qubit register
>>> qc.h(0)                      # Hadamard on qubit 0
>>> qc.cnot(0, 1)                # entangle qubits 0 and 1
>>> qc.measure_all()             # measure both qubits
>>> print(qc)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple
import math
import numpy as np
from numpy.typing import NDArray

from .gates import QuantumGates


@dataclass(frozen=True)
class Instruction:
    """
    An immutable record that binds a unitary gate matrix to one or more
    target qubit indices.

    Attributes
    ----------
    name : str
        Human-readable gate name, e.g. ``"H"``, ``"CNOT"``.
    gate : numpy.ndarray
        The unitary matrix to apply.
    qubits : tuple of int
        Indices of the qubits the gate acts on, in order.
    params : tuple of float
        Optional scalar parameters (rotation angles, etc.).
    """

    name: str
    gate: NDArray[np.complex128]
    qubits: Tuple[int, ...]
    params: Tuple[float, ...] = field(default_factory=tuple)


class QuantumCircuit:
    """
    An n-qubit quantum circuit: a register of *n* qubits and an ordered list
    of gate instructions plus measurement directives.

    Parameters
    ----------
    n_qubits : int
        Number of qubits in the register (must be ≥ 1).

    Raises
    ------
    ValueError
        If *n_qubits* < 1.

    Notes
    -----
    Qubits are indexed 0 … n_qubits−1.  The computational basis is ordered
    from qubit 0 (most-significant bit) to qubit n−1 (least-significant bit):
    |q₀ q₁ … qₙ₋₁⟩.
    """

    def __init__(self, n_qubits: int) -> None:
        if n_qubits < 1:
            raise ValueError(
                f"n_qubits must be ≥ 1, got {n_qubits}."
            )
        self._n_qubits: int = n_qubits
        self._instructions: List[Instruction] = []
        self._measurements: List[int] = []  # qubit indices to measure

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def n_qubits(self) -> int:
        """Number of qubits in the circuit register."""
        return self._n_qubits

    @property
    def instructions(self) -> List[Instruction]:
        """Ordered list of gate instructions (read-only view)."""
        return list(self._instructions)

    @property
    def measurements(self) -> List[int]:
        """List of qubit indices scheduled for measurement."""
        return list(self._measurements)

    # ------------------------------------------------------------------
    # Internal helper
    # ------------------------------------------------------------------

    def _check_qubit(self, index: int) -> None:
        if not (0 <= index < self._n_qubits):
            raise ValueError(
                f"Qubit index {index} out of range for {self._n_qubits}-qubit circuit."
            )

    def _add(self, name: str, gate: NDArray, qubits: Sequence[int],
             params: Sequence[float] = ()) -> "QuantumCircuit":
        for q in qubits:
            self._check_qubit(q)
        self._instructions.append(
            Instruction(
                name=name,
                gate=gate,
                qubits=tuple(qubits),
                params=tuple(params),
            )
        )
        return self  # enable method chaining

    # ==================================================================
    # Single-qubit gate methods
    # ==================================================================

    def i(self, qubit: int) -> "QuantumCircuit":
        """Apply the Identity gate to *qubit*."""
        return self._add("I", QuantumGates.identity(), [qubit])

    def x(self, qubit: int) -> "QuantumCircuit":
        """Apply Pauli-X (NOT) gate to *qubit*."""
        return self._add("X", QuantumGates.pauli_x(), [qubit])

    def y(self, qubit: int) -> "QuantumCircuit":
        """Apply Pauli-Y gate to *qubit*."""
        return self._add("Y", QuantumGates.pauli_y(), [qubit])

    def z(self, qubit: int) -> "QuantumCircuit":
        """Apply Pauli-Z gate to *qubit*."""
        return self._add("Z", QuantumGates.pauli_z(), [qubit])

    def h(self, qubit: int) -> "QuantumCircuit":
        """Apply Hadamard gate to *qubit*."""
        return self._add("H", QuantumGates.hadamard(), [qubit])

    def s(self, qubit: int) -> "QuantumCircuit":
        """Apply Phase (S) gate to *qubit*."""
        return self._add("S", QuantumGates.phase(), [qubit])

    def sdg(self, qubit: int) -> "QuantumCircuit":
        """Apply S† gate to *qubit*."""
        return self._add("Sdg", QuantumGates.phase_dagger(), [qubit])

    def t(self, qubit: int) -> "QuantumCircuit":
        """Apply T gate to *qubit*."""
        return self._add("T", QuantumGates.t_gate(), [qubit])

    def tdg(self, qubit: int) -> "QuantumCircuit":
        """Apply T† gate to *qubit*."""
        return self._add("Tdg", QuantumGates.t_dagger(), [qubit])

    def rx(self, theta: float, qubit: int) -> "QuantumCircuit":
        """
        Apply RX(θ) rotation around the X-axis to *qubit*.

        Parameters
        ----------
        theta : float
            Rotation angle in radians.
        qubit : int
        """
        return self._add("RX", QuantumGates.rx(theta), [qubit], [theta])

    def ry(self, theta: float, qubit: int) -> "QuantumCircuit":
        """
        Apply RY(θ) rotation around the Y-axis to *qubit*.

        Parameters
        ----------
        theta : float
            Rotation angle in radians.
        qubit : int
        """
        return self._add("RY", QuantumGates.ry(theta), [qubit], [theta])

    def rz(self, theta: float, qubit: int) -> "QuantumCircuit":
        """
        Apply RZ(θ) rotation around the Z-axis to *qubit*.

        Parameters
        ----------
        theta : float
            Rotation angle in radians.
        qubit : int
        """
        return self._add("RZ", QuantumGates.rz(theta), [qubit], [theta])

    def p(self, lam: float, qubit: int) -> "QuantumCircuit":
        """
        Apply a general phase shift P(λ) to *qubit*.

        Parameters
        ----------
        lam : float
            Phase angle in radians.
        qubit : int
        """
        return self._add("P", QuantumGates.phase_shift(lam), [qubit], [lam])

    # ==================================================================
    # Two-qubit gate methods
    # ==================================================================

    def cnot(self, control: int, target: int) -> "QuantumCircuit":
        """
        Apply CNOT (Controlled-X) gate.

        Parameters
        ----------
        control : int
            Control qubit index.
        target : int
            Target qubit index.
        """
        if control == target:
            raise ValueError("Control and target qubits must be different.")
        return self._add("CNOT", QuantumGates.cnot(), [control, target])

    def cx(self, control: int, target: int) -> "QuantumCircuit":
        """Alias for :meth:`cnot`."""
        return self.cnot(control, target)

    def cz(self, control: int, target: int) -> "QuantumCircuit":
        """
        Apply Controlled-Z gate.

        Parameters
        ----------
        control, target : int
        """
        if control == target:
            raise ValueError("Control and target qubits must be different.")
        return self._add("CZ", QuantumGates.cz(), [control, target])

    def swap(self, qubit_a: int, qubit_b: int) -> "QuantumCircuit":
        """
        Apply SWAP gate to exchange the states of *qubit_a* and *qubit_b*.

        Parameters
        ----------
        qubit_a, qubit_b : int
        """
        if qubit_a == qubit_b:
            raise ValueError("SWAP requires two distinct qubit indices.")
        return self._add("SWAP", QuantumGates.swap(), [qubit_a, qubit_b])

    def cu(
        self, u: NDArray[np.complex128], control: int, target: int
    ) -> "QuantumCircuit":
        """
        Apply a controlled-U gate using an arbitrary single-qubit unitary *u*.

        Parameters
        ----------
        u : numpy.ndarray, shape (2, 2)
        control, target : int
        """
        if control == target:
            raise ValueError("Control and target qubits must be different.")
        return self._add("CU", QuantumGates.controlled(u), [control, target])

    # ==================================================================
    # Three-qubit gate methods
    # ==================================================================

    def toffoli(
        self, control1: int, control2: int, target: int
    ) -> "QuantumCircuit":
        """
        Apply Toffoli (CCX) gate.

        Parameters
        ----------
        control1, control2 : int
            Control qubit indices.
        target : int
            Target qubit index.
        """
        qubits = [control1, control2, target]
        if len(set(qubits)) != 3:
            raise ValueError("Toffoli gate requires three distinct qubit indices.")
        return self._add("Toffoli", QuantumGates.toffoli(), qubits)

    def fredkin(
        self, control: int, target1: int, target2: int
    ) -> "QuantumCircuit":
        """
        Apply Fredkin (CSWAP) gate.

        Parameters
        ----------
        control : int
        target1, target2 : int
        """
        qubits = [control, target1, target2]
        if len(set(qubits)) != 3:
            raise ValueError("Fredkin gate requires three distinct qubit indices.")
        return self._add("Fredkin", QuantumGates.fredkin(), qubits)

    # ==================================================================
    # Measurement directives
    # ==================================================================

    def measure(self, qubit: int) -> "QuantumCircuit":
        """
        Schedule qubit *qubit* for measurement at the end of the circuit.

        Parameters
        ----------
        qubit : int
        """
        self._check_qubit(qubit)
        if qubit not in self._measurements:
            self._measurements.append(qubit)
        return self

    def measure_all(self) -> "QuantumCircuit":
        """Schedule all qubits for measurement."""
        for q in range(self._n_qubits):
            if q not in self._measurements:
                self._measurements.append(q)
        return self

    # ==================================================================
    # Convenience / introspection
    # ==================================================================

    def depth(self) -> int:
        """Return the number of gate instructions in the circuit."""
        return len(self._instructions)

    def draw(self) -> str:
        """
        Return a simple ASCII representation of the circuit.

        Returns
        -------
        str
        """
        lines = [f"QuantumCircuit({self._n_qubits} qubits)"]
        lines.append("─" * 40)
        for idx, instr in enumerate(self._instructions):
            qstr = ", ".join(f"q{q}" for q in instr.qubits)
            pstr = (
                " (" + ", ".join(f"{p:.4f}" for p in instr.params) + ")"
                if instr.params
                else ""
            )
            lines.append(f"  [{idx:02d}] {instr.name}{pstr} @ {qstr}")
        if self._measurements:
            mstr = ", ".join(f"q{q}" for q in sorted(self._measurements))
            lines.append(f"  [M]  Measure: {mstr}")
        lines.append("─" * 40)
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"QuantumCircuit(n_qubits={self._n_qubits}, "
            f"depth={self.depth()}, "
            f"measurements={self._measurements})"
        )

    def __str__(self) -> str:
        return self.draw()
