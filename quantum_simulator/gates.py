"""
gates.py — Standard unitary quantum-gate matrices
===================================================

Every quantum gate is a *unitary* matrix U satisfying U†U = I.  This module
provides a factory class (``QuantumGates``) whose static methods return NumPy
matrices that can be applied directly to state vectors or composed into larger
circuits.

Gate taxonomy implemented here
--------------------------------
Single-qubit gates
    Pauli-X (NOT), Pauli-Y, Pauli-Z
    Hadamard (H)
    Phase (S) and its conjugate (Sdg)
    π/8 gate (T) and its conjugate (Tdg)
    Rotation gates RX(θ), RY(θ), RZ(θ)
    General single-qubit phase shift P(λ)
    Identity (I)

Two-qubit gates
    CNOT  (Controlled-NOT / CX)
    CZ    (Controlled-Z)
    SWAP

Three-qubit gates
    Toffoli (CCX / CCNOT)
    Fredkin (CSWAP)

References
----------
Nielsen, M. A. & Chuang, I. L. (2010). *Quantum Computation and Quantum
    Information* (10th anniversary ed.). Cambridge University Press.
    Section 4.2 (single-qubit gates), Section 4.3 (multi-qubit gates).

IBM Quantum Documentation — Circuit Library:
    https://quantum.cloud.ibm.com/docs/en/api/qiskit/circuit_library

Barenco, A. et al. (1995). Elementary gates for quantum computation.
    *Physical Review A*, 52(5), 3457–3467.
    https://doi.org/10.1103/PhysRevA.52.3457
"""

from __future__ import annotations

import math
import numpy as np
from numpy.typing import NDArray

# Convenient type alias
_Matrix = NDArray[np.complex128]


class QuantumGates:
    """
    Factory for standard quantum-gate matrices.

    All methods are static and return (N×N) complex128 NumPy arrays.
    The convention follows Nielsen & Chuang with computational basis
    ordering  |0…0⟩, |0…1⟩, …, |1…1⟩.
    """

    # ==================================================================
    # Single-qubit gates (2 × 2 unitary matrices)
    # ==================================================================

    @staticmethod
    def identity() -> _Matrix:
        """Identity gate I = [[1, 0], [0, 1]]."""
        return np.eye(2, dtype=np.complex128)

    @staticmethod
    def pauli_x() -> _Matrix:
        """
        Pauli-X gate (quantum NOT / bit-flip).

        X = [[0, 1],
             [1, 0]]

        Maps |0⟩ → |1⟩ and |1⟩ → |0⟩.
        """
        return np.array([[0, 1], [1, 0]], dtype=np.complex128)

    @staticmethod
    def pauli_y() -> _Matrix:
        """
        Pauli-Y gate (bit-flip + phase-flip).

        Y = [[0, -i],
             [i,  0]]
        """
        return np.array([[0, -1j], [1j, 0]], dtype=np.complex128)

    @staticmethod
    def pauli_z() -> _Matrix:
        """
        Pauli-Z gate (phase-flip).

        Z = [[ 1, 0],
             [ 0,-1]]

        Maps |0⟩ → |0⟩ and |1⟩ → −|1⟩.
        """
        return np.array([[1, 0], [0, -1]], dtype=np.complex128)

    @staticmethod
    def hadamard() -> _Matrix:
        """
        Hadamard gate H = (1/√2) [[1, 1], [1, -1]].

        Creates equal superposition from a basis state:
            H|0⟩ = |+⟩ = (|0⟩ + |1⟩) / √2
            H|1⟩ = |−⟩ = (|0⟩ − |1⟩) / √2
        """
        s = 1.0 / math.sqrt(2)
        return np.array([[s, s], [s, -s]], dtype=np.complex128)

    @staticmethod
    def phase() -> _Matrix:
        """
        Phase gate S = [[1, 0], [0, i]].

        Equivalent to a π/2 rotation around the Z-axis.
        """
        return np.array([[1, 0], [0, 1j]], dtype=np.complex128)

    @staticmethod
    def phase_dagger() -> _Matrix:
        """Conjugate of the phase gate: S† = [[1, 0], [0, -i]]."""
        return np.array([[1, 0], [0, -1j]], dtype=np.complex128)

    @staticmethod
    def t_gate() -> _Matrix:
        """
        T gate (π/8 gate): T = [[1, 0], [0, e^{iπ/4}]].

        Equivalent to a π/4 rotation around Z.  Together with H, T forms a
        universal gate set for quantum computation (Boykin et al., 2000).
        """
        return np.array(
            [[1, 0], [0, np.exp(1j * math.pi / 4)]], dtype=np.complex128
        )

    @staticmethod
    def t_dagger() -> _Matrix:
        """Conjugate of the T gate: T† = [[1, 0], [0, e^{-iπ/4}]]."""
        return np.array(
            [[1, 0], [0, np.exp(-1j * math.pi / 4)]], dtype=np.complex128
        )

    @staticmethod
    def rx(theta: float) -> _Matrix:
        """
        X-rotation gate RX(θ) = e^{-iθX/2}.

        RX(θ) = [[cos(θ/2),  -i·sin(θ/2)],
                 [-i·sin(θ/2), cos(θ/2) ]]

        Parameters
        ----------
        theta : float
            Rotation angle in radians.
        """
        c = math.cos(theta / 2)
        s = math.sin(theta / 2)
        return np.array([[c, -1j * s], [-1j * s, c]], dtype=np.complex128)

    @staticmethod
    def ry(theta: float) -> _Matrix:
        """
        Y-rotation gate RY(θ) = e^{-iθY/2}.

        RY(θ) = [[cos(θ/2), -sin(θ/2)],
                 [sin(θ/2),  cos(θ/2)]]

        Parameters
        ----------
        theta : float
            Rotation angle in radians.
        """
        c = math.cos(theta / 2)
        s = math.sin(theta / 2)
        return np.array([[c, -s], [s, c]], dtype=np.complex128)

    @staticmethod
    def rz(theta: float) -> _Matrix:
        """
        Z-rotation gate RZ(θ) = e^{-iθZ/2}.

        RZ(θ) = [[e^{-iθ/2}, 0         ],
                 [0,          e^{iθ/2}  ]]

        Parameters
        ----------
        theta : float
            Rotation angle in radians.
        """
        return np.array(
            [
                [np.exp(-1j * theta / 2), 0],
                [0, np.exp(1j * theta / 2)],
            ],
            dtype=np.complex128,
        )

    @staticmethod
    def phase_shift(lam: float) -> _Matrix:
        """
        General single-qubit phase shift P(λ).

        P(λ) = [[1, 0       ],
                [0, e^{iλ}  ]]

        Parameters
        ----------
        lam : float
            Phase angle λ in radians.
        """
        return np.array(
            [[1, 0], [0, np.exp(1j * lam)]], dtype=np.complex128
        )

    # ==================================================================
    # Two-qubit gates (4 × 4 unitary matrices)
    # ==================================================================

    @staticmethod
    def cnot() -> _Matrix:
        """
        Controlled-NOT gate (CX / CNOT).

        CNOT = [[1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 0, 1],
                [0, 0, 1, 0]]

        Basis ordering: |00⟩, |01⟩, |10⟩, |11⟩.
        Flips the *target* qubit when the *control* qubit is |1⟩.
        """
        return np.array(
            [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 0, 1],
                [0, 0, 1, 0],
            ],
            dtype=np.complex128,
        )

    @staticmethod
    def cz() -> _Matrix:
        """
        Controlled-Z gate.

        CZ = [[1, 0, 0,  0],
              [0, 1, 0,  0],
              [0, 0, 1,  0],
              [0, 0, 0, -1]]

        Applies a Z (phase-flip) to the target qubit when the control is |1⟩.
        """
        return np.array(
            [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, -1],
            ],
            dtype=np.complex128,
        )

    @staticmethod
    def swap() -> _Matrix:
        """
        SWAP gate — exchanges the states of two qubits.

        SWAP = [[1, 0, 0, 0],
                [0, 0, 1, 0],
                [0, 1, 0, 0],
                [0, 0, 0, 1]]
        """
        return np.array(
            [
                [1, 0, 0, 0],
                [0, 0, 1, 0],
                [0, 1, 0, 0],
                [0, 0, 0, 1],
            ],
            dtype=np.complex128,
        )

    @staticmethod
    def controlled(u: _Matrix) -> _Matrix:
        """
        Build a *controlled-U* gate from an arbitrary single-qubit unitary U.

        The resulting 4×4 matrix applies U to the target qubit only when the
        control qubit is |1⟩:

            CU = [[I₂ | 0 ],
                  [ 0 | U ]]

        Parameters
        ----------
        u : numpy.ndarray, shape (2, 2)
            A single-qubit unitary matrix.

        Returns
        -------
        numpy.ndarray, shape (4, 4)
        """
        if u.shape != (2, 2):
            raise ValueError(f"Expected a 2×2 matrix, got shape {u.shape}.")
        cu = np.eye(4, dtype=np.complex128)
        cu[2:, 2:] = u
        return cu

    # ==================================================================
    # Three-qubit gates (8 × 8 unitary matrices)
    # ==================================================================

    @staticmethod
    def toffoli() -> _Matrix:
        """
        Toffoli gate (CCX / CCNOT).

        Flips the *target* qubit when *both* control qubits are |1⟩.
        The first 6 basis states are unchanged; |110⟩ ↔ |111⟩.

        Basis ordering: |000⟩, |001⟩, |010⟩, |011⟩, |100⟩, |101⟩, |110⟩, |111⟩.
        """
        m = np.eye(8, dtype=np.complex128)
        m[6, 6] = 0
        m[7, 7] = 0
        m[6, 7] = 1
        m[7, 6] = 1
        return m

    @staticmethod
    def fredkin() -> _Matrix:
        """
        Fredkin gate (CSWAP — Controlled-SWAP).

        When the control qubit is |1⟩ the states of the two target qubits are
        swapped: |101⟩ ↔ |110⟩.

        Basis ordering: |000⟩ … |111⟩.
        """
        m = np.eye(8, dtype=np.complex128)
        m[5, 5] = 0
        m[6, 6] = 0
        m[5, 6] = 1
        m[6, 5] = 1
        return m

    # ==================================================================
    # Utility
    # ==================================================================

    @staticmethod
    def is_unitary(matrix: _Matrix, tol: float = 1e-9) -> bool:
        """
        Return True if *matrix* is unitary (U†U ≈ I) within *tol*.

        Parameters
        ----------
        matrix : numpy.ndarray
        tol : float
        """
        n = matrix.shape[0]
        product = matrix.conj().T @ matrix
        return bool(np.allclose(product, np.eye(n), atol=tol))

    @staticmethod
    def tensor_product(a: _Matrix, b: _Matrix) -> _Matrix:
        """
        Compute the tensor (Kronecker) product A ⊗ B.

        Used to construct multi-qubit gate operators from single-qubit gates
        acting on independent qubits.

        Parameters
        ----------
        a, b : numpy.ndarray
        """
        return np.kron(a, b)
