"""
qubit.py — Single-qubit state-vector representation
=====================================================

A qubit is the fundamental unit of quantum information.  Unlike a classical
bit that is deterministically 0 or 1, a qubit can exist in a *superposition*
of both basis states simultaneously:

    |ψ⟩ = α|0⟩ + β|1⟩

where α, β ∈ ℂ and the *normalisation* (born-rule) constraint holds:

    |α|² + |β|² = 1

Geometrically this corresponds to a point on the *Bloch sphere* (Bloch 1946).

References
----------
Nielsen, M. A. & Chuang, I. L. (2010). *Quantum Computation and Quantum
    Information* (10th anniversary ed.). Cambridge University Press.
    Sections 1.2–1.3.

IBM Quantum Documentation — "Bits, gates, and circuits":
    https://qiskit.qotlabs.org/learning/courses/
        utility-scale-quantum-computing/bits-gates-and-circuits

Krantz, P. et al. (2019). A quantum engineer's guide to superconducting
    qubits. *Applied Physics Reviews*, 6(2), 021318.
    https://doi.org/10.1063/1.5089550
"""

from __future__ import annotations

import math
import numpy as np
from numpy.typing import NDArray


class Qubit:
    """
    A single-qubit state vector.

    The state is stored as a (2,) complex128 NumPy array:

        state[0] = α  (amplitude of |0⟩)
        state[1] = β  (amplitude of |1⟩)

    Parameters
    ----------
    alpha : complex, optional
        Amplitude of |0⟩.  Defaults to 1 (qubit in state |0⟩).
    beta : complex, optional
        Amplitude of |1⟩.  Defaults to 0.

    Raises
    ------
    ValueError
        If the supplied amplitudes do not satisfy the normalisation condition
        |α|² + |β|² ≈ 1 (tolerance 1 × 10⁻⁶).

    Examples
    --------
    >>> q = Qubit()            # |0⟩
    >>> q = Qubit(0, 1)        # |1⟩
    >>> import math
    >>> s = 1 / math.sqrt(2)
    >>> q = Qubit(s, s)        # |+⟩ = H|0⟩
    """

    _NORM_TOL: float = 1e-6

    def __init__(
        self,
        alpha: complex = 1.0 + 0j,
        beta: complex = 0.0 + 0j,
    ) -> None:
        self._state: NDArray[np.complex128] = np.array(
            [alpha, beta], dtype=np.complex128
        )
        self._validate()

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate(self) -> None:
        norm = float(np.real(np.vdot(self._state, self._state)))
        if abs(norm - 1.0) > self._NORM_TOL:
            raise ValueError(
                f"Qubit state is not normalised: |α|²+|β|² = {norm:.8f}. "
                "Amplitudes must satisfy |α|²+|β|² = 1."
            )

    # ------------------------------------------------------------------
    # Constructors (alternative / convenience)
    # ------------------------------------------------------------------

    @classmethod
    def zero(cls) -> "Qubit":
        """Return |0⟩ (computational basis state zero)."""
        return cls(1.0 + 0j, 0.0 + 0j)

    @classmethod
    def one(cls) -> "Qubit":
        """Return |1⟩ (computational basis state one)."""
        return cls(0.0 + 0j, 1.0 + 0j)

    @classmethod
    def plus(cls) -> "Qubit":
        """Return |+⟩ = (|0⟩ + |1⟩) / √2 (the Hadamard basis state)."""
        s = 1.0 / math.sqrt(2)
        return cls(s + 0j, s + 0j)

    @classmethod
    def minus(cls) -> "Qubit":
        """Return |−⟩ = (|0⟩ − |1⟩) / √2."""
        s = 1.0 / math.sqrt(2)
        return cls(s + 0j, -s + 0j)

    @classmethod
    def from_bloch(cls, theta: float, phi: float) -> "Qubit":
        """
        Create a qubit from Bloch-sphere angles.

        The Bloch-sphere parametrisation (Nielsen & Chuang, eq. 1.4) is:

            |ψ⟩ = cos(θ/2)|0⟩ + e^{iφ} sin(θ/2)|1⟩

        Parameters
        ----------
        theta : float
            Polar angle in [0, π].
        phi : float
            Azimuthal angle in [0, 2π).
        """
        alpha = math.cos(theta / 2) + 0j
        beta = np.exp(1j * phi) * math.sin(theta / 2)
        return cls(alpha, beta)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def state(self) -> NDArray[np.complex128]:
        """Return the (2,) state vector (read-only copy)."""
        return self._state.copy()

    @property
    def alpha(self) -> complex:
        """Amplitude of |0⟩."""
        return complex(self._state[0])

    @property
    def beta(self) -> complex:
        """Amplitude of |1⟩."""
        return complex(self._state[1])

    @property
    def prob_zero(self) -> float:
        """Probability of measuring |0⟩ = |α|²."""
        return float(abs(self._state[0]) ** 2)

    @property
    def prob_one(self) -> float:
        """Probability of measuring |1⟩ = |β|²."""
        return float(abs(self._state[1]) ** 2)

    # ------------------------------------------------------------------
    # Measurement simulation
    # ------------------------------------------------------------------

    def measure(self, rng: np.random.Generator | None = None) -> int:
        """
        Simulate a single projective measurement in the computational basis.

        The qubit collapses to |0⟩ with probability |α|² or to |1⟩ with
        probability |β|² (Born rule).  The internal state is updated
        accordingly.

        Parameters
        ----------
        rng : numpy.random.Generator, optional
            A seeded random-number generator for reproducible results.
            Defaults to ``numpy.random.default_rng()``.

        Returns
        -------
        int
            0 or 1 — the measurement outcome.
        """
        if rng is None:
            rng = np.random.default_rng()
        outcome = int(rng.choice([0, 1], p=[self.prob_zero, self.prob_one]))
        if outcome == 0:
            self._state = np.array([1.0 + 0j, 0.0 + 0j], dtype=np.complex128)
        else:
            self._state = np.array([0.0 + 0j, 1.0 + 0j], dtype=np.complex128)
        return outcome

    # ------------------------------------------------------------------
    # Tensor product
    # ------------------------------------------------------------------

    def tensor(self, other: "Qubit") -> NDArray[np.complex128]:
        """
        Return the tensor (Kronecker) product of this qubit with *other*.

        The result is a (4,) state vector representing the two-qubit product
        state  |self⟩ ⊗ |other⟩.

        Parameters
        ----------
        other : Qubit

        Returns
        -------
        numpy.ndarray, shape (4,), dtype complex128
        """
        return np.kron(self._state, other._state)

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        a, b = self._state
        return (
            f"Qubit(α={a:.4f}, β={b:.4f}) "
            f"[P(0)={self.prob_zero:.4f}, P(1)={self.prob_one:.4f}]"
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Qubit):
            return NotImplemented
        return bool(np.allclose(self._state, other._state))
