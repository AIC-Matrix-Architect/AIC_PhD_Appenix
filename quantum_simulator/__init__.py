"""
Quantum Computing Simulator
============================
A pure-Python / NumPy implementation of a statevector-based quantum computing
simulator, designed to model and test quantum circuits.

Core concepts follow the formalism described in:
  - Nielsen & Chuang, "Quantum Computation and Quantum Information" (Cambridge, 2010)
  - IBM Quantum Documentation: https://quantum.cloud.ibm.com/docs
  - Javadi-Abhari et al., "Quantum computing with Qiskit", arXiv:2405.08810 (2024)

Public API
----------
Qubit             -- single-qubit state-vector representation
QuantumGates      -- factory for standard unitary gate matrices
QuantumCircuit    -- multi-qubit register + gate scheduling
QuantumSimulator  -- statevector simulator that executes QuantumCircuits
"""

from .qubit import Qubit
from .gates import QuantumGates
from .circuit import QuantumCircuit
from .simulator import QuantumSimulator

__all__ = ["Qubit", "QuantumGates", "QuantumCircuit", "QuantumSimulator"]
__version__ = "1.0.0"
