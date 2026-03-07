# Quantum Computing Simulator

A **pure-Python / NumPy** statevector-based quantum computing simulator
designed to model and test quantum circuits.  
The simulator defines qubit behaviour as a first-class abstraction and
provides all standard quantum operations required to build and test quantum
algorithms from scratch.

---

## Theoretical Background

### What is a Qubit?

A qubit (quantum bit) is the fundamental unit of quantum information.
Unlike a classical bit that is deterministically **0** or **1**, a qubit can
exist in a *superposition* of both basis states simultaneously:

```
|ψ⟩ = α|0⟩ + β|1⟩
```

where α, β ∈ ℂ (complex amplitudes) satisfying the **normalisation
(Born-rule) constraint**:

```
|α|² + |β|² = 1
```

Geometrically, a qubit corresponds to a point on the **Bloch sphere**
(Bloch, 1946), parameterised by polar angle θ ∈ [0, π] and azimuthal
angle φ ∈ [0, 2π):

```
|ψ⟩ = cos(θ/2)|0⟩ + e^{iφ} sin(θ/2)|1⟩
```

### Quantum Gates

All quantum operations are represented by **unitary matrices** U satisfying
U†U = I.  Unitarity guarantees that quantum operations are reversible and
preserve normalisation.

| Gate      | Symbol | Matrix                                               | Action                          |
|-----------|--------|------------------------------------------------------|---------------------------------|
| Pauli-X   | X      | `[[0,1],[1,0]]`                                      | Bit-flip: \|0⟩↔\|1⟩            |
| Pauli-Y   | Y      | `[[0,-i],[i,0]]`                                     | Bit+phase flip                  |
| Pauli-Z   | Z      | `[[1,0],[0,-1]]`                                     | Phase-flip                      |
| Hadamard  | H      | `(1/√2)[[1,1],[1,-1]]`                               | Superposition: \|0⟩→\|+⟩       |
| Phase     | S      | `[[1,0],[0,i]]`                                      | π/2 Z-rotation                  |
| T gate    | T      | `[[1,0],[0,e^{iπ/4}]]`                               | π/4 Z-rotation                  |
| RX(θ)     | RX     | `[[cos θ/2, -i sin θ/2],[-i sin θ/2, cos θ/2]]`     | X-axis rotation                 |
| RY(θ)     | RY     | `[[cos θ/2, -sin θ/2],[sin θ/2, cos θ/2]]`           | Y-axis rotation                 |
| RZ(θ)     | RZ     | `[[e^{-iθ/2},0],[0,e^{iθ/2}]]`                       | Z-axis rotation                 |
| CNOT      | CX     | 4×4 controlled-NOT                                   | Entangles two qubits            |
| CZ        | CZ     | 4×4 controlled-Z                                     | Conditional phase flip          |
| SWAP      | SWAP   | 4×4 exchange                                         | Swaps two qubit states          |
| Toffoli   | CCX    | 8×8 doubly-controlled NOT                            | Classical NAND / universal gate |
| Fredkin   | CSWAP  | 8×8 controlled-SWAP                                  | Conditional swap                |

### Multi-Qubit Systems

For an *n*-qubit register the state space is **2ⁿ-dimensional**.  The
combined state is the **tensor product** (Kronecker product) of individual
qubit states:

```
|ψ₁ ψ₂ … ψₙ⟩ = |ψ₁⟩ ⊗ |ψ₂⟩ ⊗ … ⊗ |ψₙ⟩
```

### Measurement (Born Rule)

A projective measurement in the computational basis yields outcome *x* with
probability:

```
P(x) = |⟨x|ψ⟩|²
```

After measurement the state collapses to the corresponding basis state
|x⟩.

### Simulation Algorithm

1. Initialise state vector **|0…0⟩** (length 2ⁿ, first element = 1).  
2. For each gate acting on qubits q₁…qₖ, reshape the state tensor and
   contract the gate along the target axes (tensor-network contraction).  
3. At measurement draw samples from the probability distribution using the
   Born rule, returning a frequency histogram (counts dict).

---

## Installation

```bash
pip install -r requirements.txt
```

**Requirements:** Python ≥ 3.10, NumPy ≥ 1.24.

---

## Package Structure

```
quantum_simulator/
├── __init__.py    Public API
├── qubit.py       Qubit class — single-qubit state vector
├── gates.py       QuantumGates factory — all standard gate matrices
├── circuit.py     QuantumCircuit — register and gate scheduling
└── simulator.py   QuantumSimulator — statevector execution engine

tests/
└── test_quantum_simulator.py   72-test suite (pytest)
```

---

## Quick Start

### 1 — Single Qubit Operations

```python
from quantum_simulator import Qubit, QuantumGates
import numpy as np

# Create qubit in |0⟩
q = Qubit()
print(q)   # Qubit(α=1.0000+0.0000j, β=0.0000+0.0000j) [P(0)=1.0000, P(1)=0.0000]

# Create common states
q_zero  = Qubit.zero()       # |0⟩
q_one   = Qubit.one()        # |1⟩
q_plus  = Qubit.plus()       # |+⟩ = (|0⟩+|1⟩)/√2
q_minus = Qubit.minus()      # |−⟩ = (|0⟩−|1⟩)/√2

# Create qubit from Bloch sphere angles (θ=π/2, φ=π)
import math
q_bloch = Qubit.from_bloch(theta=math.pi / 2, phi=math.pi)

# Measurement probabilities
print(f"P(0) = {q_plus.prob_zero:.4f}")   # 0.5000
print(f"P(1) = {q_plus.prob_one:.4f}")    # 0.5000

# Simulate a single measurement (collapses state)
rng = np.random.default_rng(42)
result = q_plus.measure(rng=rng)
print(f"Measured: {result}")              # 0 or 1

# Tensor product of two qubits
state_00 = Qubit.zero().tensor(Qubit.zero())
print(state_00)  # [1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j]
```

### 2 — Quantum Gates

```python
from quantum_simulator import QuantumGates
import numpy as np

# Single-qubit gates
H   = QuantumGates.hadamard()     # Hadamard
X   = QuantumGates.pauli_x()      # Pauli-X (NOT)
Y   = QuantumGates.pauli_y()      # Pauli-Y
Z   = QuantumGates.pauli_z()      # Pauli-Z
S   = QuantumGates.phase()        # Phase (S)
T   = QuantumGates.t_gate()       # T gate

# Parameterised rotation gates
RX  = QuantumGates.rx(math.pi / 4)    # 45° X-rotation
RY  = QuantumGates.ry(math.pi / 2)    # 90° Y-rotation
RZ  = QuantumGates.rz(math.pi)        # 180° Z-rotation

# Two-qubit gates
CNOT  = QuantumGates.cnot()
CZ    = QuantumGates.cz()
SWAP  = QuantumGates.swap()

# Build a controlled-H gate from any unitary
CH  = QuantumGates.controlled(QuantumGates.hadamard())

# Three-qubit gates
TOF = QuantumGates.toffoli()   # Toffoli (CCX)
FRE = QuantumGates.fredkin()   # Fredkin (CSWAP)

# Tensor product (I ⊗ H: apply H to second qubit only)
IH = QuantumGates.tensor_product(np.eye(2), H)

# Verify unitarity
assert QuantumGates.is_unitary(H)

# Apply gate directly to a state vector
state = np.array([1.0, 0.0], dtype=complex)  # |0⟩
print(H @ state)   # [0.707..., 0.707...] = |+⟩
```

### 3 — Building Quantum Circuits

```python
from quantum_simulator import QuantumCircuit

# Two-qubit Bell state circuit
qc = QuantumCircuit(2)           # create 2-qubit register
qc.h(0)                          # Hadamard on qubit 0
qc.cnot(0, 1)                    # CNOT: control=q0, target=q1
qc.measure_all()                 # measure both qubits

print(qc)
# QuantumCircuit(2 qubits)
# ────────────────────────────────────────
#   [00] H @ q0
#   [01] CNOT @ q0, q1
#   [M]  Measure: q0, q1
# ────────────────────────────────────────

# Chained gate application
qc2 = QuantumCircuit(3)
qc2.h(0).h(1).h(2)              # Hadamard on all 3 qubits
qc2.toffoli(0, 1, 2)            # Toffoli gate
qc2.measure(2)                   # measure only qubit 2

# Rotation gates
import math
qc3 = QuantumCircuit(1)
qc3.rx(math.pi / 4, 0)          # RX(π/4) on qubit 0
qc3.ry(math.pi / 2, 0)          # RY(π/2) on qubit 0
qc3.p(math.pi, 0)               # Phase shift P(π) on qubit 0
qc3.measure(0)
```

### 4 — Running the Simulator

```python
from quantum_simulator import QuantumCircuit, QuantumSimulator

sim = QuantumSimulator(seed=42)  # seeded for reproducibility

# --- Bell state ---
qc = QuantumCircuit(2)
qc.h(0).cnot(0, 1).measure_all()
counts = sim.run(qc, shots=1024)
print(counts)   # {'00': ~512, '11': ~512}

# --- GHZ state (3-qubit entanglement) ---
ghz = QuantumCircuit(3)
ghz.h(0).cnot(0, 1).cnot(0, 2).measure_all()
print(sim.run(ghz, shots=2048))  # {'000': ~1024, '111': ~1024}

# --- Inspect the raw statevector ---
qc_sv = QuantumCircuit(2)
qc_sv.h(0).cnot(0, 1)
sv = sim.get_statevector(qc_sv)
print(sv)   # [0.707+0j, 0, 0, 0.707+0j]

# --- Probability distribution (no measurement collapse) ---
probs = sim.get_probabilities(qc_sv)
print(probs)   # {'00': 0.5, '11': 0.5}
```

---

## Example Algorithms

### Bell State (2-qubit Entanglement)

```python
from quantum_simulator import QuantumCircuit, QuantumSimulator

sim = QuantumSimulator(seed=0)
qc  = QuantumCircuit(2)
qc.h(0).cnot(0, 1).measure_all()
print(sim.run(qc, shots=1024))
# Expected: {'00': ~512, '11': ~512}
```

The Bell state |Φ+⟩ = (|00⟩ + |11⟩) / √2 demonstrates **quantum
entanglement**: measuring one qubit instantly determines the other.

### GHZ State (3-qubit Entanglement)

```python
qc = QuantumCircuit(3)
qc.h(0).cnot(0, 1).cnot(0, 2).measure_all()
print(sim.run(qc, shots=2048))
# Expected: {'000': ~1024, '111': ~1024}
```

The Greenberger–Horne–Zeilinger (GHZ) state
|GHZ⟩ = (|000⟩ + |111⟩) / √2 is the maximal 3-qubit entangled state.

### Quantum Superposition and Interference

```python
# Hadamard creates equal superposition, then X destroys it
qc = QuantumCircuit(1)
qc.h(0).h(0)   # H·H = I → back to |0⟩
sv = sim.get_statevector(qc)
print(sv)   # [1+0j, 0+0j]  — constructive / destructive interference
```

---

## Running Tests

```bash
pytest tests/test_quantum_simulator.py -v
```

72 tests covering: Qubit initialisation, Bloch sphere, measurement collapse,
all gate unitarity, gate action, circuit validation, Bell state, GHZ state,
Toffoli, SWAP, partial measurement, rotation gates, and determinism.

---

## References

1. **Nielsen, M. A. & Chuang, I. L.** (2010). *Quantum Computation and
   Quantum Information* (10th anniversary ed.). Cambridge University Press.
   — Foundational text for the qubit model, gate formalism, and circuit model
   (Chapters 1–4).

2. **IBM Quantum Documentation — "Bits, gates, and circuits"**.
   <https://qiskit.qotlabs.org/learning/courses/utility-scale-quantum-computing/bits-gates-and-circuits>
   — Industry reference for qubit and gate definitions.

3. **Javadi-Abhari, A. et al.** (2024). Quantum computing with Qiskit.
   *arXiv:2405.08810*. <https://arxiv.org/abs/2405.08810>
   — Architecture of a production quantum SDK used to inform the
   circuit/simulator split.

4. **Krantz, P. et al.** (2019). A quantum engineer's guide to
   superconducting qubits. *Applied Physics Reviews*, 6(2), 021318.
   <https://doi.org/10.1063/1.5089550>
   — Physical basis for qubit representations and Bloch-sphere
   parameterisation.

5. **Barenco, A. et al.** (1995). Elementary gates for quantum computation.
   *Physical Review A*, 52(5), 3457–3467.
   <https://doi.org/10.1103/PhysRevA.52.3457>
   — Gate-set universality proofs; basis for Toffoli and Fredkin gate
   inclusion.

6. **Steiger, D. S., Häner, T., & Troyer, M.** (2018). ProjectQ: an open
   source software framework for quantum computing. *Quantum*, 2, 49.
   <https://doi.org/10.22331/q-2018-01-31-49>
   — Tensor-network contraction approach for statevector simulation.

7. **IBM Quantum Documentation — Simulators overview**.
   <https://docs.quantum.ibm.com/guides/simulators>
   — Design principles for statevector simulation backends.

8. **Bloch, F.** (1946). Nuclear Induction. *Physical Review*, 70(7–8),
   460–474. <https://doi.org/10.1103/PhysRev.70.460>
   — Original Bloch sphere paper.
