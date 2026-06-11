# QuantumComputer

A REPL-style quantum computer simulator with easy usage and high versatility.

**Feature highlights**
- **Easy-to-use** interface to quickly implement algorithms
- Switch betwween simulating **state vector and density matrix**
- Track the **effective quantum channel** as Kraus operators or Choi matrix
- **Add and remove qubits** mid-circuit
- Decohering measurement
- A range of **quantum info tools** for state analysis

**Example usage**
```python
from QuantumComputer import QC

state, energy = range(2), range(2, 5)
unitary = QC().x(0).cx(0,1).U   # operator tracking is active by default for small system

qc = QC('00+11')
qc.add(energy)                  # Add ancilla qubits
qc.pe(unitary, state, energy)
qc.measure(energy, collapse=False)
print(qc[0,1])                  # Look at the state RDM
ops = qc.get_operators()        # Obtain the effective channel
print("Number of Kraus operators:", len(ops))
```
```
[[ 0.25 +0.j -0.   +0.j  0.125+0.j  0.125+0.j]
 [-0.   -0.j  0.25 +0.j  0.125-0.j  0.125-0.j]
 [ 0.125-0.j  0.125+0.j  0.25 +0.j  0.   +0.j]
 [ 0.125-0.j  0.125+0.j  0.   -0.j  0.25 -0.j]]
Number of Kraus operators: 4
```

### Installation

```bash
pip install .
```

`scipy` is optional, but recommended for sparse matrices.

### Initialization from arbitrary states

```python
qc = QC(2)             # 2 qubits in all-zero state |00>
qc = QC([1, 0, 0, 1])  # Provide an array / ndarray
qc = QC('00 + 2*11')   # String notation
```

### State handling

```python
qc[0:3]              # reduced density matrix of the given qubits
qc.get_state()       # state vector or density matrix
qc.to_dm()           # convert to density matrix
qc.to_ket('purify')  # convert back to ket (collapse by default)
qc.purify()          # equivalent to above

qc.add([2,3])        # extend the state with new qubits, all-zero state by default
qc.remove([2,3])     # remove qubits, converts to density matrix by default
QC('abcdefg')        # qubits may be named with strings
QC(['a1', 'a2'])
```

### State evolution

| Gate |    Method    |   Gate   |     Method     |
|------|--------------|----------|----------------|
| X    | `.x(q)`      | Pauli-Z  | `.z(q)`        |
| Y    | `.y(q)`      | Hadamard | `.h(q)`        |
| S    | `.s(q)`      | T        | `.t(q)`        |
| S†   | `.sdg(q)`    | T†       | `.tdg(q)`      |
| Rx   | `.rx(θ,q)`   | Ry       | `.crx(θ, q)`   |
| CNOT | `.cx(c,t)`   | CZ       | `.cz(c, t)`    |
| C-U  | `.c(U,c,t)`  | Neg. control | `.c(U, c, t, True)` |
| NCX  | `.nx(c,t)`   | Toffoli  | `.ccx(c1, c2, t)` |
| SWAP | `.swap(a,b)` | CSWAP    | `.cswap(c, a, b)` |

... and much more.

```python
qc(U, qubits)                 # arbitrary set of Kraus operators
qc('CX @ HI')                 # string notation for unitaries
qc.apply_qiskit_circuit(qc2)  # apply a circuit defined by a qiskit object

qc.measure()                  # collapse, return basis state as binary string
qc.measure([0,1], obs=XX)     # `obs` allows other observables than Z basis
qc.measure(collapse=False)    # decoherence (no collapse) → auto-conversion to density matrix
qc.decohere()                 # shorthand for above
qc.sample([0,1], shots=1000)  # sample without collapse
qc.probs_pp()                 # pretty-print probabilities
```

Use the context maanger `qc.observable(obs, qubits)` to change the default observable.

### Basic algorithms

```python
qc.qft([0,1,2])          # Quantum Fourier Transform
qc.iqft([0,1,2])         # Inverse QFT
qc.pe(U, state, energy)  # Phase estimation
```

### Operator tracking

```python
qc = QC(track_operators=False)  # deactivate operator tracking
qc = QC(auto_compress=False)    # deactivate auto compression of Kraus operators
qc.compress_operators()         # trigger compression of Kraus operators
qc.choi_matrix()                # obtain choi matrix representation
qc.to_choi()                    # convert to choi matrix tracking

qc = QC(2, as_superoperator=True)  # track as choi matrix
qc.cx(0, 1).x(0)
qc.choi_matrix()
qc.get_operators()                 # obtain Kraus operators
```

The Choi matrix representation may show much high performances when `scipy` is installed (using its sparse arrays).

### Noise

```python
qc.noise('depolarizing', 0, p=0.01)
qc.noise('amplitude_damping', [0,1], p=0.05)
qc.noise('bitflip', 0, p=0.1)

# Noise schedule (add noise after initialization and each gate and measurement)
qc = QC(2, noise_schedule=lambda qubits, process, qc: qc.noise('depolarizing', qubits, p=0.01))
```

Available default noise models: `depolarizing`, `amplitude_damping`, `phase_damping`, `bitflip`, `phaseflip`, `zdrift`. See `create_benchmark_noise_scheduler` for a scheduler factory. The context manager `qc.no_noise()` can be used to temporarily disable the noise schedule.

### Quantum info

```python
qc.von_neumann_entropy(0)        # entropy of subsystem
qc.entanglement_entropy(0)       # entanglement entropy
qc.purity(0)                     # purity of subsystem
qc.schmidt_decomposition([0,1])  # Schmidt coefficients & vectors
qc.mutual_information(0, 1)      # mutual information
qc.ev(Z, 0)                      # expectation value of Z on qubit 0
qc.std(X, 0)                     # standard deviation of X on qubit 0
```

### Test

```bash
python test.py
```

### License

Copyright (C) 2026 Rae Müller

Licensed under the GNU GPL v3 or later — see [`LICENSE`](LICENSE).
