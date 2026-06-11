import numpy as np
from QuantumComputer import QuantumComputer as QC
from utils import *

def _test_QuantumComputer():
    # test basic functionality
    qc = QC(track_operators=False)  # remove() would turn it into density matrix mode, but we want to keep state vector here
    qc.x(0)
    assert unket(qc.get_state()) == '1', f"{unket(qc.get_state())} ≠ '1'"
    qc.cx(0, 1)
    assert unket(qc.get_state()) == '11', f"{unket(qc.get_state())} ≠ '11'"
    qc.reset()
    assert unket(qc.get_state()) == '00', f"{unket(qc.get_state())} ≠ '00'"
    qc.init('00 + 11')
    outcome = qc.resetv([1])
    assert outcome in ['0', '1']
    qc.resetv(0)
    assert unket(qc.get_state()) == '00', f"{unket(qc.get_state())} ≠ '00'"
    qc.h()
    qc.x(2)
    qc.init(2, [0,1])
    assert unket(qc.get_state()) == '101', f"{unket(qc.get_state())} ≠ '101'"
    qc.z([0,2])  # noop
    qc.swap(1,2)
    assert unket(qc.get_state()) == '110', f"{unket(qc.get_state())} ≠ '110'"
    qc.reset(1)
    assert unket(qc.get_state()) == '100', f"{unket(qc.get_state())} ≠ '100'"
    qc.remove([0,2])
    assert unket(qc.get_state()) == '0', f"{unket(qc.get_state())} ≠ '0'"
    qc.reset(1)
    qc.x(2)
    result = qc.measure('all')
    assert result == '01'
    qc = QC(2)
    qc.reset([1,2])
    assert qc.n == 3, f"qc.n = {qc.n} ≠ 3"
    qc.copy()
    qc1 = QC(state=1)
    assert qc1.qubits == (0,), f"{qc1.qubits} ≠ (0,)"

    k1 = random_ket(2)
    k2 = random_ket(3)
    qc = QC(np.kron(k1, k2))
    qc.remove([0,1])
    assert np.allclose(abs(qc.get_state().conj() @ k2), 1), f"{qc.get_state()} ≠ {k2}"

    qc = QC(2)
    qc[:2] = random_dm(2)
    assert qc.n == 2, qc._qubits
    assert qc.get_state().shape == (4,4), qc._state.shape
    qc.remove(1)
    qc[-1] = '0'
    assert qc.is_pure()
    assert qc._operators[0].shape == (2,4), qc._operators

    # Heisenberg uncertainty principle
    qc = QC(1, 'random')
    assert qc.std(X) * qc.std(Z) >= abs(qc.ev(1j*(X@Z - Z@X)))/2

    # test functions
    global XX, YY
    qc = QC(2)
    qc.h(0)
    qc.cx(0, 1)
    assert np.allclose(qc.get_state(), normalize([1,0,0,1]))  # Bell state |00> + |11>
    qc(I, 0)  # let it reshape to [2,2]
    assert np.allclose(qc[0], I/2)
    assert np.allclose(qc[1], I/2)
    assert np.isclose(qc.purity(0), 0.5)
    p = qc.probs(obs=XX)  # Bell basis is the basis of XX / YY observable -> deterministic outcome
    assert np.isclose(entropy(p), 0), f"p = {p}"
    with qc.observable(YY):  # test context manager
        p = qc.probs()
        assert np.isclose(entropy(p), 0), f"p = {p}"
    U_expected = parse_unitary('CX @ HI')  # check generated unitary
    assert np.allclose(qc.get_unitary(), U_expected), f"Incorrect unitary:\n{qc.get_unitary()}\n ≠\n{U_expected}"

    assert np.isclose(qc.std(X, 0), 1)
    assert np.isclose(qc.std(Y, 0), 1)
    assert np.isclose(qc.std(Z, 0), 1)
    assert np.isclose(qc.ev(X, 0), 0)
    assert np.isclose(qc.ev(Y, 0), 0)
    assert np.isclose(qc.ev(Z, 0), 0)

    assert np.isclose(qc.entanglement_entropy(1), 1)
    assert np.isclose(qc.correlation(0, 1, Z, Z), 1)
    S = qc.schmidt_decomposition(0, coeffs_only=True)
    assert np.allclose(S, [fs2, fs2])
    assert np.isclose(qc.mutual_information(0, 1), 2)

    # test density matrix
    qc.decohere(0)
    assert np.isclose(qc.mutual_information(0, 1), 1)
    assert qc.is_matrix_mode()
    assert np.allclose(qc.get_state(), (dm('00') + dm('11'))/2)
    qc.purify()
    assert np.allclose(qc.get_state(), ket('000 + 111'))  # purification only needs one ancilla qubit in this case
    assert not qc.is_matrix_mode()
    assert qc[:].shape == (8,8)
    assert not qc.is_unitary()
    assert qc.is_isometric()
    qc.to_dm()
    rdm0 = qc.get_state([0], collapse=True)
    assert rdm0.shape == (2,2), f"{rdm0.shape} ≠ (2,2)"
    qc.remove([0])
    qc.remove(1)
    QC(random_dm(3, rank=3)).purify()

    qc = QC('00 + 01')
    assert np.allclose(qc.schmidt_coefficients([1]), [1])
    rdm0 = qc.to_dm().get_state(0, collapse=True)
    assert_dm(rdm0, n=1)
    qc = QC('00 + 01 + 11')
    assert np.isclose(trace_product(qc[0], qc[0]), 7/9)
    assert np.isclose(trace_product(qc[1], qc[1]), 7/9)
    qc = QC(3, 'random')
    assert not qc.is_matrix_mode()
    assert np.isclose(np.sum(qc.schmidt_coefficients([0])**2), 1)

    assert qc.is_unitary()
    U = random_unitary(2**2)
    qc(U, [0,2])
    assert qc.is_unitary([0,1,2])
    assert not qc.is_unitary([0,1])
    assert qc.is_unitary([0,2])
    # E = random_channel(1)
    # qc(E, 1)
    # assert not qc.is_unitary(1)  # not implemented yet
    # assert qc.is_unitary([0,2])

    # more complex test
    qc = QC(15)
    U = parse_unitary('XYZCZYX')
    qc(U, choice(qc.qubits, 7, False))
    qc.remove([5,7,6,10,2])
    U = random_unitary(2**5)
    qc(U, choice(qc.qubits, 5, False))
    assert qc.n == 10

    # test phase estimation
    H = ph(f'{1/8}*(IZ + ZI + II)')
    U = matexp(2j*np.pi*H)
    assert np.isclose(np.trace(H @ dm('00')), float_from_binstr('.011'))  # 00 is eigenstate with energy 0.375 = '011'
    state_qubits = ['s0', 's1']
    qc = QC(state_qubits)
    E_qubits = ['e0', 'e1', 'e2']
    qc.pe(U, state_qubits, E_qubits)
    res = qc.measure(E_qubits)
    assert res == '011', f"measurement result was {res} ≠ '011'"

    qc.remove('s0')
    assert np.allclose(qc.get_state(), ket('0011'))
    qc.remove('all')
    qc.add(0)

    # test schmidt decomposition
    qc = QC(5, 'random')
    bip = choice([i for i,o in bipartitions(range(5))])
    S, U, V = qc.schmidt_decomposition(bip)
    # check RDM for subsystem A
    rho_expect = qc[bip]
    rho_actual = np.sum([l_i**2 * np.outer(A_i, A_i.conj()) for l_i, A_i in zip(S, U)], axis=0)
    assert np.allclose(rho_actual, rho_expect), f"rho_expect - rho_actual = {rho_expect - rho_actual}"
    # check RDM for subsystem B
    rho_expect = qc[[i for i in qc.qubits if i not in bip]]
    rho_actual = np.sum([l_i**2 * np.outer(B_i, B_i.conj()) for l_i, B_i in zip(S, V)], axis=0)
    assert np.allclose(rho_actual, rho_expect), f"rho_expect - rho_actual = {rho_expect - rho_actual}"

    # test density matrix
    qc = QC('0100 + 1010')
    assert np.isclose(qc.entanglement_entropy(3), 0)
    qc.remove(3)
    assert not qc.is_matrix_mode()
    qc.remove(1)
    assert qc.is_matrix_mode()
    qc = QC('0100 + 1010')
    qc.KEEP_VECTOR = False
    qc.remove(3)
    assert qc.is_matrix_mode()
    assert np.isclose(qc.entanglement_entropy(1), 1)
    qc.remove(1)
    qc.x(2)
    assert np.allclose(qc.get_state(), (dm('01') + dm('10'))/2)
    qc.to_ket(kind='sample')

    # test kraus operators
    qc = QC(2)
    qc.h(0)
    qc.noise('depolarizing', 0)
    ops = qc.get_operators()
    assert len(ops) == 4
    assert_kraus(ops, n=(2,2))
    assert np.allclose(QC(2)(ops)[:], qc[:])

    qc.cx(0,1)
    qc.measure(0, collapse=False)
    ops = qc.get_operators()
    assert len(ops) == 8, len(ops)
    assert np.allclose(qc[:], QC(2)(ops)[:])
    qc.reset(0, collapse=False)
    ops = qc.get_operators()
    assert np.allclose(qc[:], QC(2)(ops)[:])

    # test choi matrix / superoperator
    qc.compress_operators()
    assert len(qc.get_operators()) == 4

    # if "scipy" in sys.modules:
    qc = QC(1)
    qc.to_dm()  # just to mute the warning, but `to_choi` would/should do this automatically
    qc.to_choi()
    assert qc.is_matrix_mode()
    choi_actual = qc.choi_matrix()
    if hasattr(choi_actual, 'toarray'):
        choi_actual = choi_actual.toarray()
    choi_expected = np.array([[1, 0, 0, 1],
                              [0, 0, 0, 0],
                              [0, 0, 0, 0],
                              [1, 0, 0, 1]])
    assert np.allclose(choi_actual, choi_expected), f"choi_actual = {choi_actual}\nchoi_expected = {choi_expected}"
    qc(X)
    choi_actual = qc.choi_matrix()
    if hasattr(choi_actual, 'toarray'):
        choi_actual = choi_actual.toarray()
    choi_expected = np.array([[0, 0, 0, 0],
                              [0, 1, 1, 0],
                              [0, 1, 1, 0],
                              [0, 0, 0, 0]])
    assert np.allclose(choi_actual, choi_expected), f"choi_actual = {choi_actual}\nchoi_expected = {choi_expected}"
    ops = noise_models['depolarizing'](0.1)
    qc(ops)
    qc.remove(0)
    qc.add(0)
    qc = QC(2, as_superoperator=True)
    U = random_unitary(2**2)
    qc(U)
    C1 = qc.choi_matrix()
    qc = QC(2, as_superoperator=False)
    qc(U)
    C2 = qc.choi_matrix()
    assert np.allclose(C1, C2), f"C1 = {C1}\nC2 = {C2}"
    qc(X, 1)  # partial application

    # test n_out > n_in
    qc = QC(1)
    qc.init(0, [1], collapse=False)
    qc(random_unitary(2**2))
    qc.measure([1], collapse=False)
    qc.compress_operators()

    if "qiskit" in sys.modules:
        warnings.filterwarnings("ignore")  # ignore deprecation warnings
        from qiskit.circuit.library import phase_estimation, RYGate, QFTGate
        U = QuantumCircuit(2)  # RYGate(np.pi/12) fails for pi/(2**(n-2)) for n >= 3
        U.rx(0.1, 0)
        n = randint(1,8)
        s_reg, e_reg = [0,1], list(range(2, n+2))
        U_PE1 = get_unitary(phase_estimation(n, U))
        U_PE1 = reorder_qubits(U_PE1, (e_reg + s_reg)[::-1])  # qiskit outputs qubits backwards
        qc2 = QC(2+n, track_operators=True).pe(get_unitary(U), s_reg[::-1], e_reg)
        U_PE2 = qc2.get_unitary()
        assert np.allclose(U_PE1, U_PE2)

        U_QFT1 = get_unitary(QFTGate(n))
        U_QFT2 = get_unitary(QC(n, track_operators=True).qft(range(n)))
        assert np.allclose(U_QFT1, U_QFT2)

        def check_apply_qiskit_circuit(qc, qubits='all'):
            U1 = reverse_qubit_order(get_unitary(qc))

            for elementary in [True, False]:
                qc2 = QC()
                qc2.apply_qiskit_circuit(qc, qubits, include_phases=True, elementary_only=elementary)
                U2 = get_unitary(qc2)
                assert np.allclose(U1, U2), np.linalg.norm(U1 - U2)

            qc2 = QC()
            qc2.apply_qiskit_circuit(qc, include_phases=False, elementary_only=True)
            U3 = get_unitary(qc2)
            qc2 = QC()
            qc2.apply_qiskit_circuit(qc, include_phases=False, elementary_only=False)
            U4 = get_unitary(qc2)
            assert np.allclose(U3, U4), np.linalg.norm(U3 - U4)

        qc1 = QuantumCircuit(2)
        qc1.u(0.1, 0.2, 0.3, 0)
        qc1.h(0)
        qc1.cx(0,1)
        check_apply_qiskit_circuit(qc1)

        qc1 = QuantumCircuit(2, 4)
        qc1.x(0)  # state '10'
        qc1.measure([0,1], [0,1])
        qc1.x([0,1])  # state '01'
        qc1.measure([0,1], [2,3])
        qc2 = QC()
        outcome = qc2.apply_qiskit_circuit(qc1)
        assert outcome == '1001', f"Outcome was: {outcome}"

        from qiskit.circuit.library import QFT
        check_apply_qiskit_circuit(QFT(4, do_swaps=False))
        check_apply_qiskit_circuit(QFT(4, do_swaps=True))
        warnings.filterwarnings("default")

if __name__ == '__main__':
    print_header('QuantumComputer')
    print("Running _test_QuantumComputer ... ", end="")
    _test_QuantumComputer()
    print("Test succeeded!")
    test_all()
    print("All tests have successfully passed!")
