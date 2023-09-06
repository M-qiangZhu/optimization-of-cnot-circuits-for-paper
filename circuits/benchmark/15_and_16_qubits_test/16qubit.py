from qiskit import QuantumCircuit


def extract_cnot_subcircuits(qasm_file_path):
    # 读取QASM文件
    circuit = QuantumCircuit.from_qasm_file(qasm_file_path)

    # 提取CNOT子线路
    cnot_subcircuits = QuantumCircuit(circuit.num_qubits)

    for instruction, qargs, _ in circuit:
        if instruction.name == 'cx':  # CNOT门
            cnot_subcircuits.append(instruction, qargs)

    return cnot_subcircuits



if __name__ == '__main__':
    qasm = 'E:/python/Distributed_quantum_circuits_scheduling/qasm/urf6_160.qasm'
    cnot_subcircuits = extract_cnot_subcircuits(qasm)
    cnot_qasm_file = cnot_subcircuits.qasm()

    qasm_file_path = "urf6_160.qasm"

    with open(qasm_file_path, "w") as qasm_file:
        qasm_file.write(cnot_qasm_file)