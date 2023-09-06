from qiskit import QuantumCircuit, Aer, execute

# 辅助函数用于创建给定函数的Oracle电路
def create_oracle(f, n):
    oracle_circuit = QuantumCircuit(n+1)
    # 将函数f传递给量子计算机
    if f(0) == 1:
        oracle_circuit.x(n)  # 将量子比特|n⟩设置为|1⟩
    for i in range(n):
        if f(1 << i) == 1:
            oracle_circuit.cx(i, n)
    return oracle_circuit

# Deutsch-Jozsa算法
def deutsch_josza_algorithm(f, n):
    qc = QuantumCircuit(n+1, n)
    # 将最后一个量子比特设置为|1⟩状态
    qc.x(n)
    qc.h(range(n+1))
    # 应用Oracle
    oracle = create_oracle(f, n)
    qc.compose(oracle, range(n+1), inplace=True)  # 在这里添加了CNOT门
    qc.h(range(n))
    qc.measure(range(n), range(n))
    return qc

# 例子：常量函数
def constant_function(x):
    count_ones = bin(x).count('1')
    return count_ones % 2

if __name__ == '__main__':
    # 实际生成的量子线路有n+1个量子位
    n_qubits = 126
    dj_circuit = deutsch_josza_algorithm(constant_function, n_qubits)

    # # 模拟量子计算机
    # backend = Aer.get_backend('qasm_simulator')
    # result = execute(dj_circuit, backend=backend).result()
    # counts = result.get_counts(dj_circuit)
    #
    # print("结果:", counts)

    print(dj_circuit)
    qc_qasm = dj_circuit.qasm()  # str类型
    print(qc_qasm)

    qasm_file_path = "Deutsch_Josza-"+str(n_qubits+1)+"qubits.qasm"

    with open(qasm_file_path, "w") as qasm_file:
        qasm_file.write(qc_qasm)

