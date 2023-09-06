import sys
from .my_circuit import Circuit, gate_types
from .my_linalg import Mat2

import numpy as np

# debug = True
debug = False


class CNOT_tracker(Circuit):
    def __init__(self, n_qubits, **kwargs):
        super().__init__(n_qubits, **kwargs)
        self.matrix = Mat2(np.identity(n_qubits, dtype=np.int32).tolist())  # 生成一个恒等矩阵/单位矩阵
        self.row_perm = np.arange(n_qubits)
        self.col_perm = np.arange(n_qubits)
        self.n_qubits = n_qubits

    def count_cnots(self):
        return len([g for g in self.gates if hasattr(g, "name") and g.name == "CNOT"])

    def row_add(self, q0, q1):
        self.add_gate("CNOT", q0, q1)
        self.matrix.row_add(q0, q1)

    def col_add(self, q0, q1):
        self.prepend_gate("CNOT", q1, q0)
        self.matrix.col_add(q0, q1)

    @staticmethod
    def get_metric_names():
        return ["n_cnots"]

    def gather_metrics(self):
        metrics = {}
        metrics["n_cnots"] = self.count_cnots()
        return metrics

    def prepend_gate(self, gate, *args, **kwargs):
        """Adds a gate to the circuit. ``gate`` can either be 
        an instance of a :class:`Gate`, or it can be the name of a gate,
        in which case additional arguments should be given.

        Example::

            circuit.add_gate("CNOT", 1, 4) # adds a CNOT gate with control 1 and target 4
            circuit.add_gate("ZPhase", 2, phase=Fraction(3,4)) # Adds a ZPhase gate on qubit 2 with phase 3/4
        """
        if isinstance(gate, str):
            gate_class = gate_types[gate]
            gate = gate_class(*args, **kwargs)
        self.gates.insert(0, gate)

    def to_qasm(self):
        qasm = super().to_qasm()
        initial_perm = "// Initial wiring: " + str(self.row_perm.tolist())
        end_perm = "// Resulting wiring: " + str(self.col_perm.tolist())
        return '\n'.join([initial_perm, end_perm, qasm])

    @staticmethod
    def from_circuit(circuit):
        new_circuit = CNOT_tracker(circuit.qubits, name=circuit.name)
        new_circuit.gates = circuit.gates
        new_circuit.update_matrix()

        debug and print("new_circuit如下:")
        debug and print(new_circuit.matrix)
        print("new_circuit如下:")
        print(new_circuit.matrix)

        return new_circuit

    def update_matrix(self):
        self.matrix = Mat2(np.identity(self.n_qubits))  # 返回一个 n_qubits * n_qubits 的矩阵
        for gate in self.gates:
            if hasattr(gate, "name") and gate.name == "CNOT":  # hasattr() 函数用于判断对象是否包含对应的属性
                self.matrix.row_add(gate.control, gate.target)
            else:
                print("Warning: CNOT tracker can only be used for circuits with only CNOT gates!")

    @staticmethod
    def from_qasm_file(fname):
        circuit = Circuit.from_qasm_file(fname)
        return CNOT_tracker.from_circuit(circuit)


