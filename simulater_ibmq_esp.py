# -*- coding: utf-5 -*-

"""
    @Author kungfu
    @Date 2023/5/24 10:08
    @Describe
    @Version 1.0
"""
import time
from qiskit import IBMQ, transpile, execute
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator, Aer
from qiskit.tools.visualization import plot_histogram
from matplotlib import pyplot as plt
from qiskit.providers.fake_provider import FakeYorktown, FakeGuadalupe, FakeManhattan, FakeQuito, FakeKolkata, \
    FakeCairo, FakeAlmaden, FakeCasablanca, FakeManila, FakeJohannesburg, FakeMelbourne, FakeRueschlikon, FakeTokyo, \
    FakeLagos


def load_qasm_file(file_path):
    with open(file_path, "r") as f:
        qasm_str = f.read()
    return QuantumCircuit.from_qasm_str(qasm_str)

def count_cnot_gates(circuit):
    cnot_count = 0
    for gate in circuit:
        if gate[0].name == "cx":  # CNOT gate
            cnot_count += 1
    return cnot_count

def count_layers(circuit):
    return circuit.depth()


def new_str0(n):
    str = ''
    for i in range(n):
        str += '0'
    return str

def calculate_esp_tvd(ideal_counts, noisy_counts):
    # 计算估计成功概率（ESP）
    esp = sum(min(ideal_counts.get(k, 0), noisy_counts.get(k, 0)) for k in ideal_counts) / sum(ideal_counts.values())

    # 计算总方差距离（TVD）
    total_shots = sum(noisy_counts.values())
    ideal_probs = {k: v / sum(ideal_counts.values()) for k, v in ideal_counts.items()}
    noisy_probs = {k: v / total_shots for k, v in noisy_counts.items()}
    all_keys = set(ideal_probs.keys()).union(noisy_probs.keys())
    tvd = 0.5 * sum(abs(ideal_probs.get(k, 0) - noisy_probs.get(k, 0)) for k in all_keys)

    return esp, tvd



if __name__ == '__main__':
    start_time = time.time()
    N = 10
    Qubit = 20
    # IBM量子设备使用真实的噪声数据，使用存储在Qiskit Terra中的数据。这里使用的设备是'ibmq_Guadalupe '。

    # backend = FakeManhattan()  # 65
    # backend = FakeGuadalupe()  # 16
    # backend = FakeQuito()  # 5
    # backend = FakeCasablanca()  # 7
    # backend = FakeLagos() # 7
    # backend = FakeManila() # 5 线性
    # backend = FakeMelbourne()  #14
    # backend = FakeRueschlikon()
    backend = FakeTokyo()
    # backend = FakeAlmaden()  #20
    # backend = FakeKolkata() # 27
    # backend = FakeCairo()  # 27
    # backend = FakeTokyo()  # 20
    # 构建量子线路
    # file_list = ['cnt3-5_179', 'cnt3-5_180', 'inc_237', 'mlp4_245']
    # file_list = ['4mod5-v1_22', '4mod5-v1_24', 'mod5mils_65','alu-v0_27', 'alu-v3_35',  'alu-v4_37', '4gt13_92',  '4mod5-v1_23', 'decod24-v2_43',
    #               '4gt5_75', '4gt13_91',  'alu-v4_36', '4gt13_90', 'hwb4_49',  'mod10_171']

    # file_list = ['5_20_10%', '5_20_20%', '5_20_30%', '5_20_40%', '5_20_50%', '5_20_60%', '5_20_70%',
    #              '5_20_80%', '5_20_90%', '5_20_100%']
    # file_list = ['5_50_10%', '5_50_20%', '5_50_30%', '5_50_40%', '5_50_50%', '5_50_60%', '5_50_70%',
    #              '5_50_80%', '5_50_90%', '5_50_100%']
    # file_list = ['5_100_10%', '5_100_20%', '5_100_30%', '5_100_40%', '5_100_50%', '5_100_60%', '5_100_70%',
    #              '5_100_80%', '5_100_90%', '5_100_100%']
    # file_list = ['5_200_10%', '5_200_20%', '5_200_30%', '5_200_40%', '5_200_50%', '5_200_60%', '5_200_70%',
    #              '5_200_80%', '5_200_90%', '5_200_100%']
    # file_list = ['5_500_10%', '5_500_20%', '5_500_30%', '5_500_40%', '5_500_50%', '5_500_60%', '5_500_70%',
    #              '5_500_80%', '5_500_90%', '5_500_100%']
    # file_list = ['5_1000_10%', '5_1000_20%', '5_1000_30%', '5_1000_40%', '5_1000_50%', '5_1000_60%', '5_1000_70%',
    #              '5_1000_80%', '5_1000_90%', '5_1000_100%']
    # file_list = ['5_2000_10%', '5_2000_20%', '5_2000_30%', '5_2000_40%', '5_2000_50%', '5_2000_60%', '5_2000_70%',
    #              '5_2000_80%', '5_2000_90%', '5_2000_100%']
    # file_list = ['5_5000_10%', '5_5000_20%', '5_5000_30%', '5_5000_40%', '5_5000_50%', '5_5000_60%', '5_5000_70%',
    #              '5_5000_80%', '5_5000_90%', '5_5000_100%']
    # file_list = ['5_10000_10%', '5_10000_20%', '5_10000_30%', '5_10000_40%', '5_10000_50%', '5_10000_60%', '5_10000_70%',
    #              '5_10000_80%', '5_10000_90%', '5_10000_100%']

    # file_list = ['5_10','5_20','5_30','5_40','5_50','5_100','5_200','5_500','5_1000','5_2000','5_5000','5_10000','5_20000','5_50000','5_100000']

    # file_list = ['16_10', '16_20', '16_30', '16_40', '16_50', '16_100', '16_200', '16_500', '16_1000', '16_2000',
    #              '16_5000', '16_10000', '16_20000', '16_50000', '16_100000']

    # file_list = ['16_200_10%', '16_200_20%', '16_200_30%', '16_200_40%', '16_200_50%', '16_200_60%', '16_200_70%',
    #              '16_200_80%', '16_200_90%', '16_200_100%']

    # file_list = ['5_10','5_20','5_30','5_40','5_50','5_100','5_200','5_500','5_1000','5_2000','5_5000','5_10000','5_20000','5_50000','5_100000']
    # file_list = ['5_20000']
    # file_list = ['5_1000_71%', '5_1000_72%', '5_1000_73%', '5_1000_74%', '5_1000_74%', '5_1000_76%', '5_1000_77%',
    #              '5_1000_78%', '5_1000_79%']
    # file_list = ['14_1000_100%']

    # file_list = ['7_1000_81%', '7_1000_82%', '7_1000_83%', '7_1000_84%', '7_1000_85%', '7_1000_86%', '7_1000_87%',
    #              '7_1000_88%', '7_1000_89%']

    # file_list = ['7_1000_10%', '7_1000_20%', '7_1000_30%', '7_1000_40%', '7_1000_50%', '7_1000_60%', '7_1000_70%',
    #              '7_1000_80%', '7_1000_81%','7_1000_82%', '7_1000_83%', '7_1000_84%', '7_1000_85%', '7_1000_86%', '7_1000_87%',
    #              '7_1000_88%', '7_1000_89%','7_1000_90%','7_1000_100%']
    # file_list = ['20_1000_100%']

    # file_list = ['20_1000_91%',
    #              '20_1000_92%',
    #              '20_1000_93%', '20_1000_94%', '20_1000_95%', '20_1000_96%', '20_1000_97%', '20_1000_98%',
    #              '20_1000_99%']

    file_list = ['20_1000_98%', '20_1000_99%']

    for file in file_list:
        frequency_list = []
        cir_path = f"circuits/benchmark/20_qubit_circuit_include_single_gate/get_fed/{file}.qasm"
        # cir_path = f"E:/python/Distributed_quantum_circuits_scheduling/synthesis/16_qubit_circuit_without_single_gate/random/syn_tokyo/{file}_syn.qasm"
        # cir_path = f"E:/python/optimization-of-cnot-circuits-for-paper/result/01234-02134/qasm-trans/mod10_171_eli.qasm"
        # print(f"cir_path : {cir_path}")
        circuit = QuantumCircuit.from_qasm_file(cir_path)

        print(file)
        # circuit.draw('mpl', filename=f"{file}-circuit.png")
        cnot_gate_count = count_cnot_gates(circuit)
        layer_count = count_layers(circuit)

        print("未编译CNOT门数：", cnot_gate_count)
        # print("未编译总门数：",sum(circuit.count_ops().values()))
        print("未编译线路层数：", layer_count)
        # print(layer_count-1)
        # Transpile the ideal circuit to a circuit that can be directly executed by the backend

        # circuit.measure_all()

        transpiled_circuit = transpile(circuit, backend)
        # print(transpiled_circuit)
        print("CNOT门数：", count_cnot_gates(transpiled_circuit))
        print("线路层数：", count_layers(transpiled_circuit))

        transpiled_circuit.measure_all()

        simulator = Aer.get_backend('qasm_simulator')
        ideal_result = execute(transpiled_circuit, simulator, shots=2048).result()
        ideal_counts = ideal_result.get_counts()

        # Run the transpiled circuit using the simulated fake backend
        job = backend.run(transpiled_circuit,shots=2048)
        # print(job.result())
        noisy_counts = job.result().get_counts()

         # 计算ESP和TVD
        esp, tvd = calculate_esp_tvd(ideal_counts, noisy_counts)

        print(f"Estimated Success Probability (ESP): {esp}")
        print(f"Total Variance Distance (TVD): {tvd}")

    end_time = time.time()
    execution_time = end_time - start_time

    print("Execution Time:", execution_time, "seconds")
