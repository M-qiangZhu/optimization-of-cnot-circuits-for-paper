from qiskit import QuantumCircuit
import re

from circuits_synthetic_without_HamiltonPath_backup import eli_of_ibmq_quatio, eli_of_ibmq_guadalupe, eli_of_ibmq_almaden, eli_of_ibmq_tokyo, eli_of_ibmq_lagos, eli_of_ibmq_melbourne

# 存储qasm文件的地址 文件名cnot_subcircuit.qasm
qasm_address = './cnot_subcircuit.qasm'


def get_data(str):
    pattern = re.compile("[\d]+")
    result = re.findall(pattern, str)
    return result


'''
读取qasm文件
'''


def converter_circ_from_qasm(input_file_name):
    gate_list = []
    qbit = 0  # 量子位
    qasm_file = open(input_file_name, 'r')
    iter_f = iter(qasm_file)
    reserve_line = 0
    num_line = 0
    for line in iter_f:  # 遍历文件，一行行遍历，读取文本
        num_line += 1
        if num_line <= reserve_line:
            continue
        else:
            if 'qreg' in line:
                qbit = get_data(line)[0]
            if line[0:1] == 'x' or line[0:1] == 'X':
                '''获取X门'''
                x = get_data(line)
                x_target = x[0]
                listSingle = ['x', int(x_target)]
                gate_list.append(listSingle)
            if line[0:1] == 'h' or line[0:1] == 'H':
                '''获取h门'''
                x = get_data(line)
                x_target = x[0]
                listSingle = ['h', int(x_target)]
                gate_list.append(listSingle)

            if line[0:2] == 'rx' or line[0:1] == 'RX':
                '''获取rx门'''

                listSingle = [line[:line.index(')') + 1], int(line[line.index('[') + 1:line.index(']')])]
                gate_list.append(listSingle)
            if line[0:2] == 'ry' or line[0:1] == 'RY':
                '''获取ry门'''

                listSingle = [line[:line.index(')') + 1], int(line[line.index('[') + 1:line.index(']')])]
                gate_list.append(listSingle)
            if line[0:2] == 'rz' or line[0:1] == 'RZ':
                '''获取rz门'''

                listSingle = [line[:line.index(')') + 1], int(line[line.index('[') + 1:line.index(']')])]
                gate_list.append(listSingle)

            if line[0:1] == 't' or line[0:1] == 'T':
                if line[0:3] == 'tdg' or line[0:1] == 'TDG':
                    '''获取tdg门'''
                    x = get_data(line)
                    x_target = x[0]
                    listSingle = ['tdg', int(x_target)]
                    gate_list.append(listSingle)
                else:
                    '''获取t门'''
                    x = get_data(line)
                    x_target = x[0]
                    listSingle = ['t', int(x_target)]
                    gate_list.append(listSingle)
            if line[0:1] == 'S' or line[0:1] == 's':
                '''获取s门'''
                x = get_data(line)
                x_target = x[0]
                listSingle = ['s', int(x_target)]
                gate_list.append(listSingle)

            if line[0:2] == 'CX' or line[0:2] == 'cx':
                '''获取CNOT'''
                cnot = get_data(line)
                cnot_control = cnot[0]
                cnot_target = cnot[1]
                listSingle = [int(cnot_control), int(cnot_target)]
                gate_list.append(listSingle)

    return qbit, gate_list


#
def split_list(input_list):
    # 当前的子序列
    current_sequence = []

    # 遍历输入的每个元素
    for item in input_list:
        # 如果当前元素的第一个元素是整数
        if isinstance(item[0], int):
            # 如果当前子序列不为空且子序列的第一个元素是字符
            if current_sequence and isinstance(current_sequence[0][0], str):
                # 将当前子序列添加到字符对列表中
                print("连续单门:", current_sequence)  # 输出当前字符序列
                single_gate_qasm = generate_single_gate_circuit(current_sequence)
                print(single_gate_qasm)
                ##############################################################
                # 这地方根据初始映射方式，更新single_gate_qasm  {0: 0, 1: 3, 2: 3, 3: 1, 3: 2}
                # 16qubits {0: 0, 1: 1, 2: 2, 3: 3, 3: 3, 5: 5, 6: 6, 7: 7, 5: 9, 9: 5, 10: 10, 11: 11, 12: 14, 13: 1 3, 14: 12, 15: 15}
                # 7qubits {0: 0, 1: 2, 2: 1, 3: 3, 3: 3, 5: 5, 6: 6}
                # 20qubits almaden {0: 0, 1: 1, 2: 2, 3: 3, 3: 3, 5: 5, 6: 6, 7: 7, 5: 5, 9: 9, 10: 10, 11: 11, 12: 12, 13: 14, 14: 13, 15: 15, 16: 16, 17: 17, 18: 18, 19: 19}
                # 20qubits tokyo {0: 0, 1: 1, 2: 2, 3: 3, 3: 3, 5: 9, 6: 5, 7: 7, 5: 6, 9: 5, 10: 10, 11: 11, 12: 12, 13: 13, 14: 18, 15: 14, 16: 15, 17: 16, 18: 17, 19: 19}
                # 14qubits melbourne {0: 0, 1: 1, 2: 2, 3: 3, 3: 3, 5: 5, 6: 6, 7: 7, 5: 5, 9: 9, 10: 10, 11: 11, 12: 12, 13: 13}
                ##############################################################
                # 重置当前子序列
                current_sequence = []
            # 将当前元素添加到当前子序列中
            current_sequence.append(item)
        else:
            # 如果当前元素的第一个元素是字符
            if current_sequence and isinstance(current_sequence[0][0], int):
                # 将当前子序列添加到数字对列表中
                print("连续CNOT门:", current_sequence)  # 输出当前数字序列
                cnot_qasm = generate_cnot_circuit(current_sequence)
                # print(cnot_qasm)
                # 将qasm存入本地地址qasm_address
                write_qasm_to_address(qasm_address, cnot_qasm)
                ##############################################################
                # 这地方对cnot_qasm做综合
                # cnot_circuits = eli_of_ibmq_quatio(qasm_address)  # 5qubits
                # cnot_circuits = eli_of_ibmq_guadalupe(qasm_address)  # 16qubits
                # cnot_circuits = eli_of_ibmq_lagos(qasm_address)  # 7qubits
                # cnot_circuits = eli_of_ibmq_almaden(qasm_address)  # 20qubits almaden
                # cnot_circuits = eli_of_ibmq_tokyo(qasm_address)  # 20qubits tokyo
                cnot_circuits = eli_of_ibmq_melbourne(qasm_address)  # 14qubits melbourne

                print("------------------------当前CNOT子线路综合结果如下------------------------")
                print(cnot_circuits)
                ##############################################################
                # 重置当前子序列
                current_sequence = []
            # 将当前元素添加到当前子序列中
            current_sequence.append(item)

    # 检查最后一个子序列并将其添加到相应的列表中
    if current_sequence:
        if isinstance(current_sequence[0][0], int):
            print("连续CNOT门:", current_sequence)  # 输出当前数字序列
            cnot_qasm = generate_cnot_circuit(current_sequence)
            print(cnot_qasm)
            ##############################################################
            # 这地方对cnot_qasm做综合
            # cnot_circuits = eli_of_ibmq_quatio(qasm_address)  # 5qubits
            # cnot_circuits = eli_of_ibmq_guadalupe(qasm_address)  # 16qubits
            # cnot_circuits = eli_of_ibmq_lagos(qasm_address)  # 7qubits
            # cnot_circuits = eli_of_ibmq_almaden(qasm_address)  # 20qubits almaden
            # cnot_circuits = eli_of_ibmq_tokyo(qasm_address)  # 20qubits tokyo
            cnot_circuits = eli_of_ibmq_melbourne(qasm_address)  # 14qubits melbourne

            print("------------------------当前CNOT子线路综合结果如下------------------------")
            print(cnot_circuits)
            ##############################################################
        else:
            print("连续单门:", current_sequence)  # 输出当前字符序列
            single_gate_qasm = generate_single_gate_circuit(current_sequence)
            print(single_gate_qasm)
            ##############################################################
            # 这地方根据初始映射方式，更新single_gate_qasm  {0: 0, 1: 3, 2: 3, 3: 1, 3: 2}
            # 16qubits {0: 0, 1: 1, 2: 2, 3: 3, 3: 3, 5: 5, 6: 6, 7: 7, 5: 9, 9: 5, 10: 10, 11: 11, 12: 14, 13: 1 3, 14: 12, 15: 15}
            # 7qubits {0: 0, 1: 2, 2: 1, 3: 3, 3: 3, 5: 5, 6: 6}
            # 20qubits almaden {0: 0, 1: 1, 2: 2, 3: 3, 3: 3, 5: 5, 6: 6, 7: 7, 5: 5, 9: 9, 10: 10, 11: 11, 12: 12, 13: 14, 14: 13, 15: 15, 16: 16, 17: 17, 18: 18, 19: 19}
            # 20qubits tokyo {0: 0, 1: 1, 2: 2, 3: 3, 3: 3, 5: 9, 6: 5, 7: 7, 5: 6, 9: 5, 10: 10, 11: 11, 12: 12, 13: 13, 14: 18, 15: 14, 16: 15, 17: 16, 18: 17, 19: 19}
            # 14qubits melbourne {0: 0, 1: 1, 2: 2, 3: 3, 3: 3, 5: 5, 6: 6, 7: 7, 5: 5, 9: 9, 10: 10, 11: 11, 12: 12, 13: 13}
            ##############################################################


# 生成cnot子线路
def generate_cnot_circuit(number_sequences):
    # 初始化量子电路，假设5个量子比特
    num_qubits = 14
    qc = QuantumCircuit(num_qubits)
    # 遍历数字序列并添加量子门
    for gate in number_sequences:
        qc.cx(gate[0], gate[1])
    print(qc)
    return qc.qasm()


# 生成单量子门子线路
def generate_single_gate_circuit(char_sequences):
    # 初始化量子电路，假设5个量子比特
    num_qubits = 14
    qc = QuantumCircuit(num_qubits)
    # 遍历数字序列并添加量子门
    for gate in char_sequences:
        control, target = gate
        if control == 'h':
            qc.h(target)
        elif control == 's':
            qc.s(target)
        elif control == 'sdg':
            qc.sdg(target)
        elif control == 't':
            qc.t(target)
        elif control == 'tdg':
            qc.tdg(target)
    print(qc)
    return qc.qasm


# 将QASM代码写入文件
def write_qasm_to_address(qasm_address, qc_qasm):
    with open(qasm_address, 'w') as f:
        f.write(qc_qasm)


if __name__ == '__main__':
    # circuit = QuantumCircuit(5)

    # input_filename = 'circuits/benchmark/5qubits/initial_qasm/4gt5_75.qasm'
    # input_filename = 'circuits/benchmark/16/initial_qasm/dc2_222.qasm'
    # input_filename = 'circuits/benchmark/7qubits/random_7/7_1000_10%.qasm'
    # input_filename = 'circuits/benchmark/20_qubit_circuit_include_single_gate/random/20_1000_10%.qasm'
    input_filename = 'circuits/benchmark/14_qubit_circuit_include_single_gate/random/14_1000_10%.qasm'
    gate_list = converter_circ_from_qasm(input_filename)[1]
    print(gate_list)

    split_list(gate_list)
