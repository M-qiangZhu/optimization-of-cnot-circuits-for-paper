# -*- coding: utf-8 -*-

"""
    @Author kungfu
    @Date 2023/6/4 13:38
    @Describe 
    @Version 1.0
"""

from itertools import combinations

import numpy as np
from matplotlib import pyplot as plt

from qiskit import QuantumCircuit, IBMQ
from qiskit import Aer, transpile
from qiskit.providers.ibmq.runtime import IBMRuntimeService
from qiskit.visualization import plot_histogram
from qiskit.test.mock import FakeYorktown

from my_tools.graph import IbmQuito, Liner
from my_tools.my_parity_maps import CNOT_tracker
from networkx.algorithms import approximation
from my_tools.my_linalg import Mat2
from my_tools.tree import Tree


def rotate(mat):
    """
    生成中心旋转的numpy二维数组
    :param mat: 需要处理的矩阵
    :return: 旋转后的二维数组
    """
    m = mat.data
    # 倒着访问，行倒着数，列也倒着数
    m = m[::-1, ::-1]
    return m


def get_circuits_to_matrix(file_name, **kwargs):
    """
    从文件读量子线路
    :param file_name:
    :param kwargs:
    :return:
    """
    circuit = CNOT_tracker.from_qasm_file(file_name)
    print(f"初始门数: {len(circuit.gates)}")
    mat = circuit.matrix
    return mat


def get_center_flipped_matrix(mat):
    """
    获取一个矩阵的中心翻转矩阵
    :param mat: 原始矩阵
    :return: 经过中心翻转的矩阵
    """
    # 生成一个新的矩阵, 用来存更新后的数据
    fli_mat = mat.copy()
    # 获取旋转后的二维数组
    rotate_arr = rotate(mat)
    # 将旋转后的二维数组赋值给原矩阵
    fli_mat.data = rotate_arr
    return fli_mat


def get_col(m, col_index):
    """
    根据矩阵获取指定的列
    :param m: 矩阵
    :param col_index: 需要获取的列的索引
    :return:
    """
    return m.data[:, col_index]


def get_row(m, row_index):
    """
    根据矩阵获取指定的行
    :param m: 矩阵
    :param row_index: 需要获取的行的索引
    :return:
    """
    return m.data[row_index, :].tolist()


def get_ones_index_col(row_index, col_list):
    """
    从当前列索引为 row_index 处向下获取值为1的元素所在行的编号, 包括 row_index
    :param col_list: 矩阵当前列
    :param row_index: 起始行索引
    :return: 值为 1 的索引列表
    """
    v_list = []
    # for num in range(row_index, len(col_list)):
    #     if col_list[num] == 1:
    #         v_list.append(num)
    for num in range(len(col_list)):
        if col_list[num] == 1:
            v_list.append(num)
    return v_list


def confirm_steiner_point(old_list, new_list):
    """
    得到用来生成树的顶点集合 {0: False, 1: True}
    当前为1的元素, steiner树给出的顶点元素 => 生成 True 和 False 的集合
    :param old_list: 矩阵中当前列值为1的顶点
    :param new_list: nx得到的steiner树的顶点
    :return:
    """
    v_dict = {}
    for new_v in new_list:
        if new_v in old_list:
            v_dict[new_v] = False
        else:
            v_dict[new_v] = True
    return v_dict


def is_cut_point(g, n):
    """
    判断当前点是否是割点
    :param g: 当前图
    :param n: 当前顶点
    :return: 是否为割点
    """
    degree = g.degree[n]
    if degree > 1:
        return True
    return False


def row_add(row1, row2):
    """Add r0 to r1"""
    for i, v in enumerate(row1):
        if v:
            row2[i] = 0 if row2[i] else 1  # 当row1中某个值为1时, 将row2中的对应位置的值取反
    return row2


def col_eli_set_steiner_point(m, node, col, cnot_list):
    """
    列消元第一步: Steiner点置1
    :param m:
    :param node:
    :param col:
    :param cnot_list:
    :return:
    """
    if node is None:
        return
    if node.left_child is not None:
        m, cnot_list = col_eli_set_steiner_point(m, node.left_child, col, cnot_list)
    if node.right_child is not None:
        m, cnot_list = col_eli_set_steiner_point(m, node.right_child, col, cnot_list)
    # 获取当前列对应当前索引处的值
    if node.parent is not None:
        j = node.val
        k = node.parent.val
        if col[j] == 1 and col[k] == 0:
            m.row_add(j, k)
            cnot_list.append((j, k))
            # print("new matrix")
            # print(matrix)
            # print(f"CNOT_list : {cnot_list}")
            # return matrix
    return m, cnot_list


def col_eli_down_elim(m, node, cnot_list):
    """
    列消元第二步, 向下消元
    :param m:
    :param node:
    :param cnot_list:
    :return:
    """
    if node is None:
        return
    if node.left_child is not None:
        m, cnot_list = col_eli_down_elim(m, node.left_child, cnot_list)
    if node.right_child is not None:
        m, cnot_list = col_eli_down_elim(m, node.right_child, cnot_list)
    # 将当前节点对应行, 加到孩子节点对应行上
    parent = node.val
    if node.left_child is not None:
        left = node.left_child.val
        m.row_add(parent, left)
        cnot_list.append((parent, left))
    if node.right_child is not None:
        right = node.right_child.val
        m.row_add(parent, right)
        cnot_list.append((parent, right))
    return m, cnot_list


def col_elim(m, start_node, col, cnot_list):
    step1_m, step1_cnots = col_eli_set_steiner_point(m, start_node, col, cnot_list)
    print("列消除step1_m :")
    print(step1_m)
    print(f"列消除step1_cnots : {step1_cnots}")
    result_m, cnot = col_eli_down_elim(m, start_node, cnot_list)
    # tmp_cnot += cnot
    return result_m, cnot


def get_ei(m, i):
    # 生成对应大小的恒等矩阵
    n_qubits = m.rank()
    matrix = Mat2(np.identity(n_qubits))
    # 从恒等矩阵中取出对应的行作为ei
    ei = matrix.data[i].tolist()
    return ei


def is_row_eql(row1, row2):
    if len(row1) == len(row2):
        length = len(row1)
        if row1 == row2:
            print("两行相等")
        else:
            print("两行不相等")
    else:
        print("两行数据不匹配!!!")


def find_set_j(m, tar_row_index, row_tar, ei):
    # 根据目标行, 生成待遍历的列表
    length = m.rank()
    all_set = []
    print()
    for i in range(1, length):
        all_set += list(combinations([j for j in range(tar_row_index, length)], i))
    for j_set in all_set:
        # 暂存ei, 用来恢复ei
        tmp_row = ei.copy()
        for i in j_set:
            row = get_row(m, i)
            row_add(row, tmp_row)
        if tmp_row == row_tar:
            return list(j_set)


def row_elim_step1(m, node, cnot_list):
    if node is None:
        return
    # 获取当前列对应当前索引处的值
    if node.parent is not None and node.is_steiner_point is True:
        j = node.val
        k = node.parent.val
        m.row_add(j, k)
        cnot_list.append((j, k))
    if node.left_child is not None:
        m, cnot_list = row_elim_step1(m, node.left_child, cnot_list)
    if node.right_child is not None:
        m, cnot_list = row_elim_step1(m, node.right_child, cnot_list)
    return m, cnot_list


def row_elim_step2(m, node, cnot_list):
    if node is None:
        return
    if node.left_child is not None:
        m, cnot_list = row_elim_step2(m, node.left_child, cnot_list)
    if node.right_child is not None:
        m, cnot_list = row_elim_step2(m, node.right_child, cnot_list)
    if node.parent is not None:
        # 将当前节点对应行, 加到父节点对应行上
        parent = node.parent.val
        child = node.val
        m.row_add(child, parent)
        cnot_list.append((child, parent))
    return m, cnot_list


def row_elim(m, node, cnot_list):
    step1_m, step1_cnots = row_elim_step1(m, node, cnot_list)
    print("行消除step1_m :")
    print(step1_m)
    print(f"行消除step1_cnots : {step1_cnots}")
    result_m, cnot = row_elim_step2(m, node, cnot_list)
    # tmp_cnot += cnot
    return result_m, cnot


def get_node_eli_order(g):
    nodes = list(g.nodes)
    print(nodes)
    print(type(nodes))
    node_eli_order = []
    while len(nodes) != 0:
        for node in nodes:
            if is_cut_point(g, node) is False:
                node_eli_order.append(node)
                g.remove_node(node)
                nodes.remove(node)
            else:
                break
    return node_eli_order


def get_gate(filepath):
    gate_list = []
    f = open(filepath, 'r')
    data = f.readlines()
    for line in data:
        xy = line.split()
        gate_list.append([eval(xy[0]), eval(xy[1])])
    return gate_list


def get_qiskit_circ(gate_list):
    in_circ = QuantumCircuit(5)
    for a in gate_list:
        in_circ.cx(a[0], a[1])
    return in_circ


def test_one_col_eli():
    # circuit_file = "./circuits/steiner/5qubits/10/Original9.qasm"  # 3
    circuit_file = "./circuits/steiner/5qubits/10/Original11.qasm"  # 1, 4
    matrix = get_circuits_to_matrix(circuit_file)
    print("matrix :")
    print(matrix)
    # 获取 ibmq_quito 架构的图
    ibmq_quito = IbmQuito()
    graph = ibmq_quito.get_graph()
    # 画图
    # ibmq_quito.draw_graph()

    # 设置当前索引
    index = 0
    # 获取当前列数据
    col_list = get_col(matrix, index)
    print(f"col_list{col_list}")
    # 获取当前列中为1的顶点
    col_ones = get_ones_index_col(index, col_list)
    print(f"col_ones : {col_ones}")
    # 如果对角线元素为0, 需要单独处理
    if col_list[index] == 0:
        # 用来生成Steiner树的顶点
        # col_ones.append(int(col_list[index]))
        v_st = col_ones + [int(index)]
        v_st = sorted(v_st)
        print(f"对角线元素为 0 时 v_st : {v_st}")
    else:
        v_st = col_ones
        print(f"对角线元素不为 0 时 v_st : {v_st}")
    # --------------------------------------------------------------------
    # 根据值为 1 的顶点集合, 生成Steiner树
    tree_from_nx = approximation.steiner_tree(graph, v_st)
    # 获取Steiner树中的顶点
    tmp_v = tree_from_nx.nodes
    print(f"tmp_v : {tmp_v}")
    # 获取用来生成树的顶点集合
    vertex = confirm_steiner_point(col_ones, tmp_v)
    # 获取用来生成树的边集合
    edges = [e for e in tree_from_nx.edges]
    print(f"vertex : {vertex}")
    print(f"edges : {edges}")
    # 生成树
    tree = Tree(vertex, edges)
    root = tree.gen_tree()
    # print(root.get_value())
    col = get_col(matrix, index)
    CNOT_list = []
    matrix, cnot = col_elim(matrix, root, col, CNOT_list)
    print(f"列消元后的矩阵 : ")
    print(matrix)
    print(f"列消元过程中使用的CNOT门: {cnot}")
    print("-" * 100)
    return matrix


def test_one_row_eli(m):
    # 获取 ibmq_quito 架构的图
    ibmq_quito = IbmQuito()
    graph = ibmq_quito.get_graph()
    index = 0
    ei = get_ei(m, index)
    print(f"ei : {ei}")
    print(f"ei类型: {type(ei)}")
    print(f"ei中数据的类型: {type(ei[0])}")
    # 获取当前被消除的行
    row_target = get_row(m, index)
    # print(f"row_{index} : {row_i}")
    # print(f"row_{index}类型: {type(row_i)}")
    # # 因为数据为引用型, row_i会被覆盖
    # row_target = row_add(ei, row_i)
    # print(f"row_target : {row_target}")
    # print(f"row_{index}: {row_i}")
    # print(ei == row_i)
    # is_row_eql(ei, row_i)
    # print(row_target == row_i)
    # is_row_eql(row_target, row_i)

    # 手动测试 row1 + row2 + row4
    # row_1 = get_row(m, 1)
    # row_2 = get_row(m, 2)
    # row_4 = get_row(m, 4)
    # print(f"row_1 : {row_1}")
    # print(f"row_2 : {row_2}")
    # print(f"row_4 : {row_4}")
    # row_add(row_1, row_2)
    # print(f"row_2更新为 : {row_2}")
    # row_add(row_2, row_4)
    # print(f"row_4更新为 : {row_4}")
    # print(row_4 == row_target)
    # 从剩余行中找到满足条件的集合{j}
    j_set = find_set_j(m, index + 1, row_target, ei)
    print(f"j_set : {j_set}")
    print(f"j_set长度为 : {len(j_set)}")

    # j_set = [1, 4, 2]
    # 根据j和i生成Steiner树
    node_set = sorted([index] + j_set)
    print(f"node_set : {node_set}")
    tree_from_nx = approximation.steiner_tree(graph, node_set)
    # 获取Steiner树中的顶点
    tmp_v = tree_from_nx.nodes
    print(f"tmp_v : {tmp_v}")
    # 获取用来生成树的顶点集合
    vertex = confirm_steiner_point(node_set, tmp_v)
    # 获取用来生成树的边集合
    edges = [e for e in tree_from_nx.edges]
    print(f"vertex : {vertex}")
    print(f"edges : {edges}")
    # 生成树
    tree = Tree(vertex, edges)
    root = tree.gen_tree()
    print(f"root.get_value() : {root.get_value()}")
    # 记录CNOT门
    CNOT_list = []
    # # 第一步: 根据j集合消元, 从根节点开始遍历树, 遇到Steiner点后, 将Steiner点对应行加到它父节点所在行
    # m, cnot = row_elim_step1(m, root, CNOT_list)
    # print(f"当前matrix : ")
    # print(m)
    # print(f"cnot : {cnot}")
    #
    # # 第二步: 从根节点开始遍历树, 将每一行加到父节点
    # m, cnot = row_elim_step2(m, root, CNOT_list)
    # print(f"当前matrix : ")
    # print(m)
    # print(f"cnot : {cnot}")

    # 执行 行消元
    m, cnot = row_elim(m, root, CNOT_list)
    print(f"行消元后的矩阵 : ")
    print(m)
    print(f"行消元过程中使用的CNOT门: {cnot}")
    print("-" * 100)
    print(f"列消元后的矩阵 : ")
    print(m)
    print(f"列消元过程中使用的CNOT门: {cnot}")
    print("-" * 100)
    return m


def test_matrix_rows_add():
    circuit_file = "./circuits/steiner/5qubits/10/Origina7.qasm"
    matrix = get_circuits_to_matrix(circuit_file)
    print("matrix :")
    print(matrix)
    matrix.row_add(1, 0)
    print("new matrix:")
    print(matrix)


def test_cut_point():
    circuit_file = "./circuits/steiner/5qubits/10/Original4.qasm"
    matrix = get_circuits_to_matrix(circuit_file)
    print("matrix :")
    print(matrix)
    # 获取当前列数据
    col_list = get_col(matrix, 0)
    print(col_list)
    # 获取当前列中为1的顶点
    col_ones = get_ones_index_col(0, col_list)
    print(col_ones)
    # 获取 ibmq_quito 架构的图
    ibmq_quito = IbmQuito()
    graph = ibmq_quito.get_graph()
    # 根据值为 1 的顶点集合, 生成Steiner树
    tree_from_nx = approximation.steiner_tree(graph, col_ones)
    # 获取Steiner树中的顶点
    tmp_v = tree_from_nx.nodes
    for v in tmp_v:
        print(v, is_cut_point(graph, v))


def test_col_eli():
    circuit_file = "./circuits/steiner/5qubits/10/Original9.qasm"
    # circuit_file = "./circuits/steiner/5qubits/10/Original11.qasm"  # 1, 4
    matrix = get_circuits_to_matrix(circuit_file)
    print("matrix :")
    print(matrix)
    # 获取 ibmq_quito 架构的图
    ibmq_quito = IbmQuito()
    graph = ibmq_quito.get_graph()
    # 画图
    # ibmq_quito.draw_graph()

    for index in range(matrix.rank()):
        print(f"index : {index}")
        # 获取当前列数据
        col_list = get_col(matrix, index)
        print(col_list)
        # 获取当前列中为1的顶点
        col_ones = get_ones_index_col(index, col_list)
        print(f"col_ones : {col_ones}")
        # 如果对角线元素为0, 需要单独处理
        if col_list[index] == 0:
            # 用来生成Steiner树的顶点
            # col_ones.append(int(col_list[index]))
            v_st = col_ones + [int(col_list[index])]
            v_st = sorted(v_st)
            print(f"对角线元素为 0 时 v_st : {v_st}")
        else:
            v_st = col_ones
            print(f"对角线元素不为 0 时 v_st : {v_st}")
        # --------------------------------------------------------------------
        # 根据值为 1 的顶点集合, 生成Steiner树
        tree_from_nx = approximation.steiner_tree(graph, v_st)
        # 获取Steiner树中的顶点
        tmp_v = tree_from_nx.nodes
        print(f"tmp_v : {tmp_v}")
        # 获取用来生成树的顶点集合
        vertex = confirm_steiner_point(col_ones, tmp_v)
        # 获取用来生成树的边集合
        edges = [e for e in tree_from_nx.edges]
        print(f"vertex : {vertex}")
        print(f"edges : {edges}")
        # 生成树
        tree = Tree(vertex, edges)
        root = tree.gen_tree()
        print(f"root.get_value() : {root.get_value()}")
        col = get_col(matrix, index)
        CNOT_list = []
        # 竖直消元
        matrix, cnot = col_elim(matrix, root, col, CNOT_list)
        print(f"matrix : ")
        print(matrix)
        print(f"cnot : {cnot}")
        # 水平消元


def test_eli_one_cul_one_row():
    matrix = test_one_col_eli()
    print("用来行消元的矩阵 :")
    print(matrix)
    matrix = test_one_row_eli(matrix)
    print("消元后的矩阵为:")
    print(matrix)


def test_get_node_eli_order():
    ibmq_quito = IbmQuito()
    graph = ibmq_quito.get_graph()
    print(get_node_eli_order(graph))


def col_row_eli(file_name):
    """
    综合消元
    :param file_name:
    :return:
    """
    # 1. 获取 ibmq_quito 架构的图
    liner = Liner()
    graph = liner.get_graph()
    # 2. 读取线路生成矩阵
    circuit_file = file_name
    # circuit_file = "./circuits/steiner/5qubits/10/Original11.qasm"  # 1, 4
    matrix = get_circuits_to_matrix(circuit_file)

    print("matrix :")
    print(matrix)
    # 3. 根据是否是割点, 生成消元序列
    eli_order = get_node_eli_order(graph.copy())
    print(f"eli_order : {eli_order}")
    print(f"eli_order类型 : {type(eli_order)}")
    # 4. 记录CNOT门用来生成线路
    CNOT = []
    # 5. 进入循环
    eli_order = [0, 1, 2, 3, 4]
    # 默认进行行列消元
    col_flag = True
    for index in eli_order:
        # 列消元
        print(f"***********************************消除第{index}列和第{index}行**************************************")
        # 获取当前列数据
        col_list = get_col(matrix, index)
        print(f"col_list{col_list}")
        # 获取当前列中为1的顶点
        col_ones = get_ones_index_col(index, col_list)
        print(f"col_ones : {col_ones}")
        # 如果对角线元素为0, 需要单独处理
        if col_list[index] == 0:
            # 用来生成Steiner树的顶点
            # col_ones.append(int(col_list[index]))
            v_st = col_ones + [index]
            v_st = sorted(v_st)
            print(f"对角线元素为 0 时 v_st : {v_st}")
        else:
            v_st = col_ones
            print(f"对角线元素不为 0 时 v_st : {v_st}")
        # --------------------------------------------------------------------
        # 根据值为 1 的顶点集合, 生成Steiner树
        if len(v_st) > 1:
            tree_from_nx = approximation.steiner_tree(graph, v_st)
            # 获取Steiner树中的顶点
            tmp_v = tree_from_nx.nodes
            print(f"tmp_v : {tmp_v}")
            if len(tmp_v) == 0:
                print("只有根节点, 无需生成steiner树")
                # 是否进行列消元
                col_flag = False
            if col_flag:
                # 获取用来生成树的顶点集合
                vertex = confirm_steiner_point(col_ones, tmp_v)
                # 获取用来生成树的边集合
                edges = [e for e in tree_from_nx.edges]
                print(f"vertex : {vertex}")
                print(f"edges : {edges}")
                # 指定根节点
                root_node = index
                print(f"根节点: {root_node}")
                # 生成树
                tree = Tree(vertex, edges, root_node)
                root = tree.gen_tree()
                print(f"当前根节点为 : {root.get_value()}")
                col = get_col(matrix, index)
                CNOT_list = []
                matrix, cnot = col_elim(matrix, root, col, CNOT_list)
                CNOT += cnot
                print(f"列消元后的矩阵 : ")
                print(matrix)
                print(f"列消元过程中使用的CNOT门: {cnot}")
                print("-" * 60)
        # 行消元
        ei = get_ei(matrix, index)
        print(f"ei : {ei}")
        print(f"ei类型: {type(ei)}")
        print(f"ei中数据的类型: {type(ei[0])}")
        # 获取当前被消除的行
        row_target = get_row(matrix, index)
        j_set = find_set_j(matrix, index + 1, row_target, ei)
        print(f"j_set : {j_set}")
        # print(f"j_set长度为 : {len(j_set)}")
        if j_set is not None:
            # 根据j和i生成Steiner树
            node_set = sorted([index] + j_set)
            print(f"node_set : {node_set}")
            tree_from_nx = approximation.steiner_tree(graph, node_set)
            # 获取Steiner树中的顶点
            tmp_v = tree_from_nx.nodes
            print(f"tmp_v : {tmp_v}")
            # 获取用来生成树的顶点集合
            vertex = confirm_steiner_point(node_set, tmp_v)
            # 获取用来生成树的边集合
            edges = [e for e in tree_from_nx.edges]
            print(f"vertex : {vertex}")
            print(f"edges : {edges}")
            # 生成树
            tree = Tree(vertex, edges, index)
            root = tree.gen_tree()
            print(f"root.get_value() : {root.get_value()}")
            # 记录CNOT门
            CNOT_list = []
            # 执行 行消元
            m, cnot = row_elim(matrix, root, CNOT_list)
            CNOT += cnot
            print(f"行消元后的矩阵 : ")
            print(m)
            print(f"行消元过程中使用的CNOT门: {cnot}")
            print("删除当前顶点")
        graph.remove_node(index)
        # 恢复 列消元标志位
        col_flag = True
    print(f"所有CNOT门: {CNOT}")
    # 将 CNOT 根据映射转换
    map_dict = {0: 20, 1: 13, 2: 19, 3: 25, 4: 31}
    new_CNOT = []
    for cnot in CNOT:
        control = map_dict.get(cnot[0])
        target = map_dict.get(cnot[1])
        new_CNOT.append((control, target))
    print(new_CNOT)
    return new_CNOT


def test_gen_circuit():
    """
    根据CNOT门序列, 生成线路测试
    :return:
    """
    file = "/Users/kungfu/PycharmWorkspace/Optimization_of_CNOT_circuits/circuits/steiner/5qubits/5/Original11.qasm"
    cnot = col_row_eli(file)
    # 根据cnot门列表, 生成线路
    circuit = QuantumCircuit(5)
    for cnot_gate in cnot:
        control = cnot_gate[0]
        target = cnot_gate[1]
        circuit.cx(control, target)
    circuit.measure_all()
    circuit.draw("mpl")
    print(circuit)
    circuit.qasm(filename="circuit-test1-1.qasm")

    # device_backend = FakeYorktown()
    #
    # # 生成一个模拟设备的模拟器, 调用' from_backend '为' ibmq_vigo '创建一个模拟器
    # sim_yorktown = AerSimulator.from_backend(device_backend)
    #
    # # 一旦我们基于真实设备创建了一个嘈杂的模拟器后端，我们就可以使用它来运行嘈杂的模拟
    # # 重要提示：在运行噪声模拟时，为后端转译电路至关重要，以便将电路转译为后端的正确噪声基门集。
    # tcirc = transpile(circuit, sim_yorktown)
    # print(tcirc)
    #
    # # Execute noisy simulation and get counts
    # result_noise = sim_yorktown.run(tcirc).result()
    # counts_noise = result_noise.get_counts(0)
    # plot_histogram(counts_noise, figsize=(14, 9))
    # # plt.savefig('CNOT circuits.png')
    # # plot_histogram(counts_noise, title="Counts for 3-qubit GHZ state with device noise model", figsize=(14, 9))
    # plt.show()


def test_read_cir():
    """
    读取线路测试
    :return:
    """
    circuit = QuantumCircuit(5)
    circuit = circuit.from_qasm_file("circuit-test2.qasm")
    # circuit.draw("mpl")
    print(circuit)


def execute_circuit():
    gates_list = [2, 4, 5, 8, 10, 15, 20, 30, 40, 80, 100, 200]
    for gate in gates_list:
        for i in range(20):
            print(f">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>第{i}个文件<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
            file_name = f"./circuits/steiner/5qubits/{gate}/Original{i}.qasm"
            origin_cnot_list = col_row_eli(file_name)
            circuit = QuantumCircuit(5)
            for cnot_gate in origin_cnot_list:
                control = cnot_gate[0]
                target = cnot_gate[1]
                circuit.cx(control, target)
            circuit.measure_all()
            circuit.draw("mpl")
            print(circuit)
            circuit.qasm(filename=f"./result/01234-02134/{gate}/circuit{i}.qasm")


def execute_benchmark_quito():
    file_list = ['4gt5_75', '4gt13_90', '4gt13_91', '4gt13_92', '4mod5-v1_22', '4mod5-v1_23', '4mod5-v1_24', 'alu-v0_27', 'alu-v3_35', 'alu-v4_36', 'alu-v4_37', 'decod24-v2_43',
                 'hwb4_49', 'mod5mils_65', 'mod10_171']
    for file in file_list:
        file_name = f"./circuits/benchmark/5qubits/qasm/{file}.qasm"
        origin_cnot_list = col_row_eli(file_name)
        circuit = QuantumCircuit(5)
        for cnot_gate in origin_cnot_list:
            control = cnot_gate[0]
            target = cnot_gate[1]
            circuit.cx(control, target)
        circuit.measure_all()
        circuit.draw("mpl")
        print(circuit)
        circuit.qasm(filename=f"./result/01234-43210/qasm-trans/{file}_eli.qasm")


def execute_benchmark_on_liner():
    file_list = ['4gt5_75', '4gt13_90', '4gt13_91', '4gt13_92', '4mod5-v1_22', '4mod5-v1_23', '4mod5-v1_24', 'alu-v0_27', 'alu-v3_35', 'alu-v4_36', 'alu-v4_37', 'decod24-v2_43',
                 'hwb4_49', 'mod5mils_65', 'mod10_171']
    for file in file_list:
        file_name = f"./circuits/benchmark/5qubits/qasm/{file}.qasm"
        origin_cnot_list = col_row_eli(file_name)
        circuit = QuantumCircuit(66)
        for cnot_gate in origin_cnot_list:
            control = cnot_gate[0]
            target = cnot_gate[1]
            circuit.cx(control, target)
        circuit.measure_all()
        circuit.draw("mpl")
        print(circuit)
        circuit.qasm(filename=f"./synthetic_result/{file}_synthetic.qasm")


if __name__ == '__main__':
    # execute_benchmark()
    # test_gen_circuit()
    # col_row_eli("./circuits/benchmark/5qubits/qasm/4gt5_75.qasm")
    execute_benchmark_on_liner()
