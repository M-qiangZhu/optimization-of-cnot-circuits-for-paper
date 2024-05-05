import time
from itertools import combinations

import numpy as np
from matplotlib import pyplot as plt

from qiskit import QuantumCircuit, IBMQ
from qiskit import Aer, transpile
from qiskit.providers.ibmq.runtime import IBMRuntimeService
from qiskit.visualization import plot_histogram
# from qiskit.test.mock import FakeYorktown
from qiskit.providers.fake_provider import FakeYorktown

from my_tools.graph import IbmQuito, IbmqGuadalupe, IbmqGuadalupe_new, IbmqKolkata_new, IbmqManhattan_new, IbmqLagos_new, IbmqAlmaden_new, IbmqTokyo_new, WuKong, WuKong_new
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
    print(type(mat))
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


class TreeNode:
    def __init__(self, value, level, path):
        self.value = value
        self.level = level
        self.path = path  # 当前节点的路径，表示从根到当前节点的列表
        self.left = None
        self.right = None


def build_tree(level, path, max_level):
    if level == max_level:
        return None  # 达到最大层，停止构建

    # 创建当前层的节点
    node = TreeNode(path[-1] if path else 0, level, path)

    # 递归构建左右子树
    node.left = build_tree(level + 1, path + [0], max_level) if level < max_level else None
    node.right = build_tree(level + 1, path + [1], max_level) if level < max_level else None

    return node  # 返回当前节点


def print_tree(node, indent=""):
    if node:
        # 打印节点及其路径
        print(f"{indent}{node.value} (Path: {'->'.join(map(str, node.path))})")
        # 递归打印左子树
        print_tree(node.left, indent + "  ")
        # 递归打印右子树
        print_tree(node.right, indent + "  ")


def dfs_search(node, target):
    if node is None:
        return None

    # 如果路径与目标匹配，则找到目标
    if node.path == target:
        return node.path

    # 继续深度优先搜索
    left_result = dfs_search(node.left, target)
    if left_result:
        return left_result

    right_result = dfs_search(node.right, target)
    if right_result:
        return right_result

    return None


def find_set_j_new(m, tar_row_index, row_tar, ei):
    """
    利用剪枝算法实现
    :param m:
    :param tar_row_index:
    :param row_tar:
    :param ei:
    :return:
    """
    # 1. 根据 matrix 的秩, 生成左孩子节点为0, 右孩子节点为1的满二叉树
    root = build_tree(0, [], m + 1)  # 构建树的根节点

    # 2. 从第一层开始向下,进行深度遍历查找
    # 搜索目标列表
    target = [0, 1, 0, 1]
    result = dfs_search(root, target)
    print(result)
    # 3. 判断当前层,当前列和已选择路径上的列是否满足 Ri + ei 对应列的值
    # 4. 如果满足, 找下一层
    # 5. 如果不满足, 剪枝,  返回上一层, 继续查找

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


# def find_set_j(m, tar_row_index, row_tar, ei):
#     length = m.rank()
#
#     def generate_combinations():
#         for i in range(1, length):
#             for j_set in combinations(range(tar_row_index, length), i):
#                 yield j_set
#
#     for j_set in generate_combinations():
#         tmp_row = ei.copy()
#         for i in j_set:
#             row = get_row(m, i)
#             row_add(row, tmp_row)
#         if tmp_row == row_tar:
#             return list(j_set)


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
            # else:
            #     break
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


def col_row_eli_of_ibmquatio(file_name):
    # 1. 获取 ibmq_quito 架构的图
    ibmq_quito = IbmQuito()
    graph = ibmq_quito.get_graph()
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
    # for index in range(rank):
    eli_order = [0, 1, 2, 3, 4]
    # eli_order = [0, 4, 3, 1, 2]
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
    map_dict = {0: 0, 1: 4, 2: 3, 3: 1, 4: 2}
    # map_dict = {0: 0, 1: 2, 2: 1, 3: 3, 4: 4}
    # map_dict = {0: 2, 1: 0, 2: 1, 3: 3, 4: 4}
    # map_dict = {0: 2, 1: 4, 2: 3, 3: 1, 4: 0}
    # map_dict = {0: 4, 1: 3, 2: 0, 3: 1, 4: 2}
    # map_dict = {0: 4, 1: 3, 2: 2, 3: 1, 4: 0}
    new_CNOT = []
    for cnot in CNOT:
        control = map_dict.get(cnot[0])
        target = map_dict.get(cnot[1])
        new_CNOT.append((control, target))
    print(new_CNOT)
    return new_CNOT


def col_row_eli_of_ibmq_lagos(file_name):
    # 1. 获取 ibmq_quito 架构的图
    ibmq_lagos = IbmqLagos_new()
    graph = ibmq_lagos.get_graph()
    # 2. 读取线路生成矩阵
    circuit_file = file_name
    matrix = get_circuits_to_matrix(circuit_file)
    print("matrix :")
    print(matrix)
    # 3. 根据是否是割点, 生成消元序列
    # eli_order = get_node_eli_order(graph.copy())
    # 4. 记录CNOT门用来生成线路
    CNOT = []
    # 5. 进入循环
    # for index in range(rank):
    order = [0, 2, 1, 3, 4, 5, 6]
    # eli_order = [0, 4, 3, 1, 2]
    update_matrix(matrix, order)
    print(matrix)
    eli_order = [0, 1, 2, 3, 4, 5, 6]
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
    map_dict = {0: 0, 1: 2, 2: 1, 3: 3, 4: 4, 5: 5, 6: 6}
    # map_dict = {0: 0, 1: 4, 2: 3, 3: 1, 4: 2}
    # map_dict = {0: 0, 1: 2, 2: 1, 3: 3, 4: 4}
    # map_dict = {0: 2, 1: 0, 2: 1, 3: 3, 4: 4}
    # map_dict = {0: 2, 1: 4, 2: 3, 3: 1, 4: 0}
    # map_dict = {0: 4, 1: 3, 2: 0, 3: 1, 4: 2}
    # map_dict = {0: 4, 1: 3, 2: 2, 3: 1, 4: 0}
    new_CNOT = []
    for cnot in CNOT:
        control = map_dict.get(cnot[0])
        target = map_dict.get(cnot[1])
        new_CNOT.append((control, target))
    print(new_CNOT)
    return new_CNOT


def col_row_eli_of_ibmq_guadalupe(file_name):
    # 1. 获取 ibmq_quito 架构的图
    ibmq_guadalupe = IbmqGuadalupe_new()
    graph = ibmq_guadalupe.get_graph()
    # 2. 读取线路生成矩阵
    circuit_file = file_name
    # circuit_file = "./circuits/steiner/5qubits/10/Original11.qasm"  # 1, 4
    matrix = get_circuits_to_matrix(circuit_file)
    print("matrix :")
    print(matrix)
    # 3. 根据是否是割点, 生成消元序列
    # eli_order = get_node_eli_order(graph.copy())

    # print(f"eli_order : {eli_order}")
    # print(f"eli_order类型 : {type(eli_order)}")
    # 4. 记录CNOT门用来生成线路
    CNOT = []
    # 5. 进入循环
    # for index in range(rank):
    order = [0, 1, 2, 3, 4, 5, 6, 7, 9, 8, 10, 11, 14, 13, 12, 15]
    # eli_order = [0, 4, 3, 1, 2]
    update_matrix(matrix, order)
    print(matrix)
    eli_order = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
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
    map_dict = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 9, 9: 8, 10: 10, 11: 11, 12: 14, 13: 13, 14: 12, 15: 15}
    # map_dict = {0: 0, 1: 4, 2: 3, 3: 1, 4: 2}
    # map_dict = {0: 0, 1: 2, 2: 1, 3: 3, 4: 4}
    # map_dict = {0: 2, 1: 0, 2: 1, 3: 3, 4: 4}
    # map_dict = {0: 2, 1: 4, 2: 3, 3: 1, 4: 0}
    # map_dict = {0: 4, 1: 3, 2: 0, 3: 1, 4: 2}
    # map_dict = {0: 4, 1: 3, 2: 2, 3: 1, 4: 0}
    new_CNOT = []
    for cnot in CNOT:
        control = map_dict.get(cnot[0])
        target = map_dict.get(cnot[1])
        new_CNOT.append((control, target))
    print(new_CNOT)
    return new_CNOT


def col_row_eli_of_ibmq_almaden(file_name):
    # 1. 获取 ibmq_quito 架构的图
    ibmq_almaden = IbmqAlmaden_new()
    graph = ibmq_almaden.get_graph()
    # 2. 读取线路生成矩阵
    circuit_file = file_name
    matrix = get_circuits_to_matrix(circuit_file)
    print("matrix :")
    print(matrix)
    # 3. 根据是否是割点, 生成消元序列
    # eli_order = get_node_eli_order(graph.copy())
    # 4. 记录CNOT门用来生成线路
    CNOT = []
    # 5. 进入循环
    # for index in range(rank):
    order = [0, 1, 2, 4, 3, 5, 6, 7, 8, 9, 10, 11, 12, 14, 13, 15, 16, 17, 18, 19]
    # eli_order = [0, 4, 3, 1, 2]
    update_matrix(matrix, order)
    print(matrix)
    eli_order = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
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
    map_dict = {0: 0, 1: 1, 2: 2, 3: 4, 4: 3, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9, 10: 10, 11: 11, 12: 12, 13: 14, 14: 13, 15: 15, 16: 16, 17: 17, 18: 18, 19: 19}
    new_CNOT = []
    for cnot in CNOT:
        control = map_dict.get(cnot[0])
        target = map_dict.get(cnot[1])
        new_CNOT.append((control, target))
    print(new_CNOT)
    return new_CNOT


def col_row_eli_of_ibmq_tokyo(file_name):
    # 1. 获取 ibmq_quito 架构的图
    ibmq_tokyo = IbmqTokyo_new()
    graph = ibmq_tokyo.get_graph()
    # 2. 读取线路生成矩阵
    circuit_file = file_name
    matrix = get_circuits_to_matrix(circuit_file)
    print("matrix :")
    print(matrix)
    # 3. 根据是否是割点, 生成消元序列
    # eli_order = get_node_eli_order(graph.copy())
    # 4. 记录CNOT门用来生成线路
    CNOT = []
    # 5. 进入循环
    # for index in range(rank):
    order = [0, 1, 2, 3, 4, 9, 8, 7, 6, 5, 10, 11, 12, 14, 18, 14, 15, 16, 17, 19]
    # eli_order = [0, 4, 3, 1, 2]
    update_matrix(matrix, order)
    print(matrix)
    eli_order = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
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
    map_dict = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 9, 6: 8, 7: 7, 8: 6, 9: 5, 10: 10, 11: 11, 12: 12, 13: 13, 14: 18, 15: 14, 16: 15, 17: 16, 18: 17, 19: 19}
    new_CNOT = []
    for cnot in CNOT:
        control = map_dict.get(cnot[0])
        target = map_dict.get(cnot[1])
        new_CNOT.append((control, target))
    print(new_CNOT)
    return new_CNOT


def col_row_eli_of_ibmq_kolkata(file_name):
    # 1. 获取 ibmq_quito 架构的图
    ibmq_kolkata = IbmqKolkata_new()
    graph = ibmq_kolkata.get_graph()
    # 2. 读取线路生成矩阵
    circuit_file = file_name
    matrix = get_circuits_to_matrix(circuit_file)
    print("matrix :")
    print(matrix)
    print(type(matrix))
    # 3. 根据是否是割点, 生成消元序列
    # eli_order = get_node_eli_order(graph.copy())
    # 4. 记录CNOT门用来生成线路
    CNOT = []
    # 5. 进入循环
    # for index in range(rank):
    order = [0, 1, 2, 3, 4, 5, 6, 7, 9, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 20, 19, 21, 22, 23, 24, 25, 26]
    # eli_order = [0, 4, 3, 1, 2]
    update_matrix(matrix, order)
    print(matrix)
    eli_order = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26]
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
    map_dict = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 9, 9: 8, 10: 10, 11: 11, 12: 12, 13: 13,
                14: 14, 15: 15, 16: 16, 17: 17, 18: 18, 19: 20, 20: 19, 21: 21, 22: 22, 23: 23, 24: 24, 25: 25, 26: 26}
    # map_dict = {0: 0, 1: 4, 2: 3, 3: 1, 4: 2}
    # map_dict = {0: 0, 1: 2, 2: 1, 3: 3, 4: 4}
    # map_dict = {0: 2, 1: 0, 2: 1, 3: 3, 4: 4}
    # map_dict = {0: 2, 1: 4, 2: 3, 3: 1, 4: 0}
    # map_dict = {0: 4, 1: 3, 2: 0, 3: 1, 4: 2}
    # map_dict = {0: 4, 1: 3, 2: 2, 3: 1, 4: 0}
    new_CNOT = []
    for cnot in CNOT:
        control = map_dict.get(cnot[0])
        target = map_dict.get(cnot[1])
        new_CNOT.append((control, target))
    print(new_CNOT)
    return new_CNOT


def col_row_eli_of_ibmq_manhattan(file_name):
    # 1. 获取 ibmq_quito 架构的图
    ibmq_manhattan = IbmqManhattan_new()
    graph = ibmq_manhattan.get_graph()
    # 2. 读取线路生成矩阵
    circuit_file = file_name
    matrix = get_circuits_to_matrix(circuit_file)
    print("matrix :")
    print(matrix)

    # 3. 根据是否是割点, 生成消元序列
    # eli_order = get_node_eli_order(graph.copy())
    # 4. 记录CNOT门用来生成线路
    CNOT = []
    # 5. 进入循环
    # for index in range(rank):
    order = [0, 1, 2, 3, 4, 5, 6, 7, 9, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41,
             42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64]
    update_matrix(matrix, order)
    print(matrix)
    eli_order = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41,
                 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64]
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
    map_dict = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 9, 9: 8, 10: 10, 11: 11, 12: 12, 13: 13, 14: 14, 15: 15, 16: 16, 17: 17, 18: 18, 19: 19, 20: 20, 21: 21, 22: 22,
                23: 23, 24: 24, 25: 25, 26: 26, 27: 27, 28: 28, 29: 29, 30: 30, 31: 31, 32: 32, 33: 33, 34: 34, 35: 35, 36: 36, 37: 37, 38: 38, 39: 39, 40: 40, 41: 41, 42: 42,
                43: 43, 44: 44, 45: 45, 46: 46, 47: 47, 48: 48, 49: 49, 50: 50, 51: 51, 52: 52, 53: 53, 54: 54, 55: 55, 56: 56, 57: 57, 58: 58, 59: 59, 60: 60, 61: 61, 62: 62,
                63: 63, 64: 64}
    new_CNOT = []
    for cnot in CNOT:
        control = map_dict.get(cnot[0])
        target = map_dict.get(cnot[1])
        new_CNOT.append((control, target))
    print(new_CNOT)
    return new_CNOT


def col_row_eli_of_wukong(file_name):
    # 1. 获取 ibmq_quito 架构的图
    wukong = WuKong_new()
    graph = wukong.get_graph()
    # 2. 读取线路生成矩阵
    circuit_file = file_name
    matrix = get_circuits_to_matrix(circuit_file)
    print("matrix :")
    print(matrix)
    # 3. 根据是否是割点, 生成消元序列
    # eli_order = get_node_eli_order(graph.copy())
    # 4. 记录CNOT门用来生成线路
    CNOT = []
    # 5. 进入循环
    # for index in range(rank):
    order = [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 9, 11]
    update_matrix(matrix, order)
    print(matrix)
    eli_order = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
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
    map_dict = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 10, 10: 9, 11: 11}
    new_CNOT = []
    for cnot in CNOT:
        control = map_dict.get(cnot[0])
        target = map_dict.get(cnot[1])
        new_CNOT.append((control, target))
    print(new_CNOT)
    return new_CNOT


def test_gen_circuit_old(qubits, file):
    # file = "/Users/kungfu/PycharmWorkspace/Optimization_of_CNOT_circuits/circuits/steiner/5qubits/5/Original11.qasm"
    cnot = col_row_eli_of_ibmq_guadalupe(file)
    # 根据cnot门列表, 生成线路
    circuit = QuantumCircuit(qubits)
    for cnot_gate in cnot:
        control = cnot_gate[0]
        target = cnot_gate[1]
        circuit.cx(control, target)
    circuit.measure_all()
    circuit.draw("mpl")
    print(circuit)
    circuit.qasm(filename=f"add-exam/result-circuits/16qubits/hwb_12.qasm")

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


def test_gen_circuit_new(qubits, cnots, file_name):
    # 根据cnot门列表, 生成线路
    circuit = QuantumCircuit(qubits)
    for cnot_gate in cnots:
        control = cnot_gate[0]
        target = cnot_gate[1]
        circuit.cx(control, target)
    circuit.measure_all()
    circuit.draw("mpl")
    print(circuit)
    circuit.qasm(filename=f"add-exam/result-circuits/B&D_circuits_synthesize/{file_name}-{qubits}qubits_synthesis.qasm")

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
    circuit = QuantumCircuit(16)
    circuit = circuit.from_qasm_file("./circuits/benchmark/16/cnt3-5_179.qasm")
    # circuit.draw("mpl")
    print(circuit)


def execute_circuit():
    gates_list = [2, 4, 5, 8, 10, 15, 20, 30, 40, 80, 100, 200]
    for gate in gates_list:
        for i in range(20):
            print(f">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>第{i}个文件<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
            file_name = f"./circuits/steiner/5qubits/{gate}/Original{i}.qasm"
            origin_cnot_list = col_row_eli_of_ibmquatio(file_name)
            circuit = QuantumCircuit(5)
            for cnot_gate in origin_cnot_list:
                control = cnot_gate[0]
                target = cnot_gate[1]
                circuit.cx(control, target)
            circuit.measure_all()
            circuit.draw("mpl")
            print(circuit)
            circuit.qasm(filename=f"./result/01234-02134/{gate}/circuit{i}.qasm")


def execute_benchmark():
    file_list = ['4gt5_75', '4gt13_90', '4gt13_91', '4gt13_92', '4mod5-v1_22', '4mod5-v1_23', '4mod5-v1_24', 'alu-v0_27', 'alu-v3_35', 'alu-v4_36', 'alu-v4_37', 'decod24-v2_43',
                 'hwb4_49', 'mod5mils_65', 'mod10_171']
    for file in file_list:
        file_name = f"./circuits/benchmark/5qubits/qasm/{file}.qasm"
        origin_cnot_list = col_row_eli_of_ibmquatio(file_name)
        circuit = QuantumCircuit(5)
        for cnot_gate in origin_cnot_list:
            control = cnot_gate[0]
            target = cnot_gate[1]
            circuit.cx(control, target)
        circuit.measure_all()
        circuit.draw("mpl")
        print(circuit)
        circuit.qasm(filename=f"./result/01234-43210/qasm-trans/{file}_eli.qasm")


def update_matrix(matrix, order):
    """
    根据消元路径更新矩阵
    :param order:
    :return:
    """
    matrix_rank = matrix.rank()
    print(matrix_rank)
    new_matrix = Mat2.id(matrix_rank)
    print(new_matrix)
    print(new_matrix.data[1][1])
    for i in range(matrix_rank):
        for j in range(matrix_rank):
            new_matrix.data[i][j] = matrix.data[order[i]][order[j]]
    print(new_matrix)
    return new_matrix


if __name__ == '__main__':
    # execute_benchmark()

    # test_gen_circuit()
    # col_row_eli_of_ibmq_guadalupe("./circuits/benchmark/15_and_16_qubits_test/16qubit_circuit/cnt3-5_179.qasm")
    # circuit_file = "./circuits/benchmark/15_and_16_qubits_test/16qubit_circuit/cnt3-5_179.qasm"
    # circuit_file = "./circuits/steiner/5qubits/10/Original11.qasm"  # 1, 4
    # matrix = get_circuits_to_matrix(circuit_file)
    # order = [0, 1, 2, 3, 4, 5, 6, 7, 9, 8, 10, 11, 14, 13, 12, 15]
    # update_matrix(matrix, order)

    # file_list = ['ham15_107', 'dc2_222', 'ham15_108', 'rd84_143', 'ham15_109', 'misex1_241', 'rd84_142']
    # for file in file_list:
    #     test_gen_circuit(16, f"./circuits/benchmark/15_and_16_qubits_test/15qubit_circuit/{file}.qasm")
    # test_gen_circuit(16, f"./circuits/benchmark/15_and_16_qubits_test/16qubit_circuit/hwb_12.qasm")

    # 遍历执行7量子位线路
    # circuits_name_list = ["Bernstein-Vazirani"]
    # qubits = 7
    # for cir in circuits_name_list:
    #     cnots = col_row_eli_of_ibmq_lagos(f'./circuits/benchmark/B&D/B&D_circuits/{cir}-{qubits}qubits-delete-singlegate.qasm')
    #     test_gen_circuit_new(qubits, cnots, cir)

    # 遍历执行12量子位线路, 悟空架构
    # circuits_name_list = ["Bernstein-Vazirani"]
    # qubits = 12
    # for cir in circuits_name_list:
    #     cnots = col_row_eli_of_wukong(f'./circuits/benchmark/B&D/B&D_circuits/{cir}-{qubits}qubits-delete-singlegate.qasm')
    #     test_gen_circuit_new(qubits, cnots, cir)

    # 遍历执行20量子位线路
    # circuits_name_list = ["Bernstein-Vazirani"]
    # qubits = 20
    # for cir in circuits_name_list:
    #     # cnots = col_row_eli_of_ibmq_almaden(f'./circuits/benchmark/B&D/B&D_circuits/{cir}-{qubits}qubits-delete-singlegate.qasm')
    #     cnots = col_row_eli_of_ibmq_tokyo(f'./circuits/benchmark/B&D/B&D_circuits/{cir}-{qubits}qubits-delete-singlegate.qasm')
    #     test_gen_circuit_new(qubits, cnots, cir)

    # 一次执行5, 16, 27, 65, 127量子位线路
    """qubits_list = [5, 16, 27, 65, 127]
    for qubits in qubits_list:
        if qubits == 5:
            cnots = col_row_eli_of_ibmquatio(f'./circuits/benchmark/B&D/B&D_circuits/Bernstein-Vazirani-{qubits}qubits.qasm')
            # 生成线路
            test_gen_circuit_new(qubits, cnots)
        elif qubits == 16:
            cnots = col_row_eli_of_ibmq_guadalupe(f'./circuits/benchmark/B&D/B&D_circuits/Bernstein-Vazirani-{qubits}qubits.qasm')
            test_gen_circuit_new(qubits, cnots)
        elif qubits == 27:
            cnots = col_row_eli_of_ibmq_kolkata(f'./circuits/benchmark/B&D/B&D_circuits/Bernstein-Vazirani-{qubits}qubits.qasm')
            test_gen_circuit_new(qubits, cnots)
        else:
            pass"""

    # 遍历执行27量子位线路
    # circuits_name_list = ["Bernstein-Vazirani", "Deutsch_Josza"]
    # qubits = 27
    # for cir in circuits_name_list:
    #     cnots = col_row_eli_of_ibmq_kolkata(f'./circuits/benchmark/B&D/B&D_circuits/{cir}-27qubits-delete-singlegate.qasm')
    #     test_gen_circuit_new(qubits, cnots, cir)

    # 单独执行 Deutsch_Josza 27qubits 线路
    # cnots = col_row_eli_of_ibmq_kolkata(f'./circuits/benchmark/B&D/B&D_circuits/Deutsch_Josza-27qubits-delete-singlegate.qasm')
    # test_gen_circuit_new(27, cnots, "Deutsch_Josza")

    # 遍历执行65量子位线路
    # circuits_name_list = ["Bernstein-Vazirani"]
    # qubits = 65
    # for cir in circuits_name_list:
    #     cnots = col_row_eli_of_ibmq_manhattan(f'./circuits/benchmark/B&D/B&D_circuits/{cir}-65qubits-delete-singlegate.qasm')
    #     test_gen_circuit_new(qubits, cnots, cir)


    # ---------------------------------------------------------------------------
    cir = "Bernstein-Vazirani"
    qubits = 27
    start_time = time.time()
    cnots = col_row_eli_of_ibmq_kolkata(f'./circuits/benchmark/B&D/B&D_circuits/{cir}-{qubits}qubits-delete-singlegate.qasm')
    test_gen_circuit_new(qubits, cnots, cir)
    end_time = time.time()
    execution_time = end_time - start_time
    print("Execution Time:", execution_time, "seconds")
