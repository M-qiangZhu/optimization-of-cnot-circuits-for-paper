# -*- coding: utf-8 -*-

"""
    @Author kungfu
    @Date 2022/7/20 15:12
    @Describe 使用Python中NetworkX包绘制深度神经网络结构图
    @Version 1.0
"""
# 导入相应包
import networkx as nx
from networkx.algorithms import approximation
import matplotlib.pyplot as plt


# 绘制DAG图
def draw_graph(G, pos):
    # figsize:指定figure的宽和高，单位为英寸
    plt.figure(figsize=(20, 10), dpi=200)
    # plt.title('Network Structure')  # 神经网络结构图标题
    # plt.xlim(-10, 170)  # 设置X轴坐标范围
    # plt.ylim(-10, 150)  # 设置Y轴坐标范围
    nx.draw(
        G,
        pos=pos,  # 点的位置
        node_color='#2D3EB7',  # 顶点颜色
        edge_color='black',  # 边的颜色
        font_color='#FFF',  # 字体颜色
        font_size=40,  # 文字大小
        font_family='Arial Unicode MS',  # 字体样式
        node_size=5000,  # 顶点大小
        with_labels=True,  # 显示顶点标签
        width=8.0,  # 边的宽度
        linewidths=8.0,  # 线宽
    )
    # 保存图片，图片大小为640*480
    plt.savefig('/Users/kungfu/Desktop/graph.png')

    # 显示图片
    plt.show()


class Graph:
    def __init__(self, name, vertices, edges, *weights, weighted=False):
        self.name = name
        self.vertices = vertices
        if not weighted:
            self.edges = edges
        else:
            self.edges = [e + w for e in self.edges for w in weights]

    def gen_graph(self):
        graph = nx.Graph()
        graph.add_nodes_from(self.vertices)
        graph.add_edges_from(self.edges)


class IbmQuito:
    def __init__(self):
        self.vertex = [0, 1, 2, 3, 4]
        self.edges = [(0, 3), (3, 4), (3, 2), (2, 1)]
        # self.edges = [(0, 2), (2, 1), (2, 3), (3, 4)]
        self.pos = {0: [0, 0], 3: [2, 0], 4: [4, 0], 2: [2, -2], 1: [2, -4]}
        # self.pos = {0: [0, 0], 2: [2, 0], 1: [4, 0], 3: [2, -2], 4: [2, -4]}
        self.graph = nx.Graph()
        self.graph.add_nodes_from(self.vertex)
        self.graph.add_edges_from(self.edges)

    def get_graph(self):
        return self.graph

    def draw_graph(self):
        draw_graph(self.graph, self.pos)

    def get_degree(self):
        return self.graph.degree()


class IbmqGuadalupe:
    def __init__(self):
        self.vertex = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
        self.edges = [(0, 1), (1, 4), (1, 2), (4, 7), (7, 6), (7, 10), (10, 12), (12, 15), (12, 13), (13, 14), (2, 3), (3, 5), (5, 8), (8, 9), (8, 11), (11, 14)]
        self.pos = {6: [6, 2],
                    0: [0, 0], 1: [2, 0], 4: [4, 0], 7: [6, 0], 10: [8, 0], 12: [10, 0], 15: [12, 0],
                    2: [2, -2], 13: [10, -2],
                    3: [2, -4], 5: [4, -4], 8: [6, -4], 11: [8, -4], 14: [10, -4],
                    9: [6, -6]}
        self.graph = nx.Graph()
        self.graph.add_nodes_from(self.vertex)
        self.graph.add_edges_from(self.edges)

    def get_graph(self):
        return self.graph

    def draw_graph(self):
        draw_graph(self.graph, self.pos)

    def get_degree(self):
        return self.graph.degree()


class IbmqGuadalupe_new:
    def __init__(self):
        self.vertex = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
        self.edges = [(0, 1), (1, 4), (1, 2), (4, 7), (7, 6), (7, 10), (10, 14), (14, 15), (14, 13), (13, 12), (2, 3), (3, 5), (5, 9), (9, 8), (9, 11), (11, 12)]
        self.pos = {6: [6, 2],
                    0: [0, 0], 1: [2, 0], 4: [4, 0], 7: [6, 0], 10: [8, 0], 14: [10, 0], 15: [12, 0],
                    2: [2, -2], 13: [10, -2],
                    3: [2, -4], 5: [4, -4], 9: [6, -4], 11: [8, -4], 12: [10, -4],
                    8: [6, -6]}
        self.graph = nx.Graph()
        self.graph.add_nodes_from(self.vertex)
        self.graph.add_edges_from(self.edges)

    def get_graph(self):
        return self.graph

    def draw_graph(self):
        draw_graph(self.graph, self.pos)

    def get_degree(self):
        return self.graph.degree()


class IbmqKolkata:
    def __init__(self):
        self.vertex = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26]
        self.edges = [(0, 1), (1, 4), (1, 2), (4, 7), (7, 6), (7, 10), (10, 12), (12, 15), (12, 13), (15, 18), (18, 17), (18, 21), (21, 23), (23, 24), (2, 3),
                      (3, 5), (5, 8), (8, 9), (8, 11), (11, 14), (14, 13), (14, 16), (16, 19), (19, 20), (19, 22), (22, 25), (25, 24), (25, 26)]
        self.pos = {6: [6, 2], 17: [14, 2],
                    0: [0, 0], 1: [2, 0], 4: [4, 0], 7: [6, 0], 10: [8, 0], 12: [10, 0], 15: [12, 0], 18: [14, 0], 21: [16, 0], 23: [18, 0],
                    2: [2, -2], 13: [10, -2], 24: [18, -2],
                    3: [2, -4], 5: [4, -4], 8: [6, -4], 11: [8, -4], 14: [10, -4], 16: [12, -4], 19: [14, -4], 22: [16, -4], 25: [18, -4], 26: [20, -4],
                    9: [6, -6], 20: [14, -6]}
        self.graph = nx.Graph()
        self.graph.add_nodes_from(self.vertex)
        self.graph.add_edges_from(self.edges)

    def get_graph(self):
        return self.graph

    def draw_graph(self):
        draw_graph(self.graph, self.pos)

    def get_degree(self):
        return self.graph.degree()


class IbmqKolkata_new:
    def __init__(self):
        self.vertex = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26]
        self.edges = [(0, 1), (1, 4), (1, 2), (4, 7), (7, 6), (7, 10), (10, 12), (12, 15), (12, 13), (15, 18), (18, 17), (18, 21), (21, 23), (23, 24), (2, 3),
                      (3, 5), (5, 9), (9, 8), (9, 11), (11, 14), (14, 13), (14, 16), (16, 20), (20, 19), (20, 22), (22, 25), (25, 24), (25, 26)]
        self.pos = {6: [6, 2], 17: [14, 2],
                    0: [0, 0], 1: [2, 0], 4: [4, 0], 7: [6, 0], 10: [8, 0], 12: [10, 0], 15: [12, 0], 18: [14, 0], 21: [16, 0], 23: [18, 0],
                    2: [2, -2], 13: [10, -2], 24: [18, -2],
                    3: [2, -4], 5: [4, -4], 9: [6, -4], 11: [8, -4], 14: [10, -4], 16: [12, -4], 20: [14, -4], 22: [16, -4], 25: [18, -4], 26: [20, -4],
                    8: [6, -6], 19: [14, -6]}
        self.graph = nx.Graph()
        self.graph.add_nodes_from(self.vertex)
        self.graph.add_edges_from(self.edges)

    def get_graph(self):
        return self.graph

    def draw_graph(self):
        draw_graph(self.graph, self.pos)

    def get_degree(self):
        return self.graph.degree()


class Xiaohong66:
    def __init__(self):
        self.vertex = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41,
                       42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66]
        self.edges = [(1, 7), (8, 2), (8, 13), (8, 14), (8, 1), (9, 2), (9, 14), (9, 15), (9, 3), (10, 4), (10, 15), (10, 3), (11, 5), (11, 4), (12, 5), (12, 6), (13, 7), (16, 23),
                      (16, 22), (16, 10), (16, 11), (17, 23), (17, 12), (17, 11), (18, 12), (19, 13), (20, 13), (20, 14), (21, 15), (21, 14), (22, 15), (24, 30), (24, 18),
                      (24, 29), (24, 17), (25, 20), (25, 31), (25, 19), (26, 21), (26, 20), (27, 21), (27, 22), (28, 23), (28, 22), (29, 23), (32, 26), (32, 25), (32, 37),
                      (32, 38), (33, 38), (33, 39),
                      (33, 27), (33, 26), (34, 28), (34, 39), (34, 27), (35, 28), (35, 29), (36, 30), (36, 29), (37, 31), (40, 35), (40, 47), (40, 46), (40, 34), (41, 36),
                      (41, 35), (41, 47),
                      (42, 36), (43, 37), (44, 37), (44, 38), (45, 38), (45, 39), (46, 39), (48, 41), (48, 54), (48, 53), (48, 42), (49, 55), (49, 44), (49, 43), (50, 45),
                      (50, 44), (51, 45),
                      (51, 46), (52, 47), (52, 46), (53, 47), (56, 49), (56, 50), (56, 61), (56, 62), (57, 51), (57, 50), (57, 63), (57, 62), (58, 63), (58, 52), (58, 51),
                      (59, 52), (59, 53),
                      (60, 54), (60, 53), (61, 55), (64, 58), (64, 59), (65, 60), (65, 59), (66, 60)]
        self.pos = {61: [1, 0], 62: [3, 0], 63: [5, 0], 64: [7, 0], 65: [9, 0], 66: [11, 0], 55: [0, 1], 56: [2, 1], 57: [4, 1], 58: [6, 1], 59: [8, 1], 60: [10, 1], 49: [1, 2],
                    50: [3, 2], 51: [5, 2], 52: [7, 2], 53: [9, 2], 54: [11, 2], 43: [0, 3], 44: [2, 3], 45: [4, 3], 46: [6, 3], 47: [8, 3], 48: [10, 3], 37: [1, 4], 38: [3, 4],
                    39: [5, 4],
                    40: [7, 4], 41: [9, 4], 42: [11, 4], 31: [0, 5], 32: [2, 5], 33: [4, 5], 34: [6, 5], 35: [8, 5], 36: [10, 5], 25: [1, 6], 26: [3, 6], 27: [5, 6], 28: [7, 6],
                    29: [9, 6],
                    30: [11, 6], 19: [0, 7], 20: [2, 7], 21: [4, 7], 22: [6, 7], 23: [8, 7], 24: [10, 7], 13: [1, 8], 14: [3, 8], 15: [5, 8], 16: [7, 8], 17: [9, 8], 18: [11, 8],
                    7: [0, 9], 8: [2, 9],
                    9: [4, 9], 10: [6, 9], 11: [8, 9], 12: [10, 9], 1: [1, 10], 2: [3, 10], 3: [5, 10], 4: [7, 10], 5: [9, 10], 6: [11, 10]}
        self.graph = nx.Graph()
        self.graph.add_nodes_from(self.vertex)
        self.graph.add_edges_from(self.edges)

    def get_graph(self):
        return self.graph

    def draw_graph(self):
        draw_graph(self.graph, self.pos)

    def get_degree(self):
        return self.graph.degree()


class Liner:
    def __init__(self):
        self.vertex = [0, 1, 2, 3, 4]
        self.edges = [(0, 1), (1, 2), (2, 3), (3, 4)]
        self.pos = {0: [0, 0], 1: [1, 0], 2: [2, 0], 3: [3, 0], 4: [4, 0]}
        # self.pos = {0: [0, 0], 2: [2, 0], 1: [4, 0], 3: [2, -2], 4: [2, -4]}
        self.graph = nx.Graph()
        self.graph.add_nodes_from(self.vertex)
        self.graph.add_edges_from(self.edges)

    def get_graph(self):
        return self.graph

    def draw_graph(self):
        draw_graph(self.graph, self.pos)

    def get_degree(self):
        return self.graph.degree()


def test_st1():
    """ 测试Steiner树1 """
    # ver = ['0', '1', '2', '3', '4']
    # edges = [('0', '1'), ('1', '2'), ('1', '3'), ('3', '4')]
    # pos = {'0': [0, 0], '1': [2, 0], '2': [4, 0], '3': [2, -2], '4': [2, -4]}

    # ver = [0, 1, 2, 3, 4]
    # edges = [(0, 1), (1, 2), (1, 3), (3, 4)]
    # pos = {0: [0, 0], 1: [2, 0], 2: [4, 0], 3: [2, -2], 4: [2, -4]}
    # graph = nx.Graph()
    # graph.add_nodes_from(ver)
    # graph.add_edges_from(edges)
    # draw_graph(graph, pos)
    # nx.draw(graph)
    # plt.show()
    # tree = graph.subgraph([0, 1, 4])
    # print(tree.edges)

    # G = nx.dodecahedral_graph()
    # nx.draw(G)
    # nx.draw(G, pos=nx.spring_layout(G))  # use spring layout
    # limits = plt.axis("off")  # turn off axis
    # plt.show()

    ibmq_quito = IbmQuito()
    graph = ibmq_quito.get_graph()
    # draw_graph(graph, ibmq_quito.pos)
    ibmq_quito.draw_graph()
    print(graph.degree[0])
    print(graph.degree[1])
    print(graph.degree[2])
    print(graph.degree[3])
    print(graph.degree[4])

    st = approximation.steiner_tree(graph, [0, 1, 4])
    print(st.nodes)
    print(st.edges)


def test_st2():
    """ 测试Steiner树2 """
    ibmq_quito = IbmQuito()
    graph = ibmq_quito.get_graph()
    nodes = list(graph.nodes)
    print(nodes)
    print(type(nodes))
    graph.remove_node()


def test_Xiaohong66():
    xh66 = Xiaohong66()
    xh66.draw_graph()


def test_Liner():
    liner = Liner()
    liner.draw_graph()


def test_IbmqGuadalupe():
    guadalupe = IbmqGuadalupe()
    guadalupe.draw_graph()


def test_IbmqKolkata():
    kolkata = IbmqKolkata_new()
    kolkata.draw_graph()


if __name__ == '__main__':
    # test_st1()
    # test_Xiaohong66()
    # test_IbmqGuadalupe()
    test_IbmqKolkata()
