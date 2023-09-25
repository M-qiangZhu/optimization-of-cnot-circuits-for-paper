from itertools import combinations


def test01():
    v = []
    for i in range(1, 67):
        v.append(i)
    print(v)


def test02():
    v_e_dcit = {1: [7, 8], 7: [1, 13], 8: [1, 2, 13, 14], 2: [8, 9], 9: [2, 3, 14, 15], 3: [9, 10],
                10: [3, 4, 15, 16],
                4: [10, 11], 11: [4, 5, 16, 17], 5: [11, 12], 12: [5, 6, 17, 18], 6: [12], 13: [7, 8, 19, 20],
                14: [8, 9, 20, 21], 15: [9, 10, 21, 22], 16: [10, 11, 22, 23], 17: [11, 12, 23, 24],
                18: [12, 24],
                19: [13, 25], 20: [13, 14, 25, 26], 21: [14, 15, 26, 27], 22: [15, 16, 27, 28],
                23: [16, 17, 28, 29],
                24: [17, 18, 29, 30], 25: [19, 20, 31, 32], 26: [20, 21, 32, 33], 27: [21, 22, 33, 34],
                28: [22, 23, 34, 35], 29: [23, 24, 35], 30: [24, 36], 31: [25, 37], 32: [25, 26, 37, 38],
                33: [26, 27, 38, 39], 34: [27, 28, 39, 40], 35: [28, 29, 40, 41], 39: [36, 33, 34, 45, 46],
                36: [39, 30, 41, 42], 37: [31, 32, 43, 44], 38: [32, 33, 44, 45], 40: [34, 35, 46, 47],
                41: [35, 36, 47, 48], 42: [36, 48], 43: [37, 49], 44: [37, 38, 49, 50], 45: [38, 39, 50, 51],
                46: [39, 40, 51, 52], 47: [40, 41, 52, 53], 48: [41, 42, 53, 54], 49: [43, 44, 55, 56],
                50: [44, 45, 56, 57], 51: [45, 46, 57, 58], 52: [46, 47, 58, 59], 53: [47, 48, 59, 60],
                54: [48, 60],
                55: [49, 61], 56: [49, 50, 61, 62], 57: [50, 51, 62, 63], 58: [51, 52, 63, 64],
                59: [52, 53, 64, 65],
                60: [53, 54, 65, 66], 61: [55, 56], 62: [56, 57], 63: [57, 58], 64: [58, 59], 65: [59, 60],
                66: [60]}
    edges = []
    for v, next_v_list in v_e_dcit.items():
        print(v)
        for next_v in next_v_list:
            print(next_v)
            edges.append((v, next_v))
        print(edges)
        print("-" * 20)
    sort_edges = sorted(edges, key=lambda x: x[0])
    print(sort_edges)

    # 将有向边转换为无向边
    undirected_edges = set()
    for edge in sort_edges:
        undirected_edges.add(frozenset(edge))
        undirected_edges.add(frozenset(reversed(edge)))

    # 输出无向边集合
    print(len(undirected_edges))
    new_edges = []
    for u_e in undirected_edges:
        new_edges.append(tuple(u_e))
    sort_new_edges = sorted(new_edges, key=lambda x: x[0])
    print(sort_new_edges)


def test03():
    my_list = [i for i in range(1, 67)]

    sublists = [my_list[i:i + 6] for i in range(0, 66, 6)]
    print(sublists)

    dict_pos = {}
    for row in [10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0]:
        row_v_list = sublists[row]
        print(row_v_list)
        for i in range(len(sublists[row])):
            if row % 2 == 0:
                dict_pos[row_v_list[i]] = [i * 2 + 1, 10 - row]
            if row % 2 != 0:
                dict_pos[row_v_list[i]] = [i * 2, 10 - row]
    print(dict_pos)


def row_add(row1, row2):
    """Add r0 to r1"""
    for i, v in enumerate(row1):
        if v:
            row2[i] = 0 if row2[i] else 1  # 当row1中某个值为1时, 将row2中的对应位置的值取反
    return row2


def get_row(m, row_index):
    """
    根据矩阵获取指定的行
    :param m: 矩阵
    :param row_index: 需要获取的行的索引
    :return:
    """
    return m.data[row_index, :].tolist()


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


if __name__ == '__main__':
    # print([i for i in range(65)])
    dic1 = {}
    for i in range(20):
        dic1[i] = i
    print(dic1)

#     dict = {}
#     for i in range(10):
#         dict[i] = [2*i, 0]
#     print(dict)
#     for i in range(11):
#         dict[i+13] = [2*i, -4]
#     print(dict)
#     for i in range(11):
#         dict[i+27] = [2*i, -8]
#     print(dict)
#     for i in range(11):
#         dict[i+41] = [2*i, -12]
#     print(dict)
#     for i in range(10):
#         dict[i+55] = [2*i+2, -16]
#     print(dict)

    # print('0'*27)

    # dict = {}
    # for i in range(5):
    #     dict[i] = [2*i, 0]
    # print(dict)
    # for i in range(5):
    #     dict[i+5] = [2*i, -2]
    # print(dict)
    # for i in range(5):
    #     dict[i+10] = [2*i, -4]
    # print(dict)
    # for i in range(5):
    #     dict[i+15] = [2*i, -6]
    # print(dict)


