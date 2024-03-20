# -*- coding: utf-8 -*-

"""
    @Author kungfu
    @Date 2024/3/10 19:09
    @Describe 
    @Version 1.0
"""
import numpy as np

def create_unit_vector(size, index):
    return [1 if i == index else 0 for i in range(size)]



def find_matching_rows(matrix):
    num_rows, num_cols = len(matrix), len(matrix[0])
    unit_vectors = [create_unit_vector(num_cols, i) for i in range(num_rows)]
    coefficients = np.array(unit_vectors)

    matching_rows = []
    for i in range(num_rows):
        target_vector = np.array(unit_vectors[i])
        other_vectors = np.delete(coefficients, i, axis=0)
        b = np.ones(num_rows - 1)  # 右边的向量都是单位向量，和为1
        try:
            x = np.linalg.solve(other_vectors, b)
            # 检查是否所有系数都是非负数
            if all(coef >= 0 for coef in x):
                matching_rows.append(i)
        except np.linalg.LinAlgError:
            # 如果无法求解线性方程组，表示无法找到对应的行
            pass

    return matching_rows

# 测试
matrix = [
    [1, 0, 0, 0, 0],
    [0, 1, 0, 0, 0],
    [0, 0, 1, 0, 1],
    [0, 0, 0, 1, 1],
    [0, 0, 0, 0, 1]
]

matching_rows = find_matching_rows(matrix)
print("Matching rows:", matching_rows)
