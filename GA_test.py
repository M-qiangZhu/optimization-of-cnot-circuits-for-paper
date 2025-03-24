import time

import numpy as np
from circuits_synthetic_without_HamiltonPath_backup import get_row, row_add, get_circuits_to_matrix, get_ei
from concurrent.futures import ThreadPoolExecutor, as_completed
from concurrent.futures import ProcessPoolExecutor, as_completed


# 生成种群
def initialize_population(population_size, length):
    return [np.random.choice([0, 1], size=length).tolist() for _ in range(population_size)]

# 适应度函数
def fitness(chromosome, m, row_tar, ei):
    length = m.rank()
    tmp_row = ei.copy()
    for i in range(len(chromosome)):
        if chromosome[i] == 1:
            row = get_row(m, i)
            row_add(row, tmp_row)
    if tmp_row == row_tar:
        cost = 1
    else:
        cost = 0
    fitness_score = cost * (length - sum(tmp_row))
    # print(fitness_score,end=' ')
    return fitness_score


def compute_fitness_concurrently(population, m, row_tar, ei):
    with ProcessPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(fitness, individual, m, row_tar, ei) for individual in population]
        fitness_scores = []
        for future in as_completed(futures):
            fitness_scores.append(future.result())
    return fitness_scores

# 选择
def select(population, fitness_scores):
    sorted_indices = sorted(range(len(fitness_scores)), key=lambda i: fitness_scores[i], reverse=True)
    # 选择前N个最优的个体
    selected_indices = sorted_indices[:2]
    return [population[i] for i in selected_indices]


#交叉
def crossover(parent1, parent2):
    point = np.random.randint(1, len(parent1))  # 随机选择交叉点
    child1 = parent1[:point] + parent2[point:]
    child2 = parent2[:point] + parent1[point:]
    return child1, child2

# 变异
def mutate(chromosome, mutation_rate=0.1):
    for i in range(len(chromosome)):
        if np.random.rand() < mutation_rate:  # 以一定概率进行变异
            chromosome[i] = 1 - chromosome[i]  # 基因变异


def genetic_algorithm(m, row_tar, ei, population_size=20, generations=100, mutation_rate=0.01):
    chromosome_length = m.rank()
    # 初始化种群
    population = [np.random.choice([0, 1], size=chromosome_length).tolist() for _ in range(population_size)]
    for generation in range(generations):
        # 计算适应度
        fitness_scores = [fitness(individual, m, row_tar, ei) for individual in population]
        # fitness_scores = compute_fitness_concurrently(population, m, row_tar, ei)
        # 选择
        population = select(population, fitness_scores)
        # 创建下一代
        next_generation = []
        while len(next_generation) < population_size:
            parent1, parent2 = select(population, fitness_scores)  # 选择两个父代
            child1, child2 = crossover(parent1, parent2)  # 交叉
            mutate(child1, mutation_rate)  # 变异
            mutate(child2, mutation_rate)
            next_generation.extend([child1, child2])
        population = next_generation[:population_size]
        # 可以在这里添加代码打印每代的最佳适应度，以观察算法进展
    # 找出最终种群中适应度最高的个体
    final_fitness_scores = [fitness(individual, m, row_tar, ei)  for individual in population]
    best_index = np.argmax(final_fitness_scores)
    return population[best_index]


if __name__ == '__main__':
    start_time = time.time()
    matrix = get_circuits_to_matrix(r'./circuits/benchmark/B&D/B&D_circuits/Deutsch_Josza-5qubits-delete-singlegate.qasm')
    print(matrix)
    print(matrix.data)
    index = 2
    ei = get_ei(matrix, index)
    population = genetic_algorithm(matrix, get_row(matrix, index), ei, population_size=200, generations=10000, mutation_rate=0.1)
    print(population)
    print(fitness(population, matrix, get_row(matrix, index), ei))
    end_time = time.time()
    execution_time = end_time - start_time
    print(execution_time)
