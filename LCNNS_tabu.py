import itertools
import random
import time


import numpy as np
from qiskit import QuantumCircuit

from circuits_synthetic_without_HamiltonPath_backup import  col_row_eli_of_ibmq_square9Q, \
    col_row_eli_of_ibmq_qx5, col_row_eli_of_ibmq_tokyo
from my_tools.graph import Square9Q, Rigetti16qAspen_old
import networkx as nx
import matplotlib.pyplot as plt
import sys
import os
import warnings


warnings.filterwarnings("ignore")


def draw(edges):
    G = nx.Graph()
    G.add_edges_from(edges)

    plt.figure(figsize=(10, 6))
    nx.draw(G,with_labels=True, node_color='lightblue', node_size=800, font_size=10, font_weight='bold',
            edge_color='gray')
    plt.show()


def is_valid_order(edges, order):
    G = nx.Graph()
    G.add_edges_from(edges)

    for vertex in order:
        # print(f"Before removing {vertex}, edges:", list(G.edges()))
        G.remove_node(vertex)
        # print(f"After removing {vertex}, edges:", list(G.edges()))

        if G.nodes():
            if not nx.is_connected(G):
                return False

    return True



def cnot_to_qasm(cnot_list, num_qubits, output_file="circuit.qasm"):
    """
    Converts a CNOT gate list to a Qiskit circuit and writes it to a QASM file.

    Parameters.
        cnot_list (list of tuples): list of CNOT gates, each tuple is an index of control and target bits.
        num_qubits (int): number of quantum bits of the quantum circuit.
        output_file (str): path to the output QASM file, default is “circuit.qasm”.
    """
    qc = QuantumCircuit(num_qubits)

    for control, target in cnot_list:
        qc.cx(control, target)

    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(output_file, "w") as qasm_file:
        qasm_file.write(qc.qasm())



# Taboo Search Optimization
def objective_function(pi_order, edges, C_v, C_ij, weights):
    n = len(pi_order)
    term1 = 1
    for i in range(n):
        for j in range(i + 1, n):
            numerator = sum(C_v[v][(pi_order[i], pi_order[j])] / C_v[v]['total']
                            for v in range(n) if v != pi_order[i] and v != pi_order[j])
            denominator = C_ij[(pi_order[i], pi_order[j])]
            term1 *= numerator / denominator

    term2 = sum((m + 1) * np.mean([weights.get((pi_order[m], neighbor), 0.05) for neighbor in range(n) if
                                   (pi_order[m], neighbor) in edges or (neighbor, pi_order[m]) in edges]) for m in
                range(n))

    return term1 - term2


def tabu_search_order(n, edges, C_v, C_ij, weights, iterations, tabu_size, suitable_order_size):
    current_order = list(range(n))
    random.shuffle(current_order)

    # Make sure the initial current_order is valid
    while not is_valid_order(edges, current_order):
        random.shuffle(current_order)

    best_order = current_order[:]
    best_cost = objective_function(best_order, edges, C_v, C_ij, weights)

    tabu_list = []
    suitable_order = []

    for _ in range(iterations):
        neighborhood = []

        for i in range(n):
            for j in range(i + 1, n):
                neighbor = current_order[:]
                neighbor[i], neighbor[j] = neighbor[j], neighbor[i]

                # The generated neighbor must satisfy is_valid_order.
                if is_valid_order(edges, neighbor) and neighbor not in tabu_list:
                    neighborhood.append(neighbor)

        if not neighborhood:
            continue  # If there are no valid neighbors, the round is skipped

        neighborhood_costs = [objective_function(neigh, edges, C_v, C_ij, weights) for neigh in neighborhood]
        max_cost = max(neighborhood_costs)
        best_candidate = neighborhood[neighborhood_costs.index(max_cost)]

        if max_cost > best_cost:
            best_order, best_cost = best_candidate, max_cost

        tabu_list.append(best_candidate)

        # Ensure that the order in the added suitable_order satisfies is_valid_order.
        if is_valid_order(edges, best_candidate):
            suitable_order.append((best_candidate, max_cost))

        # If tabu_list exceeds the maximum length, remove the oldest element.
        if len(tabu_list) > tabu_size:
            tabu_list.pop(0)

        # Ensure that all orders in suitable_order satisfy is_valid_order
        while len(suitable_order) > suitable_order_size:
            suitable_order.sort(key=lambda x: x[1], reverse=True)
            suitable_order.pop()

        current_order = best_candidate

    suitable_order.sort(key=lambda x: x[1], reverse=True)
    return best_order, best_cost, suitable_order


if __name__ == '__main__':

    start_time = time.time()

    qubit = 16

    # # qx5
    edges = [(0, 1), (0, 15), (1, 2), (1, 14), (2, 3), (2, 13), (3, 4), (3, 12), (4, 5), (4, 11), (5, 6), (5, 10),
             (6, 7), (6, 9), (7, 8), (8, 9), (9, 10), (10, 11), (11, 12), (12, 13), (13, 14), (14, 15)]

    weights = {
        (1, 0): 0.024921342030678917, (1, 2): 0.04386952992715626,
        (2, 3): 0.030605771841479945, (3, 4): 0.035429301943697866,
        (3, 14): 0.04481252727157908, (5, 4): 0.03419031200973366,
        (6, 5): 0.05560385413564303, (6, 7): 0.0430544071637424,
        (6, 11): 0.026085358646657064, (7, 10): 0.01785777064032057,
        (8, 7): 0.021063772069955355, (9, 8): 0.029009283686826065,
        (9, 10): 0.026498444370525315, (11, 10): 0.027993700199691968,
        (12, 5): 0.051557316888027144, (12, 11): 0.027514912275282494,
        (12, 13): 0.031489707292669866, (13, 4): 0.029074231952134683,
        (13, 14): 0.014733467690550478, (15, 0): 0.018433175203418,
        (15, 2): 0.03419031200973366, (15, 14): 0.05560385413564303
    }

    file_name2 = f'../circuits/steiner/16qubits/8/Original0.qasm'

    # Pre-calculated Cv and Cij
    C_v = {
        v: {'total': 10, **{(i, j): 1 for i in range(qubit) for j in range(qubit) if i != j}}
        for v in range(qubit)
    }

    C_ij = {(i, j): 2 for i in range(qubit) for j in range(qubit) if i != j}

    best_order, best_cost, suitable_orders = tabu_search_order(16, edges, C_v, C_ij, weights, iterations=50,
                                                               tabu_size=100, suitable_order_size=50)
    # print(suitable_orders)

    list_of_orders = [item[0] for item in suitable_orders]

    min_len = float('inf')
    min_cnot_list = None

    for loo in list_of_orders:
        print(loo,end=' ')
        # Redirection of standard output to avoid output interference; If output is required, cancel the sys.
        try:
            sys.stdout = open(os.devnull, 'w')
            cnot_list = col_row_eli_of_ibmq_qx5(file_name2, loo)
            sys.stdout = sys.__stdout__

            if len(cnot_list) < min_len:
                min_len = len(cnot_list)
                min_cnot_list = cnot_list

            print(len(cnot_list))

        except Exception as e:
            continue

    # Output the cnot_list with the smallest length
    if min_cnot_list is not None:
        print("The cnot_list with the smallest length is:", min_cnot_list)
        print("Its length is:", min_len)
    else:
        print("No valid cnot_list found.")






