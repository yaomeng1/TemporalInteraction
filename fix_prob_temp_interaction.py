import networkx as nx
import random
import numpy as np
import pickle
import matplotlib.pyplot as plt
from numba import jit
import os
import multiprocessing
from multiprocessing import Process, Manager
import functools
import time
import math
from utils import rand_pick_list, nbr_array_mat, edge_pair_array
import scipy.io as scio


@jit(nopython=True)
def single_round(state_dict, payoff_array, game_matrix, edge_mat, deg_array):
    """
    play game on each edge
    """

    for i in range(edge_mat.shape[0]):
        nodex = edge_mat[i, 0]
        nodey = edge_mat[i, 1]
        payoff_array[nodex] += game_matrix[state_dict[nodex]][state_dict[nodey]]
        payoff_array[nodey] += game_matrix[state_dict[nodey]][state_dict[nodex]]

    return payoff_array / deg_array


@jit(nopython=True)
def replicate_dynamic(state_array, payoff_array, nbr_mat, deg_array, nodesnum, w):
    """
    replicator dynamic after single round game: DB

    """

    update_node = np.random.choice(np.arange(nodesnum))
    nbrs_num = deg_array[update_node]
    nbrs_array = nbr_mat[update_node][:nbrs_num]

    fitness_array = 1 + w * payoff_array[nbrs_array]
    prob_array = fitness_array / np.sum(fitness_array)
    state_array[update_node] = state_array[rand_pick_list(nbrs_array, prob_array)]

    return state_array


def evolution(game_matrix, edge_mat_I_list, deg_array_I_list, nbrs_mat_R, deg_array_R, nodesnum, w, g):
    """
    whole process of evolution for 1e10 times of generation, we calculate the average payoff here
    """

    total_generation = int(1e10)
    snapshot_length = len(edge_mat_I_list)
    payoff_array = np.zeros(nodesnum, dtype=np.float_)
    state_array = np.zeros(nodesnum, dtype=np.int_)
    coop_ini = np.random.choice(nodesnum)
    state_array[coop_ini] = 1

    for step in range(total_generation):
        idx = int(step / g) % snapshot_length
        payoff_array = single_round(state_array, payoff_array, game_matrix, edge_mat_I_list[idx], deg_array_I_list[idx])
        state_array = replicate_dynamic(state_array, payoff_array, nbrs_mat_R, deg_array_R, nodesnum, w)
        payoff_array[:] = 0
        coord = np.sum(state_array)
        if coord > nodesnum - 1:
            return 1
        if coord == 0:
            return 0

    return coord / nodesnum


def process(core, b, edge_mat_I_list, deg_array_I_list, nbrs_mat_R, deg_array_R, nodesnum, g):
    w = 0.01

    game_matrix = np.zeros((2, 2))
    game_matrix[0][0] = 0  # P defect--defect
    game_matrix[0][1] = b  # T d-c
    game_matrix[1][0] = -1  # S
    game_matrix[1][1] = b - 1  # R

    repeat_time = int(1e7)
    repeat_array = np.zeros(repeat_time)

    for rep in range(repeat_time):
        coord_freq = evolution(game_matrix, edge_mat_I_list, deg_array_I_list, nbrs_mat_R, deg_array_R, nodesnum, w, g)
        repeat_array[rep] = coord_freq

    return np.sum(repeat_array == 1) / (np.sum(repeat_array == 1) + np.sum(repeat_array == 0))


if __name__ == "__main__":
    with open('./temp_BA_generate_snap_mat_n20_k6_0.pk', 'rb') as f:
        network_mat = pickle.load(f)  # stack of matrix of snapshot of interaction

    # obtain the mat of edges on two interaction graph
    edge_mat_list = [edge_pair_array(network_mat[idx, :, :]) for idx in range(network_mat.shape[0])]
    deg_array_list = [np.sum(network_mat[idx, :, :], axis=1) for idx in range(network_mat.shape[0])]
    # to avoid dividing by 0, replace the degree of isolated node in deg_array by 1
    for idx in range(len(deg_array_list)):
        deg_array_I = deg_array_list[idx]
        deg_array_I[deg_array_I == 0] = 1
        deg_array_list[idx] = deg_array_I
    # obtain the mat of neighbor: idx:-> neighbor array of replacement graph
    mAdj_aggregated = np.sum(network_mat, axis=0)  # taken as the replacement graph
    nbrs_mat_R, deg_array_R = nbr_array_mat(mAdj_aggregated)
    nodesnum = mAdj_aggregated.shape[0]

    # parallel computation of fixation probability
    g_list = [nodesnum]
    for g in g_list:
        print("g = " + str(g))
        if g == nodesnum:
            b_para_list = np.round(np.arange(19.4, 23.4, 2), decimals=1)
        fix_prob_b_list = []
        for b_para in b_para_list:
            core_list = np.arange(16)  # 64-cpu core

            pool = multiprocessing.Pool()
            t1 = time.time()

            pt = functools.partial(process, b=b_para, edge_mat_I_list=edge_mat_list, deg_array_I_list=deg_array_list,
                                   nbrs_mat_R=nbrs_mat_R, deg_array_R=deg_array_R, nodesnum=nodesnum, g=g)
            coor_freq_list = pool.map(pt, core_list)

            coor_freq_core = sum(coor_freq_list) / len(coor_freq_list)

            pool.close()
            pool.join()
            t2 = time.time()
            print("Total time:" + (t2 - t1).__str__())
            print((b_para, coor_freq_core))
            fix_prob_b_list.append(coor_freq_core)
        file = "./temp_BA_generate_snap_mat_n"+str(nodesnum)+"_k6_0_g" + str(int(g)) + "_b" + str(
            min(b_para_list)) + "_" + str(
            max(b_para_list)) + "_2.pk"
        with open(file, 'wb') as f:
            pickle.dump(fix_prob_b_list, f)
