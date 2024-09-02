import numpy as np
import torch
import pickle
import time
import scipy.io as scio

device = 'cuda'

def getBCratioTempInteract(mAdj: np.ndarray, mInt: np.ndarray, g: int):
    """
    without the need of T
    :param mAdj:  replacement adjacent matrix
    :param mInt:  interaction adjacent matrix: n*n*T
    :param g:  a snapshot hold for "g" time steps before switching

    :return:
    """
    mAdj = mAdj.astype(np.float32)
    mInt = mInt.astype(np.float32)

    w = np.sum(mAdj, axis=1).reshape(-1, 1)
    n = (torch.from_numpy(np.array([len(w)]).astype(np.float32))).to(device)
    pi = torch.from_numpy(w / np.sum(w)).to(device)
    P10 = torch.from_numpy(mAdj / w).to(device)
    P20 = torch.matmul(P10, P10)

    deg_I_time = np.sum(mInt, axis=1)  # n*T
    Qj = torch.from_numpy((deg_I_time > 0).astype(np.float32)).to(device)  # n*T
    deg_I_time[deg_I_time == 0] = 1  # n*T
    Q10_time = torch.from_numpy(mInt / np.expand_dims(deg_I_time, axis=1)).to(device)  # n*n*T

    # P(T_coal>t)  Prob_mat(:,:,1)-->P(T_coal>0)
    # Prob_mat = torch.from_numpy(np.ones(mAdj.shape) - np.eye(len(w))).to(device)
    Prob_mat = torch.ones(len(w), len(w)).to(device) - torch.eye(len(w), device=device)

    T20 = (torch.zeros(1)).to(device)
    T21 = (torch.zeros(1)).to(device)
    T01 = (torch.zeros(1)).to(device)
    eye_matrix = torch.eye(len(w), dtype=torch.bool, device=device)
    idx = 0
    while Prob_mat[0,1] > 1e-4:
        t = int(idx / g) % mInt.shape[2]
        T20 += torch.sum(pi * P20 * torch.reshape(Qj[:, t], [1, -1]) * Prob_mat)
        T21 += torch.sum(torch.matmul(pi * P20, Q10_time[:, :, t]) * Prob_mat)
        T01 += torch.sum(pi * Q10_time[:, :, t] * Prob_mat)

        # caluculate the P(T_coal>t) at time "t"
        Prob_mat = 1 / n * torch.matmul(P10, Prob_mat) \
                   + 1 / n * torch.matmul(Prob_mat, P10.T) + (n - 2) / n * Prob_mat
        Prob_mat[eye_matrix] = 0
        idx += 1

    result_device = T20 / (T21 - T01)

    # move result to CPU and return
    return result_device.to('cpu').item()




if __name__ == "__main__":

    with open('./temp_BA_generate_snap_mat_n100_k6_0.pk', 'rb') as f:
        mInt = pickle.load(f)  # stack of matrix of snapshot of interaction
    mInt = np.transpose(mInt, (1, 2, 0))
    mAdj = (np.sum(mInt, axis=2) > 0).astype(np.float_)  # taken as the replacement graph


    bcr = getBCratioTempInteract(mAdj, mInt, g=mAdj.shape[0])


    print("(b/c)* = ", bcr)


