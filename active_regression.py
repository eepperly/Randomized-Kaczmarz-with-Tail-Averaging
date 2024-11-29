#!/usr/bin/env python3

from tark import tark, mylstsq
import numpy as np

def volume_sample_orth(Q): # rpcholesky on Q @ Q.T
    d = np.sum(Q**2, axis=1)
    F = np.zeros(Q.shape)
    idx = np.zeros(Q.shape[1], dtype=int)
    for i in range(Q.shape[1]):
        idx[i] = np.random.choice(len(d), p=d/sum(d))
        F[:,i] = Q @ Q[idx[i],:].T - F[:,:i] @ F[idx[i],:i].T
        F[:,i] /= np.sqrt(F[idx[i],i])
        d -= F[:,i]**2
        d[d < 0] = 0
    return idx

def leverage_sample(A, num, input_is_orth=False):
    if not input_is_orth:
        Q, _ = np.linalg.qr(A, mode="reduced")
    else:
        Q = A
    d = np.sum(Q**2, axis=1)
    idx = np.random.choice(A.shape[0], size=num, p=d/sum(d), replace=True)
    return idx, d

def active_tark(A, b = None, b_fun = None, entry_budget = None, output_history=False):
    if not (b is None):
        if not (b_fun is None):
            raise ValueError("Cannot input both b and b_fun arguments")
        b_fun = lambda idx: b[idx]
    if b_fun is None:
        raise ValueError("Must input either b or b_fun")
    if entry_budget is None:
        entry_budget = A.shape[0] // 10
    if entry_budget < A.shape[1]:
        raise ValueError("Entry budget must be higher than number of columns of A")

    Q, R = np.linalg.qr(A, mode="reduced")

    idx = volume_sample_orth(Q)
    y0 = np.linalg.solve(Q[idx,:], b_fun(idx))

    num_steps = entry_budget - A.shape[1]
    if num_steps > 0:
        y = tark(Q, b_fun = b_fun, num_steps = num_steps, burn_in = num_steps//2, output_history = output_history)
    else:
        y = [] if output_history else y0

    if output_history:
        return [np.linalg.solve(R, y0)] + [np.linalg.solve(R, yy) for yy in y]
    else:
        return np.linalg.solve(R, y)
    
def active_leverage(A, b = None, b_fun = None, entry_budget = None):
    if not (b is None):
        if not (b_fun is None):
            raise ValueError("Cannot input both b and b_fun arguments")
        b_fun = lambda idx: b[idx]
    if b_fun is None:
        raise ValueError("Must input either b or b_fun")
    if entry_budget is None:
        entry_budget = A.shape[0] // 10
    if entry_budget < A.shape[1]:
        raise ValueError("Entry budget must be higher than number of columns of A")

    idx, d = leverage_sample(A, entry_budget)
    AS = A[idx,:] / np.sqrt(d[idx,np.newaxis])
    bS = b_fun(idx) / np.sqrt(d[idx])

    return mylstsq(AS, bS)
