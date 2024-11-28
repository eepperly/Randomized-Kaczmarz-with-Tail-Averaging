#!/usr/bin/env python

import numpy as np
from tqdm import tqdm

def tark(A, b, num_steps = None, burn_in = None, output_history = True, num_threads=1, under_relax = lambda T: 1.0, mu_reg = None, lamb_reg = None, dual = False):
    if len(b.shape) > 1 and b.shape[1] > 1:
        raise ValueError("RK methods only implemented for vectors b")
    
    if num_steps is None:
        num_steps = A.shape[0]
    if burn_in is None:
        burn_in = num_steps // 2
        
    if mu_reg is None:
        if lamb_reg is None:
            mu = 1.0
            lamb = 0.0
        else:
            mu = 1.0 / (1.0 + lamb_reg / np.linalg.norm(A, 'fro')**2)
            lamb = lamb_reg
    else:
        if lamb_reg is None:
            mu = mu_reg
            lamb = (1 - mu) / mu * np.linalg.norm(A, 'fro')**2
        else:
            raise ValueError("Cannot set both mu_reg and lamb_reg parameters")

    if dual:
        dual_vector = np.zeros(b.shape)
        if num_threads > 1:
            raise ValueError("Dual RK not implemented with multiple threads")

    x = np.zeros([A.shape[1]] + list(b.shape[1:]))
    output = np.zeros([A.shape[1]] + list(b.shape[1:]))

    weights = np.sum(A**2, axis=1)
    weights /= sum(weights)
    idxs = np.random.choice(np.arange(A.shape[0]), size = num_threads * num_steps, p=weights)
    history = []

    for t in tqdm(range(num_steps)):
        if num_threads > 1:
            i = idxs[t*num_threads:(t+1)*num_threads]
            x += under_relax(t) * np.mean(((b[i] - A[i,:] @ x)[:,np.newaxis] *  A[i,:]) / np.sum(A[i,:]**2,axis=1)[:,np.newaxis], axis=0)
        elif dual:
            i = idxs[t]
            update = under_relax(t) * (b[i] - A[i,:] @ x - lamb * dual_vector[i]) / (np.linalg.norm(A[i,:])**2 + lamb)
            dual_vector[i] += update
            x += update * A[i,:]
        else:
            i = idxs[t]
            x += under_relax(t) * (A[i,:] * (b[i] - A[i,:] @ x)) / np.linalg.norm(A[i,:])**2
        if mu != 1.0 and not dual:
            x *= mu
        if t >= burn_in:
            output += x
        if output_history:
            if t >= burn_in:
                history.append(output / (t+1-burn_in))
            else:
                history.append(x.copy())
    output /= (num_steps - burn_in)
    if output_history:
        return history
    else:
        return output

def rk(A, b, **kwargs):
    if not ("num_steps" in kwargs):
        kwargs["num_steps"] = A.shape[0]
    kwargs["burn_in"] = kwargs["num_steps"] - 1
    return tark(A, b, **kwargs)

def rka(A, b, **kwargs):
    if not ("num_threads" in kwargs):
        kwargs["num_threads"] = 10
    return rk(A, b, **kwargs)

def rku(A, b, **kwargs):
    if not ("under_relax" in kwargs):
        kwargs["under_relax"] = lambda t: 1/np.sqrt(1+t)
    return rk(A, b, **kwargs)

def mylstsq(A,b,lamb=None):
    if lamb is None:
        Q, R = np.linalg.qr(A, mode='reduced')
        return np.linalg.solve(R, Q.T @ b)
    else:
        U, S, Vt = np.linalg.svd(A, full_matrices=False)
        return Vt.T @ ((U.T @ b) * (S / (S**2 + lamb)))
