#!/usr/bin/env python

import numpy as np
import sys
sys.path.append('../')
from tark import tark, rk, mylstsq

A = np.random.randn(1000000, 10)
y = np.random.randn(10)
b = A @ y + 0.1*np.random.rand(1000000)
x = mylstsq(A, b)
tark_history = tark(A,b,num_steps=A.shape[0],burn_in=1000,output_history=True)
rk_history = rk(A,b,num_steps=A.shape[0],output_history=True)

import matplotlib.pyplot as plt
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern"],
    "text.latex.preamble": r"\usepackage{amsmath}"  # Ensure amsmath is included
})

plt.figure()
plt.plot(range(1,A.shape[0]+1),[np.linalg.norm(xx - x) / np.linalg.norm(x) for xx in tark_history], label="TARK")
plt.plot(range(1,A.shape[0]+1),[np.linalg.norm(xx - x) / np.linalg.norm(x) for xx in rk_history], label="RK")
plt.xscale("log")
plt.yscale("log")
plt.xlabel("Iteration")
plt.ylabel(r"$\|\boldsymbol{x}-\boldsymbol{x}_\star\| / \|\boldsymbol{x}_\star\|$")
plt.legend()
plt.savefig("../figs/random_example.png", dpi=300, bbox_inches='tight')
