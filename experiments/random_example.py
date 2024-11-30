#!/usr/bin/env python

import numpy as np
import sys
sys.path.append('../')
from tark import tark, rk, mylstsq

n = 100000
d = 100
A = np.random.randn(n, d)
y = np.random.randn(d)
b = A @ y + 1e-6*np.random.rand(n)
x = mylstsq(A, b)
num_steps = n
burn_in = 3000
num_threads = 10

tark_history = tark(A,b=b,num_steps=num_steps,output_history=True,burn_in=burn_in)
rk_history = rk(A,b=b,num_steps=num_steps,output_history=True)
rka_history = rk(A,b=b,num_steps=num_steps//num_threads,output_history=True,num_threads=num_threads)
rku_history = rk(A,b=b,num_steps=num_steps,output_history=True,under_relax=lambda t:1/(t+1)**(1/2))

import matplotlib.pyplot as plt
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern"],
    "text.latex.preamble": r"\usepackage{amsmath}"  # Ensure amsmath is included
})
rkcolor = "#648FFF"
rkacolor = "#785EF0"
rkucolor = "#DC267F"
tarkcolor = "#FE6100"

plt.figure()
plt.plot(range(1,num_steps+1),[np.linalg.norm(xx - x) / np.linalg.norm(x) for xx in rk_history], label="RK", color=rkcolor, linewidth=2, linestyle="dotted")
plt.plot(range(num_threads,num_steps+1,num_threads),[np.linalg.norm(xx - x) / np.linalg.norm(x) for xx in rka_history], label="RKA", color=rkacolor, linewidth=2, linestyle="dashdot")
plt.plot(range(1,num_steps+1),[np.linalg.norm(xx - x) / np.linalg.norm(x) for xx in rku_history], label="RKU", color=rkucolor, linewidth=2, linestyle="dashed")
plt.plot(range(burn_in+1,num_steps+1),[np.linalg.norm(xx - x) / np.linalg.norm(x) for xx in tark_history[burn_in:]], label="TARK", color=tarkcolor, linewidth=2)
plt.xscale("log")
plt.yscale("log")
plt.xlabel("Iteration")
plt.ylabel(r"$\|\boldsymbol{x}-\boldsymbol{x}_\star\| / \|\boldsymbol{x}_\star\|$")
plt.legend()
plt.savefig("../figs/random_example.png", dpi=300, bbox_inches='tight')
