#!/usr/bin/env python3

import numpy as np
import sys
sys.path.append('../')
from active_regression import active_tark, active_leverage
from tark import mylstsq

f = lambda y: np.abs(np.sin(np.pi * y)) * np.exp(-2 * y) + np.sign(np.cos(4 * np.pi * y))

n = 10000
d = 25

pts = np.linspace(-1,1,num=n)
kk = np.arange(d)
A = np.cos(kk[np.newaxis,:] * np.arccos(pts[:,np.newaxis]))
b = f(pts) + 0.01 * np.random.randn(n)
num_steps = 19
num_trials = 100

x = mylstsq(A, b)
resnorm = np.linalg.norm(b - A @ x)
tark = np.zeros((num_steps,num_trials))
leverage = np.zeros((num_steps,num_trials))

measurements = np.array([i*d for i in range(2,2*num_steps+1,2)], dtype=int)
for j, meas in enumerate(measurements):
    for i in range(num_trials):
        x_tark = active_tark(A, b=b, entry_budget=meas)
        tark[j,i] = np.linalg.norm(b - A @ x_tark) / resnorm - 1
        x_leverage = active_leverage(A, b=b, entry_budget=meas)
        leverage[j,i] = np.linalg.norm(b - A @ x_leverage) / resnorm - 1

import matplotlib.pyplot as plt
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern"],
    "text.latex.preamble": r"\usepackage{amsmath}"  # Ensure amsmath is included
})
plt.rcParams.update({'font.size': 16})
rkcolor = "#648FFF"
rkacolor = "#785EF0"
rkucolor = "#DC267F"
tarkcolor = "#FE6100"

fig, (ax1, ax2) = plt.subplots(1, 2, sharey=False, figsize=(15, 5))  # Adjust figsize as needed
middle = np.median(tark, axis=1)
upper = np.quantile(tark, 0.9, axis=1)
lower = np.quantile(tark, 0.1, axis=1)
ax1.plot(measurements/float(d),middle,label="Preconditioned TARK", color=tarkcolor, linewidth=2)
ax1.fill_between(measurements/float(d),lower,upper,color=tarkcolor,alpha=0.1)
middle = np.median(leverage, axis=1)
upper = np.quantile(leverage, 0.9, axis=1)
lower = np.quantile(leverage, 0.1, axis=1)
ax1.plot(measurements/float(d),middle,label="Leverage Score",color=rkucolor, linewidth=2, linestyle="dotted")
ax1.fill_between(measurements/float(d),lower,upper,color=rkucolor,alpha=0.1)
ax1.set_xscale("log")
ax1.set_yscale("log")
ax1.set_xlabel("Labels Read / $d$")
ax1.set_ylabel(r"Suboptimality $\|\boldsymbol{b}-\boldsymbol{A}\boldsymbol{x}\| / \|\boldsymbol{b}-\boldsymbol{A}\boldsymbol{x}_\star\|-1$")
ax1.legend()

ax2.plot(pts, A @ x, color=rkcolor, linewidth=2, linestyle="dashdot")
ax2.plot(pts, A @ x_tark, color=tarkcolor, linewidth=2)
ax2.plot(pts, A @ x_leverage, color=rkucolor, linewidth=2, linestyle="dotted")
ax2.plot(pts, f(pts), color="black", linestyle="dashed", linewidth=3)

fig.savefig("../figs/active_regression_example.png", dpi=300, bbox_inches='tight')

