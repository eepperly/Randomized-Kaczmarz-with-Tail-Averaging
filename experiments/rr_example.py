#!/usr/bin/env python

import numpy as np
import sys
sys.path.append('../')
from tark import tark, rk, rku, mylstsq

n = 1000000
d = 10

U, _ = np.linalg.qr(np.random.randn(n,d), mode="reduced")
V, _ = np.linalg.qr(np.random.randn(d,d))
A = U @ np.diag([1000/(i+1)**4 for i in range(d)]) @ V.T
b = A @ np.random.randn(d) + 0.01 * np.random.randn(n)

mu = 0.999
lamb = (1 - mu) / mu * np.linalg.norm(A, 'fro')**2
A_aug = np.vstack([A, np.sqrt(lamb) * np.identity(d)])
b_aug = np.concatenate([b, np.zeros(d)])

x = mylstsq(A, b)
x_mu = mylstsq(A, b, lamb=lamb)

num_steps = n
burn_in = 1000

tark_history = tark(A, b=b, num_steps=num_steps, burn_in=burn_in, output_history=True)
rk_history = rk(A, b=b, num_steps=num_steps, output_history=True)

rk_aug_history = rk(A_aug, b=b_aug, num_steps=num_steps, output_history=True)
tark_aug_history = tark(A_aug, b=b_aug, num_steps=num_steps, burn_in=burn_in, output_history=True)
tark_rr_history = tark(A, b=b, num_steps=num_steps, mu_reg=mu, burn_in=burn_in, output_history=True)

import matplotlib.pyplot as plt
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern"],
    "text.latex.preamble": r"\usepackage{amsmath}"  # Ensure amsmath is included
})
plt.rcParams.update({'font.size': 20})
rkcolor = "#648FFF"
rkacolor = "#785EF0"
rkucolor = "#DC267F"
tarkcolor = "#FE6100"

fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(15, 5))  # Adjust figsize as needed
x_ticks = [10**0, 10**2, 10**4, 10**6]

# First subplot
ax1.plot(range(1, num_steps+1), [np.linalg.norm(xx - x) / np.linalg.norm(x) for xx in rk_history], label="RK", color=rkcolor)
ax1.plot(range(burn_in+1, num_steps+1), [np.linalg.norm(xx - x) / np.linalg.norm(x) for xx in tark_history[burn_in:]], label="TARK", color=tarkcolor)
ax1.set_xscale("log")
ax1.set_yscale("log")
ax1.set_xticks(x_ticks)
ax1.set_xlabel("Rows Accessed")
ax1.set_ylabel(r"$\|\boldsymbol{x}-\boldsymbol{x}_\star\| / \|\boldsymbol{x}_\star\|$")
ax1.legend()

# Second subplot
ax2.plot(range(1, num_steps+1), [np.linalg.norm(xx - x_mu) / np.linalg.norm(x_mu) for xx in rk_aug_history], label="RK on (3.2)", color=rkcolor)
ax2.plot(range(burn_in+1, num_steps+1), [np.linalg.norm(xx - x_mu) / np.linalg.norm(x_mu) for xx in tark_aug_history[burn_in:]], label="TARK on (3.2)", color=rkacolor)
ax2.plot(range(burn_in+1, num_steps+1), [np.linalg.norm(xx - x_mu) / np.linalg.norm(x_mu) for xx in tark_rr_history[burn_in:]], label="TARK-RR", color=tarkcolor)
ax2.set_xscale("log")
ax2.set_yscale("log")
ax2.set_xticks(x_ticks)
ax2.set_xlabel("Rows Accessed")
ax2.set_ylabel(r"$\|\boldsymbol{x}-\boldsymbol{x}_\mu\| / \|\boldsymbol{x}_\mu\|$")
ax2.legend()

plt.tight_layout()
plt.savefig("../figs/rr_example.png", dpi=300, bbox_inches='tight')
