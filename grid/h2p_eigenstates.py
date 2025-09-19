from matplotlib import pyplot as plt
import numpy as np
import time
from scipy.special import erf
from sympy.physics.wigner import gaunt

from grid_lib.pseudospectral_grids.gauss_legendre_lobatto import (
    GaussLegendreLobatto,
    Linear_map,
)
from grid_lib.pseudospectral_grids.femdvr import FEMDVR

Z = 1.0
a = 1.997 / 2.0

nodes = np.array([0, a, 2 * a, 40])  # Example nodes for testing
n_points_pr_element = 31
n_points = (
    np.ones((len(nodes) - 1,), dtype=int) * n_points_pr_element
)  # Example number of points per element

print()
print(f"r_max: {nodes[-1]}, n_elem: {len(nodes)}, n_tot: {np.sum(n_points)}")

femdvr = FEMDVR(nodes, n_points, Linear_map, GaussLegendreLobatto)

# Get nodes and weights and differentiation matrix.
r = femdvr.r
r_dot = femdvr.r_dot
w = femdvr.weights
D = femdvr.D1
D2 = femdvr.D2[1:-1, 1:-1]  # Exclude boundary points

r = r[1:-1]  # Exclude boundary points
r_dot = r_dot[1:-1]  # Exclude boundary points
w = w[1:-1]  # Exclude boundary points

ei = femdvr.edge_indices
N = len(r) + 1
#########################################################################################
def compute_r_inv_l(r, a, l):
    r_min = np.minimum(r, abs(a))
    r_max = np.maximum(r, abs(a))
    if a > 0:
        return np.sqrt(4 * np.pi / (2 * l + 1)) * r_min**l / r_max ** (l + 1)
    else:
        return (
            np.sqrt(4 * np.pi / (2 * l + 1))
            * r_min**l
            / r_max ** (l + 1)
            * (-1) ** l
        )


print()
print(
    f"** Groundstate energy of Hydrogenic atom centered at (0,0,{a:.6f}) with charge Z={Z} **"
)
E0_H2p = []
E0_H_a = []
E0_erfH_a = []

l_max = 10
tic = time.time()
L_max = l_max + 1

r_inv_L = np.zeros((L_max, N - 1))
r_inv_L_am = np.zeros((L_max, N - 1))
for L in range(L_max):
    r_inv_L[L, :] = compute_r_inv_l(r, a, L)
    r_inv_L_am[L, :] = compute_r_inv_l(r, -a, L)

H = np.zeros((L_max * (N - 1), L_max * (N - 1)))
for L in range(0, L_max):
    D2_L = D2 - np.diag(L * (L + 1) / r**2)
    T_L = -0.5 * D2_L
    H[
        (L * (N - 1)) : ((L + 1) * (N - 1)), (L * (N - 1)) : ((L + 1) * (N - 1))
    ] = T_L  # -np.diag(erf(mu*r)/r)

tic_g = time.time()
gaunt_coeffs = np.zeros((L_max, L_max, L_max))
for l1 in range(L_max):
    for l2 in range(L_max):
        for L in range(L_max):
            gaunt_coeffs[l1, L, l2] = float(gaunt(l1, L, l2, 0, 0, 0, prec=128))

toc_g = time.time()
print(f"Time for Gaunt coefficients: {toc_g - tic_g:.6f} s")

V_r_inv = np.einsum("lLk, La -> lka", gaunt_coeffs, r_inv_L, optimize=True)
V_r_inv_am = np.einsum(
    "lLk, La -> lka", gaunt_coeffs, r_inv_L_am, optimize=True
)

nl = L_max
nr = N - 1

Vr_inv = np.zeros((nl, nl, nr, nr))
Vr_inv[:, :, np.arange(nr), np.arange(nr)] = V_r_inv
Vr_inv = Vr_inv.swapaxes(1, 2).swapaxes(0, 1).swapaxes(2, 3).T
Vr_inv = Vr_inv.reshape(nr * nl, nr * nl)

Vr_inv_am = np.zeros((nl, nl, nr, nr))
Vr_inv_am[:, :, np.arange(nr), np.arange(nr)] = V_r_inv_am
Vr_inv_am = Vr_inv_am.swapaxes(1, 2).swapaxes(0, 1).swapaxes(2, 3).T
Vr_inv_am = Vr_inv_am.reshape(nr * nl, nr * nl)

toc = time.time()
print(f"Time setup matrix: {toc - tic:.6f} s")

print(f"dim(H): {H.shape}")

tic = time.time()
eps_H2p, C_H2p = np.linalg.eig(H - Z * Vr_inv - Z * Vr_inv_am)
toc = time.time()
print(
    f"Time for diagonalization: {toc - tic:.2f} s, Nr of radial points = {N}, L_max = {l_max}"
)

idx_H2p = eps_H2p.argsort()
eps_H2p = eps_H2p[idx_H2p]
C_H2p = C_H2p[:, idx_H2p]
print(
    f"E0_H2p({l_max},R={2*a},{n_points_pr_element}): {(eps_H2p[0]+1/(2*a)):.15f}"
)

E0_H2p.append(eps_H2p[0] + 1 / (2 * a))
u0 = C_H2p[:, 0].reshape((L_max, N - 1))
norm_u0 = np.sum(u0**2 * w * r_dot)
u0 = u0 / np.sqrt(norm_u0)

P0_r = np.einsum("Li,Li->i", u0.conj(), u0)
int_P0_r = np.sum(P0_r * w * r_dot)
print(f"int P0(r)dr: {int_P0_r:.8f}")

fig, axs = plt.subplots(1, 2, figsize=(14, 10))
for L in range(L_max):
    axs[0].plot(r, np.abs(u0[L, :]) ** 2, label=r"$u_{%d}(r)$" % L)
axs[0].set_xlabel("r")
axs[0].set_ylabel("u_L(r)")
axs[0].legend()
axs[1].plot(r, P0_r, label=r"$P_0(r)$")
axs[1].set_xlabel("r")
axs[1].set_ylabel(r"$P_0(r)$")
axs[1].legend()
plt.tight_layout()
plt.show()
