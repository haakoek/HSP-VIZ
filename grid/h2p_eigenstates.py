from matplotlib import pyplot as plt
import numpy as np
import time
from scipy.special import erf, sph_harm
from sympy.physics.wigner import gaunt
from scipy.sparse.linalg import LinearOperator, bicgstab
import tqdm

from opt_einsum import contract

from grid_lib.pseudospectral_grids.gauss_legendre_lobatto import (
    GaussLegendreLobatto,
    Linear_map,
)
from grid_lib.pseudospectral_grids.femdvr import FEMDVR
from grid_lib.spherical_coordinates.angular_matrix_elements import (
    AngularMatrixElements,
)
from grid_lib.spherical_coordinates.lasers import sine_square_laser, sine_laser
from grid_lib.spherical_coordinates.utils import (
    Counter,
)
from grid_lib.spherical_coordinates.utils import mask_function


Z = 1.0
a = 1.997 / 2.0

nodes = np.array([0, a, 2 * a, 15])  # Example nodes for testing
n_points_pr_element = 16
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

mask_r = mask_function(r, r[-1], 0.8 * r[-1])

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
Tl = np.zeros((L_max, N - 1, N - 1))
for L in range(0, L_max):
    D2_L = D2 - np.diag(L * (L + 1) / r**2)
    T_L = -0.5 * D2_L
    Tl[L] = T_L
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

H += -Z * Vr_inv - Z * Vr_inv_am 

tic = time.time()
eps_H2p, C_H2p = np.linalg.eig(H)
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
norm_u0 = 0.0
for l in range(L_max):
    norm_u0 += np.dot(w, u0[l, :] * u0[l, :])
print(f"Norm of the groundstate: {norm_u0:.6f}")

u0 /= np.sqrt(norm_u0)
psi0 = contract("li, i->li", u0, 1/r)

P0_r = np.einsum("li,li->i", u0.conj(), u0)
int_P0_r = np.dot(w, P0_r)
print(f"Integral of the radial density: {int_P0_r:.6f}")

fig, axs = plt.subplots(2,2, figsize=(14,10))
axs[0, 0].plot(r, P0_r, label=f"$P_0(r)$")
axs[0, 0].axvline(a, color="k", ls="--", label="Proton positions")
axs[0, 0].legend()

for l in range(L_max):
    axs[0, 1].plot(r, np.abs(u0[l, :])**2, label=f"$|u_{l}(r)|^2$")
    axs[1, 0].plot(r, np.abs(psi0[l, :])**2, label=f"$|\psi_{l}(r)|^2$")
axs[0,1].legend()
axs[1,0].legend()
plt.show()

from scipy.interpolate import CubicSpline

theta = np.linspace(0,np.pi,21)
theta = theta[1:-1]
n_theta = len(theta)
r_uniform = np.linspace(0.01, r[-1], 101)
ul_cs = [CubicSpline(r, u0[l, :]) for l in range(L_max)]
ul = np.array([ul_cs[l](r_uniform) for l in range(L_max)])

psi_cs = np.zeros(ul.shape)
for l in range(l_max):
    psi_cs[l, :] = ul[l, :] / r_uniform

"""
For m=0 Y_l(theta, phi)  is independent of phi
"""

nr_uniform = len(r_uniform)
psi = np.zeros((nr_uniform, n_theta))

for l in range(l_max):
    Y_l0 = sph_harm(0, l, 0, theta).real  # (n_theta,)
    for k in range(nr_uniform):
        for th in range(n_theta):
            psi[k, th] += psi_cs[l, k] * Y_l0[th]

density = psi**2
print(np.min(density.ravel()), np.max(density.ravel()))
