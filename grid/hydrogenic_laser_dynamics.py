import numpy as np
from matplotlib import pyplot as plt
import tqdm
import time
from scipy.linalg import expm
from scipy.sparse.linalg import LinearOperator, bicgstab
import sys
from scipy.integrate import solve_ivp


from opt_einsum import contract

from grid_lib.pseudospectral_grids.gauss_legendre_lobatto import (
    GaussLegendreLobatto,
    Linear_map,
)
from grid_lib.spherical_coordinates.angular_matrix_elements import (
    AngularMatrixElements,
)
from grid_lib.spherical_coordinates.lasers import sine_square_laser, sine_laser
from grid_lib.spherical_coordinates.utils import (
    Counter,
)
from grid_lib.spherical_coordinates.utils import mask_function

###########################################################################################################
"""
Setup radial grid, Hamiltonian, determine eigenstates and set initial state.
"""
Z = 1.0
r_min = 0
r_max = 40
N = int(1.5 * r_max)  # Number of grid points, should be large enough

GLL = GaussLegendreLobatto(N, Linear_map(r_min, r_max), symmetrize=False)
D1 = GLL.D1
D2 = np.dot(D1, D1)[1:-1, 1:-1]

r = GLL.r[1:-1]
r_dot = GLL.r_dot[1:-1]
w_r = GLL.weights[1:-1]
w_r_rdot = w_r * r_dot
n_r = len(r)

# Setup mask function that absorbs the wavefunction at the grid boundary.
mask_r = mask_function(r, r[-1], 0.8 * r[-1])

plt.figure()
plt.plot(r, mask_r)
plt.axvline(0.8 * r_max, color="k", ls="--", label=r"$r_{mask}$")
plt.show()

l_max = 5
n_l = l_max + 1

Tl = np.zeros((n_l, n_r, n_r))
Hl = np.zeros((n_l, n_r, n_r))

V = -Z / r
for l in range(0, n_l):
    Tl[l] = -0.5 * D2 + np.diag(l * (l + 1) / (2 * r**2))
    Hl[l] = Tl[l] + np.diag(V)

eps, C = np.linalg.eig(Hl[0])
idx = np.argsort(eps)
eps = eps[idx]
C = C[:, idx]
print("Ground state energy:", eps[0])
norm_C = np.sum(C[:, 0] ** 2 * w_r_rdot)
u0 = C[:, 0] / np.sqrt(norm_C)

psi_t = np.zeros((n_l, n_r), dtype=np.complex128)
print(f"Size of the wavefunction array: {psi_t.nbytes/1e6:.2g} MB")
psi_t[0] = np.complex128(u0)

Pr_t0 = contract("Ia, Ia->a", psi_t.conj(), psi_t)
#############################################################################################################

# pulse inputs
E0 = 0.03  # 0.06, 0.12
omega = 0.057
n_cycles = 3

t_cycle = 2 * np.pi / omega
e_field_z = sine_square_laser(
    E0=E0, omega=omega, td=n_cycles * t_cycle, phase=0
)

ame = AngularMatrixElements(l_max=l_max, N=101)
Z_omega = np.zeros((n_l, n_l))
for l1 in range(0, n_l):
    for l2 in range(0, n_l):
        Z_omega[l1, l2] = ame.l1m1_costh_l2m2(l1, 0, l2, 0)


dt = 0.1
tfinal = n_cycles * t_cycle
num_steps = int(tfinal / dt) + 1

###########################################################################################################
"""
Setup time-dependent Hamiltonian.
"""


class Hamiltonian:
    def __init__(self, Hl, z_omega, r, e_field):
        self.Hl = Hl
        self.z_omega = z_omega
        self.r = r
        self.e_field = e_field
        self.n_l, self.n_r = Hl.shape[0], Hl.shape[1]

    def __call__(self, psi, t, ravel=True):
        psi = psi.reshape((self.n_l, self.n_r))
        H_psi = contract("Iab, Ib -> Ia", self.Hl, psi)
        tmp_r = contract("a, Ja -> Ja", self.r, psi)
        Z_psi = contract("IJ, Ja->Ia", self.z_omega, tmp_r)
        if ravel:
            return (H_psi + self.e_field(t) * Z_psi).ravel()
        else:
            return H_psi + self.e_field(t) * Z_psi


Ht = Hamiltonian(Hl, Z_omega, r, e_field_z)

###########################################################################################################
"""
Setup preconditioner Crank-Nicolson propagation.
"""
M_H = np.zeros((n_l, n_r, n_r), dtype=np.complex128)
M_T = np.zeros((n_l, n_r, n_r), dtype=np.complex128)

for l in range(0, n_l):
    M_H[l] = np.linalg.inv(np.eye(n_r) + 1j * dt / 2 * Hl[l])
    M_T[l] = np.linalg.inv(np.eye(n_r) + 1j * dt / 2 * Tl[l])

M = M_H.copy()


def preconditioner(psi):
    """Preconditioner for the linear system."""

    psi = psi.reshape((n_l, n_r))
    Mpsi = contract("Iab, Ib->Ia", M, psi)
    return Mpsi.ravel()


M_linear = LinearOperator((n_r * n_l, n_r * n_l), matvec=preconditioner)
############################################################################################################


# sampling arrays
time_points = np.linspace(0, tfinal, num_steps)
expec_z = np.zeros(num_steps, dtype=np.complex128)
norm_t = np.zeros(num_steps, dtype=np.complex128)
norm_t[0] = contract("Ia,Ia,a->", psi_t.conj(), psi_t, w_r_rdot)
nr_its_conv = np.zeros(num_steps)
psi_history = np.zeros((num_steps, n_l, n_r), dtype=np.complex128)
psi_history[0] = psi_t.copy()
psit0_psit = np.zeros(num_steps, dtype=np.complex128)

psi_t0 = psi_t.copy()
psit0_psit[0] = contract("Ia, Ia, a->", psi_t0.conj(), psi_t, w_r_rdot)


for i in tqdm.tqdm(range(num_steps - 1)):

    ti = time_points[i]
    ti_mid = time_points[i] + dt / 2

    Ap_lambda = lambda psi_in, ti=ti_mid: psi_in.ravel() + 1j * dt / 2 * Ht(
        psi_in, ti
    )
    Ap_linear = LinearOperator((n_r * n_l, n_r * n_l), matvec=Ap_lambda)
    z = psi_t.ravel() - 1j * dt / 2 * Ht(psi_t, ti_mid)
    local_counter = Counter()
    psi_t, info = bicgstab(
        Ap_linear,
        z,
        M=M_linear,
        x0=psi_t.ravel(),
        tol=1e-12,
        callback=local_counter,
    )
    # print(f"Converged after {local_counter.counter} iterations")
    nr_its_conv[i] = local_counter.counter
    psi_t = psi_t.reshape((n_l, n_r))
    psi_t = contract("Ik, k->Ik", psi_t, mask_r)

    tmp_z = contract("IJ, Ja->Ia", Z_omega, psi_t)
    expec_z[i + 1] = contract(
        "a, a, Ia, Ia->", w_r_rdot, r, psi_t.conj(), tmp_z
    )
    norm_t[i + 1] = contract("Ia,Ia,a->", psi_t.conj(), psi_t, w_r_rdot)
    psit0_psit[i + 1] = contract("Ia, Ia, a->", psi_t0.conj(), psi_t, w_r_rdot)


samples = dict()
samples["time_points"] = time_points
samples["expec_z"] = expec_z
samples["norm_t"] = norm_t
samples["psi"] = psi_history
samples["r"] = r
samples["w_r"] = w_r
samples["r_dot"] = r_dot
samples["E0"] = E0
samples["omega"] = omega
samples["ncycles"] = n_cycles
samples["potential"] = V

np.savez(
    f"CN_dt={dt}_rmax={r_max}_N={N}_hydrogenic_E0={E0}_omega={omega}_ncycles={n_cycles}",
    **samples,
)

Pr = contract("Ia, Ia->a", psi_t.conj(), psi_t)
print(f"Max(Im(Pr)): {np.max(Pr.imag):.3e}")
int_Pr = contract("a, a->", w_r_rdot, Pr)
print(f"Norm of the wavefunction: {int_Pr:.12f}")

fig, axs = plt.subplots(3, 2, figsize=(10, 8))
axs[0, 0].plot(
    time_points, e_field_z(time_points), color="red", label=r"$E(t)$"
)
axs[0, 0].grid()
axs[0, 0].legend()

axs[0, 1].plot(time_points, expec_z.real, label=r"$\langle z(t) \rangle$")
axs[0, 1].grid()
axs[0, 1].legend()

axs[1, 0].plot(r, Pr_t0.real, label=r"$P(r,t=0)$")
axs[1, 0].plot(r, Pr.real, label=r"$P(r,t=t_f)$")
axs[1, 0].grid()
axs[1, 0].legend()
axs[1, 0].set_xlabel(r"$r[a.u.]$")

# axs[1,1].semilogy(time_points, np.abs(1-norm_t.real), label=r'|1-$\langle \psi(t) | \psi(t) \rangle$|')
# axs[1,1].grid()
# axs[1,1].legend()
axs[1, 1].plot(
    time_points,
    np.abs(psit0_psit) ** 2,
    label=r"$|\langle \psi(0) | \psi(t) \rangle|^2$",
)
axs[1, 1].grid()
axs[1, 1].legend()

axs[2, 0].semilogy(
    time_points,
    np.abs(1 - norm_t.real),
    label=r"$|1-\langle \psi(t) | \psi(t) \rangle$|",
)
axs[2, 0].grid()
axs[2, 0].legend()

plt.show()
