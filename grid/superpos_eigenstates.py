import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import CubicSpline

# REQUIREMENT: https://github.com/hyqD/grid-lib
from grid_lib.pseudospectral_grids.gauss_legendre_lobatto import (
    GaussLegendreLobatto,
    Linear_map,
)
from grid_lib.spherical_coordinates.potentials import (
    Coulomb,
)
from grid_lib.spherical_coordinates.angular_matrix_elements import (
    AngularMatrixElements_l,
    AngularMatrixElements_lm,
)
from grid_lib.spherical_coordinates.properties import (
    expec_x_i,
)

# Setup Legendre-Lobatto grid
N = 120
nr = N - 1
r_min = 0
r_max = 40
l_max = 1
n_states = 3

gll = GaussLegendreLobatto(N, Linear_map(r_min, r_max), symmetrize=False)

r = gll.r[1:-1]
r_uniform = np.linspace(r[0], r[-1], 400)

D2 = gll.D2[1:-1, 1:-1]
ddr = -0.5 * D2
w_r = gll.weights[1:-1]
r_dot = gll.r_dot[1:-1]

Z = 1  # Nuclear charge
Hydrogen_potential = Coulomb(Z)
V_Hydrogen = np.diag(Hydrogen_potential(r))

eigenstates = np.zeros((l_max + 1, nr, n_states))
eigenenergies = np.zeros((l_max + 1, n_states))

print()
for l in range(0, l_max + 1):

    """
    l=0: s-orbitals
    l=1: p-orbitals
    l=2: d-orbitals
    ...
    """

    T = ddr + np.diag(l * (l + 1) / (2 * r**2))
    H_l_Hydrogen = T + V_Hydrogen

    energy_l, u_l = np.linalg.eig(H_l_Hydrogen)

    idx_e = np.argsort(energy_l)
    energy_l = energy_l[idx_e]
    u_l = u_l[:, idx_e]

    """
    numpy.linalg.eig gives eigenvectors that are normalized w.r.t to the vector product.

    The wavefunction should be normalized w.r.t the integral, such that, 

        \int |\psi_i(r, theta, phi)|^2 r^2 sin(\theta) dr d\theta d\phi = 1
    """
    for i in range(u_l.shape[1]):
        norm_psi = np.dot(w_r, r_dot * u_l[:, i] * u_l[:, i])
        u_l[:, i] /= np.sqrt(norm_psi)

    eigenstates[l] = u_l[:, :n_states]
    eigenenergies[l] = energy_l[:n_states]

    print(f"** The 3 lowest lying eigenvalues for l={l} **")
    print(energy_l[0:3])
    print()

fig, axs = plt.subplots(l_max + 1, 1, figsize=(15, 4), sharex=True)
for states in range(n_states):
    for l in range(l_max + 1):
        axs[l].plot(
            r,
            eigenstates[l, :, states] ** 2,
            label=r"$|u_{%d, %d}(r)|^2$" % (states + l + 1, l),
        )
        axs[l].set_xlabel("r (a.u.)")
        axs[l].set_ylabel(r"$u_{n,l}(r)$")
        axs[l].legend()
        axs[l].grid()


u_t = np.zeros((l_max + 1, nr), dtype=np.complex128)

c = np.zeros(l_max + 1, dtype=np.complex128)
c[0] = 1.0
c[1] = 1.0
norm_c = np.sqrt(np.sum(np.abs(c) ** 2))
c /= norm_c
c1 = c[0]
c2 = c[1]

tfinal = 16.0
dt = 0.8

num_steps = int(tfinal / dt) + 1
print(f"Number of time steps: {num_steps}")

time_points = np.zeros(num_steps)
expec_r = np.zeros(num_steps, dtype=np.complex128)

expec_r_1s = np.dot(w_r, r_dot * r * np.abs(eigenstates[0, :, 0]) ** 2)
expec_r_2p = np.dot(w_r, r_dot * r * np.abs(eigenstates[1, :, 0]) ** 2)


angular_matrix_elements = AngularMatrixElements_l(
    arr_to_calc=["z_Omega"], l_max=l_max
)
z_Omega = angular_matrix_elements("z_Omega")
expec_z = np.zeros(num_steps, dtype=np.complex128)


for n in range(num_steps):

    tn = n * dt
    time_points[n] = tn

    u_t[0] = (
        c1 * np.exp(-1j * tn * eigenenergies[0, 0]) * eigenstates[0, :, 0]
    )  # 1s
    u_t[1] = (
        c2 * np.exp(-1j * tn * eigenenergies[1, 0]) * eigenstates[1, :, 0]
    )  # 2p

    expec_z[n] = expec_x_i(u_t, w_r, r, z_Omega)


plt.figure()
plt.plot(time_points, expec_z.real)
plt.show()
