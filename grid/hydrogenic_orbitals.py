import numpy as np

from grid_lib.pseudospectral_grids.gauss_legendre_lobatto import (
    GaussLegendreLobatto,
    Linear_map
)
from grid_lib.spherical_coordinates.potentials import (
    Coulomb,
)
from matplotlib import pyplot as plt
from scipy.interpolate import CubicSpline

# Setup Legendre-Lobatto grid
N = 120
nr = N - 1
r_min = 0
r_max = 40


gll = GaussLegendreLobatto(N, Linear_map(r_min, r_max), symmetrize=False)

r = gll.r[1:-1]
r_uniform = np.linspace(r[0], r[-1], 400)

D2 = gll.D2[1:-1, 1:-1]
ddr = -0.5 * D2
w_r = gll.weights[1:-1]
r_dot = gll.r_dot[1:-1]

Z = 1 #Nuclear charge
Hydrogen_potential = Coulomb(Z)
V_Hydrogen = np.diag(Hydrogen_potential(r))



print()
for l in range(0, 3):

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

    print(f"** The 3 lowest lying eigenvalues for l={l} **")
    print(energy_l[0:3])
    print()

    plt.figure(l)
    plt.title(f"(Radial distribution / r^2) Hydrogenic orbitals for l={l}, Z={Z}")
    for k in range(3):
        
        """
        The orbitals are represented on a non-uniform set of grid points r.

        To obtain "higher resolution" for visualization, we can perform a cubic spline (cs) interpolation.
        """

        cs_psi_k = CubicSpline(r, u_l[:, k])

        plt.plot(r, (u_l[:,k]/r)**2+energy_l[k], linestyle='dashdot', label=r"$|\psi_{%d, %d}(r)|^2$" % (k, l))
        plt.plot(r_uniform, (cs_psi_k(r_uniform)/r_uniform)**2 + energy_l[k], label=r"$|\psi^{cs}_{%d, %d}(r)|^2$" % (k, l))
        #plt.axhline(energy_l[k], linestyle="--", color="red", linewidth=0.5, label=r"$E_{%d, %d}$" % (k, l) + f" = {energy_l[k]:.3f}")
    plt.grid()
    plt.legend()
plt.show()