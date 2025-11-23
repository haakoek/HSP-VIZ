import numpy as np
from scipy.interpolate import CubicSpline
from scipy.special import sph_harm
from grid_lib.spherical_coordinates.angular_momentum import (
    number_of_lm_states,
    LM_to_I,
    setup_y_and_ybar_sympy,
)
from matplotlib import pyplot as plt
import math
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons
from scipy.special import sph_harm
import scipy.special
from scipy.special import assoc_laguerre
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.colors import ListedColormap
from skimage import measure
import skimage

import scipy as sp
import sys


def orbital(r, B, X, Y, Z, l_max, m_max):

    r2 = X**2 + Y**2
    R = np.sqrt(r2 + Z**2)

    cos_theta = np.zeros_like(R)
    np.divide(Z, R, where=R > 0.0, out=cos_theta)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)

    Theta_2 = np.arccos(cos_theta)
    Theta = np.arctan2(np.sqrt(r2), Z)
    Theta = Theta_2.copy()
    Phi = np.arctan2(Y, X) + np.pi
    # print(np.min(Phi), np.max(Phi))

    orbital = np.zeros_like(R, dtype=np.complex128)

    for m in range(-m_max, m_max + 1):
        for l in range(abs(m), l_max + 1):
            # print(l,m)
            I_lm = LM_to_I(l, m, l_max, m_max)
            R_lm_cs = CubicSpline(r, B[:, I_lm] / r)
            R_lm_cs_R = R_lm_cs(R.ravel()).reshape(R.shape)
            Y_lm = sph_harm(m, l, Phi, Theta)
            # print(Y_lm.ravel())
            orb_lm = R_lm_cs_R * Y_lm
            orbital += orb_lm

    # orbital = np.nan_to_num(orbital)
    return orbital


l_max = 1
r_max = 10.0
nr_ratio = 4
m_max = l_max

L_max = 2 * l_max
M_max = 2 * m_max

n_e = 4  # number of electron orbitals
n_p = 4  # number of proton orbitals

Nr = int(r_max * nr_ratio)
dat = np.load(
    f"hdplus_data/hd+_fci_rmax=10_N={Nr}_lmax={l_max}_mmax={m_max}_ne={n_e}_np={n_p}.npz"
)

A = dat["A"]
B = dat["B"]
r = dat["r"]

gamma_e = dat["rho_e"]
gamma_p = dat["rho_p"]

"""
The electron and proton densities are given by

    rho_e(\vec{r}) = \sum_{pq} \phi_p^*(\vec{r}) gamma_e_{pq} phi_q(\vec{r})
    rho_p(\vec{r}) = \sum_{pq} \psi_p^*(\vec{r}) gamma_p_{pq} \psi_q(\vec{r})

"""

dz = 0.2
zmin = -6
zmax = 6
x = np.arange(zmin, zmax, dz)
y = np.arange(zmin, zmax, dz)
z = np.arange(zmin, zmax, dz)
X, Y, Z = np.meshgrid(x, y, z)

density_e = np.zeros(X.shape, dtype=np.complex128)
density_p = np.zeros(X.shape, dtype=np.complex128)

for p in range(n_e):
    phi_p = orbital(r, A[p], X, Y, Z, l_max, m_max)
    psi_p = orbital(r, B[p], X, Y, Z, l_max, m_max)
    for q in range(n_e):
        phi_q = orbital(r, A[q], X, Y, Z, l_max, m_max)
        psi_q = orbital(r, B[q], X, Y, Z, l_max, m_max)
        density_e += np.conj(phi_p) * gamma_e[p, q] * phi_q
        density_p += np.conj(psi_p) * gamma_p[p, q] * psi_q

print(np.linalg.norm(density_e.imag))
print(np.linalg.norm(density_p.imag))
