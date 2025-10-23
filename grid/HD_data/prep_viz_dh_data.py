import numpy as np
import matplotlib.pyplot as plt

r_max = 30.0
l_max = 0

nr_ratio = [2,3,4,5,6]

ratio = nr_ratio[-1]
Nr = int(r_max * ratio)
dat = np.load(f"hd_data/DH_GS_neorb=1_nporb=1_rmax={r_max}_lmax={l_max}_mmax=0_Nr={Nr}_dt=1.0_nsteps=8000.npz")

"""
A: np.ndarray, shape (n_e, Nr, n_lm)
B: np.ndarray, shape (n_p, Nr, n_lm)

    n_e: number of electron orbitals (here 1)
    n_p: number of positron orbitals (here 1)
    n_lm: number of angular momentum numbers
"""

A = dat["A"]
B = dat["B"] 
r = dat["r"]

