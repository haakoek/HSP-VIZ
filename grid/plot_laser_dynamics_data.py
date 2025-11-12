import numpy as np
from matplotlib import pyplot as plt
import numpy as np
from grid_lib.spherical_coordinates.angular_momentum import LM_to_I

atom = "Be"
dt = 0.1
m_max = 0
M_max = 2 * m_max
E0 = 0.0925
omega = 0.057 # Corresponds to 800 nm

r_max = 320.0
N = int(2.75 * r_max)

n_cycles = 3
t_c = 2*np.pi/omega
td = n_cycles * t_c

l_max = 10
n_l = l_max + 1
L_max = 2 * l_max

dat = np.load(
    f"laser_dynamics_data/{atom}_Gauss_s=1_epsgauss=1e-10_maxitgauss=200_bicgstab_atol=1e-12_initguess=prev_y_anderson=True_dt={dt}_rmax={r_max}_N={N}_l_max={l_max}_m_max={m_max}_Lmax={L_max}_Mmax={M_max}_E0={E0}_omega={omega}_ncycles={n_cycles}_field_gauge=velocity_maskon=True.npz",
    allow_pickle=True,
)

time_points = dat["time_points"]
time_points_td = time_points[time_points <= td]
expec_z = dat["expec_z"]
expec_z_td = expec_z[time_points <= td]

r = dat["r"]
w_r_rdot = dat["w_r_rdot"]

"""
The shape of U_final is (n_docc, n_r, n_lm)
    - n_docc: number of doubly occupied orbitals
    - n_r: number of radial grid points
    - n_lm: number of (l,m) combinations in the spherical harmonics basis
"""
U_final = dat["A"] # The orbitals at the final time
U_t = dat["A_history"] # The orbitals at time = 0.0, 25.0, 50.0, ...



n_docc = U_final.shape[0]
n_electrons = 2 * n_docc
P_ionization = dat["P_single_ionization"]/n_electrons


plt.figure()
plt.plot(time_points, P_ionization, label="Single Ionization Probability: " + r"$P_{ion}(t) = \int_0^{20} \rho(r,t) dr$")
plt.legend()
plt.grid()


for p in range(n_docc):
    norm_Up = 0.0
    for l in range(n_l):
        norm_Up_l = np.sum(np.abs(U_final[p, :, l])**2*w_r_rdot)
        norm_Up += norm_Up_l
    print(f"Norm of |phi_{p}> at final time: {norm_Up}")

fig, ax = plt.subplots(2, 1, figsize=(8, 6))
for l in range(n_l):
    ax[0].semilogy(
        r,
        np.abs(U_final[0, :, LM_to_I(l, 0, l_max, m_max)])**2, label=r"$|u_{1, %d}(r)|^2$" % l
    )
    ax[1].semilogy(
        r,
        np.abs(U_final[1, :, LM_to_I(l, 0, l_max, m_max)])**2, label=r"$|u_{2, %d}(r)|^2$" % l
    )
ax[0].grid()
ax[0].legend()
ax[1].grid()
ax[1].legend()
plt.show()