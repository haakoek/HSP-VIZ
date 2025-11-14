import numpy as np
from matplotlib import pyplot as plt

rmax = 20
N = 80
l_max = 40 #2, 10, 20, 30, 40
m_max = 0
Lmax = 2*l_max
Mmax = 2*m_max

dat = np.load(f"diatomics/LiH_gs_rmax={rmax}_N={N}_l_max={l_max}_m_max={m_max}_Lmax={Lmax}_Mmax={Mmax}.npz")

r = dat["r"]
idx_r = np.argmin(np.abs(r - 3.05 / 2))
a_z = r[idx_r]
w_r = dat["w_r"]
r_dot = dat["r_dot"]
w_r_rdot = w_r * r_dot
A = dat["A"]
energy_t = dat["energy_t"]
time_points = dat["time_points"]

orbital_1 = A[0, :, :]
orbital_2 = A[1, :, :]

Pr = 2.0 * np.einsum(
    "paI->a", np.abs(A) ** 2, optimize=True
)
int_Pr = np.einsum("a, a->", Pr, w_r_rdot, optimize=True)
print(f"Integral of P(r) dr = {int_Pr:.6f}")
print(f"ERHF: {energy_t[-1]:.6f} a.u.")
print()
for p in range(A.shape[0]):
    for l in range(l_max + 1):
        int_upl = np.einsum(
            "a, a->",
            np.abs(A[p, :, l]) ** 2,
            w_r_rdot,
            optimize=True,
        )
        print(f"int |u({p},{l})(r)|^2 dr = {int_upl:.2g}")
    print()
print()

fig, ax = plt.subplots(1, 1, figsize=(10, 8))
ax.plot(r, Pr, "-o", color="#17d9ff", label=r"$P^{HF}(r)$")
ax.fill_between(
    r,
    Pr,
    color="#17d9ff",
    alpha=0.3,
    label=r"$\int P(r) dr$" + f" = {int_Pr:.1f}",
)
ax.axvline(a_z, color="green", linestyle="--", label="Nuclear position")
ax.set_xlabel("r (a.u.)")
ax.legend()
ax.grid()

fig, ax = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
for l in range(0, l_max+1):
    ax[0].plot(
        r,np.abs(A[0,:,l])**2, label=r"$|u_{1,%d}(r)|^2$" % l)
    ax[1].plot(
        r,np.abs(A[1,:,l])**2, label=r"$|u_{2,%d}(r)|^2$" % l)
ax[0].set_title(r"$1\sigma$" + " l-components")
ax[0].axvline(a_z, color="green", linestyle="--", label="Nuclear position")
ax[0].grid()
ax[1].set_title(r"$2\sigma$" + " l-components")
ax[1].axvline(a_z, color="green", linestyle="--", label="Nuclear position")
ax[1].grid()


# plt.figure()
# plt.plot(time_points, energy_t, label=r"$E(\tau)$")
# plt.grid()
# plt.xlabel("Imaginary time (a.u.)")
# plt.legend()

plt.show()