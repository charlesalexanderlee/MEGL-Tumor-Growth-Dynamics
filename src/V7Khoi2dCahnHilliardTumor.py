import os
from pathlib import Path
import numpy as np
from scipy.fft import dct, idct, dst, idst
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
 
print(os.getcwd())
 
# ------------------------------------------------------------------
# 1D two-field tumor model, semi-implicit spectral method with
# physical boundary conditions:
#   p = phi_p (tumor):   Neumann   p_x = 0 at x = 0, L   -> cosine transform (DCT-II)
#   s = phi_sigma:       Dirichlet s = 1 at x = 0, L     -> sine transform of (s-1) (DST-II)
#
#   p_t = Mp * lap(mu_p) + Sp,   mu_p = Psi'(p) - eps^2 lap(p) - chi0 * s
#   s_t = Ms * lap(mu_s) + Ss,   mu_s = s/f - chi0 * p,   Ms = f*D
#
# Cos and sin bases differ, so the chi0 coupling terms are treated
# explicitly (each field keeps its own implicit scalar divisor).
# ------------------------------------------------------------------
 
# ---- grid (cell-centred, boundaries at the cell faces x = 0 and x = L) ----
N = 256
L = 25.6
dx = L / N
x = (np.arange(N) + 0.5) * dx
 
# ---- parameters (from the write-up) ----
Mp   = 200.0
f    = 0.01
D    = 1.0
eps  = np.sqrt(0.005)   # eps^2 = 0.005 (interface resolved); write-up literal eps = 0.005 is unresolved here
# eps = 0.005          # write-up literal value: interface far thinner than dx, gives spikes/ringing
Ebar = 0.045
chi0 = 0.035
lamP = 0.5
lamA = 0.0
saturate = True     # multiply growth by (1-p). NOT in the write-up: without it p grows
                    # past 1 and the run blows up near t ~ 12 (no hypoxic/necrotic cap)
 
# ---- time ----
dt = 1.0e-2
T = 20.0
nsteps = int(T / dt)
nsave = 10
 
# ---- wavenumbers ----
n = np.arange(N)
kc  = np.pi * n / L            # cosine modes (Neumann):   0, pi/L, 2pi/L, ...
ks  = np.pi * (n + 1) / L      # sine modes   (Dirichlet): pi/L, 2pi/L, ...
kc2, kc4 = kc**2, kc**4
ks2 = ks**2
 
# Psi'(p) = A*p + Ebar*(4p^3 - 6p^2 - p),  A = 3*Ebar
A = 3.0 * Ebar
divisor_p = 1.0 + dt * Mp * (A * kc2 + eps**2 * kc4)
divisor_q = 1.0 + dt * D * ks2          # q = s - 1
 
# transforms (orthonormal)
C  = lambda u: dct(u, type=2, norm='ortho')
Ci = lambda u: idct(u, type=2, norm='ortho')
S_ = lambda u: dst(u, type=2, norm='ortho')
Si = lambda u: idst(u, type=2, norm='ortho')
 
# ---- initial condition: tumor blob, p + s = 1 ----
xc, R, w = L / 2, 2.0, 0.5
p = 0.5 * (1.0 - np.tanh((np.abs(x - xc) - R) / w))
s = 1.0 - p
p0, s0 = p.copy(), s.copy()
 
p_hat = C(p)
q_hat = S_(s - 1.0)
 
pdata = [p.copy()]; sdata = [s.copy()]; tdata = [0.0]
mass = [np.sum(p) * dx]
 
for step in range(nsteps):
 
    # explicit pieces
    Nl = Ebar * (4 * p**3 - 6 * p**2 - p)
    grow = lamP * s * p * ((1.0 - p) if saturate else 1.0)
    Sp = grow - lamA * p
    Ss = -grow + lamA * p
 
    # explicit coupling:  Laplacians of the *other* field, in physical space
    # p equation: -Mp*chi0*lap(s) with the no-flux (Neumann) Laplacian, i.e. the
    # cosine-basis Laplacian applied to s. This keeps grad(mu).n = 0, so the
    # tumor mass is only changed by the source terms.
    Cp_hat = Mp * chi0 * kc2 * C(s)
    # s equation: -f*D*chi0*lap(p), evaluated in physical space
    lap_p = Ci(-kc2 * p_hat)
    Cs = -f * D * chi0 * lap_p
 
    # update each field in its own basis
    p_hat = (p_hat + dt * (-Mp * kc2 * C(Nl) + C(Sp) + Cp_hat)) / divisor_p
    q_hat = (q_hat + dt * (S_(Ss) + S_(Cs))) / divisor_q
 
    p = Ci(p_hat)
    s = 1.0 + Si(q_hat)
 
    mass.append(np.sum(p) * dx)
    if (step + 1) % nsave == 0:
        pdata.append(p.copy()); sdata.append(s.copy()); tdata.append((step + 1) * dt)
 
print("p range:", p.min(), p.max(), "| s range:", s.min(), s.max())
print("max(p + s) over run:", max(np.max(pp + ss) for pp, ss in zip(pdata, sdata)))
print("tumor mass: start", mass[0], " end", mass[-1])
 
# ------------------------------------------------------------------
# Movie
# ------------------------------------------------------------------
fig, ax = plt.subplots()
line_p, = ax.plot([], [], lw=2, color='red',  label=r"$\phi_p$ (tumor)")
line_s, = ax.plot([], [], lw=2, color='blue', label=r"$\phi_\sigma$ (nutrient)")
ax.set_xlim(0, L)
ax.set_ylim(-0.1, 1.3)
ax.set_xlabel("x")
ax.legend(loc="upper right")
title = ax.set_title("t = 0.00")
 
def init():
    line_p.set_data([], [])
    line_s.set_data([], [])
    return line_p, line_s
 
def update(frame):
    line_p.set_data(x, pdata[frame])
    line_s.set_data(x, sdata[frame])
    title.set_text(f"Tumor model, t = {tdata[frame]:.2f}")
    return line_p, line_s, title
 
ani = FuncAnimation(fig, update, frames=len(pdata), init_func=init, blit=True)
writer = FFMpegWriter(fps=40, bitrate=1800)
 
# Save next to this script (cwd may be "/" and read-only when run from a terminal)
try:
    outdir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    outdir = str(Path.home() / "Desktop")
out = os.path.join(outdir, "tumor_2field_bc.mp4")
ani.save(out, writer=writer)
print("Saved to:", out)