import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter

N = 256               
L = 10.1               
dt = 5.0e-5        
T = 1.30                
nsteps = int(T / dt)

# Constants
eps = 0.04  
M = 0.1   
sigma = 3.0e-5
M_sigma = 0.20
delta_sigma = 0.10
chi0 = 0.035
eps_T = 0.25
M0 = 0.90 
lambda_p = 0.05
lambda_A = 0.01

# Parameters from Lima model
Mp = 200
Mh = 20
Mn = 0
D = 1
lambda_p = 0.5
lambda_a = 0.0
lambda_ph = 0.5
lambda_pH = 0.5
lambda_Hp = 0.5
lambda_hn = 0.5
eps_T = 0.005
chi0 = 0.035
delta_sigma = 0.01
sigma_ph = 0.4
sigma_hp = 0.5
sigma_hn = 0.3
Ebar = 0.045

x = np.linspace(0,L, N)
dx = L / N

# Wavenumbers 
k = 2.0 * np.pi * np.fft.fftfreq(N, d=dx)
k2 = k**2
k4 = k**4

rng = np.random.default_rng(300)

phi_sigma = 0.04 * np.abs(rng.standard_normal(N))
phi_T = 0.05 * np.abs(rng.standard_normal(N))
#phi_T = (.1)*1.0/( (x - 2.0)**2 +1)
phi_H = 0.05 * np.abs(rng.standard_normal(N))
phi_N = 0.05 * np.abs(rng.standard_normal(N))
phi_sigma = 1.0 - phi_T - phi_H - phi_N

phi_sigma_hat = np.fft.fft(phi_sigma)
phi_T_hat = np.fft.fft(phi_T)
phi_H_hat = np.fft.fft(phi_H)
phi_N_hat = np.fft.fft(phi_N)

phi_sigma_data = []; phi_sigma_data.append(phi_sigma)
phi_T_data = []; phi_T_data.append(phi_T)
phi_H_data = []; phi_T_data.append(phi_H)
phi_N_data = []; phi_T_data.append(phi_N)
tdata = []; tdata.append(0.0)

# adding noise for later
#noise_scale = np.sqrt( 2.0 * M * sigma * dt / dx)

def fft(u):
    return np.fft.fft(u)

def ifft(u_hat):
    return np.fft.ifft(u_hat).real

def derivative(u, order=1):
    return ifft((1j * k)**order * fft(u))

def laplacian(u):
    return ifft(-k2 * fft(u))

def Phi_prime(phi):
    #return phi**3 - phi
    return 2*phi**3 - 3*phi**2 + phi
    #return phi*(1.0 - phi) * (1.0 - 2 * phi)

def heaviside(x):
    return np.heaviside(x, 0.0)

def mobility(phi):
    return M0 * np.ones_like(phi)

def S_sigma(phi_sigma, phi_T, phi_H, phi_N):
    phi_P = phi_T - phi_H - phi_N
    return -lambda_p * phi_sigma * phi_P - lambda_ph * phi_sigma * phi_H + lambda_a * (phi_P + phi_H)

def S_T(phi_sigma, phi_T, phi_H, phi_N):
    return lambda_p * phi_sigma * (phi_T - phi_H - phi_N) - lambda_A * (phi_T - phi_N)

def S_H(phi_sigma, phi_T, phi_H, phi_N):
    phi_P = phi_T - phi_H - phi_N
    return -lambda_A * phi_H + lambda_pH * heaviside(sigma_ph - phi_sigma) * phi_P -\
        heaviside( phi_sigma - sigma_hp) * phi_H - lambda_hn * heaviside(sigma_hn - phi_sigma) * phi_H

def S_N(phi_sigma, phi_T, phi_H, phi_N):
    phi_P = phi_T - phi_H - phi_N
    return lambda_hn * heaviside( sigma_hn - phi_sigma) * phi_H

def S_P(phi_sigma, phi_T, phi_H, phi_N):
    phi_P = phi_T - phi_H - phi_N
    return -lambda_p * phi_sigma * phi_P - \
                lambda_A * phi_P - lambda_ph * heaviside( sigma_ph - phi_sigma)*phi_sigma +\
                lambda_Hp * heaviside(phi_sigma - sigma_hp) * phi_H

def Mobility(phi_T, phi_H = 0, phi_N = 0):
    return Mp * (phi_T - phi_H - phi_N)**2 + Mh * phi_H**2 + Mn * phi_N**2

def Mobility_h(phi_H):
    return Mh * phi_H**2

def Mobility_n(phi_N):
    return Mn * phi_N**2

den_sigma = 1.0 + dt * M_sigma * delta_sigma * k2
den_T = 1.0 + dt * M0 * eps_T**2 * k4
den_H = 1.0 + dt * M0 * eps_T**2 * k4
den_N = 1.0 + dt * M0 * eps_T**2 * k4

for n in range(nsteps):
    phi_sigma = ifft(phi_sigma_hat)
    phi_T = ifft(phi_T_hat)
    phi_H = ifft(phi_H_hat)
    phi_N = ifft(phi_N_hat)

    mu =  Phi_prime(phi_T) - eps_T**2 * laplacian(phi_T) - chi0 * phi_sigma
    
    rhs_sigma = phi_sigma_hat + dt * M_sigma * chi0 * k2 * phi_T_hat + dt * fft(S_sigma(phi_sigma, phi_T, phi_H, phi_N))  
    phi_sigma_hat_new = rhs_sigma / den_sigma

    rhs_T =  -M0 * k2 * fft(Phi_prime(phi_T)) + M0 * chi0 * k2 * phi_sigma_hat 
    rhs_H =  -M0 * k2 * fft(Phi_prime(phi_T)) + M0 * chi0 * k2 * phi_sigma_hat 
    rhs_N =  -M0 * k2 * fft(Phi_prime(phi_T)) + M0 * chi0 * k2 * phi_sigma_hat 

    d_mu = derivative(mu)
    M_current_T = Mobility(phi_T, phi_H, phi_N)
    variable_flux_T = (M_current_T - M0) * d_mu
    M_current_H = Mobility_h(phi_H)
    variable_flux_H = (M_current_H - M0) * d_mu
    M_current_N = Mobility_n(phi_N)
    variable_flux_N = (M_current_N - M0) * d_mu

    rhs_T += fft(derivative(variable_flux_T))
    rhs_T += fft(S_T(phi_sigma, phi_T, phi_H, phi_N))
    rhs_H += fft(derivative(variable_flux_H))
    rhs_H += fft(S_H(phi_sigma, phi_T, phi_H, phi_N))
    rhs_N += fft(derivative(variable_flux_N))
    rhs_N += fft(S_N(phi_sigma, phi_T, phi_H, phi_N))

    eta_T = rng.standard_normal(N)*.3
    eta_T_hat = np.fft.fft(eta_T)
    noise_hat_T = ( np.sqrt(2.0 * M * sigma * dt)* 1j * k * eta_T_hat / np.sqrt(dx) )*0

    eta_H = rng.standard_normal(N)*.3
    eta_H_hat = np.fft.fft(eta_H)
    noise_hat_H = ( np.sqrt(2.0 * M * sigma * dt)* 1j * k * eta_H_hat / np.sqrt(dx) )*0

    eta_N = rng.standard_normal(N)*.3
    eta_N_hat = np.fft.fft(eta_N)
    noise_hat_N = ( np.sqrt(2.0 * M * sigma * dt)* 1j * k * eta_N_hat / np.sqrt(dx) )*0

    phi_T_hat_new = ( phi_T_hat + dt * rhs_T + noise_hat_T) / den_T
    phi_H_hat_new = ( phi_H_hat + dt * rhs_H + noise_hat_H) / den_H
    phi_N_hat_new = ( phi_N_hat + dt * rhs_N + noise_hat_N) / den_N

    phi_sigma_hat = phi_sigma_hat_new
    phi_T_hat = phi_T_hat_new
    phi_H_hat = phi_H_hat_new
    phi_N_hat = phi_N_hat_new

    phi_sigma_data.append(phi_sigma)
    phi_T_data.append(phi_T)
    phi_H_data.append(phi_H)
    phi_N_data.append(phi_N)
    tdata.append(n*dt)

# Create movie from output
phi_sigma_data_small = phi_sigma_data[::50]
phi_T_data_small = phi_T_data[::50]
phi_H_data_small = phi_H_data[::50]
phi_N_data_small = phi_N_data[::50]
#------------------------------

fig, ax = plt.subplots()
line1, = ax.plot([], [], lw=2, color='blue')  # Initialize an empty line
line2, = ax.plot([], [], lw=2, color='red')  # Initialize an empty line
line3, = ax.plot([], [], lw=2, color='green')  # Initialize an empty line
line4, = ax.plot([], [], lw=2, color='yellow')  # Initialize an empty line
ax.grid(True, linestyle='--')
ax.set_facecolor('lightgray')
fig.set_facecolor('whitesmoke')
ax.set_xlim(0, L)
ax.set_ylim(-1.2, 1.2)
ax.set_title("Tumor_CH_FULL_V1")

def init():
    line1.set_data([], [])
    line2.set_data([], [])
    line3.set_data([], [])
    line4.set_data([], [])
    return line1, line2, line3, line4

def update(frame):
    y1 = phi_T_data_small[frame]
    y2 = phi_H_data_small[frame]
    y3 = phi_N_data_small[frame]
    y4 = phi_sigma_data_small[frame]
    line1.set_data(x, y1)
    line2.set_data(x, y2)
    line3.set_data(x, y3)
    line4.set_data(x, y4)
    return line1,line2, line3, line4

ani = FuncAnimation(fig, update, frames=len(phi_H_data_small), init_func=init, blit=True)
writer = FFMpegWriter(fps=40, bitrate=1800)
ani.save("tumor_full_v1.mp4", writer=writer)
