import numpy as np

def fft(u):
    return np.fft.fft(u)

def ifft(u_hat):
    return np.fft.ifft(u_hat).real

def derivative(u, k, order=1):
    return ifft((1j * k)**order * fft(u))

def laplacian(u, k2):
    return ifft(-k2 * fft(u))

def Phi_prime(phi):
    #return phi**3 - phi
    return 2*phi**3 - 3*phi**2 + phi
    #return phi*(1.0 - phi) * (1.0 - 2 * phi)

def heaviside(x):
    return np.heaviside(x, 0.0)

def mobility(phi, M0):
    return M0 * np.ones_like(phi)

def S_sigma(phi_sigma, phi_T, phi_H, phi_N, lambda_p, lambda_ph, lambda_a):
    phi_P = phi_T - phi_H - phi_N
    return -lambda_p * phi_sigma * phi_P - lambda_ph * phi_sigma * phi_H + lambda_a * (phi_P + phi_H)

def S_T(phi_sigma, phi_T, phi_H, phi_N, lambda_p, lambda_A):
    return lambda_p * phi_sigma * (phi_T - phi_H - phi_N) - lambda_A * (phi_T - phi_N)

def S_H(phi_sigma, phi_T, phi_H, phi_N, lambda_A, lambda_pH, sigma_ph, lambda_hn, sigma_hp, sigma_hn):
    phi_P = phi_T - phi_H - phi_N
    return -lambda_A * phi_H + lambda_pH * heaviside(sigma_ph - phi_sigma) * phi_P -\
        heaviside( phi_sigma - sigma_hp) * phi_H - lambda_hn * heaviside(sigma_hn - phi_sigma) * phi_H

def S_N(phi_sigma, phi_T, phi_H, phi_N, lambda_hn, sigma_hn):
    phi_P = phi_T - phi_H - phi_N
    return lambda_hn * heaviside( sigma_hn - phi_sigma) * phi_H

def S_P(phi_sigma, phi_T, phi_H, phi_N, lambda_p, lambda_A, lambda_ph, sigma_ph, lambda_Hp, sigma_hp):
    phi_P = phi_T - phi_H - phi_N
    return -lambda_p * phi_sigma * phi_P - \
                lambda_A * phi_P - lambda_ph * heaviside( sigma_ph - phi_sigma)*phi_sigma +\
                lambda_Hp * heaviside(phi_sigma - sigma_hp) * phi_H

def Mobility(phi_T, Mp, Mh, Mn, phi_H = 0, phi_N = 0):
    return Mp * (phi_T - phi_H - phi_N)**2 + Mh * phi_H**2 + Mn * phi_N**2

def Mobility_h(phi_H, Mh):
    return Mh * phi_H**2

def Mobility_n(phi_N, Mn):
    return Mn * phi_N**2
