"""
SPATIOTEMPORAL MODELING OF TUMOR GROWTH DYNAMICS
https://github.com/charlesalexanderlee/MEGL-Tumor-Growth-Dynamics

Mason Experimental Geometry Laboratory
George Mason University
Department of Mathematical Sciences
Fall 2026

[Authors]
* Dr. Daniel Anderson
* Dr. Scott Cochran
* Elisha Erzoah
* Charles Alexander Lee
* Lia McPherson
* Khoi Vu
"""

from helpers import (
    fft,
    ifft,
    derivative,
    laplacian,
    Phi_prime,
    S_T,
    S_H,
    S_N,
    S_sigma,
    Mobility,
    Mobility_h,
    Mobility_n,
)
import numpy as np
import pathlib
import json
import time

# A run counts as "blown up" if any field becomes NaN/inf or its magnitude
# exceeds this value. The phi fields should stay roughly in [0, 1], so 1e3
# is a clear sign of divergence and catches it well before overflow.
BLOWUP_THRESHOLD = 1.0e3


def _blew_up(*fields):
    """Return True if any field has a non-finite value or exceeds the threshold."""
    for f in fields:
        if not np.all(np.isfinite(f)):
            return True
        if np.max(np.abs(f)) > BLOWUP_THRESHOLD:
            return True
    return False


def run_simulation(sim_num, timestamp):
    print(f"[Simulation {sim_num+1}] Starting simulation")
    start_time = time.perf_counter()

    # Create folder for current simulation
    folder_path = pathlib.Path(f"./output/{timestamp}/sim_{sim_num+1}")
    folder_path.mkdir(parents=True, exist_ok=True)

    # Simulation settings
    N = 256
    L = 10.0
    dt = 5.0e-5
    T = 1.0
    nsteps = int(T / dt)

    # Constants
    eps = .04
    M = .1
    sigma = 3.0e-5
    M_sigma = .20
    delta_sigma = .10
    chi0 = 0.035
    eps_T = .25
    M0 = .90
    lambda_p = .05
    lambda_A = .01

    # Random Number Generator
    rng = np.random.default_rng()

    # Simulation parameters (Monte Carlo)
    Mp = rng.choice(np.linspace(start=100, stop=300, num=N))
    Mh = rng.choice(np.linspace(start=10, stop=30, num=N))
    Mn = rng.choice(np.linspace(start=0, stop=10, num=N))
    D = rng.choice(np.linspace(start=0.5, stop=1.5, num=N))
    lambda_p = rng.choice(np.linspace(start=0, stop=1, num=N))
    lambda_a = rng.choice(np.linspace(start=0, stop=1, num=N))
    lambda_ph = rng.choice(np.linspace(start=0, stop=1, num=N))
    lambda_pH = rng.choice(np.linspace(start=0, stop=1, num=N))
    lambda_Hp = rng.choice(np.linspace(start=0, stop=1, num=N))
    lambda_hn = rng.choice(np.linspace(start=0, stop=1, num=N))
    eps_T = rng.choice(np.linspace(start=0, stop=0.01, num=N))
    chi0 = rng.choice(np.linspace(start=0.01, stop=0.05, num=N))
    delta_sigma = rng.choice(np.linspace(start=0.0001, stop=1, num=N))
    sigma_ph = rng.choice(np.linspace(start=0.1, stop=1, num=N))
    sigma_hp = rng.choice(np.linspace(start=0.1, stop=1, num=N))
    sigma_hn = rng.choice(np.linspace(start=0.1, stop=1, num=N))
    Ebar = rng.choice(np.linspace(start=0.025, stop=0.05, num=N))

    # Cast to plain floats so the dict is JSON/pickle friendly
    parameters = {
        "Mp": float(Mp),
        "Mh": float(Mh),
        "Mn": float(Mn),
        "D": float(D),
        "lambda_p": float(lambda_p),
        "lambda_a": float(lambda_a),
        "lambda_ph": float(lambda_ph),
        "lambda_pH": float(lambda_pH),
        "lambda_Hp": float(lambda_Hp),
        "lambda_hn": float(lambda_hn),
        "eps_T": float(eps_T),
        "chi0": float(chi0),
        "delta_sigma": float(delta_sigma),
        "sigma_ph": float(sigma_ph),
        "sigma_hp": float(sigma_hp),
        "sigma_hn": float(sigma_hn),
        "Ebar": float(Ebar),
    }

    # Create 1-dimensional line
    # x = np.linspace(0, L, N)
    dx = L / N

    # Wave Numbers
    k = 2.0 * np.pi * np.fft.fftfreq(N, d=dx)
    k2 = k**2
    k4 = k**4

    # Initial conditions
    rng = np.random.default_rng(300)
    phi_T = 0.05 * np.abs(rng.standard_normal(N))
    phi_H = 0.05 * np.abs(rng.standard_normal(N))
    phi_N = 0.05 * np.abs(rng.standard_normal(N))
    phi_sigma = np.ones(N) - phi_T - phi_H - phi_N

    # Perform FFT on initial condition
    phi_T_hat = np.fft.fft(phi_T)
    phi_H_hat = np.fft.fft(phi_H)
    phi_N_hat = np.fft.fft(phi_N)
    phi_sigma_hat = np.fft.fft(phi_sigma)

    # Create empty data arrays and store initial states
    phi_sigma_data = np.empty((nsteps, N))
    phi_T_data = np.empty((nsteps, N))
    phi_H_data = np.empty((nsteps, N))
    phi_N_data = np.empty((nsteps, N))
    t_data = np.arange(0, T, nsteps)

    phi_sigma_data[0] = phi_sigma
    phi_T_data[0] = phi_T
    phi_H_data[0] = phi_H
    phi_N_data[0] = phi_N

    # Caclulate density constants
    den_sigma = 1.0 + dt * M_sigma * delta_sigma * k2
    den_T = 1.0 + dt * M0 * eps_T**2 * k4
    den_H = 1.0 + dt * M0 * eps_T**2 * k4
    den_N = 1.0 + dt * M0 * eps_T**2 * k4

    # Blow-up bookkeeping
    blew_up = False
    blowup_step = None
    steps_completed = nsteps

    # Begin solving by iterating.
    # np.errstate silences overflow warnings; we detect blow-up ourselves.
    with np.errstate(all="ignore"):
        for n in range(1, nsteps):
            phi_sigma = ifft(phi_sigma_hat)
            phi_T = ifft(phi_T_hat)
            phi_H = ifft(phi_H_hat)
            phi_N = ifft(phi_N_hat)

            rhs_sigma = phi_sigma_hat + dt * M_sigma * chi0 * k2 * phi_T_hat + dt * fft(S_sigma(phi_sigma, phi_T, phi_H, phi_N, lambda_p, lambda_ph, lambda_a))
            rhs_T = -M0 * k2 * fft(Phi_prime(phi_T)) + M0 * chi0 * k2 * phi_sigma_hat
            rhs_H = -M0 * k2 * fft(Phi_prime(phi_T)) + M0 * chi0 * k2 * phi_sigma_hat
            rhs_N = -M0 * k2 * fft(Phi_prime(phi_T)) + M0 * chi0 * k2 * phi_sigma_hat

            # Approximate mu (Equation 6.3)
            mu = Phi_prime(phi_T) - eps_T**2 * laplacian(phi_T, k2) - chi0 * phi_sigma
            d_mu = derivative(mu, k, order=1)

            # Approximate phi_T (Equation 6.2)
            M_current_T = Mobility(phi_T, phi_H, phi_N, Mp, Mh, Mn)
            variable_flux_T = (M_current_T - M0) * d_mu
            rhs_T += 1j*k * fft(variable_flux_T)
            rhs_T += fft(S_T(phi_sigma, phi_T, phi_H, phi_N, lambda_p, lambda_A))

            # Approximate phi_H (Equation 6.4)
            M_current_H = Mobility_h(phi_H, Mh)
            variable_flux_H = (M_current_H - M0) * d_mu
            rhs_H += 1j*k * fft(variable_flux_H)
            rhs_H += fft(S_H(phi_sigma, phi_T, phi_H, phi_N, lambda_A, lambda_pH, sigma_ph, sigma_hp, lambda_hn, sigma_hn))

            # Approximate phi_N (Equation 6.5)
            M_current_N = Mobility_n(phi_N, Mn)
            variable_flux_N = (M_current_N - M0) * d_mu
            rhs_N += 1j*k * fft(variable_flux_N)
            rhs_N += fft(S_N(phi_sigma, phi_H, lambda_hn, sigma_hn))

            # Approximate phi_sigma (Equation 6.1)
            phi_sigma_hat = rhs_sigma / den_sigma
            phi_T_hat = (phi_T_hat + dt * rhs_T) / den_T
            phi_H_hat = (phi_H_hat + dt * rhs_H) / den_H
            phi_N_hat = (phi_N_hat + dt * rhs_N) / den_N

            # Store results in data matrices
            phi_sigma_data[n] = phi_sigma
            phi_T_data[n] = phi_T
            phi_H_data[n] = phi_H
            phi_N_data[n] = phi_N

            # Blow-up check: look at the freshly updated state in physical space
            if _blew_up(
                ifft(phi_sigma_hat),
                ifft(phi_T_hat),
                ifft(phi_H_hat),
                ifft(phi_N_hat),
            ):
                blew_up = True
                blowup_step = n
                steps_completed = n + 1
                break

    end_time = time.perf_counter()
    elapsed = end_time - start_time

    if blew_up:
        print(
            f"[Simulation {sim_num+1}] BLEW UP at step {blowup_step} "
            f"(t = {blowup_step * dt:.6f}) after {elapsed:.2f} secs"
        )
    else:
        print(f"[Simulation {sim_num+1}] Simulation complete (took {elapsed} secs)")

    # Only keep the steps that were actually computed
    phi_sigma_data = phi_sigma_data[:steps_completed]
    phi_T_data = phi_T_data[:steps_completed]
    phi_H_data = phi_H_data[:steps_completed]
    phi_N_data = phi_N_data[:steps_completed]

    # Save results to corresponding simulation folder
    print(f"[Simulation {sim_num+1}] Saving simulation")

    start_time = time.perf_counter()
    np.save(f"./output/{timestamp}/sim_{sim_num+1}/phi_sigma_data", phi_sigma_data)
    np.save(f"./output/{timestamp}/sim_{sim_num+1}/phi_T_data", phi_T_data)
    np.save(f"./output/{timestamp}/sim_{sim_num+1}/phi_H_data", phi_H_data)
    np.save(f"./output/{timestamp}/sim_{sim_num+1}/phi_N_data", phi_N_data)
    np.save(f"./output/{timestamp}/sim_{sim_num+1}/time_data", t_data)

    result = {
        "sim_num": sim_num + 1,
        "blew_up": blew_up,
        "blowup_step": blowup_step,
        "blowup_time": None if blowup_step is None else blowup_step * dt,
        "steps_completed": steps_completed,
        "parameters": parameters,
    }

    with open(f"./output/{timestamp}/sim_{sim_num+1}/params.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=4)
    end_time = time.perf_counter()

    print(f"[Simulation {sim_num+1}] Saving complete (took {end_time - start_time} secs)")

    return result
