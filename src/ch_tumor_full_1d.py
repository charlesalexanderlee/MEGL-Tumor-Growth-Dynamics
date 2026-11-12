"""
SPATIOTEMPORAL MODELING OF TUMOR GROWTH DYNAMICS
https://github.com/charlesalexanderlee/MEGL-Tumor-Growth-Dynamics

Mason Experimental Geometry Laboratory
George Mason University
Fall 2026

[Faculty Members]
* Dr. Daniel Anderson
* Dr. Scott Cochran

[Graduate Students]
* Elisha Erzoah

[Undergraduate Students]
* Charles Alexander Lee
* Lia McPherson
* Khoi Vu
"""

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from helpers import *
from datetime import datetime
import numpy as np
import pandas as pd
import pathlib
import json
import time

def save_results(data, label, timestamp, sim_num):
    print(f"[Simulation {sim_num}] Saving {label}.csv")
    
    # Write dataframe to file
    pd.DataFrame(data).to_csv(f"./output/{timestamp}/sim_{sim_num+1}/{label}.csv")

    print(f"[Simulation {sim_num}] Saving {label}.csv complete")

def run_simulation(sim_num, timestamp):
    # Simulation Settings
    N = 256
    L = 10.1               
    dt = 5.0e-5
    T = 1.30                
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

    # Create folder for current simulation
    folder_path = pathlib.Path(f"./output/{timestamp}/sim_{sim_num+1}")
    folder_path.mkdir(parents=True, exist_ok=True)

    print(f"[Simulation {sim_num+1}] Started")

    Mp = np.random.choice(np.linspace(start=100, stop=300, num=N))
    Mh = np.random.choice(np.linspace(start=10, stop=30, num=N))
    Mn = np.random.choice(np.linspace(start=0, stop=10, num=N))
    D = np.random.choice(np.linspace(start=0.5, stop=1.5, num=N))
    lambda_p = np.random.choice(np.linspace(start=0, stop=1, num=N))
    lambda_a = np.random.choice(np.linspace(start=0, stop=1, num=N))
    lambda_ph = np.random.choice(np.linspace(start=0, stop=1, num=N))
    lambda_pH = np.random.choice(np.linspace(start=0, stop=1, num=N))
    lambda_Hp = np.random.choice(np.linspace(start=0, stop=1, num=N))
    lambda_hn = np.random.choice(np.linspace(start=0, stop=1, num=N))
    eps_T = np.random.choice(np.linspace(start=0, stop=0.01, num=N))
    chi0 = np.random.choice(np.linspace(start=0.01, stop=0.05, num=N))
    delta_sigma = np.random.choice(np.linspace(start=0.0001, stop=1, num=N))
    sigma_ph = np.random.choice(np.linspace(start=0.1, stop=1, num=N))
    sigma_hp = np.random.choice(np.linspace(start=0.1, stop=1, num=N))
    sigma_hn = np.random.choice(np.linspace(start=0.1, stop=1, num=N))
    Ebar = np.random.choice(np.linspace(start=0.025, stop=0.05, num=N))

    parameters = {
        "Mp": Mp,
        "Mh": Mh,
        "Mn": Mn,
        "D": D,
        "lambda_p": lambda_p,
        "lambda_a": lambda_a,
        "lambda_ph": lambda_ph,
        "lambda_pH": lambda_pH,
        "lambda_Hp": lambda_Hp,
        "lambda_hn": lambda_hn,
        "eps_T": eps_T,
        "chi0": chi0,
        "delta_sigma": delta_sigma,
        "sigma_ph": sigma_ph,
        "sigma_hp": sigma_hp,
        "sigma_hn": sigma_hn,
        "Ebar": Ebar
    }

    # Create 1-dimensional line
    x = np.linspace(0, L, N)
    dx = L / N

    # Wave Numbers 
    k = 2.0 * np.pi * np.fft.fftfreq(N, d=dx)
    k2 = k**2
    k4 = k**4

    # Random Number Generator
    rng = np.random.default_rng(300)

    # Initial conditions
    phi_T = 0.05 * np.abs(rng.standard_normal(N))
    phi_H = 0.05 * np.abs(rng.standard_normal(N))
    phi_N = 0.05 * np.abs(rng.standard_normal(N))
    phi_sigma = np.ones(N) - phi_T - phi_H - phi_N

    # Perform FFT on initial condition
    phi_T_hat = np.fft.fft(phi_T)
    phi_H_hat = np.fft.fft(phi_H)
    phi_N_hat = np.fft.fft(phi_N)
    phi_sigma_hat = np.fft.fft(phi_sigma)

    # Initial state
    phi_sigma_data = []
    phi_sigma_data.append(phi_sigma)

    phi_T_data = []
    phi_T_data.append(phi_T)

    phi_H_data = []
    phi_H_data.append(phi_H)

    phi_N_data = []
    phi_N_data.append(phi_N)
    
    tdata = []
    tdata.append(0.0)

    # Noise addition (for later)
    # noise_scale = np.sqrt( 2.0 * M * sigma * dt / dx)

    den_sigma = 1.0 + dt * M_sigma * delta_sigma * k2
    den_T = 1.0 + dt * M0 * eps_T**2 * k4
    den_H = 1.0 + dt * M0 * eps_T**2 * k4
    den_N = 1.0 + dt * M0 * eps_T**2 * k4

    for n in range(nsteps):
        phi_sigma = ifft(phi_sigma_hat)
        phi_T = ifft(phi_T_hat)
        phi_H = ifft(phi_H_hat)
        phi_N = ifft(phi_N_hat)

        mu =  Phi_prime(phi_T) - eps_T**2 * laplacian(phi_T, k2) - chi0 * phi_sigma
        
        rhs_sigma = phi_sigma_hat + dt * M_sigma * chi0 * k2 * phi_T_hat + dt * fft(S_sigma(phi_sigma, phi_T, phi_H, phi_N, lambda_p, lambda_ph, lambda_a))  
        phi_sigma_hat_new = rhs_sigma / den_sigma

        rhs_T =  -M0 * k2 * fft(Phi_prime(phi_T)) + M0 * chi0 * k2 * phi_sigma_hat 
        rhs_H =  -M0 * k2 * fft(Phi_prime(phi_T)) + M0 * chi0 * k2 * phi_sigma_hat 
        rhs_N =  -M0 * k2 * fft(Phi_prime(phi_T)) + M0 * chi0 * k2 * phi_sigma_hat 

        d_mu = derivative(mu, k, order=1)
        M_current_T = Mobility(phi_T, phi_H, phi_N, Mp, Mh, Mn)
        variable_flux_T = (M_current_T - M0) * d_mu
        M_current_H = Mobility_h(phi_H, Mh)
        variable_flux_H = (M_current_H - M0) * d_mu
        M_current_N = Mobility_n(phi_N, Mn)
        variable_flux_N = (M_current_N - M0) * d_mu

        rhs_T += fft(derivative(variable_flux_T, k, order=1))
        rhs_T += fft(S_T(phi_sigma, phi_T, phi_H, phi_N, lambda_p, lambda_A))
        rhs_H += fft(derivative(variable_flux_H, k, order=1))
        rhs_H += fft(S_H(phi_sigma, phi_T, phi_H, phi_N, lambda_A, lambda_pH, sigma_ph, sigma_hp, lambda_hn, sigma_hn))
        rhs_N += fft(derivative(variable_flux_N, k, order=1))
        rhs_N += fft(S_N(phi_sigma, phi_T, phi_H, phi_N, lambda_hn, sigma_hn))

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

    # Save parameters for current simulation to file
    with open(f"./output/{timestamp}/sim_{sim_num+1}/params.json", "w", encoding="utf-8") as f:
        json.dump(parameters, f, indent=4)

    # Save results to corresponding simulation folder
    print(f"[Simulation {sim_num+1}] Saving")
    pd.DataFrame(phi_sigma_data).to_csv(f"./output/{timestamp}/sim_{sim_num+1}/phi_sigma_data.csv")
    pd.DataFrame(phi_T_data).to_csv(f"./output/{timestamp}/sim_{sim_num+1}/phi_T_data.csv")
    pd.DataFrame(phi_H_data).to_csv(f"./output/{timestamp}/sim_{sim_num+1}/phi_H_data.csv")
    pd.DataFrame(phi_N_data).to_csv(f"./output/{timestamp}/sim_{sim_num+1}/phi_N_data.csv")
    pd.DataFrame(tdata).to_csv(f"./output/{timestamp}/sim_{sim_num+1}/tdata.csv")
    print(f"[Simulation {sim_num+1}] Complete")

    # with ThreadPoolExecutor(max_workers=5) as thread_executor:
    #     thread_executor.submit(
    #         save_results,
    #         data=phi_sigma_data,
    #         label="phi_sigma_data",
    #         timestamp=timestamp,
    #         sim_num=sim_num 
    #     )

    #     thread_executor.submit(
    #         save_results,
    #         data=phi_T_data,
    #         label="phi_T_data",
    #         timestamp=timestamp,
    #         sim_num=sim_num 
    #     )

    #     thread_executor.submit(
    #         save_results,
    #         data=phi_H_data,
    #         label="phi_H_data",
    #         timestamp=timestamp,
    #         sim_num=sim_num 
    #     )

    #     thread_executor.submit(
    #         save_results,
    #         data=phi_N_data,
    #         label="phi_N_data",
    #         timestamp=timestamp,
    #         sim_num=sim_num 
    #     )

    #     thread_executor.submit(
    #         save_results,
    #         data=tdata,
    #         label="tdata",
    #         timestamp=timestamp,
    #         sim_num=sim_num 
    #     )

def main():
    NUM_SIMULATIONS = 10
    MAX_WORKERS = 10
    TIMESTAMP = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Create output folder for the run
    folder_path = pathlib.Path(f"./output/{TIMESTAMP}")
    folder_path.mkdir(parents=True, exist_ok=True)

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as process_executor:
        for sim_num in range(NUM_SIMULATIONS):
            process_executor.submit(
                run_simulation, 
                sim_num=sim_num, 
                timestamp=TIMESTAMP
            )

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(e)

