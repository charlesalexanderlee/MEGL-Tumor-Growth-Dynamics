# Processes 
# 1a. running simulation
# 1b. writing results to disk
# 2a. reading results from disk
# 2b. creating animation job
# 2c. writing animation movie to disk
from archive.sim_old.run_simulation import Simulation
import numpy as np

def main():
    NUM_SIMULATIONS = 1000000  
    N = 256
    L = 10.1               
    dt = 5.0e-5        
    T = 1.30                
    nsteps = int(T / dt)

    # Parameters from Lima model
    parameters = {
        "num_simulations": NUM_SIMULATIONS,

        "N": N,
        "L": L,
        "dt": dt,
        "T": T,
        "nsteps": nsteps,

        "eps": 0.04,
        "M": 0.1,   
        "sigma": 3.0e-5,
        "M_sigma": 0.20,
        "delta_sigma": 0.10,
        "chi0": 0.035,
        "eps_T": 0.25,
        "M0": 0.90,
        "lambda_p": 0.05,
        "lambda_A": 0.01,

        "Mp":          np.linspace(start=100, stop=300, num=N),
        "Mh":          np.linspace(start=10, stop=30, num=N),
        "Mn":          np.linspace(start=0, stop=10, num=N),
        "D":           np.linspace(start=0.5, stop=1.5, num=N),
        "lambda_p":    np.linspace(start=0, stop=1, num=N),
        "lambda_a":    np.linspace(start=0, stop=1, num=N),
        "lambda_ph":   np.linspace(start=0, stop=1, num=N),
        "lambda_pH":   np.linspace(start=0, stop=1, num=N),
        "lambda_Hp":   np.linspace(start=0, stop=1, num=N),
        "lambda_hn":   np.linspace(start=0, stop=1, num=N),
        "eps_T":       np.linspace(start=0, stop=0.01, num=N),
        "chi0":        np.linspace(start=0.01, stop=0.05, num=N),
        "delta_sigma": np.linspace(start=0.0001, stop=1, num=N),
        "sigma_ph":    np.linspace(start=0.1, stop=1, num=N),
        "sigma_hp":    np.linspace(start=0.1, stop=1, num=N),
        "sigma_hn":    np.linspace(start=0.1, stop=1, num=N),
        "Ebar":        np.linspace(start=0.025, stop=0.05, num=N)
    }

    sim = Simulation(
        params=parameters
    )

    sim.start()

if __name__ == "__main__":
    main()
