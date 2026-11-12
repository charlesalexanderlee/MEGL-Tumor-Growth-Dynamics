import numpy as np
import time
from archive.sim_old.helpers import *
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import ProcessPoolExecutor

class Simulation():
    def __init__(self, params):
        self.num_simulations = params["num_simulations"]

        self.n = params["N"]
        self.L = params["L"]
        self.dt = params["dt"]
        self.T = params["T"]
        self.nsteps = params["nsteps"]

        self.eps = params["eps"]
        self.M = params["M"]
        self.sigma = params["sigma"]
        self.M_sigma = params["M_sigma"]
        self.delta_sigma = params["delta_sigma"]
        self.chi0 = params["chi0"]
        self.eps_T = params["eps_T"]
        self.M0 = params["M0"]
        self.lambda_p = params["lambda_p"]
        self.lambda_A = params["lambda_A"]
        
        self.Mp = params["Mp"]
        self.Mh = params["Mh"]
        self.Mn = params["Mn"]
        self.D = params["D"]
        self.lambda_p = params["lambda_p"]
        self.lambda_a = params["lambda_a"]
        self.lambda_ph = params["lambda_ph"]
        self.lambda_pH = params["lambda_pH"]
        self.lambda_Hp = params["lambda_Hp"]
        self.lambda_hn = params["lambda_hn"]
        self.eps_T = params["eps_T"]
        self.chi0 = params["chi0"]
        self.delta_sigma = params["delta_sigma"]
        self.sigma_ph = params["sigma_ph"]
        self.sigma_hp = params["sigma_hp"]
        self.sigma_hn = params["sigma_hn"]
        self.Ebar = params["Ebar"]

    def simulation(self, sim_num):
        # print("sim started", sim_num)
        x = np.linspace(0, self.L, self.N)
        dx = self.L / self.N
    
        # Wave Numbers 
        k = 2.0 * np.pi * np.fft.fftfreq(self.N, d=dx)
        k2 = k**2
        k4 = k**4
    
        rng = np.random.default_rng(300)
    
        phi_sigma = 0.04 * np.abs(rng.standard_normal(self.N))
        phi_T = 0.05 * np.abs(rng.standard_normal(self.N))
        #phi_T = (.1)*1.0/( (x - 2.0)**2 +1)
        phi_H = 0.05 * np.abs(rng.standard_normal(self.N))
        phi_N = 0.05 * np.abs(rng.standard_normal(self.N))
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
        # noise_scale = np.sqrt( 2.0 * M * sigma * dt / dx)
    
        den_sigma = 1.0 + self.dt * self.M_sigma * self.delta_sigma * k2
        den_T = 1.0 + self.dt * self.M0 * self.eps_T**2 * k4
        den_H = 1.0 + self.dt * self.M0 * self.eps_T**2 * k4
        den_N = 1.0 + self.dt * self.M0 * self.eps_T**2 * k4

        for n in range(self.nsteps):
            phi_sigma = ifft(phi_sigma_hat)
            phi_T = ifft(phi_T_hat)
            phi_H = ifft(phi_H_hat)
            phi_N = ifft(phi_N_hat)
    
            mu =  Phi_prime(phi_T) - self.eps_T**2 * laplacian(phi_T) - self.chi0 * phi_sigma
            
            rhs_sigma = phi_sigma_hat + self.dt * self.M_sigma * self.chi0 * k2 * phi_T_hat + self.dt * fft(S_sigma(phi_sigma, phi_T, phi_H, phi_N))  
            phi_sigma_hat_new = rhs_sigma / den_sigma
    
            rhs_T =  -self.M0 * k2 * fft(Phi_prime(phi_T)) + self.M0 * self.chi0 * k2 * phi_sigma_hat 
            rhs_H =  -self.M0 * k2 * fft(Phi_prime(phi_T)) + self.M0 * self.chi0 * k2 * phi_sigma_hat 
            rhs_N =  -self.M0 * k2 * fft(Phi_prime(phi_T)) + self.M0 * self.chi0 * k2 * phi_sigma_hat 
    
            d_mu = derivative(mu)
            M_current_T = Mobility(phi_T, phi_H, phi_N)
            variable_flux_T = (M_current_T - self.M0) * d_mu
            M_current_H = Mobility_h(phi_H)
            variable_flux_H = (M_current_H - self.M0) * d_mu
            M_current_N = Mobility_n(phi_N)
            variable_flux_N = (M_current_N - self.M0) * d_mu
    
            rhs_T += fft(derivative(variable_flux_T))
            rhs_T += fft(S_T(phi_sigma, phi_T, phi_H, phi_N))
            rhs_H += fft(derivative(variable_flux_H))
            rhs_H += fft(S_H(phi_sigma, phi_T, phi_H, phi_N))
            rhs_N += fft(derivative(variable_flux_N))
            rhs_N += fft(S_N(phi_sigma, phi_T, phi_H, phi_N))
    
            eta_T = rng.standard_normal(self.N) * 0.3
            eta_T_hat = np.fft.fft(eta_T)
            noise_hat_T = ( np.sqrt(2.0 * self.M * self.sigma * self.dt)* 1j * k * eta_T_hat / np.sqrt(dx) ) * 0
    
            eta_H = rng.standard_normal(self.N) * 0.3
            eta_H_hat = np.fft.fft(eta_H)
            noise_hat_H = ( np.sqrt(2.0 * self.M * self.sigma * self.dt)* 1j * k * eta_H_hat / np.sqrt(dx) ) * 0
    
            eta_N = rng.standard_normal(self.N) * 0.3
            eta_N_hat = np.fft.fft(eta_N)
            noise_hat_N = ( np.sqrt(2.0 * self.M * self.sigma * self.dt)* 1j * k * eta_N_hat / np.sqrt(dx) ) * 0
    
            phi_T_hat_new = ( phi_T_hat + self.dt * rhs_T + noise_hat_T) / den_T
            phi_H_hat_new = ( phi_H_hat + self.dt * rhs_H + noise_hat_H) / den_H
            phi_N_hat_new = ( phi_N_hat + self.dt * rhs_N + noise_hat_N) / den_N
    
            phi_sigma_hat = phi_sigma_hat_new
            phi_T_hat = phi_T_hat_new
            phi_H_hat = phi_H_hat_new
            phi_N_hat = phi_N_hat_new
    
            phi_sigma_data.append(phi_sigma)
            phi_T_data.append(phi_T)
            phi_H_data.append(phi_H)
            phi_N_data.append(phi_N)
            tdata.append(n * self.dt)
        
        results = {
            "phi_sigma_data": phi_sigma_data,
            "phi_T_data": phi_T_data,
            "phi_H_data": phi_H_data,
            "phi_N_data": phi_N_data,
            "tdata": tdata
        } 

        # print(results)
        return results

        # return sim_num

    def start(self):
        items = np.arange(self.num_simulations)

        start_time = time.perf_counter()
        # with ThreadPoolExecutor(max_workers=100) as executer:
        #     results = executer.map(self.simulation, items)
        with ProcessPoolExecutor(max_workers=100) as executor:
                results = executor.map(self.simulation, items)
        end_time = time.perf_counter()
        execution_time = end_time - start_time

        # for result in results:
        #     print(result)

        print(results)
        print(f"{self.num_simulations} simulations complete ({execution_time} secs)")
