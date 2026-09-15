'''
Implementation of the RISC model
Kuhen 2025
'''
import numpy as np
import matplotlib.pyplot as plt

# Model parameters
alpha_i = 4.4
alpha_s = 0.3
alpha_c = 4.8
beta = 20
gamma = 1.5
N_i = 4.0
N_s = 4.0
N_c = 4.0

time_delta = 0.01
max_time_steps = 600

# Initial state 
initial_state = np.array([0, 1, 0, 0.05]) # (r, i, s, c) = (0, 1, 0, 0.05)

# Data matrix
X = initial_state

# Update our state
state = initial_state
for t in range(max_time_steps):
    print(state)
    r, i, s, c = state[0], state[1], state[2], state[3]

    r_p = gamma - (beta * r * i)
    i_p = (alpha_i * i * (1 - (i/N_i))) - (beta * r * i)
    s_p = (beta * r * i) + (alpha_s * s) * (1 - (s/N_s))
    c_p = (alpha_c * c * (1 - (c/N_c))) - (s * c)

    state = state + time_delta * np.array([r_p, i_p, s_p, c_p])
    X = np.vstack([X, state])

plt.title("RISC Model")
plt.plot(X[:, 0], color="red", label="r")
plt.plot(X[:, 1], color="blue", label="i")
plt.plot(X[:, 2], color="green", label="s")
plt.plot(X[:, 3], color="black", label="c")
plt.legend()
plt.show()