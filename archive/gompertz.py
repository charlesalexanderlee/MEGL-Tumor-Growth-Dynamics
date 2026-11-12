import numpy as np
import matplotlib.pyplot as plt

# Parameters
gamma = 1.5
beta = 20

alpha_i = 4.4
alpha_s = 0.3
alpha_c = 4.8

N_i = 4.0
N_s = 4.0
N_c = 4.0

# State (r, i, s, c)
initial_state = np.array([0.001, 1, 0.001, 0.05])

time_delta = 0.01
max_time_steps = 500

# Data matrix
X = initial_state

state = initial_state
print(initial_state)

for t in range(max_time_steps):
    r, i, s, c = state[0], state[1], state[2], state[3]

    r_p = gamma - beta*r*i
    i_p = alpha_i * i * np.log(N_i/i) - beta*r*i
    s_p = beta * r * i + alpha_s * s * np.log(N_s/s)
    c_p = alpha_c * c * np.log(N_c/c) - s*c

    state = state + time_delta * np.array([r_p, i_p, s_p, c_p])
    X = np.vstack([X, state])

    print(state)

# plt.title("RISC Model (Gompertz)")
# plt.plot(X[:, 0], color="red", label="r")
# plt.plot(X[:, 1], color="blue", label="i")
# plt.plot(X[:, 2], color="green", label="s")
# plt.plot(X[:, 3], color="black", label="c")
# plt.scatter(X[:, 2], X[:, 3])
# plt.legend()

fig = plt.figure(figsize=plt.figaspect(2))

# 3D Plot of R, I, S
colors = X[1:-2, 2]
ax = fig.add_subplot(2, 1, 1, projection='3d', computed_zorder=False)
sc = ax.scatter(X[0, 0], X[0, 1], X[0, 2], marker='^', color='green', s=40, zorder=2)
sc = ax.scatter(X[1:-2, 0], X[1:-2, 1], X[1:-2, 2], zorder=1, c='blue')
sc = ax.scatter(X[-1:, 0], X[-1:, 1], X[-1:, 2], marker='v', color='red', s=40, zorder=2)
ax.set_xlabel('Reservoir Cells', fontweight='bold')
ax.set_ylabel('Immune Cells', fontweight='bold')
ax.set_zlabel('Superimmune Cells', fontweight='bold')
ax.set_title("RIS Cells Over Time\n(Gompertz Growth)", fontweight='bold')
# plt.colorbar(sc, ax=ax, label='Time')

# 2D Plot of C
ax = fig.add_subplot(2, 1, 2)
ax.plot(X[:, 3], color='red')
ax.grid(True)
ax.set_xlabel('Time', fontweight='bold')
ax.set_ylabel('Cancer Cells', fontweight='bold')
plt.show()

# Forward Euler (potential instabilities with the time delta)
