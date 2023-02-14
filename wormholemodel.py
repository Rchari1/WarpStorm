import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D, axes3d
from scipy.integrate import solve_ivp
from matplotlib import cm

def f_R(R, alpha):
    # Define the specific form of f(R) gravity being used
    return alpha * R

def dxdt(t, y, mass1, mass2, alpha):
    # Define the derivatives of x, y, and z with respect to time
    x, y, z = y[:3]
    G = 6.67430 * 10**-11 # Gravitational constant
    dx = -G * mass1 * x / np.sqrt(x**2 + y**2 + z**2)**3 - G * mass2 * (x - 10) / np.sqrt((x - 10)**2 + y**2 + z**2)**3
    dy = -G * mass1 * y / np.sqrt(x**2 + y**2 + z**2)**3 - G * mass2 * y / np.sqrt((x - 10)**2 + y**2 + z**2)**3
    dz = -G * mass1 * z / np.sqrt(x**2 + y**2 + z**2)**3 - G * mass2 * z / np.sqrt((x - 10)**2 + y**2 + z**2)**3
    # Add a term to incorporate the effects of f(R) gravity
    dx = dx + alpha * x / np.sqrt(x**2 + y**2 + z**2)
    dy = dy + alpha * y / np.sqrt(x**2 + y**2 + z**2)
    dz = dz + alpha * z / np.sqrt(x**2 + y**2 + z**2)
    return np.array([dx, dy, dz, 0, 0, 0])

# Initial conditions
x0 = -5
y0 = 0
z0 = 5
vx0 = 1
vy0 = 2
vz0 = -0.5
mass1 = 0
mass2 = 0
alpha = 0.5 # Set the value of alpha based on the specific form of f(R) gravity being used

# Set the time range for the simulation
t_eval = np.linspace(0, 10, 1000)

# Solve the system of ODEs using the solve_ivp function
solution = solve_ivp(fun=lambda t, y: dxdt(t, y, mass1, mass2, alpha), t_span=[0, 10], y0=[x0, y0, z0, vx0, vy0, vz0], t_eval=t_eval, method='RK45')

# Extract the solution for x, y, and z
x = solution.y[0]
y = solution.y[1]
z = solution.y[2]

# Generate a torus shape
R = 1
r = 0.5
phi, theta = np.meshgrid(np.linspace(0, 2 * np.pi, 100), np.linspace(0, 2 * np.pi, 100))

x_torus = (R + r * np.cos(phi)) * np.cos(theta)
y_torus = (R + r * np.cos(phi)) * np.sin(theta)
z_torus = r * np.sin(phi)

%matplotlib nbagg

# Plot the position of the particle
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.plot_surface(x_torus, y_torus, z_torus, cmap=cm.coolwarm, alpha=0.5)
ax.plot(x, y, z, 'red', lw=2)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
plt.show()

