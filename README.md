# 3D Particle Simulation with f(R) Gravity

## This code simulates the motion of a particle in three-dimensional space under the influence of two point masses (a non-trvial wormhole) and an additional force due to f(R) gravity. The simulation is implemented using the scipy.integrate library and the matplotlib library is used to visualize the results.

## Requirements
numpy
scipy
matplotlib
mpl_toolkits

## Usage
To run the code, simply execute the code in a Python environment. A 3D plot of the particle's position will be displayed, along with a torus shape. The particle's position is shown in red.

## Code Explanation
## The code consists of the following main components:

*f_R function: Defines the specific form of f(R) gravity being used, where R is the scalar curvature and alpha is a constant that determines the strength of the f(R) force.

*dxdt function: Defines the derivatives of the particle's position (x, y, and z) and velocity (vx, vy, and vz) with respect to time, taking into account the gravitational forces from the two point masses and the additional f(R) force.

*Initial conditions: Specifies the initial conditions for the particle's position and velocity, as well as the intial conditions of a non-trivial wormhole.

*Time range: Sets the time range for the simulation using the np.linspace function.

*Ordinary Differential Equation solution: Solves the system of ODEs using the solve_ivp function from the scipy.integrate library.

*Plotting: Plots the solution for the particle's position as a 3D curve in a figure, along with a torus shape. The plot is created using the matplotlib library and the projection argument of the add_subplot function is set to '3d' to specify that the plot is in three dimensions. The plot_surface function is used to plot the torus shape and the plot function is used to plot the particle's position.

## Note
The specific form of f(R) gravity and the value of alpha used in the simulation are hard-coded in the code. To use a different form of f(R) gravity or to change the value of alpha, the f_R function and the dxdt function need to be modified accordingly.

## Author:

Raghav Chari, Undergraduate Student at the University of Tennessee Department of Physics and Astromony


