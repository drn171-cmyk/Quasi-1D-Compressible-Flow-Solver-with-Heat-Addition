import numpy as np
import matplotlib.pyplot as plt

# Constants
gamma = 1.4  # Ratio of specific heats for air
R = 287.0    # Specific gas constant for air (J/kg*K)
C_p = 1005.0 # Specific heat at constant pressure for air (J/kg*K)

# Boundary conditions
T_0 = 600.0    # Initial temperature (K)
p_0 = 5e6      # Initial pressure (Pa)
u = 100.0      # Initial velocity (m/s)
L = 1.0        # Length of the domain (m)
T_1 = 300.0    # Final temperature (K) (Note: Not explicitly used in initial march)
p_1 = 101325.0 # Final pressure (Pa) (Note: Not explicitly used in initial march)
q = 1000.0     # Heat transfer rate (W/m)

# Discretization
N = 100              # Number of grid points
dx = L / (N - 1)     # Grid spacing
x = np.linspace(-L/2, L/2, N)  # Spatial grid
y = 0.5 * x**2 + 1              # Parabolic profile for y
A = np.pi * (y**2)               # Cross-sectional area

# Initialize arrays for temperature, pressure, and velocity
T = np.zeros(N)
p = np.zeros(N)
u_array = np.zeros(N)
rho = np.zeros(N)
T_total = np.zeros(N)  # Total temperature array
Ma = np.zeros(N)       # Mach number array

# Set initial conditions
T[0] = T_0
p[0] = p_0
u_array[0] = u
rho[0] = p[0] / (R * T[0])                            # Initial density using ideal gas law
T_total[0] = T[0] + (u_array[0]**2) / (2 * C_p)       # Total temperature at the inlet
Ma[0] = u_array[0] / np.sqrt(gamma * R * T[0])        # Initial Mach number

m_dot = rho[0] * u_array[0] * A[0]  # Mass flow rate (kg/s)

# Compute temperature, pressure, velocity, and density along the domain
for i in range(1, N):

    T_total[i] = T_total[i-1] + (q * dx) / (m_dot * C_p)

    # Initial guess for Gauss-Seidel
    p[i] = p[i-1]
    u_array[i] = u_array[i-1]

    # Gauss-Seidel iteration for pressure and velocity
    for j in range(100):  # Maximum iterations

        p_old = p[i]
        u_old = u_array[i]

        # Update static temperature using total temperature and velocity
        T[i] = T_total[i] - (u_array[i]**2) / (2 * C_p)

        # Update density using ideal gas law
        rho[i] = p[i] / (R * T[i])

        # Update velocity using mass flow rate
        u_array[i] = m_dot / (rho[i] * A[i])

        # Update pressure using momentum equation (simplified for quasi-1D flow)
        p[i] = p[i-1] + (rho[i-1]*A[i-1]*u_array[i-1]**2 - rho[i]*A[i]*u_array[i]**2) / A[i]

        # Check for convergence based on changes in pressure and velocity
        if np.abs(p[i] - p_old) < 1e-6 and np.abs(u_array[i] - u_old) < 1e-6:
            break

    # Compute Mach number
    Ma[i] = u_array[i] / np.sqrt(gamma * R * T[i])

    # Check for normal shock conditions
    Ma_1 = Ma[i-1]

    if Ma_1 > 1 and p[i] > p[i-1]:  # Normal shock conditions
        # Compute post-shock conditions using normal shock relations
        Ma_2_sq = ((gamma-1)*Ma_1**2 + 2) / (2*gamma*Ma_1**2 - (gamma-1))  # Mach number squared after shock
        Ma_2 = np.sqrt(Ma_2_sq)

        rho[i] = rho[i-1] * ((gamma+1)*Ma_1**2) / ((gamma-1)*Ma_1**2 + 2)          # Density after shock
        p[i] = p[i-1] * (2*gamma*Ma_1**2 - (gamma-1)) / (gamma+1)                  # Pressure after shock
        T[i] = p[i] / (rho[i] * R)                                                 # Temperature after shock
        u_array[i] = Ma_2 * np.sqrt(gamma * R * T[i])                               # Velocity after shock
        Ma[i] = Ma_2                                                               # Mach number after shock

# Plot results
plt.figure(figsize=(12, 10))

plt.subplot(3, 2, 1)
plt.plot(x, T, label='Temperature (K)')
plt.xlabel('Position (m)')
plt.ylabel('Temperature (K)')
plt.title('Temperature Distribution')
plt.grid()

plt.subplot(3, 2, 2)
plt.plot(x, p, label='Pressure (Pa)')
plt.xlabel('Position (m)')
plt.ylabel('Pressure (Pa)')
plt.title('Pressure Distribution')
plt.grid()

plt.subplot(3, 2, 3)
plt.plot(x, u_array, label='Velocity (m/s)')
plt.xlabel('Position (m)')
plt.ylabel('Velocity (m/s)')
plt.title('Velocity Distribution')
plt.grid()

plt.subplot(3, 2, 4)
plt.plot(x, rho, label='Density (kg/m^3)')
plt.xlabel('Position (m)')
plt.ylabel('Density (kg/m^3)')
plt.title('Density Distribution')
plt.grid()

plt.subplot(3, 2, 5)
plt.plot(x, Ma, label='Mach Number', color='purple')
plt.axhline(y=1.0, color='red', linestyle='--', linewidth=1, label='Ma = 1')
plt.xlabel('Position (m)')
plt.ylabel('Mach Number (-)')
plt.title('Mach Number Distribution')
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()

# Contour plots for temperature and velocity

Ny = 100
y_max = np.max(y)
y_grid_1d = np.linspace(-y_max, y_max, Ny)


X_grid, Y_grid = np.meshgrid(x, y_grid_1d)

T_2D = np.full_like(X_grid, np.nan)
u_2D = np.full_like(X_grid, np.nan)

for i in range(N):
    for j in range(Ny):
        if np.abs(Y_grid[j, i]) <= y[i]:
            T_2D[j, i] = T[i]
            u_2D[j, i] = u_array[i]

plt.figure(figsize=(14, 5))

# 1. Temperature Contour
plt.subplot(1, 2, 1)
# cmap='inferno' 
contour_T = plt.contourf(X_grid, Y_grid, T_2D, levels=100, cmap='inferno')
plt.plot(x, y, 'k-', linewidth=2)  # Üst duvar
plt.plot(x, -y, 'k-', linewidth=2) # Alt duvar (simetrik)
plt.colorbar(contour_T, label='Temperature (K)')
plt.title('Temperature Contour Plot')
plt.xlabel('Position x (m)')
plt.ylabel('Position y (m)')
plt.grid(False) 

# 2. Velocity Contour
plt.subplot(1, 2, 2)
# cmap='viridis' 
contour_u = plt.contourf(X_grid, Y_grid, u_2D, levels=100, cmap='viridis')
plt.plot(x, y, 'k-', linewidth=2)  # Üst duvar
plt.plot(x, -y, 'k-', linewidth=2) # Alt duvar
plt.colorbar(contour_u, label='Velocity (m/s)')
plt.title('Velocity Contour Plot')
plt.xlabel('Position x (m)')
plt.ylabel('Position y (m)')
plt.grid(False)

plt.tight_layout()
plt.show()
