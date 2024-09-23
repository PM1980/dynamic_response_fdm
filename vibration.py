import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

def calculate_response_cds(K, M, zeta, x0, v0, tf, dt, force_type, force_param, omega_force=0.0):
    """
    Calculate the dynamic response of a damped harmonic oscillator using the Central Difference Scheme.

    Parameters:
    - K: Stiffness (N/m)
    - M: Mass (kg)
    - zeta: Damping ratio
    - x0: Initial displacement (m)
    - v0: Initial velocity (m/s)
    - tf: Final time (s)
    - dt: Time step (s)
    - force_type: "Harmonic" or "Linear"
    - force_param: Amplitude for Harmonic, Slope for Linear
    - omega_force: Frequency for Harmonic force (rad/s)

    Returns:
    - t: Time array
    - x: Displacement array
    """
    omega_n = np.sqrt(K / M)  # Natural frequency (rad/s)
    C = 2 * zeta * np.sqrt(K * M)  # Damping coefficient (N·s/m)

    t = np.arange(0, tf + dt, dt)
    n = len(t)
    x = np.zeros(n)
    
    # Initialize displacement and velocity
    x[0] = x0
    # Calculate initial acceleration
    if force_type == "Harmonic":
        F0 = force_param * np.sin(omega_force * t[0])
    else:  # Linear
        F0 = force_param * t[0]
    a0 = (F0 - C * v0 - K * x0) / M
    # Use Taylor series to estimate x[1]
    x[1] = x0 + dt * v0 + 0.5 * dt**2 * a0

    # Stability Check
    critical_dt = 2 / omega_n
    if dt >= critical_dt:
        st.warning(f"The chosen time step dt = {dt} s may be too large for stability of the Central Difference Scheme. Consider dt < {critical_dt:.4f} s.")

    # Compute response using Central Difference Scheme
    for i in range(1, n-1):
        if force_type == "Harmonic":
            F = force_param * np.sin(omega_force * t[i])
        else:  # Linear
            F = force_param * t[i]
        
        # Central Difference Formula
        x_next = (2 * x[i] - x[i-1] + dt**2 * (F - C * (x[i] - x[i-1]) / dt - K * x[i]) / M) / (1 + (C * dt) / (2 * M))
        x[i+1] = x_next

    return t, x

# Streamlit User Interface
st.title("Damped Harmonic Oscillator Simulation (Central Difference Scheme)")

st.sidebar.header("System Parameters")
K = st.sidebar.number_input("Stiffness (N/m)", value=1000.0, min_value=0.1)
M = st.sidebar.number_input("Mass (kg)", value=1.0, min_value=0.1)
zeta = st.sidebar.slider("Damping Ratio (ζ)", 0.0, 2.0, 0.05, step=0.01)
x0 = st.sidebar.number_input("Initial Displacement (m)", value=1.0)
v0 = st.sidebar.number_input("Initial Velocity (m/s)", value=0.0)
tf = st.sidebar.number_input("Simulation Time (s)", value=10.0, min_value=0.1, step=0.1)
dt = st.sidebar.number_input("Time Step (s)", value=0.01, min_value=0.001, step=0.001)

st.sidebar.header("External Force")
force_type = st.sidebar.selectbox("Force Type", ["Harmonic", "Linear"])
if force_type == "Harmonic":
    force_amplitude = st.sidebar.number_input("Force Amplitude (N)", value=0.0)
    omega_force = st.sidebar.number_input("Force Frequency (rad/s)", value=1.0, min_value=0.0)
    force_param = force_amplitude
else:
    force_slope = st.sidebar.number_input("Force Slope (N/s)", value=0.0)
    force_param = force_slope
    omega_force = 0.0  # Not used for Linear force

# Calculate response
t, x = calculate_response_cds(K, M, zeta, x0, v0, tf, dt, force_type, force_param, omega_force)

# Plotting Displacement vs Time
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(t, x, label='Displacement x(t)', color='blue')
ax.set_xlabel("Time (s)", fontsize=14)
ax.set_ylabel("Displacement (m)", fontsize=14)
ax.set_title("Oscillator Response using Central Difference Scheme", fontsize=16)
ax.grid(True)
ax.legend()

st.pyplot(fig)

# Optional: Phase Space Plot
# Calculate velocity using central difference approximation
v = np.zeros_like(x)
v[0] = v0
for i in range(1, len(x)-1):
    v[i] = (x[i+1] - x[i-1]) / (2 * dt)
v[-1] = (x[-1] - x[-2]) / dt  # Forward difference for the last point

fig_phase, ax_phase = plt.subplots(figsize=(10, 6))
ax_phase.plot(x, v, label='Phase Space Trajectory', color='green')
ax_phase.set_xlabel("Displacement (m)", fontsize=14)
ax_phase.set_ylabel("Velocity (m/s)", fontsize=14)
ax_phase.set_title("Phase Space Plot", fontsize=16)
ax_phase.grid(True)
ax_phase.legend()

st.pyplot(fig_phase)

# Optional: Energy Plot
kinetic = 0.5 * M * v**2
potential = 0.5 * K * x**2
total_energy = kinetic + potential

fig_energy, ax_energy = plt.subplots(figsize=(10, 6))
ax_energy.plot(t, kinetic, label='Kinetic Energy', color='red')
ax_energy.plot(t, potential, label='Potential Energy', color='orange')
ax_energy.plot(t, total_energy, label='Total Energy', color='purple')
ax_energy.set_xlabel("Time (s)", fontsize=14)
ax_energy.set_ylabel("Energy (J)", fontsize=14)
ax_energy.set_title("Energy of the Oscillator", fontsize=16)
ax_energy.grid(True)
ax_energy.legend()

st.pyplot(fig_energy)
