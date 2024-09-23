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
    - v: Velocity array
    - a: Acceleration array
    """
    omega_n = np.sqrt(K / M)  # Natural frequency (rad/s)
    C = 2 * zeta * np.sqrt(K * M)  # Damping coefficient (N·s/m)

    t = np.arange(0, tf + dt, dt)
    n = len(t)
    x = np.zeros(n)
    v = np.zeros(n)
    a = np.zeros(n)
    
    # Initialize displacement and velocity
    x[0] = x0
    v[0] = v0
    # Calculate initial acceleration
    if force_type == "Harmonic":
        F0 = force_param * np.sin(omega_force * t[0])
    else:  # Linear
        F0 = force_param * t[0]
    a[0] = (F0 - C * v0 - K * x0) / M
    # Use Taylor series to estimate x[1]
    x[1] = x0 + dt * v0 + 0.5 * dt**2 * a[0]

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
        
        # Calculate velocity and acceleration
        v[i] = (x[i+1] - x[i-1]) / (2 * dt)
        a[i] = (F - C * v[i] - K * x[i]) / M

    # Calculate final velocity and acceleration
    v[-1] = (x[-1] - x[-2]) / dt
    a[-1] = (F - C * v[-1] - K * x[-1]) / M

    return t, x, v, a

# Streamlit User Interface
st.title("Vibração forçada amortecida - Sistema de 1 grau de liberdade (Diferença Finita Central)")

st.sidebar.header("Parâmetros dos sistema")
K = st.sidebar.number_input("Rigidez (N/m)", value=1000.0, min_value=0.1)
M = st.sidebar.number_input("Massa (kg)", value=1.0, min_value=0.1)
zeta = st.sidebar.slider("Razão de amortecimento (ζ)", 0.0, 2.0, 0.05, step=0.01)
x0 = st.sidebar.number_input("Deslocamento inicial (m)", value=1.0)
v0 = st.sidebar.number_input("Velocidade inicial (m/s)", value=0.0)
tf = st.sidebar.number_input("Tempo de simulação (s)", value=10.0, min_value=0.1, step=0.1)
dt = st.sidebar.number_input("Passo de tempo (s)", value=0.01, min_value=0.001, step=0.001)

st.sidebar.header("Excitação externa")
force_type = st.sidebar.selectbox("Tipo de excitação", ["Fo*sin(w*t)", "Fo*x"])
if force_type == "Fo*sin(w*t)":
    force_amplitude = st.sidebar.number_input("Amplitude Fo (N)", value=0.0)
    omega_force = st.sidebar.number_input("Frequência de excitação w (rad/s)", value=1.0, min_value=0.0)
    force_param = force_amplitude
else:
    force_slope = st.sidebar.number_input("Amplitude Fo(N)", value=0.0)
    force_param = force_slope
    omega_force = 0.0  # Not used for Linear force

# Calculate response
t, x, v, a = calculate_response_cds(K, M, zeta, x0, v0, tf, dt, force_type, force_param, omega_force)

# Plotting Displacement, Velocity, and Acceleration vs Time
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 15), sharex=True)

ax1.plot(t, x, label='Displacement x(t)', color='blue')
ax1.set_ylabel("Displacement (m)", fontsize=12)
ax1.set_title("Oscillator Response using Central Difference Scheme", fontsize=14)
ax1.grid(True)
ax1.legend()

ax2.plot(t, v, label='Velocity v(t)', color='green')
ax2.set_ylabel("Velocity (m/s)", fontsize=12)
ax2.grid(True)
ax2.legend()

ax3.plot(t, a, label='Acceleration a(t)', color='red')
ax3.set_xlabel("Time (s)", fontsize=12)
ax3.set_ylabel("Acceleration (m/s²)", fontsize=12)
ax3.grid(True)
ax3.legend()

plt.tight_layout()
st.pyplot(fig)

# Phase Space Plot
fig_phase, ax_phase = plt.subplots(figsize=(10, 6))
ax_phase.plot(x, v, label='Phase Space Trajectory', color='purple')
ax_phase.set_xlabel("Displacement (m)", fontsize=12)
ax_phase.set_ylabel("Velocity (m/s)", fontsize=12)
ax_phase.set_title("Phase Space Plot", fontsize=14)
ax_phase.grid(True)
ax_phase.legend()

st.pyplot(fig_phase)

# Energy Plot
kinetic = 0.5 * M * v**2
potential = 0.5 * K * x**2
total_energy = kinetic + potential

fig_energy, ax_energy = plt.subplots(figsize=(10, 6))
ax_energy.plot(t, kinetic, label='Kinetic Energy', color='red')
ax_energy.plot(t, potential, label='Potential Energy', color='orange')
ax_energy.plot(t, total_energy, label='Total Energy', color='purple')
ax_energy.set_xlabel("Time (s)", fontsize=12)
ax_energy.set_ylabel("Energy (J)", fontsize=12)
ax_energy.set_title("Energy of the Oscillator", fontsize=14)
ax_energy.grid(True)
ax_energy.legend()

st.pyplot(fig_energy)
