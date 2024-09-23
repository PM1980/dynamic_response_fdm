import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

def calculate_response(K, M, zeta, x0, v0, tf, dt, force_type, force_amp, force_freq):
    omega_n = np.sqrt(K / M)
    C = 2 * zeta * np.sqrt(K * M)

    def oscillator_ode(t, y):
        x, v = y
        if force_type == "Harmonic":
            F = force_amp * np.sin(force_freq * t)
        else:  # Linear
            F = force_amp * t
        return [v, (F - C * v - K * x) / M]

    sol = solve_ivp(oscillator_ode, (0, tf), [x0, v0], t_eval=np.arange(0, tf + dt, dt))
    return sol.t, sol.y[0]

st.title("Damped Harmonic Oscillator Simulation")

st.sidebar.header("System Parameters")
K = st.sidebar.number_input("Stiffness (N/m)", value=1000.0, min_value=0.1)
M = st.sidebar.number_input("Mass (kg)", value=1.0, min_value=0.1)
zeta = st.sidebar.slider("Damping Ratio", 0.0, 2.0, 0.1)
x0 = st.sidebar.number_input("Initial Displacement (m)", value=1.0)
v0 = st.sidebar.number_input("Initial Velocity (m/s)", value=0.0)
tf = st.sidebar.number_input("Simulation Time (s)", value=10.0, min_value=0.1)
dt = st.sidebar.number_input("Time Step (s)", value=0.01, min_value=0.001)

st.sidebar.header("External Force")
force_type = st.sidebar.selectbox("Force Type", ["Harmonic", "Linear"])
force_amp = st.sidebar.number_input("Force Amplitude (N)", value=0.0)
if force_type == "Harmonic":
    force_freq = st.sidebar.number_input("Force Frequency (rad/s)", value=1.0, min_value=0.01)
else:
    force_freq = 0.0

t, x = calculate_response(K, M, zeta, x0, v0, tf, dt, force_type, force_amp, force_freq)

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(t, x, 'b-', linewidth=2, label='Displacement x(t)')
ax.set_xlabel("Time (s)")
ax.set_ylabel("Displacement (m)")
ax.set_title("Damped Harmonic Oscillator Response")
ax.grid(True)
ax.legend()
st.pyplot(fig)

# Velocity and Acceleration Plots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

# Velocity Plot
ax1.plot(t, np.gradient(x, t), 'g-', linewidth=2, label='Velocity v(t)')
ax1.set_xlabel("Time (s)")
ax1.set_ylabel("Velocity (m/s)")
ax1.set_title("Velocity of Damped Harmonic Oscillator")
ax1.grid(True)
ax1.legend()

# Acceleration Plot
ax2.plot(t, np.gradient(np.gradient(x, t), t), 'r-', linewidth=2, label='Acceleration a(t)')
ax2.set_xlabel("Time (s)")
ax2.set_ylabel("Acceleration (m/s²)")
ax2.set_title("Acceleration of Damped Harmonic Oscillator")
ax2.grid(True)
ax2.legend()

st.pyplot(fig)

# Phase Plane Plot
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111)
ax.plot(x, np.gradient(x, t), 'k-', linewidth=2)
ax.set_xlabel("Displacement (m)")
ax.set_ylabel("Velocity (m/s)")
ax.set_title("Phase Plane of Damped Harmonic Oscillator")
ax.grid(True)
st.pyplot(fig)

# Animated Plot (Requires FFmpeg)
fig, ax = plt.subplots(figsize=(8, 6))
line, = ax.plot([], [], lw=2)
ax.set_xlim(0, tf)
ax.set_ylim(min(x) * 1.1, max(x) * 1.1)
ax.set_xlabel("Time (s)")
ax.set_ylabel("Displacement (m)")
ax.set_title("Animated Response of Damped Harmonic Oscillator")

def animate(i):
    line.set_data(t[:i], x[:i])
    return line,

from matplotlib.animation import FuncAnimation
ani = FuncAnimation(fig, animate, len(t), interval=50, blit=True)

st.write("Animated Response:")
st.pyplot(fig)
