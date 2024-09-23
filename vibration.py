import streamlit as st
import numpy as np
import plotly.graph_objs as go
from plotly.subplots import make_subplots

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
    
    # Calculate initial force
    if force_type == "Harmonic":
        F0 = force_param * np.sin(omega_force * t[0])
    else:  # Linear
        F0 = force_param * t[0]
    
    # Calculate initial acceleration
    a[0] = (F0 - C * v0 - K * x0) / M
    
    # Use Taylor series to estimate x[1]
    x[1] = x0 + dt * v0 + 0.5 * dt**2 * a[0]
    
    # Estimate initial velocity at t=dt using central difference
    v[1] = (x[1] - x[0]) / dt
    
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
        
        # Calculate velocity using central difference
        v[i] = (x[i+1] - x[i-1]) / (2 * dt)
        
        # Calculate acceleration
        a[i] = (F - C * v[i] - K * x[i]) / M

    # For the last point, use forward difference for velocity and acceleration
    v[-1] = (x[-1] - x[-2]) / dt
    if force_type == "Harmonic":
        F_last = force_param * np.sin(omega_force * t[-1])
    else:
        F_last = force_param * t[-1]
    a[-1] = (F_last - C * v[-1] - K * x[-1]) / M

    return t, x, v, a

def create_animation(t, y, y_label, title):
    """
    Create an animated Plotly line plot for the given data.

    Parameters:
    - t: Time array
    - y: Data array (displacement, velocity, or acceleration)
    - y_label: Label for the y-axis
    - title: Title of the plot

    Returns:
    - fig: Plotly Figure object
    """
    fig = go.Figure(
        data=[go.Scatter(x=[], y=[], mode='lines', name=y_label)],
        layout=go.Layout(
            title=title,
            xaxis=dict(range=[t[0], t[-1]], title='Time (s)'),
            yaxis=dict(range=[min(y)*1.1, max(y)*1.1], title=y_label),
            updatemenus=[dict(
                type="buttons",
                buttons=[dict(label="Play",
                              method="animate",
                              args=[None, {"frame": {"duration": 20, "redraw": True},
                                           "fromcurrent": True, "transition": {"duration": 0}}])]
            )]
        ),
        frames=[go.Frame(data=[go.Scatter(x=t[:k+1], y=y[:k+1], mode='lines', name=y_label)],
                         name=str(k)) for k in range(len(t))]
    )

    fig.update_layout(showlegend=False)
    return fig

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
t, x, v, a = calculate_response_cds(K, M, zeta, x0, v0, tf, dt, force_type, force_param, omega_force)

# Create Animated Plots
st.header("Animated Plots")

# Displacement Animation
st.subheader("Displacement vs Time")
fig_displacement = create_animation(t, x, "Displacement (m)", "Displacement vs Time")
st.plotly_chart(fig_displacement, use_container_width=True)

# Velocity Animation
st.subheader("Velocity vs Time")
fig_velocity = create_animation(t, v, "Velocity (m/s)", "Velocity vs Time")
st.plotly_chart(fig_velocity, use_container_width=True)

# Acceleration Animation
st.subheader("Acceleration vs Time")
fig_acceleration = create_animation(t, a, "Acceleration (m/s²)", "Acceleration vs Time")
st.plotly_chart(fig_acceleration, use_container_width=True)

# Plotting Displacement, Velocity, and Acceleration vs Time (Static Plots)
st.header("Static Plots")

# Displacement vs Time
st.subheader("Displacement vs Time")
fig1 = go.Figure()
fig1.add_trace(go.Scatter(x=t, y=x, mode='lines', name='Displacement'))
fig1.update_layout(title='Displacement vs Time', xaxis_title='Time (s)', yaxis_title='Displacement (m)', template='plotly_dark')
st.plotly_chart(fig1, use_container_width=True)

# Velocity vs Time
st.subheader("Velocity vs Time")
fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=t, y=v, mode='lines', name='Velocity', line=dict(color='orange')))
fig2.update_layout(title='Velocity vs Time', xaxis_title='Time (s)', yaxis_title='Velocity (m/s)', template='plotly_dark')
st.plotly_chart(fig2, use_container_width=True)

# Acceleration vs Time
st.subheader("Acceleration vs Time")
fig3 = go.Figure()
fig3.add_trace(go.Scatter(x=t, y=a, mode='lines', name='Acceleration', line=dict(color='green')))
fig3.update_layout(title='Acceleration vs Time', xaxis_title='Time (s)', yaxis_title='Acceleration (m/s²)', template='plotly_dark')
st.plotly_chart(fig3, use_container_width=True)

# Phase Space Plot
st.header("Phase Space Plot")
fig_phase = go.Figure()
fig_phase.add_trace(go.Scatter(x=x, y=v, mode='lines', name='Phase Space Trajectory', line=dict(color='purple')))
fig_phase.update_layout(title='Phase Space Plot', xaxis_title='Displacement (m)', yaxis_title='Velocity (m/s)', template='plotly_dark')
st.plotly_chart(fig_phase, use_container_width=True)

# Energy Plot
st.header("Energy of the Oscillator")

kinetic = 0.5 * M * v**2
potential = 0.5 * K * x**2
total_energy = kinetic + potential

fig_energy = go.Figure()
fig_energy.add_trace(go.Scatter(x=t, y=kinetic, mode='lines', name='Kinetic Energy', line=dict(color='red')))
fig_energy.add_trace(go.Scatter(x=t, y=potential, mode='lines', name='Potential Energy', line=dict(color='orange')))
fig_energy.add_trace(go.Scatter(x=t, y=total_energy, mode='lines', name='Total Energy', line=dict(color='purple')))
fig_energy.update_layout(title='Energy of the Oscillator', xaxis_title='Time (s)', yaxis_title='Energy (J)', template='plotly_dark')
st.plotly_chart(fig_energy, use_container_width=True)

# Optional: Download Data
st.header("Download Simulation Data")
import pandas as pd

df = pd.DataFrame({
    'Time (s)': t,
    'Displacement (m)': x,
    'Velocity (m/s)': v,
    'Acceleration (m/s²)': a,
    'Kinetic Energy (J)': kinetic,
    'Potential Energy (J)': potential,
    'Total Energy (J)': total_energy
})

st.download_button(
    label="Download Data as CSV",
    data=df.to_csv(index=False).encode('utf-8'),
    file_name='oscillator_data.csv',
    mime='text/csv',
)
