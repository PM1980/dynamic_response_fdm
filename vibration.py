import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import io
import base64

# Set page config for dark theme
st.set_page_config(page_title="Resposta Dinâmica SDOF com Diferenças Centrais", layout="wide", initial_sidebar_state="expanded")

# Custom CSS for dark theme
st.markdown("""
<style>
    .reportview-container {
        background: #0e1117;
        color: #fafafa;
    }
    .sidebar .sidebar-content {
        background: #262730;
    }
    .Widget>label {
        color: #fafafa;
    }
    .stTextInput>div>div>input {
        color: #fafafa;
    }
</style>
""", unsafe_allow_html=True)

def calculate_response(M, K, zeta, load_type, load_params, tf, dt, x0, v0):
    C = 2 * zeta * np.sqrt(K * M)
    omega = np.sqrt(K/M)
    
    t = np.arange(0, tf + dt, dt)
    n = len(t)
    
    x = np.zeros(n)
    v = np.zeros(n)
    a = np.zeros(n)
    
    if load_type == "Zero":
        P = np.zeros(n)
    elif load_type == "Harmonica":
        Po, w = load_params
        P = Po * np.sin(w * t)
    elif load_type == "Linear":
        P0, P1 = load_params
        P = np.linspace(P0, P1, n)
    elif load_type == "Pulso":
        Po, duration = load_params
        P = np.where(t <= duration, Po, 0)
    
    x[0] = x0
    v[0] = v0
    a[0] = (P[0] - C*v[0] - K*x[0]) / M
    
    x[1] = x[0] + v[0]*dt + 0.5*a[0]*dt**2
    
    for i in range(2, n):
        x[i] = (2*(dt**2)*(P[i-1]-K*x[i-1]) + C*dt*x[i-2] + 4*M*x[i-1] - 2*M*x[i-2]) / (C*dt + 2*M)
    
    v[1:-1] = (x[2:] - x[:-2]) / (2*dt)
    a[1:-1] = (x[2:] - 2*x[1:-1] + x[:-2]) / (dt**2)
    
    return t, x, v, a, P

def create_plot(t, x, v, a, P):
    fig, axs = plt.subplots(4, 1, figsize=(12, 20))
    plt.style.use('dark_background')
    
    axs[0].plot(t, P, color='#00ff00')
    axs[0].set_ylabel("Load (N)")
    axs[0].set_title("Applied Load")
    axs[0].grid(True, color='#555555')

    axs[1].plot(t, x, color='#ff9900')
    axs[1].set_ylabel("Displacement (m)")
    axs[1].set_title("Displacement Response")
    axs[1].grid(True, color='#555555')

    axs[2].plot(t, v, color='#00ffff')
    axs[2].set_ylabel("Velocity (m/s)")
    axs[2].set_title("Velocity Response")
    axs[2].grid(True, color='#555555')

    axs[3].plot(t, a, color='#ff00ff')
    axs[3].set_ylabel("Acceleration (m/s²)")
    axs[3].set_title("Acceleration Response")
    axs[3].grid(True, color='#555555')

    axs[3].set_xlabel("Time (s)")

    plt.tight_layout()
    return fig, axs

st.title("Resposta Dinâmica SDOF com Diferenças Centrais")

# Sidebar for inputs
st.sidebar.header("System Parameters")
M = st.sidebar.number_input("Massa (kg)", value=275e3, format="%e")
K = st.sidebar.number_input("Rigidez (N/m)", value=2.287e5, format="%e")
zeta = st.sidebar.number_input("Razao amortecimento", value=0.05, format="%f")
x0 = st.sidebar.number_input("Deslocamento inicial (m)", value=0.0, format="%f")
v0 = st.sidebar.number_input("Velocidade inicial (m/s)", value=0.0, format="%f")

st.sidebar.header("Parâmetros da simulação")
tf = st.sidebar.number_input("Tempo da simulação (s)", value=200)
dt = st.sidebar.number_input("Passo de tempo (s)", value=0.1)

st.sidebar.header("Parâmetros da carga")
load_type = st.sidebar.selectbox("Load Type", ["Zero", "Harmonica", "Linear", "Pulso"])

if load_type == "Zero":
    load_params = None
elif load_type == "Harmonica":
    Po = st.sidebar.number_input("Load Amplitude (N)", value=1e5)
    w = st.sidebar.number_input("Load Frequency (rad/s)", value=0.1)
    load_params = (Po, w)
elif load_type == "Linear":
    P0 = st.sidebar.number_input("Initial Load (N)", value=0)
    P1 = st.sidebar.number_input("Final Load (N)", value=1e5)
    load_params = (P0, P1)
elif load_type == "Pulso":
    Po = st.sidebar.number_input("Pulse Amplitude (N)", value=1e5)
    duration = st.sidebar.number_input("Pulse Duration (s)", value=10.0)
    load_params = (Po, duration)

# Calculate response
t, x, v, a, P = calculate_response(M, K, zeta, load_type, load_params, tf, dt, x0, v0)

# Create plot
fig, axs = create_plot(t, x, v, a, P)

# Display the plot using st.pyplot
st.pyplot(fig)

# Function to handle mouse click events
def on_click(event):
    if event.xdata is not None and event.ydata is not None:
        st.session_state.clicked_x = event.xdata
        st.session_state.clicked_y = event.ydata
        st.experimental_rerun()

# Connect the click event to the plot
fig.canvas.mpl_connect('button_press_event', on_click)

# Display clicked values
if 'clicked_x' in st.session_state and 'clicked_y' in st.session_state:
    st.write(f"Clicked point: Time = {st.session_state.clicked_x:.2f} s, Value = {st.session_state.clicked_y:.2e}")

# Add download button for the plot
buf = io.BytesIO()
fig.savefig(buf, format="png")
btn = st.download_button(
    label="Download Plot",
    data=buf.getvalue(),
    file_name="sdof_response_plot.png",
    mime="image/png"
)
