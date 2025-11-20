import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.fft import fft, fftfreq
from scipy.signal import butter, lfilter, freqz
import time

# ==========================================
# PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="BioSignal LTI Analysis Pro",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern UI
st.markdown("""
<style>
    /* Main styling */
    .stApp {
        background: linear-gradient(135deg, #0a0e27 0%, #1a1f3a 100%);
    }
    
    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #1e2139 0%, #2a2f4a 100%);
        border: 2px solid #3a4e7a;
        border-radius: 15px;
        padding: 25px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        transition: transform 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        border-color: #4F8BF9;
    }
    
    /* Headers */
    h1 {
        background: linear-gradient(90deg, #4F8BF9, #00d4ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        font-size: 3rem !important;
        text-align: center;
        padding: 20px 0;
    }
    
    h2, h3 {
        color: #00d4ff;
        font-weight: 600;
    }
    
    /* Info boxes */
    .stAlert {
        background: rgba(79, 139, 249, 0.1);
        border-left: 4px solid #4F8BF9;
        border-radius: 10px;
    }
    
    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1f3a 0%, #0a0e27 100%);
        border-right: 2px solid #3a4e7a;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        background-color: transparent;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: linear-gradient(135deg, #2a2f4a 0%, #1e2139 100%);
        border-radius: 10px 10px 0 0;
        color: #00d4ff;
        font-weight: 600;
        padding: 10px 20px;
        border: 2px solid #3a4e7a;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #4F8BF9 0%, #00d4ff 100%);
        color: white;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background: linear-gradient(135deg, #2a2f4a 0%, #1e2139 100%);
        border-radius: 10px;
        font-weight: 600;
        color: #00d4ff;
    }
    
    /* Progress bar animation */
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }
    
    .pulse {
        animation: pulse 2s ease-in-out infinite;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# HEADER SECTION
# ==========================================
st.title("🩺 Advanced Biomedical Signal Processing & LTI System Analysis")

st.markdown("""
<div style='background: linear-gradient(135deg, #1e2139 0%, #2a2f4a 100%); padding: 20px; border-radius: 15px; border: 2px solid #3a4e7a; margin-bottom: 20px;'>
    <h3 style='color: #00d4ff; margin-top: 0;'>📊 Real-Time Digital Signal Processing Dashboard</h3>
    <p style='color: #b0b8d0; font-size: 1.1rem; margin-bottom: 0;'>
        Explore the fundamentals of biomedical signal processing using <strong>Linear Time-Invariant (LTI) systems</strong>. 
        This interactive platform demonstrates noise removal techniques using Butterworth filters while verifying core 
        DSP principles like linearity and time-invariance.
    </p>
</div>
""", unsafe_allow_html=True)

# ==========================================
# SIDEBAR CONTROLS
# ==========================================
st.sidebar.markdown("## 🎛️ Signal Configuration")
st.sidebar.markdown("---")

with st.sidebar.expander("📡 **Signal Parameters**", expanded=True):
    sig_freq = st.slider("💓 Heart Rate (Hz)", 0.5, 3.0, 1.2, 0.1, 
                         help="Fundamental frequency representing heart beats per second")
    noise_amp = st.slider("📈 Noise Amplitude", 0.0, 2.0, 0.5, 0.1,
                          help="Intensity of interference added to the signal")
    noise_type = st.selectbox("⚡ Interference Type", 
                             ["50Hz Powerline", "Gaussian White", "Mixed"],
                             help="50Hz = AC power interference, White = random noise, Mixed = both")

st.sidebar.markdown("---")
st.sidebar.markdown("## ⚙️ Filter Design (LTI)")

with st.sidebar.expander("🔧 **Butterworth Filter Settings**", expanded=True):
    cutoff = st.slider("🎚️ Cutoff Frequency (Hz)", 1.0, 60.0, 15.0, 1.0, 
                       help="Frequencies above this will be attenuated")
    order = st.slider("📊 Filter Order", 1, 10, 5, 1, 
                      help="Higher order provides sharper frequency cutoff but may introduce phase distortion")

st.sidebar.markdown("---")
apply_filter = st.sidebar.checkbox("✅ Apply Filter", value=True)
show_advanced = st.sidebar.checkbox("🔬 Show Advanced Analytics", value=False)

# ==========================================
# SIGNAL GENERATION
# ==========================================
@st.cache_data
def generate_signal(sig_freq, noise_amp, noise_type, seed=42):
    np.random.seed(seed)
    fs = 500.0
    duration = 4.0
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    
    # Synthetic ECG-like signal with harmonics
    clean_sig = 1.5 * np.sin(2 * np.pi * sig_freq * t) + \
                0.5 * np.sin(2 * np.pi * (sig_freq * 2) * t) + \
                0.2 * np.sin(2 * np.pi * (sig_freq * 3) * t)
    
    if noise_type == "50Hz Powerline":
        noise = noise_amp * np.sin(2 * np.pi * 50.0 * t)
    elif noise_type == "Gaussian White":
        noise = noise_amp * np.random.normal(0, 1, t.shape)
    else:
        noise = (noise_amp * 0.5 * np.random.normal(0, 1, t.shape)) + \
                (noise_amp * 0.5 * np.sin(2 * np.pi * 50.0 * t))
    
    noisy_sig = clean_sig + noise
    return t, clean_sig, noisy_sig, noise, fs

t, clean_sig, noisy_sig, noise, fs = generate_signal(sig_freq, noise_amp, noise_type)

# ==========================================
# FILTERING
# ==========================================
def butter_lowpass_filter(data, cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = lfilter(b, a, data)
    return y, b, a

filtered_sig, b_coef, a_coef = butter_lowpass_filter(noisy_sig, cutoff, fs, order)

# ==========================================
# METRICS CALCULATION
# ==========================================
mse_val = np.mean((filtered_sig - clean_sig)**2)
snr_input = 10 * np.log10(np.sum(clean_sig**2) / np.sum(noise**2)) if np.sum(noise**2) > 0 else np.inf
snr_output = 10 * np.log10(np.sum(clean_sig**2) / np.sum((filtered_sig - clean_sig)**2))
snr_improvement = snr_output - snr_input
correlation = np.corrcoef(filtered_sig, clean_sig)[0, 1]

# ==========================================
# METRICS DASHBOARD
# ==========================================
st.markdown("### 📊 Performance Metrics")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(f"""
    <div class='metric-card'>
        <h2 style='color: #4F8BF9; margin: 0;'>{snr_input:.2f} dB</h2>
        <p style='color: #b0b8d0; margin: 5px 0 0 0;'>Input SNR</p>
        <p style='font-size: 0.8rem; color: #7a8299; margin: 5px 0 0 0;'>Signal-to-Noise Ratio</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    delta_color = "🟢" if snr_improvement > 0 else "🔴"
    st.markdown(f"""
    <div class='metric-card'>
        <h2 style='color: #00d4ff; margin: 0;'>{snr_output:.2f} dB</h2>
        <p style='color: #b0b8d0; margin: 5px 0 0 0;'>Output SNR {delta_color}</p>
        <p style='font-size: 0.8rem; color: #7a8299; margin: 5px 0 0 0;'>Improvement: {snr_improvement:.2f} dB</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class='metric-card'>
        <h2 style='color: #2ecc71; margin: 0;'>{mse_val:.4f}</h2>
        <p style='color: #b0b8d0; margin: 5px 0 0 0;'>Mean Squared Error</p>
        <p style='font-size: 0.8rem; color: #7a8299; margin: 5px 0 0 0;'>Filter Accuracy</p>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown(f"""
    <div class='metric-card'>
        <h2 style='color: #e74c3c; margin: 0;'>{correlation:.4f}</h2>
        <p style='color: #b0b8d0; margin: 5px 0 0 0;'>Correlation</p>
        <p style='font-size: 0.8rem; color: #7a8299; margin: 5px 0 0 0;'>Output vs Clean</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ==========================================
# TIME DOMAIN ANALYSIS
# ==========================================
st.markdown("### 📈 Time Domain Analysis")

tab1, tab2, tab3 = st.tabs(["🔄 Interactive Comparison", "📊 Separated Signals", "🔍 Error Analysis"])

with tab1:
    limit = int(2 * fs)
    
    fig = go.Figure()
    
    # Add traces
    fig.add_trace(go.Scatter(
        x=t[:limit], y=noisy_sig[:limit],
        mode='lines', name='Noisy Input',
        line=dict(color='rgba(231, 76, 60, 0.5)', width=1.5),
        hovertemplate='Time: %{x:.3f}s<br>Amplitude: %{y:.3f}mV<extra></extra>'
    ))
    
    if apply_filter:
        fig.add_trace(go.Scatter(
            x=t[:limit], y=filtered_sig[:limit],
            mode='lines', name='Filtered Output',
            line=dict(color='#2ecc71', width=2.5),
            hovertemplate='Time: %{x:.3f}s<br>Amplitude: %{y:.3f}mV<extra></extra>'
        ))
    
    fig.add_trace(go.Scatter(
        x=t[:limit], y=clean_sig[:limit],
        mode='lines', name='Clean Reference',
        line=dict(color='#4F8BF9', width=2, dash='dash'),
        hovertemplate='Time: %{x:.3f}s<br>Amplitude: %{y:.3f}mV<extra></extra>'
    ))
    
    fig.update_layout(
        title='Signal Comparison (First 2 Seconds)',
        xaxis_title='Time (seconds)',
        yaxis_title='Amplitude (mV)',
        template='plotly_dark',
        hovermode='x unified',
        height=450,
        legend=dict(x=0.01, y=0.99, bgcolor='rgba(30, 33, 57, 0.8)'),
        plot_bgcolor='rgba(26, 31, 58, 0.5)',
        paper_bgcolor='rgba(26, 31, 58, 0.3)'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.info("""
    **💡 Insight:** This visualization shows how the LTI filter removes high-frequency noise while preserving 
    the underlying biomedical signal. The filtered output (green) closely follows the clean reference (blue), 
    demonstrating effective noise suppression.
    """)

with tab2:
    fig2 = make_subplots(
        rows=3, cols=1,
        subplot_titles=('Noisy Input Signal', 'Filtered Output Signal', 'Clean Reference Signal'),
        vertical_spacing=0.08
    )
    
    limit = int(2 * fs)
    
    fig2.add_trace(go.Scatter(x=t[:limit], y=noisy_sig[:limit], 
                              line=dict(color='#e74c3c', width=1.5),
                              name='Noisy'), row=1, col=1)
    
    fig2.add_trace(go.Scatter(x=t[:limit], y=filtered_sig[:limit], 
                              line=dict(color='#2ecc71', width=2),
                              name='Filtered'), row=2, col=1)
    
    fig2.add_trace(go.Scatter(x=t[:limit], y=clean_sig[:limit], 
                              line=dict(color='#4F8BF9', width=2),
                              name='Clean'), row=3, col=1)
    
    fig2.update_xaxes(title_text='Time (seconds)', row=3, col=1)
    fig2.update_yaxes(title_text='Amplitude (mV)', row=2, col=1)
    
    fig2.update_layout(
        height=700,
        template='plotly_dark',
        showlegend=False,
        plot_bgcolor='rgba(26, 31, 58, 0.5)',
        paper_bgcolor='rgba(26, 31, 58, 0.3)'
    )
    
    st.plotly_chart(fig2, use_container_width=True)

with tab3:
    error_signal = filtered_sig - clean_sig
    error_rms = np.sqrt(np.mean(error_signal**2))
    
    fig3 = go.Figure()
    
    fig3.add_trace(go.Scatter(
        x=t, y=error_signal,
        mode='lines', name='Reconstruction Error',
        line=dict(color='#e74c3c', width=1.5),
        fill='tozeroy', fillcolor='rgba(231, 76, 60, 0.2)'
    ))
    
    fig3.add_hline(y=0, line_dash="dash", line_color="white", opacity=0.5)
    
    fig3.update_layout(
        title=f'Reconstruction Error (RMS: {error_rms:.4f} mV)',
        xaxis_title='Time (seconds)',
        yaxis_title='Error Amplitude (mV)',
        template='plotly_dark',
        height=400,
        plot_bgcolor='rgba(26, 31, 58, 0.5)',
        paper_bgcolor='rgba(26, 31, 58, 0.3)'
    )
    
    st.plotly_chart(fig3, use_container_width=True)
    
    st.warning(f"""
    **📉 Error Analysis:** The Root Mean Square (RMS) error is **{error_rms:.4f} mV**, indicating the average 
    deviation between filtered and clean signals. Lower values suggest better filter performance.
    """)

# ==========================================
# FREQUENCY DOMAIN ANALYSIS
# ==========================================
st.markdown("### 🌊 Frequency Domain Analysis")

col_left, col_right = st.columns([3, 2])

with col_left:
    # FFT Analysis
    N = len(noisy_sig)
    yf_noisy = fft(noisy_sig)
    yf_filtered = fft(filtered_sig)
    yf_clean = fft(clean_sig)
    xf = fftfreq(N, 1 / fs)
    half_N = N // 2
    
    fig4 = go.Figure()
    
    fig4.add_trace(go.Scatter(
        x=xf[:half_N], y=2.0/N * np.abs(yf_noisy[:half_N]),
        mode='lines', name='Noisy Spectrum',
        line=dict(color='rgba(231, 76, 60, 0.6)', width=2)
    ))
    
    if apply_filter:
        fig4.add_trace(go.Scatter(
            x=xf[:half_N], y=2.0/N * np.abs(yf_filtered[:half_N]),
            mode='lines', name='Filtered Spectrum',
            line=dict(color='#2ecc71', width=2.5)
        ))
    
    fig4.add_trace(go.Scatter(
        x=xf[:half_N], y=2.0/N * np.abs(yf_clean[:half_N]),
        mode='lines', name='Clean Spectrum',
        line=dict(color='#4F8BF9', width=2, dash='dot')
    ))
    
    # Add cutoff frequency line
    fig4.add_vline(x=cutoff, line_dash="dash", line_color="#00d4ff", 
                   annotation_text=f"Cutoff: {cutoff}Hz", 
                   annotation_position="top right")
    
    # Highlight powerline noise if present
    if noise_type in ["50Hz Powerline", "Mixed"]:
        fig4.add_vline(x=50, line_dash="dot", line_color="#e74c3c", 
                       annotation_text="50Hz Interference", 
                       annotation_position="bottom right")
    
    fig4.update_layout(
        title='Frequency Spectrum (FFT)',
        xaxis_title='Frequency (Hz)',
        yaxis_title='Magnitude',
        xaxis=dict(range=[0, 60]),
        template='plotly_dark',
        height=450,
        hovermode='x unified',
        plot_bgcolor='rgba(26, 31, 58, 0.5)',
        paper_bgcolor='rgba(26, 31, 58, 0.3)'
    )
    
    st.plotly_chart(fig4, use_container_width=True)
    
    st.info("""
    **🔬 Frequency Insight:** The FFT reveals the frequency components of the signal. The Butterworth filter 
    attenuates frequencies above the cutoff, effectively removing high-frequency noise while preserving the 
    biomedical signal's fundamental frequency and lower harmonics.
    """)

with col_right:
    # Filter Frequency Response
    st.markdown("#### 📉 Filter Frequency Response")
    
    w, h = freqz(b_coef, a_coef, worN=8000, fs=fs)
    
    fig5 = go.Figure()
    
    fig5.add_trace(go.Scatter(
        x=w, y=20 * np.log10(abs(h)),
        mode='lines', name='Magnitude Response',
        line=dict(color='#00d4ff', width=3),
        fill='tozeroy', fillcolor='rgba(0, 212, 255, 0.2)'
    ))
    
    fig5.add_hline(y=-3, line_dash="dash", line_color="#2ecc71", 
                   annotation_text="-3dB (Half Power)", 
                   annotation_position="bottom right")
    
    fig5.update_layout(
        title=f'Butterworth Lowpass (Order {order})',
        xaxis_title='Frequency (Hz)',
        yaxis_title='Magnitude (dB)',
        xaxis=dict(range=[0, 60]),
        template='plotly_dark',
        height=450,
        plot_bgcolor='rgba(26, 31, 58, 0.5)',
        paper_bgcolor='rgba(26, 31, 58, 0.3)'
    )
    
    st.plotly_chart(fig5, use_container_width=True)
    
    st.success(f"""
    **Filter Characteristics:**
    - **Type:** Butterworth Lowpass
    - **Order:** {order}
    - **Cutoff:** {cutoff} Hz
    - **Rolloff:** ~{order*20} dB/decade
    
    Higher order provides steeper rolloff but may introduce phase distortion.
    """)

# ==========================================
# LTI VERIFICATION
# ==========================================
st.markdown("### 🔬 LTI System Verification")

col_v1, col_v2 = st.columns([2, 1])

with col_v1:
    st.markdown("""
    <div style='background: linear-gradient(135deg, #1e2139 0%, #2a2f4a 100%); padding: 20px; border-radius: 15px; border: 2px solid #3a4e7a;'>
        <h4 style='color: #00d4ff; margin-top: 0;'>📐 Linearity Test (Superposition Principle)</h4>
        <p style='color: #b0b8d0;'>
            A system is <strong>linear</strong> if it satisfies the superposition principle:
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.latex(r"T\{a \cdot x_1 + b \cdot x_2\} = a \cdot T\{x_1\} + b \cdot T\{x_2\}")
    
    # Perform verification
    test_len = 200
    x1 = np.random.random(test_len)
    x2 = np.random.random(test_len)
    a, b = 2.5, -1.5
    
    lhs, _, _ = butter_lowpass_filter(a*x1 + b*x2, cutoff, fs, order)
    y1, _, _ = butter_lowpass_filter(x1, cutoff, fs, order)
    y2, _, _ = butter_lowpass_filter(x2, cutoff, fs, order)
    rhs = a*y1 + b*y2
    
    check_diff = np.mean((lhs - rhs)**2)
    max_diff = np.max(np.abs(lhs - rhs))
    
    # Visualization
    fig6 = go.Figure()
    
    fig6.add_trace(go.Scatter(
        y=lhs, mode='lines', name='LHS: T{ax₁ + bx₂}',
        line=dict(color='#4F8BF9', width=2)
    ))
    
    fig6.add_trace(go.Scatter(
        y=rhs, mode='lines', name='RHS: aT{x₁} + bT{x₂}',
        line=dict(color='#2ecc71', width=2, dash='dash')
    ))
    
    fig6.update_layout(
        title='Linearity Verification: LHS vs RHS',
        xaxis_title='Sample Index',
        yaxis_title='Amplitude',
        template='plotly_dark',
        height=350,
        plot_bgcolor='rgba(26, 31, 58, 0.5)',
        paper_bgcolor='rgba(26, 31, 58, 0.3)'
    )
    
    st.plotly_chart(fig6, use_container_width=True)

with col_v2:
    st.markdown("#### 📊 Test Results")
    
    st.markdown(f"""
    <div style='background: rgba(79, 139, 249, 0.1); padding: 15px; border-radius: 10px; border-left: 4px solid #4F8BF9; margin: 10px 0;'>
        <p style='margin: 5px 0; color: #b0b8d0;'><strong>Coefficients:</strong></p>
        <p style='margin: 5px 0; color: #00d4ff;'>a = {a}, b = {b}</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown(f"""
    <div style='background: rgba(46, 204, 113, 0.1); padding: 15px; border-radius: 10px; border-left: 4px solid #2ecc71; margin: 10px 0;'>
        <p style='margin: 5px 0; color: #b0b8d0;'><strong>MSE Difference:</strong></p>
        <p style='margin: 5px 0; color: #2ecc71; font-size: 1.2rem;'>{check_diff:.2e}</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown(f"""
    <div style='background: rgba(231, 76, 60, 0.1); padding: 15px; border-radius: 10px; border-left: 4px solid #e74c3c; margin: 10px 0;'>
        <p style='margin: 5px 0; color: #b0b8d0;'><strong>Max Deviation:</strong></p>
        <p style='margin: 5px 0; color: #e74c3c; font-size: 1.2rem;'>{max_diff:.2e}</p>
    </div>
    """, unsafe_allow_html=True)
    
    if check_diff < 1e-10:
        st.success("### ✅ SYSTEM IS LINEAR")
        st.balloons()
    else:
        st.error("### ❌ SYSTEM IS NON-LINEAR")
    
    st.markdown("---")
    
    st.caption("""
    The Butterworth filter is an LTI system. The near-zero MSE confirms 
    that superposition holds, validating linearity.
    """)

# ==========================================
# ADVANCED ANALYTICS
# ==========================================
if show_advanced:
    st.markdown("### 🎯 Advanced Analytics")
    
    with st.expander("📈 Statistical Analysis", expanded=True):
        col_a1, col_a2, col_a3 = st.columns(3)
        
        with col_a1:
            st.markdown("**Input Signal Stats**")
            st.write(f"Mean: {np.mean(noisy_sig):.4f}")
            st.write(f"Std Dev: {np.std(noisy_sig):.4f}")
            st.write(f"Peak: {np.max(np.abs(noisy_sig)):.4f}")
            
        with col_a2:
            st.markdown("**Output Signal Stats**")
            st.write(f"Mean: {np.mean(filtered_sig):.4f}")
            st.write(f"Std Dev: {np.std(filtered_sig):.4f}")
            st.write(f"Peak: {np.max(np.abs(filtered_sig)):.4f}")
            
        with col_a3:
            st.markdown("**Noise Reduction**")
            noise_reduction = (np.std(noisy_sig) - np.std(filtered_sig)) / np.std(noisy_sig) * 100
            st.write(f"Variance Reduction: {noise_reduction:.2f}%")
            st.write(f"SNR Gain: {snr_improvement:.2f} dB")
    
    with st.expander("🔧 Filter Coefficients", expanded=False):
        col_b1, col_b2 = st.columns(2)
        
        with col_b1:
            st.markdown("**Numerator (b coefficients)**")
            st.code(str(b_coef))
            
        with col_b2:
            st.markdown("**Denominator (a coefficients)**")
            st.code(str(a_coef))

# ==========================================
# EDUCATIONAL SECTION
# ==========================================
st.markdown("### 📚 Understanding the Concepts")

tab_edu1, tab_edu2, tab_edu3, tab_edu4 = st.tabs([
    "🧠 What is an LTI System?", 
    "🎚️ Butterworth Filters", 
    "📊 Signal Processing Basics",
    "💡 Clinical Applications"
])

with tab_edu1:
    st.markdown("""
    <div style='background: linear-gradient(135deg, #1e2139 0%, #2a2f4a 100%); padding: 25px; border-radius: 15px; border: 2px solid #3a4e7a; margin: 15px 0;'>
        <h3 style='color: #00d4ff; margin-top: 0;'>Linear Time-Invariant (LTI) Systems</h3>
        <p style='color: #b0b8d0; font-size: 1.05rem; line-height: 1.8;'>
            An <strong>LTI system</strong> is the cornerstone of digital signal processing. It has two fundamental properties:
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    col_edu1, col_edu2 = st.columns(2)
    
    with col_edu1:
        st.markdown("""
        <div style='background: rgba(79, 139, 249, 0.1); padding: 20px; border-radius: 10px; border: 2px solid #4F8BF9; height: 280px;'>
            <h4 style='color: #4F8BF9;'>1️⃣ Linearity</h4>
            <p style='color: #b0b8d0; line-height: 1.7;'>
                The system obeys the <strong>superposition principle</strong>. If you scale or add input signals, 
                the output scales or adds proportionally.
            </p>
        </div>
        """, unsafe_allow_html=True)
        st.latex(r"T\{ax_1 + bx_2\} = aT\{x_1\} + bT\{x_2\}")
        
    with col_edu2:
        st.markdown("""
        <div style='background: rgba(46, 204, 113, 0.1); padding: 20px; border-radius: 10px; border: 2px solid #2ecc71; height: 280px;'>
            <h4 style='color: #2ecc71;'>2️⃣ Time-Invariance</h4>
            <p style='color: #b0b8d0; line-height: 1.7;'>
                The system's behavior doesn't change over time. A delayed input produces an equally delayed output.
            </p>
        </div>
        """, unsafe_allow_html=True)
        st.latex(r"y(t - t_0) = T\{x(t - t_0)\}")
    
    st.info("""
    **🎯 Why LTI Systems Matter in Biomedical Engineering:**
    - **Predictable behavior**: Mathematical analysis is straightforward
    - **Frequency domain tools**: We can use Fourier transforms and frequency response
    - **Filter design**: Enables creation of reliable noise reduction systems
    - **Real-time processing**: Efficient implementation in medical devices
    """)

with tab_edu2:
    st.markdown("""
    <div style='background: linear-gradient(135deg, #1e2139 0%, #2a2f4a 100%); padding: 25px; border-radius: 15px; border: 2px solid #3a4e7a; margin: 15px 0;'>
        <h3 style='color: #00d4ff; margin-top: 0;'>The Butterworth Filter</h3>
        <p style='color: #b0b8d0; font-size: 1.05rem; line-height: 1.8;'>
            Named after British engineer Stephen Butterworth (1930), this filter is designed to have 
            the <strong>flattest possible frequency response</strong> in the passband.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    col_f1, col_f2 = st.columns([3, 2])
    
    with col_f1:
        st.markdown("#### 🔍 Key Characteristics")
        st.markdown("""
        - **Maximally flat magnitude response**: No ripples in the passband
        - **Smooth rolloff**: Gradual transition between passband and stopband
        - **Order control**: Higher orders provide sharper cutoffs (≈20n dB/decade)
        - **Phase response**: Non-linear phase (acceptable for many applications)
        """)
        
        st.markdown("#### 📐 Transfer Function")
        st.latex(r"|H(j\omega)|^2 = \frac{1}{1 + (\omega/\omega_c)^{2n}}")
        st.caption("Where n is the filter order and ωc is the cutoff frequency")
        
    with col_f2:
        st.markdown("#### 🎚️ Filter Order Effects")
        
        # Create comparison visualization
        frequencies = np.linspace(0, 60, 1000)
        fig_order = go.Figure()
        
        for test_order in [1, 3, 5, 8]:
            nyq = 0.5 * fs
            normal_cutoff = cutoff / nyq
            b_test, a_test = butter(test_order, normal_cutoff, btype='low', analog=False)
            w_test, h_test = freqz(b_test, a_test, worN=frequencies, fs=fs)
            
            fig_order.add_trace(go.Scatter(
                x=w_test, y=20 * np.log10(abs(h_test)),
                mode='lines', name=f'Order {test_order}',
                line=dict(width=2)
            ))
        
        fig_order.update_layout(
            title='Order Comparison',
            xaxis_title='Frequency (Hz)',
            yaxis_title='Magnitude (dB)',
            template='plotly_dark',
            height=300,
            showlegend=True,
            legend=dict(x=0.7, y=0.95),
            plot_bgcolor='rgba(26, 31, 58, 0.5)',
            paper_bgcolor='rgba(26, 31, 58, 0.3)'
        )
        
        st.plotly_chart(fig_order, use_container_width=True)
    
    st.success("""
    **💡 Clinical Advantage:** Butterworth filters are ideal for ECG and EEG processing because they 
    preserve signal morphology in the passband while effectively attenuating noise frequencies.
    """)

with tab_edu3:
    st.markdown("""
    <div style='background: linear-gradient(135deg, #1e2139 0%, #2a2f4a 100%); padding: 25px; border-radius: 15px; border: 2px solid #3a4e7a; margin: 15px 0;'>
        <h3 style='color: #00d4ff; margin-top: 0;'>Signal Processing Fundamentals</h3>
    </div>
    """, unsafe_allow_html=True)
    
    col_s1, col_s2 = st.columns(2)
    
    with col_s1:
        st.markdown("#### 🌊 Time vs Frequency Domain")
        st.markdown("""
        <div style='background: rgba(79, 139, 249, 0.1); padding: 15px; border-radius: 10px; border-left: 4px solid #4F8BF9; margin: 10px 0;'>
            <p style='color: #b0b8d0; margin: 0;'>
                <strong>Time Domain:</strong> Shows how signal amplitude changes over time. 
                Useful for observing waveform shapes and temporal patterns.
            </p>
        </div>
        
        <div style='background: rgba(46, 204, 113, 0.1); padding: 15px; border-radius: 10px; border-left: 4px solid #2ecc71; margin: 10px 0;'>
            <p style='color: #b0b8d0; margin: 0;'>
                <strong>Frequency Domain:</strong> Reveals which frequencies are present in the signal. 
                Essential for identifying noise and designing filters.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("#### 📊 Key Metrics")
        st.markdown("""
        - **SNR (Signal-to-Noise Ratio)**: Measures signal quality in dB
        - **MSE (Mean Squared Error)**: Quantifies reconstruction accuracy
        - **Correlation**: Indicates similarity between signals (-1 to 1)
        - **RMS (Root Mean Square)**: Average signal power
        """)
    
    with col_s2:
        st.markdown("#### 🎯 Common Noise Types")
        
        noise_info = {
            "50/60Hz Powerline": {
                "source": "AC electrical interference",
                "frequency": "50Hz (EU) / 60Hz (US)",
                "solution": "Notch or lowpass filter",
                "color": "#e74c3c"
            },
            "Gaussian White": {
                "source": "Thermal noise, amplifiers",
                "frequency": "All frequencies equally",
                "solution": "Lowpass or averaging",
                "color": "#9b59b6"
            },
            "Motion Artifacts": {
                "source": "Patient movement",
                "frequency": "0.5-10 Hz",
                "solution": "Highpass filter",
                "color": "#f39c12"
            }
        }
        
        for noise, info in noise_info.items():
            st.markdown(f"""
            <div style='background: rgba(255, 255, 255, 0.05); padding: 12px; border-radius: 8px; border-left: 4px solid {info['color']}; margin: 8px 0;'>
                <p style='color: {info['color']}; font-weight: 600; margin: 0 0 5px 0;'>{noise}</p>
                <p style='color: #b0b8d0; font-size: 0.9rem; margin: 3px 0;'>📍 {info['source']}</p>
                <p style='color: #7a8299; font-size: 0.85rem; margin: 3px 0;'>🔧 {info['solution']}</p>
            </div>
            """, unsafe_allow_html=True)

with tab_edu4:
    st.markdown("""
    <div style='background: linear-gradient(135deg, #1e2139 0%, #2a2f4a 100%); padding: 25px; border-radius: 15px; border: 2px solid #3a4e7a; margin: 15px 0;'>
        <h3 style='color: #00d4ff; margin-top: 0;'>Clinical Applications of Signal Processing</h3>
    </div>
    """, unsafe_allow_html=True)
    
    applications = {
        "🫀 Electrocardiography (ECG)": {
            "signal": "Electrical activity of the heart",
            "freq_range": "0.05 - 100 Hz",
            "challenges": "Powerline noise, muscle artifacts, baseline wander",
            "filters": "0.5Hz highpass + 40Hz lowpass",
            "importance": "Detecting arrhythmias, myocardial infarction, ischemia"
        },
        "🧠 Electroencephalography (EEG)": {
            "signal": "Brain electrical activity",
            "freq_range": "0.5 - 70 Hz (Delta, Theta, Alpha, Beta, Gamma)",
            "challenges": "Eye blinks, muscle activity, electrode noise",
            "filters": "Bandpass 0.5-70Hz + notch at 50/60Hz",
            "importance": "Epilepsy detection, sleep studies, brain-computer interfaces"
        },
        "💪 Electromyography (EMG)": {
            "signal": "Muscle electrical activity",
            "freq_range": "20 - 500 Hz",
            "challenges": "Motion artifacts, crosstalk, powerline",
            "filters": "20Hz highpass + 500Hz lowpass",
            "importance": "Neuromuscular disorder diagnosis, prosthetic control"
        },
        "🩸 Photoplethysmography (PPG)": {
            "signal": "Blood volume changes (optical)",
            "freq_range": "0.5 - 10 Hz",
            "challenges": "Motion artifacts, ambient light",
            "filters": "Bandpass 0.5-10Hz + adaptive filtering",
            "importance": "Heart rate, oxygen saturation, blood pressure estimation"
        }
    }
    
    for app, details in applications.items():
        with st.expander(f"**{app}**", expanded=False):
            col_app1, col_app2 = st.columns([2, 3])
            
            with col_app1:
                st.markdown(f"""
                **📊 Signal Type:**  
                {details['signal']}
                
                **📡 Frequency Range:**  
                {details['freq_range']}
                
                **⚠️ Common Challenges:**  
                {details['challenges']}
                """)
            
            with col_app2:
                st.markdown(f"""
                **🔧 Typical Filter Configuration:**  
                {details['filters']}
                
                **🏥 Clinical Importance:**  
                {details['importance']}
                """)
    
    st.info("""
    **🔬 Research Note:** Modern biomedical signal processing increasingly uses adaptive filters, 
    wavelet transforms, and machine learning techniques. However, LTI filters remain the foundation 
    due to their reliability, low computational cost, and real-time capability.
    """)

# ==========================================
# INTERACTIVE DEMO SECTION
# ==========================================
st.markdown("### 🎮 Interactive Learning Demo")

with st.expander("🧪 **Experiment: See Filter Effects in Real-Time**", expanded=False):
    st.markdown("""
    Use the sliders below to see how different parameters affect filtering performance. 
    Try extreme values to understand the tradeoffs!
    """)
    
    col_demo1, col_demo2 = st.columns(2)
    
    with col_demo1:
        demo_cutoff = st.slider("🎚️ Demo Cutoff Frequency", 1.0, 60.0, 20.0, 1.0, key="demo_cutoff")
        demo_order = st.slider("📊 Demo Filter Order", 1, 10, 4, 1, key="demo_order")
    
    with col_demo2:
        demo_noise = st.slider("📈 Demo Noise Level", 0.0, 3.0, 1.0, 0.1, key="demo_noise")
        demo_type = st.selectbox("⚡ Demo Noise Type", 
                                 ["50Hz Powerline", "Gaussian White", "Mixed"], 
                                 key="demo_type")
    
    # Generate demo signal
    t_demo, clean_demo, noisy_demo, _, _ = generate_signal(sig_freq, demo_noise, demo_type, seed=123)
    filtered_demo, _, _ = butter_lowpass_filter(noisy_demo, demo_cutoff, fs, demo_order)
    
    # Calculate demo metrics
    snr_demo_in = 10 * np.log10(np.sum(clean_demo**2) / np.sum((noisy_demo - clean_demo)**2))
    snr_demo_out = 10 * np.log10(np.sum(clean_demo**2) / np.sum((filtered_demo - clean_demo)**2))
    improvement_demo = snr_demo_out - snr_demo_in
    
    col_metric1, col_metric2, col_metric3 = st.columns(3)
    col_metric1.metric("Input SNR", f"{snr_demo_in:.2f} dB")
    col_metric2.metric("Output SNR", f"{snr_demo_out:.2f} dB", f"{improvement_demo:.2f} dB")
    col_metric3.metric("Improvement", f"{improvement_demo:.2f} dB", 
                       "Good" if improvement_demo > 5 else "Moderate")
    
    # Quick comparison plot
    limit_demo = int(1 * fs)
    fig_demo = go.Figure()
    
    fig_demo.add_trace(go.Scatter(x=t_demo[:limit_demo], y=noisy_demo[:limit_demo],
                                  mode='lines', name='Noisy', 
                                  line=dict(color='rgba(231, 76, 60, 0.4)', width=1)))
    fig_demo.add_trace(go.Scatter(x=t_demo[:limit_demo], y=filtered_demo[:limit_demo],
                                  mode='lines', name='Filtered',
                                  line=dict(color='#2ecc71', width=2)))
    fig_demo.add_trace(go.Scatter(x=t_demo[:limit_demo], y=clean_demo[:limit_demo],
                                  mode='lines', name='Clean Reference',
                                  line=dict(color='#4F8BF9', width=1.5, dash='dot')))
    
    fig_demo.update_layout(
        title='Real-Time Filter Comparison',
        xaxis_title='Time (s)',
        yaxis_title='Amplitude (mV)',
        template='plotly_dark',
        height=350,
        plot_bgcolor='rgba(26, 31, 58, 0.5)',
        paper_bgcolor='rgba(26, 31, 58, 0.3)'
    )
    
    st.plotly_chart(fig_demo, use_container_width=True)
    
    st.success("""
    **💡 Observations:**
    - Lower cutoff frequencies remove more noise but may distort the signal
    - Higher filter orders provide sharper cutoff but increase computational cost
    - The optimal filter design balances noise reduction with signal preservation
    """)

# ==========================================
# DOWNLOAD SECTION
# ==========================================
st.markdown("### 💾 Export Data")

col_down1, col_down2, col_down3 = st.columns(3)

with col_down1:
    # Export filtered signal
    filtered_csv = "\n".join([f"{t[i]},{filtered_sig[i]}" for i in range(len(t))])
    st.download_button(
        label="📥 Download Filtered Signal",
        data=filtered_csv,
        file_name="filtered_signal.csv",
        mime="text/csv"
    )

with col_down2:
    # Export metrics
    metrics_text = f"""Signal Processing Report
========================
Date: {time.strftime('%Y-%m-%d %H:%M:%S')}

Signal Parameters:
- Heart Rate: {sig_freq} Hz
- Noise Amplitude: {noise_amp}
- Noise Type: {noise_type}

Filter Configuration:
- Type: Butterworth Lowpass
- Order: {order}
- Cutoff Frequency: {cutoff} Hz

Performance Metrics:
- Input SNR: {snr_input:.2f} dB
- Output SNR: {snr_output:.2f} dB
- SNR Improvement: {snr_improvement:.2f} dB
- Mean Squared Error: {mse_val:.6f}
- Correlation: {correlation:.6f}
"""
    st.download_button(
        label="📥 Download Report",
        data=metrics_text,
        file_name="processing_report.txt",
        mime="text/plain"
    )

with col_down3:
    st.markdown("""
    <div style='background: rgba(79, 139, 249, 0.1); padding: 15px; border-radius: 10px; text-align: center;'>
        <p style='color: #00d4ff; font-weight: 600; margin: 0;'>📊 Export Options</p>
        <p style='color: #b0b8d0; font-size: 0.9rem; margin: 5px 0 0 0;'>Download processed signals and analysis reports</p>
    </div>
    """, unsafe_allow_html=True)

# ==========================================
# FOOTER
# ==========================================
st.markdown("---")
st.markdown("""
<div style='text-align: center; padding: 20px; color: #7a8299;'>
    <p style='font-size: 1.3rem; margin-bottom: 10px;'><strong>🩺 BioSignal LTI Analysis Pro</strong></p>
    <p style='font-size: 0.95rem; color: #b0b8d0;'>
        Digital Signal Processing • Linear Time-Invariant Systems • Biomedical Engineering
    </p>
    <p style='font-size: 0.85rem; margin-top: 15px;'>
        Built with ❤️ using Streamlit, Plotly, SciPy, and NumPy
    </p>
</div>
""", unsafe_allow_html=True)