import streamlit as st
import plotly.graph_objects as go

# Page configuration
st.set_page_config(
    page_title="QuantumLearn - Interactive Quantum Computing Education",
    page_icon="⚛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main {
        padding-top: 2rem;
    }
    .stTitle {
        color: #1f77b4;
        font-size: 2.5rem !important;
        text-align: center;
        margin-bottom: 2rem;
    }
    .quantum-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .feature-box {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 8px;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
</style>
""", unsafe_allow_html=True)

# Sidebar navigation
with st.sidebar:
    st.markdown("## 🎯 Navigation")
    st.markdown("---")
    
    # Learning modules
    st.markdown("### 📚 Learning Modules")
    if st.button("🏠 Home", use_container_width=True):
        st.switch_page("app.py")
    
    if st.button("🌟 Quantum Basics", use_container_width=True):
        st.switch_page("pages/01_Quantum_Basics.py")
    
    if st.button("🔄 Superposition", use_container_width=True):
        st.switch_page("pages/02_Superposition.py")
    
    if st.button("🔗 Entanglement", use_container_width=True):
        st.switch_page("pages/03_Entanglement.py")
    
    if st.button("🔍 Grover's Algorithm", use_container_width=True):
        st.switch_page("pages/04_grover.py")
    
    if st.button("🎲 Quantum Gates", use_container_width=True):
        st.switch_page("pages/05_Quantum_Gates.py")
    
    if st.button("🔬 Circuit Builder", use_container_width=True):
        st.switch_page("pages/06_Circuit_Builder.py")
    
    if st.button("💻 Code Playground", use_container_width=True):
        st.switch_page("pages/07_Code_Playground.py")
    
    st.markdown("---")
    st.markdown("### 🎮 Interactive Tools")
    if st.button("⚗️ Quantum Simulator", use_container_width=True):
        st.switch_page("pages/08_Quantum_Simulator.py")

# Main content
st.markdown('<h1 class="stTitle">⚛️ QuantumLearn</h1>', unsafe_allow_html=True)
st.markdown('<div class="quantum-card"><h2>🚀 Interactive Quantum Computing Education</h2><p>Master quantum computing through hands-on learning with interactive circuits, visual simulations, and real code examples.</p></div>', unsafe_allow_html=True)

# Feature highlights
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="feature-box">
        <h3>🔧 Interactive Circuits</h3>
        <p>Build and visualize quantum circuits with drag-and-drop interface. See how gates affect qubit states in real-time.</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="feature-box">
        <h3>💻 Live Code Editor</h3>
        <p>Write and execute quantum algorithms with syntax highlighting and instant feedback. Learn by doing!</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="feature-box">
        <h3>📊 Visual Learning</h3>
        <p>Understand complex concepts through interactive visualizations, animations, and step-by-step explanations.</p>
    </div>
    """, unsafe_allow_html=True)

# Quick start section
st.markdown("## 🎯 Quick Start Guide")

quick_col1, quick_col2 = st.columns(2)

with quick_col1:
    st.markdown("""
    ### 🌱 For Complete Beginners
    1. **Start with Quantum Basics** - Learn what qubits are
    2. **Explore Superposition** - Understand quantum states
    3. **Discover Entanglement** - See quantum correlations
    4. **Try the Circuit Builder** - Create your first circuits
    """)

with quick_col2:
    st.markdown("""
    ### 🚀 For Those with Some Knowledge
    1. **Jump to Quantum Gates** - Review gate operations
    2. **Try Grover's Algorithm** - Implement search algorithms
    3. **Use the Code Playground** - Write custom quantum programs
    4. **Experiment with the Simulator** - Test complex circuits
    """)

# Progress tracking
st.markdown("## 📈 Your Learning Journey")
progress_col1, progress_col2 = st.columns([3, 1])

with progress_col1:
    # Simulated progress (in a real app, this would be stored)
    modules = ["Quantum Basics", "Superposition", "Entanglement", "Quantum Gates", "Grover's Algorithm"]
    progress = [100, 80, 60, 30, 0]  # Example progress
    
    fig = go.Figure(data=go.Bar(
        x=modules,
        y=progress,
        marker_color=['#1f77b4' if p == 100 else '#ff7f0e' if p > 0 else '#d62728' for p in progress]
    ))
    fig.update_layout(
        title="Module Completion Status",
        yaxis_title="Progress (%)",
        xaxis_title="Learning Modules",
        height=300
    )
    st.plotly_chart(fig, use_container_width=True)

with progress_col2:
    st.metric("Modules Completed", "2/5", "+1")
    st.metric("Circuits Built", "12", "+3")
    st.metric("Algorithms Learned", "1", "+1")

# Call to action
st.markdown("---")
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    if st.button("🚀 Start Your Quantum Journey!", size="large", use_container_width=True):
        st.switch_page("pages/01_Quantum_Basics.py")
