import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from qiskit import QuantumCircuit, execute, Aer
from qiskit.visualization import plot_bloch_vector
import plotly.express as px

st.set_page_config(page_title="Quantum Basics", page_icon="🌟", layout="wide")

# Custom CSS
st.markdown("""
<style>
    .concept-box {
        background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 5px solid #2196f3;
    }
    .example-box {
        background: #f1f8e9;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 5px solid #4caf50;
    }
    .warning-box {
        background: #fff3e0;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 5px solid #ff9800;
    }
</style>
""", unsafe_allow_html=True)

st.title("🌟 Quantum Computing Basics")
st.markdown("*Your first step into the quantum world*")

# Navigation tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📚 What is Quantum?", "🔹 Qubits", "🌊 Superposition", "🎯 Measurement", "🧮 Practice"])

with tab1:
    st.header("What is Quantum Computing?")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        <div class="concept-box">
        <h3>🎯 The Big Idea</h3>
        <p>Quantum computing uses the strange rules of quantum mechanics to solve problems that would take classical computers forever!</p>
        
        <h4>Classical vs Quantum Bits:</h4>
        <ul>
        <li><strong>Classical Bit:</strong> Can be 0 OR 1</li>
        <li><strong>Quantum Bit (Qubit):</strong> Can be 0 AND 1 at the same time!</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="example-box">
        <h3>🌟 Real-World Analogy: The Spinning Coin</h3>
        <p>Imagine a coin spinning in the air:</p>
        <ul>
        <li><strong>Classical bit:</strong> Coin on the table (definitely heads OR tails)</li>
        <li><strong>Quantum bit:</strong> Spinning coin (both heads AND tails until it lands)</li>
        </ul>
        <p>The magic happens while it's spinning!</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        # Interactive visualization of classical vs quantum
        fig = go.Figure()
        
        # Classical bit visualization
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[1, 0],
            mode='markers+text',
            marker=dict(size=20, color='red'),
            text=['Classical 0', 'Classical 1'],
            textposition="top center",
            name="Classical Bits"
        ))
        
        # Quantum superposition visualization
        theta = np.linspace(0, 2*np.pi, 100)
        x_quantum = 0.5 + 0.3 * np.cos(theta)
        y_quantum = 0.5 + 0.3 * np.sin(theta)
        
        fig.add_trace(go.Scatter(
            x=x_quantum, y=y_quantum,
            mode='lines',
            line=dict(color='blue', width=3),
            name="Quantum Superposition"
        ))
        
        fig.update_layout(
            title="Classical vs Quantum States",
            xaxis_title="State Space",
            yaxis_title="Probability",
            height=300,
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.header("🔹 Understanding Qubits")
    
    st.markdown("""
    <div class="concept-box">
    <h3>What is a Qubit?</h3>
    <p>A qubit (quantum bit) is the basic unit of quantum information. Unlike classical bits, qubits can exist in a <strong>superposition</strong> of states.</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎮 Interactive Qubit Explorer")
        
        # Qubit state controls
        alpha = st.slider("Real part (α)", -1.0, 1.0, 0.707, 0.01)
        beta = st.slider("Imaginary part (β)", -1.0, 1.0, 0.707, 0.01)
        
        # Normalize the state
        norm = np.sqrt(alpha**2 + beta**2)
        if norm > 0:
            alpha_norm = alpha / norm
            beta_norm = beta / norm
        else:
            alpha_norm, beta_norm = 1, 0
        
        st.markdown(f"""
        **Your Qubit State:**
        |ψ⟩ = {alpha_norm:.3f}|0⟩ + {beta_norm:.3f}|1⟩
        
        **Probabilities:**
        - P(measuring 0) = {alpha_norm**2:.3f} ({alpha_norm**2*100:.1f}%)
        - P(measuring 1) = {beta_norm**2:.3f} ({beta_norm**2*100:.1f}%)
        """)
    
    with col2:
        # Bloch sphere visualization
        st.subheader("🌐 Bloch Sphere Representation")
        
        # Calculate Bloch sphere coordinates
        theta = 2 * np.arccos(abs(alpha_norm))
        phi = np.angle(beta_norm) - np.angle(alpha_norm)
        
        x = np.sin(theta) * np.cos(phi)
        y = np.sin(theta) * np.sin(phi)
        z = np.cos(theta)
        
        # Create Bloch sphere plot
        fig = go.Figure()
        
        # Sphere surface
        u = np.linspace(0, 2 * np.pi, 50)
        v = np.linspace(0, np.pi, 50)
        x_sphere = np.outer(np.cos(u), np.sin(v))
        y_sphere = np.outer(np.sin(u), np.sin(v))
        z_sphere = np.outer(np.ones(np.size(u)), np.cos(v))
        
        fig.add_trace(go.Surface(
            x=x_sphere, y=y_sphere, z=z_sphere,
            opacity=0.3, showscale=False,
            colorscale='Blues'
        ))
        
        # State vector
        fig.add_trace(go.Scatter3d(
            x=[0, x], y=[0, y], z=[0, z],
            mode='lines+markers',
            line=dict(color='red', width=6),
            marker=dict(size=[5, 10], color=['blue', 'red']),
            name='Qubit State'
        ))
        
        fig.update_layout(
            scene=dict(
                xaxis_title='X', yaxis_title='Y', zaxis_title='Z',
                aspectmode='cube'
            ),
            height=400,
            title="Qubit State on Bloch Sphere"
        )
        
        st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.header("🌊 Superposition: The Quantum Magic")
    
    st.markdown("""
    <div class="concept-box">
    <h3>🎭 What is Superposition?</h3>
    <p>Superposition means a qubit can be in multiple states simultaneously. It's like being in two places at once!</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Probability Visualization")
        
        # Create superposition example
        states = ['|0⟩', '|1⟩', 'Equal Superposition', 'Weighted Superposition']
        prob_0 = [1, 0, 0.5, 0.8]
        prob_1 = [0, 1, 0.5, 0.2]
        
        fig = go.Figure(data=[
            go.Bar(name='Probability of |0⟩', x=states, y=prob_0, marker_color='lightblue'),
            go.Bar(name='Probability of |1⟩', x=states, y=prob_1, marker_color='lightcoral')
        ])
        
        fig.update_layout(
            barmode='stack',
            title='Different Quantum States',
            yaxis_title='Probability',
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("🔬 Create Your Own Superposition")
        
        weight = st.slider("Weight towards |1⟩", 0.0, 1.0, 0.5, 0.01)
        
        alpha_super = np.sqrt(1 - weight)
        beta_super = np.sqrt(weight)
        
        st.markdown(f"""
        **Your Superposition:**
        |ψ⟩ = {alpha_super:.3f}|0⟩ + {beta_super:.3f}|1⟩
        
        **What this means:**
        - {(1-weight)*100:.1f}% chance of measuring 0
        - {weight*100:.1f}% chance of measuring 1
        - The qubit is in BOTH states until measured!
        """)
        
        # Visual representation
        fig_super = go.Figure(data=go.Bar(
            x=['Probability of |0⟩', 'Probability of |1⟩'],
            y=[(1-weight), weight],
            marker_color=['lightblue', 'lightcoral']
        ))
        fig_super.update_layout(title="Your Superposition State", height=300)
        st.plotly_chart(fig_super, use_container_width=True)

with tab4:
    st.header("🎯 Quantum Measurement")
    
    st.markdown("""
    <div class="warning-box">
    <h3>⚠️ The Measurement Problem</h3>
    <p>When you measure a quantum state, the superposition collapses! The qubit "chooses" one of its possible states based on probability.</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎲 Measurement Simulator")
        
        # Set up a quantum state for measurement
        measure_weight = st.slider("Bias towards |1⟩", 0.0, 1.0, 0.3, 0.01, key="measure")
        num_measurements = st.slider("Number of measurements", 10, 1000, 100, 10)
        
        if st.button("🔬 Perform Measurements"):
            # Simulate measurements
            measurements = np.random.random(num_measurements) < measure_weight
            count_1 = np.sum(measurements)
            count_0 = num_measurements - count_1
            
            st.markdown(f"""
            **Results after {num_measurements} measurements:**
            - Got |0⟩: {count_0} times ({count_0/num_measurements*100:.1f}%)
            - Got |1⟩: {count_1} times ({count_1/num_measurements*100:.1f}%)
            - Expected |1⟩: {measure_weight*100:.1f}%
            """)
            
            # Plot results
            fig_measure = go.Figure(data=[
                go.Bar(x=['Measured |0⟩', 'Measured |1⟩'], 
                      y=[count_0, count_1],
                      marker_color=['lightblue', 'lightcoral'])
            ])
            fig_measure.update_layout(title="Measurement Results", height=300)
            st.plotly_chart(fig_measure, use_container_width=True)
    
    with col2:
        st.subheader("📈 Convergence to Theory")
        
        st.markdown("""
        **Key Insights:**
        1. Each measurement gives a definite result (0 or 1)
        2. Individual results seem random
        3. Over many measurements, frequencies approach theoretical probabilities
        4. Quantum mechanics is probabilistic, not deterministic!
        """)
        
        # Show convergence
        theory_prob = measure_weight
        measurements_range = range(10, 201, 10)
        measured_probs = []
        
        for n in measurements_range:
            simulated = np.random.random(n) < theory_prob
            measured_probs.append(np.mean(simulated))
        
        fig_conv = go.Figure()
        fig_conv.add_trace(go.Scatter(
            x=list(measurements_range), 
            y=measured_probs,
            mode='lines+markers',
            name='Measured Probability',
            line=dict(color='blue')
        ))
        fig_conv.add_trace(go.Scatter(
            x=list(measurements_range), 
            y=[theory_prob] * len(measurements_range),
            mode='lines',
            name='Theoretical Probability',
            line=dict(color='red', dash='dash')
        ))
        fig_conv.update_layout(
            title="Convergence to Theory",
            xaxis_title="Number of Measurements",
            yaxis_title="Probability of |1⟩",
            height=300
        )
        st.plotly_chart(fig_conv, use_container_width=True)

with tab5:
    st.header("🧮 Practice Problems")
    
    st.markdown("""
    <div class="concept-box">
    <h3>🎯 Test Your Understanding</h3>
    <p>Try these problems to reinforce what you've learned!</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Problem 1
    st.subheader("Problem 1: Probability Calculation")
    st.markdown("A qubit is in the state |ψ⟩ = 0.6|0⟩ + 0.8|1⟩")
    
    with st.expander("Click to solve"):
        answer1 = st.radio(
            "What's the probability of measuring |0⟩?",
            ["0.36", "0.64", "0.6", "0.8"]
        )
        
        if st.button("Check Answer 1"):
            if answer1 == "0.36":
                st.success("✅ Correct! P(0) = |0.6|² = 0.36")
            else:
                st.error("❌ Try again. Remember: probability = |amplitude|²")
    
    # Problem 2
    st.subheader("Problem 2: Superposition Understanding")
    st.markdown("Which statement about superposition is TRUE?")
    
    with st.expander("Click to solve"):
        answer2 = st.radio(
            "Choose the correct statement:",
            [
                "A qubit in superposition has a definite value",
                "Superposition means the qubit is both 0 and 1 simultaneously",
                "Superposition only exists in theory",
                "Measuring superposition always gives the same result"
            ]
        )
        
        if st.button("Check Answer 2"):
            if "both 0 and 1 simultaneously" in answer2:
                st.success("✅ Correct! Superposition allows qubits to be in multiple states at once!")
            else:
                st.error("❌ Think about what makes quantum different from classical...")
    
    # Problem 3
    st.subheader("Problem 3: Circuit Prediction")
    
    # Create a simple circuit
    qc = QuantumCircuit(1, 1)
    qc.h(0)  # Hadamard gate creates superposition
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Circuit:**")
        fig_circuit = qc.draw(output='mpl', style='iqp')
        st.pyplot(fig_circuit)
        plt.close()
    
    with col2:
        with st.expander("Predict the outcome"):
            st.markdown("This circuit applies a Hadamard gate to |0⟩. What's the result?")
            answer3 = st.radio(
                "Final state:",
                [
                    "|0⟩",
                    "|1⟩", 
                    "(|0⟩ + |1⟩)/√2",
                    "(|0⟩ - |1⟩)/√2"
                ]
            )
            
            if st.button("Check Answer 3"):
                if "(|0⟩ + |1⟩)/√2" in answer3:
                    st.success("✅ Perfect! The Hadamard gate creates an equal superposition!")
                else:
                    st.error("❌ The Hadamard gate creates equal superposition from |0⟩")

# Progress indicator
st.markdown("---")
st.markdown("### 🎉 Congratulations!")
st.markdown("You've completed the Quantum Basics module. Ready for the next challenge?")

col1, col2, col3 = st.columns([1, 1, 1])
with col2:
    if st.button("Continue to Superposition →", use_container_width=True):
        st.switch_page("pages/02_Superposition.py")