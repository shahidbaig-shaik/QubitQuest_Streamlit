import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from qiskit import QuantumCircuit, execute, Aer, transpile
from qiskit.visualization import plot_bloch_vector, plot_state_qsphere, plot_state_hinton
import plotly.express as px
import pandas as pd
from qiskit.quantum_info import Statevector, DensityMatrix, random_statevector

st.set_page_config(page_title="Quantum Simulator", page_icon="⚗️", layout="wide")

# Custom CSS
st.markdown("""
<style>
    .simulator-panel {
        background: linear-gradient(135deg, #e8f5e8 0%, #c8e6c9 100%);
        padding: 2rem;
        border-radius: 15px;
        margin: 1rem 0;
        border: 2px solid #4caf50;
    }
    .control-panel {
        background: #f3e5f5;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 4px solid #9c27b0;
    }
    .visualization-box {
        background: #e3f2fd;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 4px solid #2196f3;
    }
    .state-info {
        background: #fff3e0;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
        border: 1px solid #ffcc02;
        font-family: monospace;
    }
    .bloch-sphere {
        display: flex;
        justify-content: center;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

st.title("⚗️ Advanced Quantum Simulator")
st.markdown("*Visualize and simulate quantum states in real-time*")

# Initialize session state
if 'quantum_state' not in st.session_state:
    st.session_state.quantum_state = Statevector.from_label('0')
if 'simulation_history' not in st.session_state:
    st.session_state.simulation_history = []
if 'current_circuit' not in st.session_state:
    st.session_state.current_circuit = QuantumCircuit(1)

# Sidebar for simulator controls
with st.sidebar:
    st.markdown("## 🎛️ Simulator Controls")
    
    # State initialization
    st.markdown("### 🚀 Initialize State")
    
    init_method = st.selectbox(
        "Initialization method:",
        ["Computational basis", "Custom amplitudes", "Random state", "Bloch sphere angles"]
    )
    
    if init_method == "Computational basis":
        num_qubits = st.selectbox("Number of qubits:", [1, 2, 3, 4], index=0)
        basis_state = st.selectbox(
            "Initial state:",
            [f"|{format(i, f'0{num_qubits}b')}⟩" for i in range(2**num_qubits)]
        )
        
        if st.button("Initialize State"):
            label = basis_state[1:-1]  # Remove |⟩
            st.session_state.quantum_state = Statevector.from_label(label)
            st.session_state.current_circuit = QuantumCircuit(num_qubits)
            st.success(f"Initialized to {basis_state}")
    
    elif init_method == "Custom amplitudes":
        num_qubits = st.selectbox("Number of qubits:", [1, 2, 3], index=0, key="custom_qubits")
        
        st.markdown("**Enter amplitudes (will be normalized):**")
        amplitudes = []
        for i in range(2**num_qubits):
            state_label = f"|{format(i, f'0{num_qubits}b')}⟩"
            real_part = st.number_input(f"Real({state_label}):", -1.0, 1.0, 0.0, 0.1, key=f"real_{i}")
            imag_part = st.number_input(f"Imag({state_label}):", -1.0, 1.0, 0.0, 0.1, key=f"imag_{i}")
            amplitudes.append(complex(real_part, imag_part))
        
        if st.button("Create Custom State"):
            if any(abs(amp) > 1e-10 for amp in amplitudes):
                # Normalize
                norm = np.sqrt(sum(abs(amp)**2 for amp in amplitudes))
                if norm > 0:
                    amplitudes = [amp/norm for amp in amplitudes]
                    st.session_state.quantum_state = Statevector(amplitudes)
                    st.session_state.current_circuit = QuantumCircuit(num_qubits)
                    st.success("Custom state created!")
                else:
                    st.error("All amplitudes are zero!")
            else:
                st.error("At least one amplitude must be non-zero!")
    
    elif init_method == "Random state":
        num_qubits = st.selectbox("Number of qubits:", [1, 2, 3], index=0, key="random_qubits")
        seed = st.number_input("Random seed:", 0, 1000, 42, key="random_seed")
        
        if st.button("Generate Random State"):
            np.random.seed(seed)
            st.session_state.quantum_state = random_statevector(2**num_qubits)
            st.session_state.current_circuit = QuantumCircuit(num_qubits)
            st.success("Random state generated!")
    
    elif init_method == "Bloch sphere angles":
        st.markdown("**Single qubit only**")
        theta = st.slider("θ (polar angle):", 0.0, np.pi, 0.0, 0.1, key="theta")
        phi = st.slider("φ (azimuthal angle):", 0.0, 2*np.pi, 0.0, 0.1, key="phi")
        
        if st.button("Set Bloch State"):
            # Create state vector from Bloch sphere angles
            state_vector = [np.cos(theta/2), np.exp(1j*phi)*np.sin(theta/2)]
            st.session_state.quantum_state = Statevector(state_vector)
            st.session_state.current_circuit = QuantumCircuit(1)
            st.success(f"State set to θ={theta:.2f}, φ={phi:.2f}")
    
    st.markdown("---")
    
    # Gate operations
    st.markdown("### 🎲 Apply Gates")
    
    if st.session_state.quantum_state.num_qubits == 1:
        gate_type = st.selectbox(
            "Single-qubit gate:",
            ["H", "X", "Y", "Z", "S", "T", "RX", "RY", "RZ"]
        )
        
        angle = None
        if gate_type in ["RX", "RY", "RZ"]:
            angle = st.slider(f"{gate_type} angle:", 0.0, 2*np.pi, np.pi/2, 0.1)
        
        if st.button("Apply Gate"):
            qc = QuantumCircuit(1)
            if gate_type == "H":
                qc.h(0)
            elif gate_type == "X":
                qc.x(0)
            elif gate_type == "Y":
                qc.y(0)
            elif gate_type == "Z":
                qc.z(0)
            elif gate_type == "S":
                qc.s(0)
            elif gate_type == "T":
                qc.t(0)
            elif gate_type == "RX":
                qc.rx(angle, 0)
            elif gate_type == "RY":
                qc.ry(angle, 0)
            elif gate_type == "RZ":
                qc.rz(angle, 0)
            
            # Apply gate to current state
            new_state = st.session_state.quantum_state.evolve(qc)
            st.session_state.quantum_state = new_state
            st.session_state.current_circuit.compose(qc, inplace=True)
            st.success(f"{gate_type} gate applied!")
    
    else:
        st.info("Multi-qubit gate operations available in main interface")
    
    st.markdown("---")
    
    # Measurement simulation
    st.markdown("### 📏 Measurement")
    
    if st.button("Simulate Measurement"):
        # Perform measurement simulation
        probabilities = st.session_state.quantum_state.probabilities()
        num_qubits = st.session_state.quantum_state.num_qubits
        
        # Simulate 1000 measurements
        measurements = np.random.choice(
            range(2**num_qubits), 
            size=1000, 
            p=probabilities
        )
        
        # Store in session state for visualization
        st.session_state.measurement_results = {
            format(i, f'0{num_qubits}b'): np.sum(measurements == i) 
            for i in range(2**num_qubits)
        }
        st.success("Measurement simulation complete!")
    
    st.markdown("---")
    
    # Simulation history
    st.markdown("### 📚 History")
    
    if st.button("Save Current State"):
        st.session_state.simulation_history.append({
            'state': st.session_state.quantum_state.copy(),
            'circuit': st.session_state.current_circuit.copy(),
            'timestamp': str(pd.Timestamp.now())
        })
        st.success("State saved to history!")
    
    if st.button("Clear History"):
        st.session_state.simulation_history = []
        st.success("History cleared!")
    
    # Show recent history
    if st.session_state.simulation_history:
        st.markdown("**Recent states:**")
        for i, entry in enumerate(st.session_state.simulation_history[-3:]):
            if st.button(f"Load State {len(st.session_state.simulation_history) - len(st.session_state.simulation_history[-3:]) + i + 1}", key=f"load_{i}"):
                st.session_state.quantum_state = entry['state']
                st.session_state.current_circuit = entry['circuit']
                st.success("State loaded!")

# Main content area with tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🌐 State Visualization", 
    "📊 Measurement Analysis", 
    "🎛️ Interactive Controls", 
    "🧪 Quantum Experiments",
    "📈 Advanced Analytics"
])

with tab1:
    st.header("🌐 Quantum State Visualization")
    
    if st.session_state.quantum_state is not None:
        
        # State information
        st.markdown("""
        <div class="state-info">
        <h4>📋 Current State Information</h4>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Number of Qubits", st.session_state.quantum_state.num_qubits)
            st.metric("Dimension", 2**st.session_state.quantum_state.num_qubits)
        
        with col2:
            # Calculate purity (for mixed states, this would be < 1)
            purity = np.real(np.trace(st.session_state.quantum_state.to_operator() @ st.session_state.quantum_state.to_operator()))
            st.metric("Purity", f"{purity:.4f}")
            
            # Entanglement measure (for multi-qubit states)
            if st.session_state.quantum_state.num_qubits > 1:
                # Calculate von Neumann entropy of reduced state
                try:
                    reduced_state = st.session_state.quantum_state.partial_trace([1])
                    eigenvals = np.linalg.eigvals(reduced_state.data)
                    eigenvals = eigenvals[eigenvals > 1e-10]  # Remove near-zero eigenvals
                    entropy = -np.sum(eigenvals * np.log2(eigenvals))
                    st.metric("Entanglement Entropy", f"{entropy:.4f}")
                except:
                    st.metric("Entanglement Entropy", "N/A")
        
        with col3:
            # Global phase (meaningless for pure states, but interesting to show)
            st.metric("Global Phase", "Arbitrary")
        
        # State vector display
        st.markdown("### 📊 State Vector Components")
        
        state_vector = st.session_state.quantum_state.data
        num_qubits = st.session_state.quantum_state.num_qubits
        
        # Create DataFrame for better display
        state_data = []
        for i, amplitude in enumerate(state_vector):
            basis_state = format(i, f'0{num_qubits}b')
            probability = abs(amplitude)**2
            phase = np.angle(amplitude)
            
            state_data.append({
                'Basis State': f"|{basis_state}⟩",
                'Amplitude': f"{amplitude:.4f}",
                'Probability': f"{probability:.4f}",
                'Phase (rad)': f"{phase:.4f}"
            })
        
        df = pd.DataFrame(state_data)
        st.dataframe(df, use_container_width=True)
        
        # Visualization based on number of qubits
        if num_qubits == 1:
            # Bloch sphere for single qubit
            st.markdown("### 🌐 Bloch Sphere Representation")
            
            # Calculate Bloch vector
            pauli_x = np.array([[0, 1], [1, 0]])
            pauli_y = np.array([[0, -1j], [1j, 0]])
            pauli_z = np.array([[1, 0], [0, -1]])
            
            rho = st.session_state.quantum_state.to_operator().data
            
            x_coord = np.real(np.trace(rho @ pauli_x))
            y_coord = np.real(np.trace(rho @ pauli_y))
            z_coord = np.real(np.trace(rho @ pauli_z))
            
            # Create 3D Bloch sphere plot
            fig = go.Figure()
            
            # Add sphere surface
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
            
            # Add coordinate axes
            fig.add_trace(go.Scatter3d(
                x=[-1, 1], y=[0, 0], z=[0, 0],
                mode='lines', line=dict(color='red', width=3),
                name='X-axis'
            ))
            fig.add_trace(go.Scatter3d(
                x=[0, 0], y=[-1, 1], z=[0, 0],
                mode='lines', line=dict(color='green', width=3),
                name='Y-axis'
            ))
            fig.add_trace(go.Scatter3d(
                x=[0, 0], y=[0, 0], z=[-1, 1],
                mode='lines', line=dict(color='blue', width=3),
                name='Z-axis'
            ))
            
            # Add state vector
            fig.add_trace(go.Scatter3d(
                x=[0, x_coord], y=[0, y_coord], z=[0, z_coord],
                mode='lines+markers',
                line=dict(color='black', width=6),
                marker=dict(size=[5, 10], color=['gray', 'red']),
                name='State Vector'
            ))
            
            # Add labels
            fig.add_trace(go.Scatter3d(
                x=[1.1], y=[0], z=[0],
                mode='text', text=['|+⟩'], textfont=dict(size=14),
                showlegend=False
            ))
            fig.add_trace(go.Scatter3d(
                x=[-1.1], y=[0], z=[0],
                mode='text', text=['|-⟩'], textfont=dict(size=14),
                showlegend=False
            ))
            fig.add_trace(go.Scatter3d(
                x=[0], y=[1.1], z=[0],
                mode='text', text=['|i⟩'], textfont=dict(size=14),
                showlegend=False
            ))
            fig.add_trace(go.Scatter3d(
                x=[0], y=[-1.1], z=[0],
                mode='text', text=['|-i⟩'], textfont=dict(size=14),
                showlegend=False
            ))
            fig.add_trace(go.Scatter3d(
                x=[0], y=[0], z=[1.1],
                mode='text', text=['|0⟩'], textfont=dict(size=14),
                showlegend=False
            ))
            fig.add_trace(go.Scatter3d(
                x=[0], y=[0], z=[-1.1],
                mode='text', text=['|1⟩'], textfont=dict(size=14),
                showlegend=False
            ))
            
            fig.update_layout(
                title="Bloch Sphere Representation",
                scene=dict(
                    xaxis_title='X', yaxis_title='Y', zaxis_title='Z',
                    aspectmode='cube',
                    camera=dict(eye=dict(x=1.2, y=1.2, z=1.2))
                ),
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Display Bloch coordinates
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("X coordinate", f"{x_coord:.4f}")
            with col2:
                st.metric("Y coordinate", f"{y_coord:.4f}")
            with col3:
                st.metric("Z coordinate", f"{z_coord:.4f}")
        
        else:
            # Multi-qubit visualization
            st.markdown("### 📊 Probability Distribution")
            
            probabilities = st.session_state.quantum_state.probabilities()
            basis_states = [f"|{format(i, f'0{num_qubits}b')}⟩" for i in range(2**num_qubits)]
            
            # Bar chart of probabilities
            fig = go.Figure(data=go.Bar(
                x=basis_states,
                y=probabilities,
                marker_color='lightblue',
                text=[f"{p:.3f}" for p in probabilities],
                textposition='auto'
            ))
            fig.update_layout(
                title="State Probability Distribution",
                xaxis_title="Basis State",
                yaxis_title="Probability",
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Phase information
            st.markdown("### 🌊 Phase Distribution")
            
            phases = [np.angle(amp) for amp in state_vector]
            
            fig_phase = go.Figure(data=go.Bar(
                x=basis_states,
                y=phases,
                marker_color='lightcoral',
                text=[f"{p:.2f}" for p in phases],
                textposition='auto'
            ))
            fig_phase.update_layout(
                title="Phase Distribution",
                xaxis_title="Basis State",
                yaxis_title="Phase (radians)",
                height=400
            )
            st.plotly_chart(fig_phase, use_container_width=True)
    
    else:
        st.info("Initialize a quantum state using the sidebar controls to begin visualization.")

with tab2:
    st.header("📊 Measurement Analysis")
    
    if hasattr(st.session_state, 'measurement_results'):
        st.markdown("### 🎯 Measurement Results (1000 shots)")
        
        # Display measurement histogram
        states = list(st.session_state.measurement_results.keys())
        counts = list(st.session_state.measurement_results.values())
        
        fig = go.Figure(data=go.Bar(
            x=states,
            y=counts,
            marker_color='lightgreen',
            text=counts,
            textposition='auto'
        ))
        fig.update_layout(
            title="Measurement Outcome Distribution",
            xaxis_title="Measured State",
            yaxis_title="Count",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Statistical analysis
        st.markdown("### 📈 Statistical Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Compare with theoretical probabilities
            if st.session_state.quantum_state is not None:
                theoretical_probs = st.session_state.quantum_state.probabilities()
                measured_probs = np.array(counts) / 1000
                
                st.markdown("**Theoretical vs Measured Probabilities:**")
                comparison_data = []
                for i, state in enumerate(states):
                    if i < len(theoretical_probs):
                        comparison_data.append({
                            'State': state,
                            'Theoretical': f"{theoretical_probs[i]:.4f}",
                            'Measured': f"{measured_probs[i]:.4f}",
                            'Difference': f"{abs(theoretical_probs[i] - measured_probs[i]):.4f}"
                        })
                
                df_comparison = pd.DataFrame(comparison_data)
                st.dataframe(df_comparison, use_container_width=True)
        
        with col2:
            # Basic statistics
            total_measurements = sum(counts)
            entropy = -sum(p * np.log2(p) if p > 0 else 0 for p in np.array(counts)/total_measurements)
            
            st.metric("Total Measurements", total_measurements)
            st.metric("Shannon Entropy", f"{entropy:.4f}")
            st.metric("Most Likely State", max(st.session_state.measurement_results, key=st.session_state.measurement_results.get))
    
    else:
        st.info("Perform a measurement simulation using the sidebar to see analysis.")
    
    # Interactive measurement simulation
    st.markdown("---")
    st.markdown("### 🎲 Custom Measurement Simulation")
    
    if st.session_state.quantum_state is not None:
        num_shots = st.slider("Number of measurements:", 100, 10000, 1000, 100)
        
        if st.button("Run Custom Measurement"):
            probabilities = st.session_state.quantum_state.probabilities()
            num_qubits = st.session_state.quantum_state.num_qubits
            
            measurements = np.random.choice(
                range(2**num_qubits), 
                size=num_shots, 
                p=probabilities
            )
            
            custom_results = {
                format(i, f'0{num_qubits}b'): np.sum(measurements == i) 
                for i in range(2**num_qubits)
            }
            
            # Display results
            states = list(custom_results.keys())
            counts = list(custom_results.values())
            
            fig = go.Figure(data=go.Bar(
                x=states,
                y=counts,
                marker_color='orange',
                text=counts,
                textposition='auto'
            ))
            fig.update_layout(
                title=f"Custom Measurement Results ({num_shots} shots)",
                xaxis_title="Measured State",
                yaxis_title="Count",
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.header("🎛️ Interactive Quantum Controls")
    
    st.markdown("""
    <div class="control-panel">
    <h3>🎮 Real-time State Manipulation</h3>
    <p>Use these controls to interactively modify your quantum state</p>
    </div>
    """, unsafe_allow_html=True)
    
    if st.session_state.quantum_state is not None and st.session_state.quantum_state.num_qubits == 1:
        
        # Interactive Bloch sphere controls
        st.markdown("### 🌐 Interactive Bloch Sphere")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            # Real-time controls
            theta_control = st.slider("θ (Polar angle):", 0.0, np.pi, np.pi/2, 0.05, key="interactive_theta")
            phi_control = st.slider("φ (Azimuthal angle):", 0.0, 2*np.pi, 0.0, 0.05, key="interactive_phi")
            
            # Update state in real-time
            new_state_vector = [
                np.cos(theta_control/2), 
                np.exp(1j*phi_control)*np.sin(theta_control/2)
            ]
            interactive_state = Statevector(new_state_vector)
            
            # Display current coordinates
            st.markdown("**Current State:**")
            st.write(f"α = {new_state_vector[0]:.3f}")
            st.write(f"β = {new_state_vector[1]:.3f}")
            
            if st.button("Apply Interactive State"):
                st.session_state.quantum_state = interactive_state
                st.success("State updated!")
        
        with col2:
            # Real-time Bloch sphere
            pauli_x = np.array([[0, 1], [1, 0]])
            pauli_y = np.array([[0, -1j], [1j, 0]])
            pauli_z = np.array([[1, 0], [0, -1]])
            
            rho = interactive_state.to_operator().data
            
            x_coord = np.real(np.trace(rho @ pauli_x))
            y_coord = np.real(np.trace(rho @ pauli_y))
            z_coord = np.real(np.trace(rho @ pauli_z))
            
            # Create simplified Bloch sphere
            fig_interactive = go.Figure()
            
            # Sphere wireframe
            u = np.linspace(0, 2 * np.pi, 20)
            v = np.linspace(0, np.pi, 20)
            x_sphere = np.outer(np.cos(u), np.sin(v))
            y_sphere = np.outer(np.sin(u), np.sin(v))
            z_sphere = np.outer(np.ones(np.size(u)), np.cos(v))
            
            fig_interactive.add_trace(go.Surface(
                x=x_sphere, y=y_sphere, z=z_sphere,
                opacity=0.2, showscale=False,
                colorscale='Blues'
            ))
            
            # State vector
            fig_interactive.add_trace(go.Scatter3d(
                x=[0, x_coord], y=[0, y_coord], z=[0, z_coord],
                mode='lines+markers',
                line=dict(color='red', width=8),
                marker=dict(size=[5, 15], color=['gray', 'red']),
                name='State'
            ))
            
            fig_interactive.update_layout(
                title="Interactive Bloch Sphere",
                scene=dict(
                    xaxis_title='X', yaxis_title='Y', zaxis_title='Z',
                    aspectmode='cube'
                ),
                height=400,
                showlegend=False
            )
            
            st.plotly_chart(fig_interactive, use_container_width=True)
    
    # Gate sequence builder
    st.markdown("---")
    st.markdown("### 🔧 Gate Sequence Builder")
    
    if 'gate_sequence' not in st.session_state:
        st.session_state.gate_sequence = []
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.session_state.quantum_state is not None:
            num_qubits = st.session_state.quantum_state.num_qubits
            
            available_gates = ["H", "X", "Y", "Z", "S", "T"]
            if num_qubits > 1:
                available_gates.extend(["CNOT", "CZ", "SWAP"])
            
            selected_gate = st.selectbox("Add gate:", available_gates)
            
            if num_qubits > 1 and selected_gate in ["CNOT", "CZ"]:
                control = st.selectbox("Control qubit:", list(range(num_qubits)))
                target = st.selectbox("Target qubit:", [i for i in range(num_qubits) if i != control])
                gate_info = f"{selected_gate}({control},{target})"
            elif num_qubits > 1 and selected_gate == "SWAP":
                qubit1 = st.selectbox("First qubit:", list(range(num_qubits)))
                qubit2 = st.selectbox("Second qubit:", [i for i in range(num_qubits) if i != qubit1])
                gate_info = f"{selected_gate}({qubit1},{qubit2})"
            else:
                target_qubit = 0 if num_qubits == 1 else st.selectbox("Target qubit:", list(range(num_qubits)))
                gate_info = f"{selected_gate}({target_qubit})"
            
            if st.button("Add to Sequence"):
                st.session_state.gate_sequence.append(gate_info)
                st.success(f"Added {gate_info}")
    
    with col2:
        st.markdown("**Current Sequence:**")
        if st.session_state.gate_sequence:
            for i, gate in enumerate(st.session_state.gate_sequence):
                st.write(f"{i+1}. {gate}")
        else:
            st.write("No gates in sequence")
        
        if st.button("Clear Sequence"):
            st.session_state.gate_sequence = []
            st.success("Sequence cleared!")
    
    with col3:
        if st.session_state.gate_sequence and st.session_state.quantum_state is not None:
            if st.button("Execute Sequence"):
                # Build and execute the gate sequence
                num_qubits = st.session_state.quantum_state.num_qubits
                qc = QuantumCircuit(num_qubits)
                
                try:
                    for gate_info in st.session_state.gate_sequence:
                        if "CNOT" in gate_info:
                            control, target = map(int, gate_info.split('(')[1].split(')')[0].split(','))
                            qc.cx(control, target)
                        elif "CZ" in gate_info:
                            control, target = map(int, gate_info.split('(')[1].split(')')[0].split(','))
                            qc.cz(control, target)
                        elif "SWAP" in gate_info:
                            qubit1, qubit2 = map(int, gate_info.split('(')[1].split(')')[0].split(','))
                            qc.swap(qubit1, qubit2)
                        else:
                            gate_name = gate_info.split('(')[0]
                            qubit = int(gate_info.split('(')[1].split(')')[0])
                            if gate_name == "H":
                                qc.h(qubit)
                            elif gate_name == "X":
                                qc.x(qubit)
                            elif gate_name == "Y":
                                qc.y(qubit)
                            elif gate_name == "Z":
                                qc.z(qubit)
                            elif gate_name == "S":
                                qc.s(qubit)
                            elif gate_name == "T":
                                qc.t(qubit)
                    
                    # Apply sequence to current state
                    new_state = st.session_state.quantum_state.evolve(qc)
                    st.session_state.quantum_state = new_state
                    st.session_state.current_circuit.compose(qc, inplace=True)
                    st.success("Sequence executed!")
                    
                except Exception as e:
                    st.error(f"Error executing sequence: {e}")

with tab4:
    st.header("🧪 Quantum Experiments")
    
    experiment_type = st.selectbox(
        "Choose experiment:",
        [
            "Quantum Interference",
            "Bell State Analysis", 
            "Quantum Phase Estimation",
            "Quantum Walks",
            "Decoherence Simulation",
            "Quantum Fourier Transform"
        ]
    )
    
    if experiment_type == "Quantum Interference":
        st.markdown("""
        ### 🌊 Quantum Interference Experiment
        
        Explore how quantum states can interfere constructively or destructively.
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Setup:**")
            
            # Create Mach-Zehnder-like interferometer
            phase_shift = st.slider("Phase shift (radians):", 0.0, 2*np.pi, 0.0, 0.1)
            
            if st.button("Run Interference Experiment"):
                # Create interferometer circuit
                qc = QuantumCircuit(1, 1)
                qc.h(0)  # First beam splitter
                qc.rz(phase_shift, 0)  # Phase shift
                qc.h(0)  # Second beam splitter
                qc.measure(0, 0)
                
                # Simulate
                backend = Aer.get_backend('qasm_simulator')
                job = execute(qc, backend, shots=1000)
                result = job.result()
                counts = result.get_counts()
                
                st.session_state.interference_results = counts
        
        with col2:
            if hasattr(st.session_state, 'interference_results'):
                counts = st.session_state.interference_results
                prob_0 = counts.get('0', 0) / 1000
                prob_1 = counts.get('1', 0) / 1000
                
                fig = go.Figure(data=go.Bar(
                    x=['|0⟩', '|1⟩'],
                    y=[prob_0, prob_1],
                    marker_color=['lightblue', 'lightcoral']
                ))
                fig.update_layout(title="Interference Pattern", height=300)
                st.plotly_chart(fig, use_container_width=True)
                
                st.write(f"P(|0⟩) = {prob_0:.3f}")
                st.write(f"P(|1⟩) = {prob_1:.3f}")
                
                # Theoretical prediction
                theoretical_0 = np.cos(phase_shift/2)**2
                st.write(f"Theoretical P(|0⟩) = {theoretical_0:.3f}")
    
    elif experiment_type == "Bell State Analysis":
        st.markdown("""
        ### 🔗 Bell State Analysis
        
        Create and analyze all four Bell states.
        """)
        
        bell_state = st.selectbox(
            "Choose Bell state:",
            ["Φ+ (|00⟩+|11⟩)/√2", "Φ- (|00⟩-|11⟩)/√2", "Ψ+ (|01⟩+|10⟩)/√2", "Ψ- (|01⟩-|10⟩)/√2"]
        )
        
        if st.button("Create Bell State"):
            qc = QuantumCircuit(2)
            
            if bell_state.startswith("Φ+"):
                qc.h(0)
                qc.cx(0, 1)
            elif bell_state.startswith("Φ-"):
                qc.h(0)
                qc.z(0)
                qc.cx(0, 1)
            elif bell_state.startswith("Ψ+"):
                qc.h(0)
                qc.x(1)
                qc.cx(0, 1)
            elif bell_state.startswith("Ψ-"):
                qc.h(0)
                qc.x(1)
                qc.z(0)
                qc.cx(0, 1)
            
            # Create the state
            bell_statevector = Statevector.from_instruction(qc)
            st.session_state.quantum_state = bell_statevector
            st.session_state.current_circuit = qc
            
            st.success(f"Created {bell_state}!")
            
            # Display state vector
            st.write("State vector:")
            for i, amp in enumerate(bell_statevector.data):
                if abs(amp) > 1e-10:
                    basis = format(i, '02b')
                    st.write(f"|{basis}⟩: {amp:.3f}")

with tab5:
    st.header("📈 Advanced Quantum Analytics")
    
    if st.session_state.quantum_state is not None:
        
        # Quantum state tomography simulation
        st.markdown("### 🔬 Quantum State Tomography")
        
        if st.session_state.quantum_state.num_qubits == 1:
            st.markdown("**Single-qubit state tomography:**")
            
            # Simulate measurements in X, Y, Z bases
            if st.button("Perform Tomography"):
                # Pauli measurements
                pauli_circuits = []
                
                # Z measurement (computational basis)
                qc_z = QuantumCircuit(1, 1)
                qc_z.measure(0, 0)
                pauli_circuits.append(('Z', qc_z))
                
                # X measurement
                qc_x = QuantumCircuit(1, 1)
                qc_x.h(0)
                qc_x.measure(0, 0)
                pauli_circuits.append(('X', qc_x))
                
                # Y measurement
                qc_y = QuantumCircuit(1, 1)
                qc_y.sdg(0)
                qc_y.h(0)
                qc_y.measure(0, 0)
                pauli_circuits.append(('Y', qc_y))
                
                # Simulate measurements
                backend = Aer.get_backend('qasm_simulator')
                tomography_results = {}
                
                for basis, circuit in pauli_circuits:
                    # Apply measurement circuit to current state
                    full_circuit = st.session_state.current_circuit.copy()
                    full_circuit.compose(circuit, inplace=True)
                    
                    job = execute(full_circuit, backend, shots=1000)
                    result = job.result()
                    counts = result.get_counts()
                    
                    prob_0 = counts.get('0', 0) / 1000
                    prob_1 = counts.get('1', 0) / 1000
                    expectation = prob_0 - prob_1  # <σ_i> = P(0) - P(1)
                    
                    tomography_results[basis] = expectation
                
                # Reconstruct density matrix
                pauli_x = np.array([[0, 1], [1, 0]])
                pauli_y = np.array([[0, -1j], [1j, 0]])
                pauli_z = np.array([[1, 0], [0, -1]])
                identity = np.array([[1, 0], [0, 1]])
                
                reconstructed_rho = 0.5 * (
                    identity + 
                    tomography_results['X'] * pauli_x + 
                    tomography_results['Y'] * pauli_y + 
                    tomography_results['Z'] * pauli_z
                )
                
                st.markdown("**Tomography Results:**")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("⟨σ_X⟩", f"{tomography_results['X']:.3f}")
                with col2:
                    st.metric("⟨σ_Y⟩", f"{tomography_results['Y']:.3f}")
                with col3:
                    st.metric("⟨σ_Z⟩", f"{tomography_results['Z']:.3f}")
                
                st.markdown("**Reconstructed Density Matrix:**")
                st.write(f"ρ = {reconstructed_rho}")
                
                # Fidelity with original state
                original_rho = st.session_state.quantum_state.to_operator().data
                fidelity = np.real(np.trace(np.sqrt(np.sqrt(original_rho) @ reconstructed_rho @ np.sqrt(original_rho))))**2
                st.metric("Reconstruction Fidelity", f"{fidelity:.4f}")
        
        # Quantum process tomography (for simple gates)
        st.markdown("---")
        st.markdown("### ⚙️ Process Characterization")
        
        if st.session_state.current_circuit.depth() > 0:
            st.markdown("**Current circuit complexity:**")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Circuit Depth", st.session_state.current_circuit.depth())
            with col2:
                st.metric("Gate Count", len(st.session_state.current_circuit.data))
            with col3:
                gate_types = [instr[0].name for instr in st.session_state.current_circuit.data]
                unique_gates = len(set(gate_types))
                st.metric("Unique Gate Types", unique_gates)
            
            # Gate composition analysis
            gate_counts = {}
            for instr, qargs, cargs in st.session_state.current_circuit.data:
                gate_name = instr.name
                gate_counts[gate_name] = gate_counts.get(gate_name, 0) + 1
            
            if gate_counts:
                st.markdown("**Gate Composition:**")
                gate_names = list(gate_counts.keys())
                gate_frequencies = list(gate_counts.values())
                
                fig = go.Figure(data=go.Pie(
                    labels=gate_names,
                    values=gate_frequencies,
                    hole=0.3
                ))
                fig.update_layout(title="Circuit Gate Distribution", height=400)
                st.plotly_chart(fig, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("""
### 🎯 Simulator Features

- **Real-time Visualization**: Watch quantum states evolve as you apply operations
- **Interactive Controls**: Manipulate states directly with sliders and controls  
- **Measurement Simulation**: Run virtual experiments with statistical analysis
- **Advanced Analytics**: Perform quantum state and process tomography
- **Educational Experiments**: Explore fundamental quantum phenomena

**Tip**: Start with single-qubit states to understand the basics, then explore multi-qubit entanglement!
""")

# Navigation
col1, col2, col3 = st.columns(3)
with col1:
    if st.button("← Code Playground", use_container_width=True):
        st.switch_page("pages/07_Code_Playground.py")
with col2:
    if st.button("🏠 Home", use_container_width=True):
        st.switch_page("app.py")
with col3:
    if st.button("🌟 Quantum Basics →", use_container_width=True):
        st.switch_page("pages/01_Quantum_Basics.py")