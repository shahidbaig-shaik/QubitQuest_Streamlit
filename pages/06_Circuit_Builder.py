import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from qiskit import QuantumCircuit, execute, Aer
from qiskit.visualization import plot_histogram, circuit_drawer
import plotly.express as px
from streamlit_drawable_canvas import st_canvas

st.set_page_config(page_title="Circuit Builder", page_icon="🔬", layout="wide")

# Custom CSS
st.markdown("""
<style>
    .circuit-builder {
        background: linear-gradient(135deg, #e1f5fe 0%, #b3e5fc 100%);
        padding: 2rem;
        border-radius: 15px;
        margin: 1rem 0;
        border: 2px solid #0277bd;
    }
    .gate-palette {
        background: #f3e5f5;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border-left: 4px solid #7b1fa2;
    }
    .simulation-box {
        background: #e8f5e8;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 4px solid #388e3c;
    }
    .instruction-box {
        background: #fff8e1;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
        border-left: 3px solid #ffa000;
    }
    .gate-button {
        margin: 0.25rem;
        padding: 0.5rem 1rem;
        border-radius: 8px;
        border: 2px solid #ddd;
        background: linear-gradient(135deg, #ffffff 0%, #f5f5f5 100%);
        cursor: pointer;
        display: inline-block;
        text-align: center;
        font-weight: bold;
    }
    .gate-button:hover {
        background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
        border-color: #2196f3;
    }
</style>
""", unsafe_allow_html=True)

st.title("🔬 Quantum Circuit Builder")
st.markdown("*Design, build, and simulate quantum circuits interactively*")

# Initialize session state
if 'circuit_history' not in st.session_state:
    st.session_state.circuit_history = []
if 'current_circuit' not in st.session_state:
    st.session_state.current_circuit = None
if 'num_qubits' not in st.session_state:
    st.session_state.num_qubits = 2

# Sidebar for circuit configuration
with st.sidebar:
    st.markdown("## 🔧 Circuit Configuration")
    
    # Number of qubits
    num_qubits = st.selectbox(
        "Number of qubits:",
        [1, 2, 3, 4, 5],
        index=1,
        key="circuit_qubits"
    )
    
    if num_qubits != st.session_state.num_qubits:
        st.session_state.num_qubits = num_qubits
        st.session_state.current_circuit = QuantumCircuit(num_qubits, num_qubits)
    
    if st.session_state.current_circuit is None:
        st.session_state.current_circuit = QuantumCircuit(num_qubits, num_qubits)
    
    st.markdown("---")
    
    # Preset circuits
    st.markdown("## 📋 Preset Circuits")
    
    if st.button("🌟 Bell State", use_container_width=True):
        circuit = QuantumCircuit(2, 2)
        circuit.h(0)
        circuit.cx(0, 1)
        st.session_state.current_circuit = circuit
    
    if st.button("🔄 GHZ State", use_container_width=True):
        circuit = QuantumCircuit(3, 3)
        circuit.h(0)
        circuit.cx(0, 1)
        circuit.cx(1, 2)
        st.session_state.current_circuit = circuit
    
    if st.button("🎲 Random Circuit", use_container_width=True):
        circuit = QuantumCircuit(num_qubits, num_qubits)
        gates = ['h', 'x', 'y', 'z', 's', 't']
        for _ in range(np.random.randint(3, 8)):
            gate = np.random.choice(gates)
            qubit = np.random.randint(num_qubits)
            if gate == 'h':
                circuit.h(qubit)
            elif gate == 'x':
                circuit.x(qubit)
            elif gate == 'y':
                circuit.y(qubit)
            elif gate == 'z':
                circuit.z(qubit)
            elif gate == 's':
                circuit.s(qubit)
            elif gate == 't':
                circuit.t(qubit)
        st.session_state.current_circuit = circuit
    
    if st.button("🗑️ Clear Circuit", use_container_width=True):
        st.session_state.current_circuit = QuantumCircuit(num_qubits, num_qubits)
    
    st.markdown("---")
    
    # Circuit operations
    st.markdown("## ⚙️ Circuit Operations")
    
    if st.button("📥 Save to History", use_container_width=True):
        if st.session_state.current_circuit and len(st.session_state.current_circuit.data) > 0:
            st.session_state.circuit_history.append(st.session_state.current_circuit.copy())
            st.success("Circuit saved!")
    
    if st.button("🔄 Add Measurements", use_container_width=True):
        for i in range(num_qubits):
            st.session_state.current_circuit.measure(i, i)
    
    # Circuit history
    if st.session_state.circuit_history:
        st.markdown("### 📚 Circuit History")
        for i, circuit in enumerate(st.session_state.circuit_history[-5:]):  # Show last 5
            if st.button(f"Load Circuit {len(st.session_state.circuit_history) - len(st.session_state.circuit_history[-5:]) + i + 1}", key=f"load_{i}"):
                st.session_state.current_circuit = circuit.copy()

# Main content area with tabs
tab1, tab2, tab3, tab4 = st.tabs(["🔧 Builder", "📊 Analysis", "🧪 Experiments", "📚 Tutorials"])

with tab1:
    st.markdown("""
    <div class="instruction-box">
    <h3>🎯 How to Build Circuits</h3>
    <p>1. Select gates from the palette below<br>
    2. Choose target qubits for each gate<br>
    3. Build your circuit step by step<br>
    4. Simulate and analyze the results!</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Gate palette
    st.markdown("""
    <div class="gate-palette">
    <h3>🎨 Gate Palette</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Single-qubit gates
    st.subheader("Single-Qubit Gates")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("**Pauli Gates**")
        if st.button("X Gate", help="Bit flip (NOT gate)", use_container_width=True):
            st.session_state.selected_gate = "X"
        if st.button("Y Gate", help="Bit + phase flip", use_container_width=True):
            st.session_state.selected_gate = "Y"
        if st.button("Z Gate", help="Phase flip", use_container_width=True):
            st.session_state.selected_gate = "Z"
    
    with col2:
        st.markdown("**Hadamard & Phase**")
        if st.button("H Gate", help="Hadamard (superposition)", use_container_width=True):
            st.session_state.selected_gate = "H"
        if st.button("S Gate", help="Phase gate (π/2)", use_container_width=True):
            st.session_state.selected_gate = "S"
        if st.button("T Gate", help="T gate (π/4)", use_container_width=True):
            st.session_state.selected_gate = "T"
    
    with col3:
        st.markdown("**Rotation Gates**")
        if st.button("RX Gate", help="X-rotation", use_container_width=True):
            st.session_state.selected_gate = "RX"
        if st.button("RY Gate", help="Y-rotation", use_container_width=True):
            st.session_state.selected_gate = "RY"
        if st.button("RZ Gate", help="Z-rotation", use_container_width=True):
            st.session_state.selected_gate = "RZ"
    
    with col4:
        st.markdown("**Inverse Gates**")
        if st.button("S† Gate", help="S dagger", use_container_width=True):
            st.session_state.selected_gate = "SDG"
        if st.button("T† Gate", help="T dagger", use_container_width=True):
            st.session_state.selected_gate = "TDG"
    
    # Multi-qubit gates
    st.subheader("Multi-Qubit Gates")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("CNOT Gate", help="Controlled-X", use_container_width=True):
            st.session_state.selected_gate = "CNOT"
        if st.button("CZ Gate", help="Controlled-Z", use_container_width=True):
            st.session_state.selected_gate = "CZ"
    
    with col2:
        if st.button("SWAP Gate", help="Swap qubits", use_container_width=True):
            st.session_state.selected_gate = "SWAP"
        if st.button("Toffoli Gate", help="Controlled-Controlled-X", use_container_width=True):
            st.session_state.selected_gate = "TOFFOLI"
    
    with col3:
        if st.button("Fredkin Gate", help="Controlled-SWAP", use_container_width=True):
            st.session_state.selected_gate = "FREDKIN"
    
    # Gate application interface
    st.markdown("---")
    
    if 'selected_gate' in st.session_state:
        gate = st.session_state.selected_gate
        
        st.markdown(f"### Adding {gate} Gate")
        
        # Gate-specific controls
        if gate in ["X", "Y", "Z", "H", "S", "T", "SDG", "TDG"]:
            target_qubit = st.selectbox(
                "Target qubit:",
                list(range(num_qubits)),
                key="single_target"
            )
            
            # Rotation angle for rotation gates
            angle = None
            if gate in ["RX", "RY", "RZ"]:
                angle = st.slider(
                    "Rotation angle (radians):",
                    0.0, 2*np.pi, np.pi/2, 0.1,
                    key="rotation_angle"
                )
            
            if st.button(f"Add {gate} to Circuit", use_container_width=True):
                if gate == "X":
                    st.session_state.current_circuit.x(target_qubit)
                elif gate == "Y":
                    st.session_state.current_circuit.y(target_qubit)
                elif gate == "Z":
                    st.session_state.current_circuit.z(target_qubit)
                elif gate == "H":
                    st.session_state.current_circuit.h(target_qubit)
                elif gate == "S":
                    st.session_state.current_circuit.s(target_qubit)
                elif gate == "T":
                    st.session_state.current_circuit.t(target_qubit)
                elif gate == "SDG":
                    st.session_state.current_circuit.sdg(target_qubit)
                elif gate == "TDG":
                    st.session_state.current_circuit.tdg(target_qubit)
                elif gate == "RX":
                    st.session_state.current_circuit.rx(angle, target_qubit)
                elif gate == "RY":
                    st.session_state.current_circuit.ry(angle, target_qubit)
                elif gate == "RZ":
                    st.session_state.current_circuit.rz(angle, target_qubit)
                
                st.success(f"{gate} gate added!")
        
        elif gate in ["CNOT", "CZ"]:
            col1, col2 = st.columns(2)
            with col1:
                control_qubit = st.selectbox(
                    "Control qubit:",
                    list(range(num_qubits)),
                    key="control"
                )
            with col2:
                target_options = [i for i in range(num_qubits) if i != control_qubit]
                target_qubit = st.selectbox(
                    "Target qubit:",
                    target_options,
                    key="target"
                )
            
            if st.button(f"Add {gate} to Circuit", use_container_width=True):
                if gate == "CNOT":
                    st.session_state.current_circuit.cx(control_qubit, target_qubit)
                elif gate == "CZ":
                    st.session_state.current_circuit.cz(control_qubit, target_qubit)
                st.success(f"{gate} gate added!")
        
        elif gate == "SWAP":
            col1, col2 = st.columns(2)
            with col1:
                qubit1 = st.selectbox(
                    "First qubit:",
                    list(range(num_qubits)),
                    key="swap1"
                )
            with col2:
                qubit2_options = [i for i in range(num_qubits) if i != qubit1]
                qubit2 = st.selectbox(
                    "Second qubit:",
                    qubit2_options,
                    key="swap2"
                )
            
            if st.button(f"Add {gate} to Circuit", use_container_width=True):
                st.session_state.current_circuit.swap(qubit1, qubit2)
                st.success(f"{gate} gate added!")
        
        elif gate == "TOFFOLI" and num_qubits >= 3:
            col1, col2, col3 = st.columns(3)
            with col1:
                control1 = st.selectbox(
                    "Control 1:",
                    list(range(num_qubits)),
                    key="toffoli_c1"
                )
            with col2:
                control2_options = [i for i in range(num_qubits) if i != control1]
                control2 = st.selectbox(
                    "Control 2:",
                    control2_options,
                    key="toffoli_c2"
                )
            with col3:
                target_options = [i for i in range(num_qubits) if i not in [control1, control2]]
                target = st.selectbox(
                    "Target:",
                    target_options,
                    key="toffoli_target"
                )
            
            if st.button(f"Add {gate} to Circuit", use_container_width=True):
                st.session_state.current_circuit.ccx(control1, control2, target)
                st.success(f"{gate} gate added!")
    
    # Circuit visualization
    st.markdown("---")
    st.markdown("### 🔍 Current Circuit")
    
    if st.session_state.current_circuit and len(st.session_state.current_circuit.data) > 0:
        try:
            # Display circuit
            fig_circuit = st.session_state.current_circuit.draw(output='mpl', style='iqp', fold=20)
            st.pyplot(fig_circuit, use_container_width=True)
            plt.close()
            
            # Circuit statistics
            st.markdown("**Circuit Statistics:**")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Qubits", num_qubits)
            with col2:
                st.metric("Gates", len(st.session_state.current_circuit.data))
            with col3:
                st.metric("Depth", st.session_state.current_circuit.depth())
        
        except Exception as e:
            st.error(f"Error displaying circuit: {e}")
    
    else:
        st.info("🎨 Start building your circuit by selecting gates from the palette above!")

with tab2:
    st.header("📊 Circuit Analysis & Simulation")
    
    if st.session_state.current_circuit and len(st.session_state.current_circuit.data) > 0:
        
        # Simulation options
        col1, col2 = st.columns(2)
        
        with col1:
            simulation_type = st.selectbox(
                "Simulation type:",
                ["Measurement", "Statevector", "Unitary Matrix"],
                key="sim_type"
            )
            
            shots = st.slider("Number of shots:", 100, 10000, 1000, 100)
        
        with col2:
            backend_type = st.selectbox(
                "Backend:",
                ["qasm_simulator", "statevector_simulator", "unitary_simulator"],
                key="backend"
            )
        
        if st.button("🚀 Run Simulation", use_container_width=True):
            
            try:
                # Prepare circuit for simulation
                sim_circuit = st.session_state.current_circuit.copy()
                
                if simulation_type == "Measurement":
                    # Add measurements if not present
                    if not any('measure' in str(instr[0]) for instr in sim_circuit.data):
                        for i in range(num_qubits):
                            sim_circuit.measure(i, i)
                    
                    # Run measurement simulation
                    backend = Aer.get_backend('qasm_simulator')
                    job = execute(sim_circuit, backend, shots=shots)
                    result = job.result()
                    counts = result.get_counts()
                    
                    # Display results
                    st.markdown("### 📊 Measurement Results")
                    
                    # Plot histogram
                    states = list(counts.keys())
                    frequencies = list(counts.values())
                    
                    fig = go.Figure(data=go.Bar(
                        x=states,
                        y=frequencies,
                        marker_color='lightblue',
                        text=frequencies,
                        textposition='auto'
                    ))
                    fig.update_layout(
                        title=f"Measurement Results ({shots} shots)",
                        xaxis_title="Measured State",
                        yaxis_title="Count",
                        height=400
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Show probabilities
                    st.markdown("**Probabilities:**")
                    total = sum(frequencies)
                    for state, count in counts.items():
                        prob = count / total
                        st.write(f"|{state}⟩: {prob:.4f} ({prob*100:.2f}%)")
                
                elif simulation_type == "Statevector":
                    # Run statevector simulation
                    backend = Aer.get_backend('statevector_simulator')
                    job = execute(sim_circuit, backend)
                    result = job.result()
                    statevector = result.get_statevector()
                    
                    st.markdown("### 🌊 Quantum State Vector")
                    
                    # Display statevector
                    n_states = len(statevector)
                    state_labels = [format(i, f'0{num_qubits}b') for i in range(n_states)]
                    amplitudes = np.abs(statevector)
                    phases = np.angle(statevector)
                    
                    # Plot amplitudes
                    fig = go.Figure()
                    fig.add_trace(go.Bar(
                        x=state_labels,
                        y=amplitudes**2,
                        name='Probability',
                        marker_color='lightgreen'
                    ))
                    fig.update_layout(
                        title="State Probabilities",
                        xaxis_title="Basis State",
                        yaxis_title="Probability",
                        height=400
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Show complex amplitudes
                    st.markdown("**Complex Amplitudes:**")
                    for i, (label, amp) in enumerate(zip(state_labels, statevector)):
                        if abs(amp) > 0.001:  # Only show significant amplitudes
                            st.write(f"|{label}⟩: {amp:.4f}")
                
                elif simulation_type == "Unitary Matrix":
                    # Remove measurements for unitary simulation
                    unitary_circuit = QuantumCircuit(num_qubits)
                    for instr, qargs, cargs in sim_circuit.data:
                        if 'measure' not in str(instr):
                            unitary_circuit.append(instr, qargs, cargs)
                    
                    # Run unitary simulation
                    backend = Aer.get_backend('unitary_simulator')
                    job = execute(unitary_circuit, backend)
                    result = job.result()
                    unitary = result.get_unitary()
                    
                    st.markdown("### 🔄 Unitary Matrix")
                    
                    # Display unitary matrix (real and imaginary parts)
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("**Real Part:**")
                        fig_real = go.Figure(data=go.Heatmap(
                            z=np.real(unitary),
                            colorscale='RdBu',
                            zmid=0
                        ))
                        fig_real.update_layout(title="Real Part", height=400)
                        st.plotly_chart(fig_real, use_container_width=True)
                    
                    with col2:
                        st.markdown("**Imaginary Part:**")
                        fig_imag = go.Figure(data=go.Heatmap(
                            z=np.imag(unitary),
                            colorscale='RdBu',
                            zmid=0
                        ))
                        fig_imag.update_layout(title="Imaginary Part", height=400)
                        st.plotly_chart(fig_imag, use_container_width=True)
                    
                    # Verify unitarity
                    unitary_check = np.allclose(
                        np.dot(unitary, unitary.conj().T),
                        np.eye(len(unitary)),
                        atol=1e-10
                    )
                    
                    if unitary_check:
                        st.success("✅ Matrix is unitary!")
                    else:
                        st.error("❌ Matrix is not unitary!")
            
            except Exception as e:
                st.error(f"Simulation error: {e}")
        
        # Circuit properties analysis
        st.markdown("---")
        st.markdown("### 🔬 Circuit Properties")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Circuit Depth", st.session_state.current_circuit.depth())
            st.metric("Total Gates", len(st.session_state.current_circuit.data))
        
        with col2:
            # Count gate types
            gate_counts = {}
            for instr, qargs, cargs in st.session_state.current_circuit.data:
                gate_name = instr.name
                gate_counts[gate_name] = gate_counts.get(gate_name, 0) + 1
            
            st.markdown("**Gate Composition:**")
            for gate, count in gate_counts.items():
                st.write(f"{gate}: {count}")
        
        with col3:
            # Connectivity analysis
            if num_qubits > 1:
                connections = set()
                for instr, qargs, cargs in st.session_state.current_circuit.data:
                    if len(qargs) > 1:
                        for i in range(len(qargs)-1):
                            connections.add((qargs[i].index, qargs[i+1].index))
                
                st.metric("Qubit Connections", len(connections))
                if connections:
                    st.write("Connected pairs:", list(connections))
    
    else:
        st.info("🔧 Build a circuit first to analyze it!")

with tab3:
    st.header("🧪 Quantum Experiments")
    
    st.markdown("""
    <div class="simulation-box">
    <h3>🎯 Ready-to-Try Experiments</h3>
    <p>Explore quantum phenomena with these pre-built experiments!</p>
    </div>
    """, unsafe_allow_html=True)
    
    experiment = st.selectbox(
        "Choose an experiment:",
        [
            "Superposition Demo",
            "Entanglement Creation", 
            "Quantum Interference",
            "Phase Kickback",
            "Quantum Teleportation",
            "Deutsch-Jozsa Algorithm"
        ]
    )
    
    if experiment == "Superposition Demo":
        st.subheader("🌊 Superposition Demonstration")
        
        st.markdown("""
        **Experiment:** Create and measure superposition states
        
        **Goal:** Understand how measurement collapses superposition
        """)
        
        superposition_circuit = QuantumCircuit(1, 1)
        superposition_circuit.h(0)
        superposition_circuit.measure(0, 0)
        
        col1, col2 = st.columns(2)
        with col1:
            fig = superposition_circuit.draw(output='mpl', style='iqp')
            st.pyplot(fig)
            plt.close()
        
        with col2:
            if st.button("Run Superposition Experiment"):
                backend = Aer.get_backend('qasm_simulator')
                job = execute(superposition_circuit, backend, shots=1000)
                result = job.result()
                counts = result.get_counts()
                
                fig = go.Figure(data=go.Bar(
                    x=list(counts.keys()),
                    y=list(counts.values()),
                    marker_color=['lightblue', 'lightcoral']
                ))
                fig.update_layout(title="Superposition Results", height=300)
                st.plotly_chart(fig, use_container_width=True)
                
                st.write("Expected: ~50% |0⟩, ~50% |1⟩")
                for state, count in counts.items():
                    st.write(f"|{state}⟩: {count/1000:.3f}")
    
    elif experiment == "Entanglement Creation":
        st.subheader("🔗 Bell State Creation")
        
        st.markdown("""
        **Experiment:** Create maximally entangled Bell states
        
        **Goal:** Demonstrate quantum entanglement correlation
        """)
        
        bell_circuit = QuantumCircuit(2, 2)
        bell_circuit.h(0)
        bell_circuit.cx(0, 1)
        bell_circuit.measure_all()
        
        col1, col2 = st.columns(2)
        with col1:
            fig = bell_circuit.draw(output='mpl', style='iqp')
            st.pyplot(fig)
            plt.close()
        
        with col2:
            if st.button("Create Bell State"):
                backend = Aer.get_backend('qasm_simulator')
                job = execute(bell_circuit, backend, shots=1000)
                result = job.result()
                counts = result.get_counts()
                
                fig = go.Figure(data=go.Bar(
                    x=list(counts.keys()),
                    y=list(counts.values()),
                    marker_color='lightgreen'
                ))
                fig.update_layout(title="Bell State Results", height=300)
                st.plotly_chart(fig, use_container_width=True)
                
                st.write("Expected: ~50% |00⟩, ~50% |11⟩, 0% |01⟩, 0% |10⟩")
                st.write("This shows perfect correlation!")

with tab4:
    st.header("📚 Circuit Building Tutorials")
    
    tutorial = st.selectbox(
        "Choose a tutorial:",
        [
            "Basic Gates Tutorial",
            "Creating Superposition",
            "Building Entangled States",
            "Multi-Qubit Operations",
            "Quantum Algorithm Patterns"
        ]
    )
    
    if tutorial == "Basic Gates Tutorial":
        st.markdown("""
        ### 🎓 Tutorial 1: Basic Quantum Gates
        
        **Step 1:** Start with a single qubit in |0⟩ state
        
        **Step 2:** Apply different gates and observe the effects:
        - **X Gate:** Flips |0⟩ to |1⟩ (quantum NOT)
        - **H Gate:** Creates superposition (|0⟩ + |1⟩)/√2
        - **Z Gate:** Adds phase to |1⟩ state
        
        **Step 3:** Add measurement to see the results
        
        **Try this:** Build a circuit with H, then measure. Run it multiple times!
        """)
        
        if st.button("Load Tutorial Circuit"):
            tutorial_circuit = QuantumCircuit(1, 1)
            tutorial_circuit.h(0)
            tutorial_circuit.measure(0, 0)
            st.session_state.current_circuit = tutorial_circuit
            st.success("Tutorial circuit loaded! Switch to the Builder tab to see it.")
    
    elif tutorial == "Creating Superposition":
        st.markdown("""
        ### 🎓 Tutorial 2: Superposition States
        
        **Objective:** Learn to create different superposition states
        
        **Common superposition states:**
        - |+⟩ = (|0⟩ + |1⟩)/√2 → Apply H to |0⟩
        - |-⟩ = (|0⟩ - |1⟩)/√2 → Apply X then H to |0⟩
        - |i⟩ = (|0⟩ + i|1⟩)/√2 → Apply H then S to |0⟩
        
        **Exercise:** Try creating each of these states and measure them!
        """)
        
        state_choice = st.selectbox("Load example:", ["|+⟩ state", "|-⟩ state", "|i⟩ state"])
        
        if st.button("Load Example Circuit"):
            if state_choice == "|+⟩ state":
                circuit = QuantumCircuit(1, 1)
                circuit.h(0)
            elif state_choice == "|-⟩ state":
                circuit = QuantumCircuit(1, 1)
                circuit.x(0)
                circuit.h(0)
            else:  # |i⟩ state
                circuit = QuantumCircuit(1, 1)
                circuit.h(0)
                circuit.s(0)
            
            st.session_state.current_circuit = circuit
            st.success(f"{state_choice} circuit loaded!")

# Footer with tips
st.markdown("---")
st.markdown("""
### 💡 Pro Tips for Circuit Building

1. **Start Simple:** Begin with single-qubit gates before moving to multi-qubit operations
2. **Visualize:** Use the Analysis tab to understand what your circuit does
3. **Experiment:** Try the preset circuits and modify them
4. **Measure:** Always add measurements to see the quantum effects
5. **Save:** Use the circuit history to save interesting circuits

**Remember:** Quantum circuits are read left to right, just like reading text!
""")

# Navigation
col1, col2, col3 = st.columns(3)
with col1:
    if st.button("← Back to Gates", use_container_width=True):
        st.switch_page("pages/05_Quantum_Gates.py")
with col2:
    if st.button("💻 Code Playground", use_container_width=True):
        st.switch_page("pages/07_Code_Playground.py")
with col3:
    if st.button("⚗️ Quantum Simulator →", use_container_width=True):
        st.switch_page("pages/08_Quantum_Simulator.py")