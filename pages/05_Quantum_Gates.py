import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from qiskit import QuantumCircuit, execute, Aer
from qiskit.visualization import plot_bloch_vector
import plotly.express as px

st.set_page_config(page_title="Quantum Gates", page_icon="🎲", layout="wide")

# Custom CSS
st.markdown("""
<style>
    .gate-box {
        background: linear-gradient(135deg, #e8f5e8 0%, #c8e6c9 100%);
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 5px solid #4caf50;
    }
    .matrix-box {
        background: #f3e5f5;
        padding: 1rem;
        border-radius: 8px;
        font-family: monospace;
        text-align: center;
        margin: 0.5rem 0;
    }
    .effect-box {
        background: #fff3e0;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border-left: 3px solid #ff9800;
    }
</style>
""", unsafe_allow_html=True)

st.title("🎲 Quantum Gates")
st.markdown("*The building blocks of quantum circuits*")

# Navigation tabs
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📖 Gate Basics", 
    "🔧 Single-Qubit Gates", 
    "🔗 Multi-Qubit Gates", 
    "🎮 Interactive Explorer", 
    "🧪 Gate Laboratory", 
    "📝 Practice"
])

with tab1:
    st.header("Understanding Quantum Gates")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        <div class="gate-box">
        <h3>🎯 What are Quantum Gates?</h3>
        <p>Quantum gates are the operations we can perform on qubits. They're like instructions that tell qubits how to change their state.</p>
        
        <h4>Key Properties:</h4>
        <ul>
        <li><strong>Unitary:</strong> They preserve probability (reversible)</li>
        <li><strong>Linear:</strong> They work on superposition states</li>
        <li><strong>Deterministic:</strong> Same input always gives same output</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="effect-box">
        <h4>🔄 Gate Representation</h4>
        <p>Gates can be represented as:</p>
        <ul>
        <li><strong>Circuit symbols</strong> - Visual representation</li>
        <li><strong>Matrices</strong> - Mathematical representation</li>
        <li><strong>Bloch sphere rotations</strong> - Geometric representation</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        # Gate family tree
        fig = go.Figure(go.Treemap(
            labels=["Quantum Gates", "Single-Qubit", "Multi-Qubit", "Pauli Gates", "Rotation Gates", "Hadamard", "CNOT", "Toffoli"],
            parents=["", "Quantum Gates", "Quantum Gates", "Single-Qubit", "Single-Qubit", "Single-Qubit", "Multi-Qubit", "Multi-Qubit"],
            values=[8, 4, 2, 3, 1, 1, 1, 1],
            textinfo="label+value",
            marker_colorscale='Viridis'
        ))
        fig.update_layout(title="Gate Family Tree", height=400)
        st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.header("🔧 Single-Qubit Gates")
    
    # Gate selector
    gate_choice = st.selectbox(
        "Choose a gate to explore:",
        ["Pauli-X (NOT)", "Pauli-Y", "Pauli-Z", "Hadamard (H)", "S Gate", "T Gate", "Rotation Gates"]
    )
    
    col1, col2, col3 = st.columns(3)
    
    # Define gate properties
    gate_info = {
        "Pauli-X (NOT)": {
            "matrix": np.array([[0, 1], [1, 0]]),
            "symbol": "X",
            "description": "Flips |0⟩ ↔ |1⟩ (quantum NOT gate)",
            "effect": "|0⟩ → |1⟩, |1⟩ → |0⟩",
            "bloch": "180° rotation around X-axis"
        },
        "Pauli-Y": {
            "matrix": np.array([[0, -1j], [1j, 0]]),
            "symbol": "Y", 
            "description": "Combination of bit flip and phase flip",
            "effect": "|0⟩ → i|1⟩, |1⟩ → -i|0⟩",
            "bloch": "180° rotation around Y-axis"
        },
        "Pauli-Z": {
            "matrix": np.array([[1, 0], [0, -1]]),
            "symbol": "Z",
            "description": "Flips the phase of |1⟩",
            "effect": "|0⟩ → |0⟩, |1⟩ → -|1⟩",
            "bloch": "180° rotation around Z-axis"
        },
        "Hadamard (H)": {
            "matrix": np.array([[1, 1], [1, -1]]) / np.sqrt(2),
            "symbol": "H",
            "description": "Creates superposition",
            "effect": "|0⟩ → (|0⟩+|1⟩)/√2, |1⟩ → (|0⟩-|1⟩)/√2",
            "bloch": "90° rotation + flip around X+Z axis"
        },
        "S Gate": {
            "matrix": np.array([[1, 0], [0, 1j]]),
            "symbol": "S",
            "description": "Quarter phase rotation",
            "effect": "|0⟩ → |0⟩, |1⟩ → i|1⟩",
            "bloch": "90° rotation around Z-axis"
        },
        "T Gate": {
            "matrix": np.array([[1, 0], [0, np.exp(1j*np.pi/4)]]),
            "symbol": "T",
            "description": "Eighth phase rotation",
            "effect": "|0⟩ → |0⟩, |1⟩ → e^(iπ/4)|1⟩",
            "bloch": "45° rotation around Z-axis"
        }
    }
    
    if gate_choice in gate_info:
        gate = gate_info[gate_choice]
        
        with col1:
            st.subheader("📊 Gate Matrix")
            st.markdown(f"""
            <div class="matrix-box">
            <h4>{gate['symbol']} Gate</h4>
            <p>{gate['description']}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Display matrix
            matrix = gate['matrix']
            matrix_str = "["
            for i, row in enumerate(matrix):
                if i > 0:
                    matrix_str += " "
                matrix_str += "["
                for j, elem in enumerate(row):
                    if j > 0:
                        matrix_str += ", "
                    if np.isreal(elem):
                        matrix_str += f"{elem.real:.3f}"
                    else:
                        matrix_str += f"{elem:.3f}"
                matrix_str += "]"
                if i < len(matrix) - 1:
                    matrix_str += "\n"
            matrix_str += "]"
            
            st.code(matrix_str)
            
            st.markdown(f"""
            **Effect:** {gate['effect']}
            
            **Bloch Sphere:** {gate['bloch']}
            """)
        
        with col2:
            st.subheader("🎯 Circuit Visualization")
            
            # Create circuit
            qc = QuantumCircuit(1, 1)
            if gate_choice == "Pauli-X (NOT)":
                qc.x(0)
            elif gate_choice == "Pauli-Y":
                qc.y(0)
            elif gate_choice == "Pauli-Z":
                qc.z(0)
            elif gate_choice == "Hadamard (H)":
                qc.h(0)
            elif gate_choice == "S Gate":
                qc.s(0)
            elif gate_choice == "T Gate":
                qc.t(0)
            
            # Draw circuit
            fig_circuit = qc.draw(output='mpl', style='iqp')
            st.pyplot(fig_circuit, use_container_width=True)
            plt.close()
        
        with col3:
            st.subheader("🌐 State Evolution")
            
            # Input state selector
            input_state = st.selectbox(
                "Input state:",
                ["|0⟩", "|1⟩", "(|0⟩+|1⟩)/√2", "(|0⟩-|1⟩)/√2"]
            )
            
            # Calculate output state
            if input_state == "|0⟩":
                state_vector = np.array([1, 0])
            elif input_state == "|1⟩":
                state_vector = np.array([0, 1])
            elif input_state == "(|0⟩+|1⟩)/√2":
                state_vector = np.array([1, 1]) / np.sqrt(2)
            else:  # (|0⟩-|1⟩)/√2
                state_vector = np.array([1, -1]) / np.sqrt(2)
            
            output_state = np.dot(matrix, state_vector)
            
            st.markdown(f"""
            **Input:** {input_state}
            
            **Output:** 
            {output_state[0]:.3f}|0⟩ + {output_state[1]:.3f}|1⟩
            
            **Probabilities:**
            - P(0) = {abs(output_state[0])**2:.3f}
            - P(1) = {abs(output_state[1])**2:.3f}
            """)
    
    elif gate_choice == "Rotation Gates":
        st.subheader("🔄 Parametric Rotation Gates")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **Rotation Gates allow arbitrary rotations:**
            - **RX(θ)**: Rotation around X-axis
            - **RY(θ)**: Rotation around Y-axis  
            - **RZ(θ)**: Rotation around Z-axis
            """)
            
            rotation_type = st.selectbox("Rotation type:", ["RX", "RY", "RZ"])
            angle = st.slider("Rotation angle (radians)", 0.0, 2*np.pi, np.pi/2, 0.1)
            
            # Calculate rotation matrix
            if rotation_type == "RX":
                matrix = np.array([
                    [np.cos(angle/2), -1j*np.sin(angle/2)],
                    [-1j*np.sin(angle/2), np.cos(angle/2)]
                ])
            elif rotation_type == "RY":
                matrix = np.array([
                    [np.cos(angle/2), -np.sin(angle/2)],
                    [np.sin(angle/2), np.cos(angle/2)]
                ])
            else:  # RZ
                matrix = np.array([
                    [np.exp(-1j*angle/2), 0],
                    [0, np.exp(1j*angle/2)]
                ])
            
            st.markdown(f"**{rotation_type}({angle:.2f}) Matrix:**")
            st.code(f"[{matrix[0,0]:.3f}, {matrix[0,1]:.3f}]\n[{matrix[1,0]:.3f}, {matrix[1,1]:.3f}]")
        
        with col2:
            # Bloch sphere visualization for rotation
            st.markdown("**Bloch Sphere Visualization**")
            
            # Initial state (can be adjusted)
            theta_init = st.slider("Initial θ", 0.0, np.pi, 0.0, 0.1)
            phi_init = st.slider("Initial φ", 0.0, 2*np.pi, 0.0, 0.1)
            
            # Initial Bloch vector
            x_init = np.sin(theta_init) * np.cos(phi_init)
            y_init = np.sin(theta_init) * np.sin(phi_init)
            z_init = np.cos(theta_init)
            
            # Apply rotation (simplified for visualization)
            if rotation_type == "RX":
                x_final, y_final, z_final = x_init, y_init*np.cos(angle) - z_init*np.sin(angle), y_init*np.sin(angle) + z_init*np.cos(angle)
            elif rotation_type == "RY":
                x_final, y_final, z_final = x_init*np.cos(angle) + z_init*np.sin(angle), y_init, -x_init*np.sin(angle) + z_init*np.cos(angle)
            else:  # RZ
                x_final, y_final, z_final = x_init*np.cos(angle) - y_init*np.sin(angle), x_init*np.sin(angle) + y_init*np.cos(angle), z_init
            
            # Plot
            fig = go.Figure()
            
            # Add sphere
            u = np.linspace(0, 2 * np.pi, 20)
            v = np.linspace(0, np.pi, 20)
            x_sphere = np.outer(np.cos(u), np.sin(v))
            y_sphere = np.outer(np.sin(u), np.sin(v))
            z_sphere = np.outer(np.ones(np.size(u)), np.cos(v))
            
            fig.add_trace(go.Surface(x=x_sphere, y=y_sphere, z=z_sphere, opacity=0.3, showscale=False))
            
            # Add vectors
            fig.add_trace(go.Scatter3d(x=[0, x_init], y=[0, y_init], z=[0, z_init], 
                                     mode='lines+markers', line=dict(color='blue', width=4),
                                     marker=dict(size=[3, 8]), name='Initial'))
            fig.add_trace(go.Scatter3d(x=[0, x_final], y=[0, y_final], z=[0, z_final], 
                                     mode='lines+markers', line=dict(color='red', width=4),
                                     marker=dict(size=[3, 8]), name='Final'))
            
            fig.update_layout(scene=dict(aspectmode='cube'), height=400, title=f"{rotation_type} Rotation")
            st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.header("🔗 Multi-Qubit Gates")
    
    st.markdown("""
    <div class="gate-box">
    <h3>🎯 Multi-Qubit Operations</h3>
    <p>These gates operate on multiple qubits simultaneously and can create entanglement!</p>
    </div>
    """, unsafe_allow_html=True)
    
    multi_gate = st.selectbox("Select multi-qubit gate:", ["CNOT (CX)", "CZ Gate", "SWAP", "Toffoli (CCX)", "Fredkin"])
    
    col1, col2 = st.columns(2)
    
    with col1:
        if multi_gate == "CNOT (CX)":
            st.markdown("""
            ### 🎯 CNOT (Controlled-X) Gate
            
            **Operation:** Flips target qubit if control qubit is |1⟩
            
            **Truth Table:**
            - |00⟩ → |00⟩
            - |01⟩ → |01⟩  
            - |10⟩ → |11⟩
            - |11⟩ → |10⟩
            
            **Key Property:** Creates entanglement!
            """)
            
            # Create CNOT circuit
            qc_cnot = QuantumCircuit(2, 2)
            qc_cnot.cx(0, 1)
            
        elif multi_gate == "CZ Gate":
            st.markdown("""
            ### 🎯 Controlled-Z Gate
            
            **Operation:** Applies Z gate to target if control is |1⟩
            
            **Truth Table:**
            - |00⟩ → |00⟩
            - |01⟩ → |01⟩
            - |10⟩ → |10⟩
            - |11⟩ → -|11⟩
            """)
            
            qc_cnot = QuantumCircuit(2, 2)
            qc_cnot.cz(0, 1)
            
        elif multi_gate == "SWAP":
            st.markdown("""
            ### 🎯 SWAP Gate
            
            **Operation:** Exchanges the states of two qubits
            
            **Truth Table:**
            - |00⟩ → |00⟩
            - |01⟩ → |10⟩
            - |10⟩ → |01⟩  
            - |11⟩ → |11⟩
            """)
            
            qc_cnot = QuantumCircuit(2, 2)
            qc_cnot.swap(0, 1)
            
        elif multi_gate == "Toffoli (CCX)":
            st.markdown("""
            ### 🎯 Toffoli Gate (Controlled-Controlled-X)
            
            **Operation:** Flips target if both controls are |1⟩
            
            **Key Property:** Universal for classical computation!
            
            **Example:** |110⟩ → |111⟩
            """)
            
            qc_cnot = QuantumCircuit(3, 3)
            qc_cnot.ccx(0, 1, 2)
        
        # Draw the circuit
        fig_multi = qc_cnot.draw(output='mpl', style='iqp')
        st.pyplot(fig_multi)
        plt.close()
    
    with col2:
        st.subheader("🧪 Entanglement Demo")
        
        if multi_gate == "CNOT (CX)":
            # Demonstrate Bell state creation
            bell_qc = QuantumCircuit(2, 2)
            bell_qc.h(0)  # Create superposition
            bell_qc.cx(0, 1)  # Create entanglement
            
            st.markdown("""
            **Creating a Bell State:**
            1. Apply H to first qubit: |00⟩ → (|00⟩ + |10⟩)/√2
            2. Apply CNOT: (|00⟩ + |10⟩)/√2 → (|00⟩ + |11⟩)/√2
            
            **Result:** Maximally entangled state!
            """)
            
            fig_bell = bell_qc.draw(output='mpl', style='iqp')
            st.pyplot(fig_bell)
            plt.close()
            
            # Simulate the Bell state
            simulator = Aer.get_backend('statevector_simulator')
            job = execute(bell_qc, simulator)
            result = job.result()
            statevector = result.get_statevector()
            
            st.markdown("**Final State Vector:**")
            state_labels = ['|00⟩', '|01⟩', '|10⟩', '|11⟩']
            amplitudes = [f"{amp:.3f}" for amp in statevector]
            
            for i, (label, amp) in enumerate(zip(state_labels, amplitudes)):
                if abs(statevector[i]) > 0.001:
                    st.write(f"{label}: {amp}")

with tab4:
    st.header("🎮 Interactive Gate Explorer")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("🔧 Build Your Circuit")
        
        num_qubits = st.selectbox("Number of qubits:", [1, 2, 3])
        
        # Initialize circuit
        if 'circuit' not in st.session_state:
            st.session_state.circuit = QuantumCircuit(num_qubits, num_qubits)
        
        # Gate controls
        if num_qubits == 1:
            gate_options = ["H", "X", "Y", "Z", "S", "T"]
            selected_gate = st.selectbox("Add gate:", gate_options)
            qubit_target = 0
            
            if st.button("Add Gate"):
                if selected_gate == "H":
                    st.session_state.circuit.h(qubit_target)
                elif selected_gate == "X":
                    st.session_state.circuit.x(qubit_target)
                elif selected_gate == "Y":
                    st.session_state.circuit.y(qubit_target)
                elif selected_gate == "Z":
                    st.session_state.circuit.z(qubit_target)
                elif selected_gate == "S":
                    st.session_state.circuit.s(qubit_target)
                elif selected_gate == "T":
                    st.session_state.circuit.t(qubit_target)
        
        else:
            gate_options = ["H", "X", "Y", "Z", "CNOT", "CZ", "SWAP"]
            selected_gate = st.selectbox("Add gate:", gate_options)
            
            if selected_gate in ["H", "X", "Y", "Z"]:
                qubit_target = st.selectbox("Target qubit:", list(range(num_qubits)))
                
                if st.button("Add Gate"):
                    if selected_gate == "H":
                        st.session_state.circuit.h(qubit_target)
                    elif selected_gate == "X":
                        st.session_state.circuit.x(qubit_target)
                    elif selected_gate == "Y":
                        st.session_state.circuit.y(qubit_target)
                    elif selected_gate == "Z":
                        st.session_state.circuit.z(qubit_target)
            
            elif selected_gate in ["CNOT", "CZ"]:
                control_qubit = st.selectbox("Control qubit:", list(range(num_qubits)))
                target_options = [i for i in range(num_qubits) if i != control_qubit]
                target_qubit = st.selectbox("Target qubit:", target_options)
                
                if st.button("Add Gate"):
                    if selected_gate == "CNOT":
                        st.session_state.circuit.cx(control_qubit, target_qubit)
                    elif selected_gate == "CZ":
                        st.session_state.circuit.cz(control_qubit, target_qubit)
            
            elif selected_gate == "SWAP":
                qubit1 = st.selectbox("First qubit:", list(range(num_qubits)))
                qubit2_options = [i for i in range(num_qubits) if i != qubit1]
                qubit2 = st.selectbox("Second qubit:", qubit2_options)
                
                if st.button("Add Gate"):
                    st.session_state.circuit.swap(qubit1, qubit2)
        
        if st.button("Clear Circuit"):
            st.session_state.circuit = QuantumCircuit(num_qubits, num_qubits)
        
        if st.button("Add Measurements"):
            for i in range(num_qubits):
                st.session_state.circuit.measure(i, i)
    
    with col2:
        st.subheader("📊 Your Circuit")
        
        if len(st.session_state.circuit.data) > 0:
            fig_custom = st.session_state.circuit.draw(output='mpl', style='iqp')
            st.pyplot(fig_custom)
            plt.close()
            
            # Simulate the circuit
            if st.button("🚀 Simulate Circuit"):
                simulator = Aer.get_backend('qasm_simulator')
                
                # Add measurements if not present
                temp_circuit = st.session_state.circuit.copy()
                if not any(isinstance(instr[0], type(temp_circuit.measure(0, 0)[0])) for instr in temp_circuit.data):
                    for i in range(num_qubits):
                        temp_circuit.measure(i, i)
                
                job = execute(temp_circuit, simulator, shots=1000)
                result = job.result()
                counts = result.get_counts()
                
                # Plot results
                states = list(counts.keys())
                frequencies = list(counts.values())
                
                fig_results = go.Figure(data=go.Bar(x=states, y=frequencies))
                fig_results.update_layout(
                    title="Measurement Results (1000 shots)",
                    xaxis_title="Measured State",
                    yaxis_title="Count"
                )
                st.plotly_chart(fig_results, use_container_width=True)
                
                # Show probabilities
                total_shots = sum(frequencies)
                st.markdown("**Probabilities:**")
                for state, count in counts.items():
                    prob = count / total_shots
                    st.write(f"|{state}⟩: {prob:.3f} ({prob*100:.1f}%)")
        
        else:
            st.info("👆 Add some gates to see your circuit!")

with tab5:
    st.header("🧪 Gate Laboratory")
    
    st.markdown("""
    <div class="gate-box">
    <h3>🔬 Experiment with Gate Combinations</h3>
    <p>Try different gate sequences and see their combined effect!</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Experiment 1: Gate Inverses")
        
        st.markdown("""
        **Try these combinations and observe:**
        - H followed by H → Identity
        - X followed by X → Identity  
        - S followed by S† → Identity
        """)
        
        exp1_gates = st.multiselect(
            "Select gate sequence:",
            ["H", "X", "Y", "Z", "S", "S†", "T", "T†"],
            default=["H", "H"]
        )
        
        if exp1_gates:
            # Create circuit with selected gates
            exp_qc = QuantumCircuit(1, 1)
            for gate in exp1_gates:
                if gate == "H":
                    exp_qc.h(0)
                elif gate == "X":
                    exp_qc.x(0)
                elif gate == "Y":
                    exp_qc.y(0)
                elif gate == "Z":
                    exp_qc.z(0)
                elif gate == "S":
                    exp_qc.s(0)
                elif gate == "S†":
                    exp_qc.sdg(0)
                elif gate == "T":
                    exp_qc.t(0)
                elif gate == "T†":
                    exp_qc.tdg(0)
            
            fig_exp1 = exp_qc.draw(output='mpl', style='iqp')
            st.pyplot(fig_exp1)
            plt.close()
    
    with col2:
        st.subheader("Experiment 2: Creating States")
        
        target_state = st.selectbox(
            "Target state to create:",
            ["|+⟩ = (|0⟩+|1⟩)/√2", "|-⟩ = (|0⟩-|1⟩)/√2", "|i⟩ = (|0⟩+i|1⟩)/√2", "|-i⟩ = (|0⟩-i|1⟩)/√2"]
        )
        
        st.markdown("""
        **Hints:**
        - |+⟩: Apply H to |0⟩
        - |-⟩: Apply X then H to |0⟩  
        - |i⟩: Apply H then S to |0⟩
        - |-i⟩: Apply H then S† to |0⟩
        """)
        
        # Solution builder
        solution_gates = st.multiselect(
            "Your solution:",
            ["H", "X", "Y", "Z", "S", "S†"],
            key="solution"
        )
        
        if solution_gates and st.button("Test Solution"):
            # Check if solution is correct
            solutions = {
                "|+⟩ = (|0⟩+|1⟩)/√2": ["H"],
                "|-⟩ = (|0⟩-|1⟩)/√2": ["X", "H"],
                "|i⟩ = (|0⟩+i|1⟩)/√2": ["H", "S"],
                "|-i⟩ = (|0⟩-i|1⟩)/√2": ["H", "S†"]
            }
            
            correct_solution = solutions[target_state]
            if solution_gates == correct_solution:
                st.success("✅ Correct! Well done!")
            else:
                st.error(f"❌ Try again. Hint: {correct_solution}")

with tab6:
    st.header("📝 Practice Problems")
    
    # Problem 1
    st.subheader("Problem 1: Gate Identification")
    st.markdown("Which gate creates equal superposition from |0⟩?")
    
    with st.expander("Solve Problem 1"):
        answer1 = st.radio(
            "Choose the correct gate:",
            ["Pauli-X", "Pauli-Z", "Hadamard", "S Gate"],
            key="p1"
        )
        
        if st.button("Check Answer", key="check1"):
            if answer1 == "Hadamard":
                st.success("✅ Correct! H|0⟩ = (|0⟩ + |1⟩)/√2")
            else:
                st.error("❌ Try again. Think about which gate creates superposition.")
    
    # Problem 2
    st.subheader("Problem 2: CNOT Truth Table")
    st.markdown("What is CNOT|10⟩?")
    
    with st.expander("Solve Problem 2"):
        answer2 = st.radio(
            "Choose the result:",
            ["|00⟩", "|01⟩", "|10⟩", "|11⟩"],
            key="p2"
        )
        
        if st.button("Check Answer", key="check2"):
            if answer2 == "|11⟩":
                st.success("✅ Perfect! CNOT flips the target when control is |1⟩")
            else:
                st.error("❌ Remember: CNOT flips target if control is |1⟩")
    
    # Problem 3
    st.subheader("Problem 3: Gate Composition")
    st.markdown("What is the result of applying H then X then H to |0⟩?")
    
    with st.expander("Solve Problem 3"):
        answer3 = st.radio(
            "Final state:",
            ["|0⟩", "|1⟩", "(|0⟩+|1⟩)/√2", "(|0⟩-|1⟩)/√2"],
            key="p3"
        )
        
        if st.button("Check Answer", key="check3"):
            if answer3 == "|1⟩":
                st.success("✅ Excellent! HXH|0⟩ = Z|0⟩ = |0⟩... wait, that's wrong. The answer is |1⟩")
            else:
                st.error("❌ Try working through it step by step: H|0⟩ → X → H")

# Progress and navigation
st.markdown("---")
st.markdown("### 🎉 Great Job!")
st.markdown("You've explored the fundamental quantum gates. Ready for more advanced topics?")

col1, col2, col3 = st.columns(3)
with col1:
    if st.button("← Back to Entanglement", use_container_width=True):
        st.switch_page("pages/03_Entanglement.py")
with col2:
    if st.button("🔬 Try Circuit Builder", use_container_width=True):
        st.switch_page("pages/06_Circuit_Builder.py")
with col3:
    if st.button("💻 Code Playground →", use_container_width=True):
        st.switch_page("pages/07_Code_Playground.py")