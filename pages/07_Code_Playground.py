import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from qiskit import QuantumCircuit, execute, Aer, transpile
from qiskit.visualization import plot_histogram, circuit_drawer
import plotly.express as px
from streamlit_ace import st_ace
import traceback
import sys
from io import StringIO

st.set_page_config(page_title="Code Playground", page_icon="💻", layout="wide")

# Custom CSS
st.markdown("""
<style>
    .code-playground {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        padding: 2rem;
        border-radius: 15px;
        margin: 1rem 0;
        border: 2px solid #6c757d;
    }
    .example-box {
        background: #e8f5e8;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 4px solid #28a745;
    }
    .output-box {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
        border: 1px solid #dee2e6;
        font-family: monospace;
    }
    .error-box {
        background: #f8d7da;
        color: #721c24;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
        border: 1px solid #f5c6cb;
    }
    .tutorial-box {
        background: #fff3cd;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 4px solid #ffc107;
    }
</style>
""", unsafe_allow_html=True)

st.title("💻 Quantum Code Playground")
st.markdown("*Write, experiment, and learn with interactive quantum computing code*")

# Initialize session state
if 'code_output' not in st.session_state:
    st.session_state.code_output = ""
if 'execution_history' not in st.session_state:
    st.session_state.execution_history = []

# Sidebar for code templates and examples
with st.sidebar:
    st.markdown("## 📚 Code Examples")
    
    example_choice = st.selectbox(
        "Choose an example:",
        [
            "Hello Quantum World",
            "Superposition Example", 
            "Bell State Creation",
            "Quantum Teleportation",
            "Grover's Algorithm",
            "Quantum Fourier Transform",
            "Random Number Generator",
            "Deutsch-Jozsa Algorithm",
            "Shor's Algorithm (simplified)",
            "Quantum Error Correction"
        ]
    )
    
    # Code examples dictionary
    examples = {
        "Hello Quantum World": """# Your first quantum program!
from qiskit import QuantumCircuit, execute, Aer
import matplotlib.pyplot as plt

# Create a quantum circuit with 1 qubit and 1 classical bit
qc = QuantumCircuit(1, 1)

# Add a Hadamard gate to create superposition
qc.h(0)

# Measure the qubit
qc.measure(0, 0)

# Draw the circuit
print("Your first quantum circuit:")
print(qc.draw())

# Simulate the circuit
backend = Aer.get_backend('qasm_simulator')
job = execute(qc, backend, shots=1000)
result = job.result()
counts = result.get_counts()

print(f"\\nResults: {counts}")
print("Congratulations! You've run your first quantum program! 🎉")""",

        "Superposition Example": """# Exploring quantum superposition
from qiskit import QuantumCircuit, execute, Aer
import numpy as np

# Create different superposition states
def create_superposition_state(angle_degrees):
    qc = QuantumCircuit(1, 1)
    
    # Convert degrees to radians
    angle_rad = np.radians(angle_degrees)
    
    # Create custom superposition using RY rotation
    qc.ry(angle_rad, 0)
    qc.measure(0, 0)
    
    return qc

# Try different angles
angles = [0, 30, 45, 60, 90]

print("Superposition experiment results:")
print("Angle | P(|0⟩) | P(|1⟩)")
print("------|--------|-------")

backend = Aer.get_backend('qasm_simulator')

for angle in angles:
    qc = create_superposition_state(angle)
    job = execute(qc, backend, shots=1000)
    result = job.result()
    counts = result.get_counts()
    
    prob_0 = counts.get('0', 0) / 1000
    prob_1 = counts.get('1', 0) / 1000
    
    print(f"{angle:3d}°  | {prob_0:.3f}  | {prob_1:.3f}")

print("\\nNotice how the probabilities change with angle!")""",

        "Bell State Creation": """# Creating and analyzing Bell states
from qiskit import QuantumCircuit, execute, Aer
from qiskit.visualization import plot_histogram

def create_bell_state():
    # Create quantum circuit with 2 qubits
    qc = QuantumCircuit(2, 2)
    
    # Create Bell state |Φ+⟩ = (|00⟩ + |11⟩)/√2
    qc.h(0)        # Put first qubit in superposition
    qc.cx(0, 1)    # Entangle with second qubit
    
    # Measure both qubits
    qc.measure_all()
    
    return qc

# Create and simulate Bell state
bell_circuit = create_bell_state()

print("Bell State Circuit:")
print(bell_circuit.draw())

# Run simulation
backend = Aer.get_backend('qasm_simulator')
job = execute(bell_circuit, backend, shots=1000)
result = job.result()
counts = result.get_counts()

print(f"\\nMeasurement Results: {counts}")

# Analyze the results
print("\\nAnalysis:")
total_shots = sum(counts.values())
for state, count in counts.items():
    prob = count / total_shots
    print(f"P(|{state}⟩) = {prob:.3f} ({prob*100:.1f}%)")

print("\\nNotice: Only |00⟩ and |11⟩ appear - this shows entanglement!")""",

        "Quantum Teleportation": """# Quantum teleportation protocol
from qiskit import QuantumCircuit, execute, Aer
import numpy as np

def quantum_teleportation():
    # Create circuit with 3 qubits and 3 classical bits
    # Qubit 0: Alice's unknown state to teleport
    # Qubit 1: Alice's part of entangled pair
    # Qubit 2: Bob's part of entangled pair
    qc = QuantumCircuit(3, 3)
    
    # Step 1: Prepare unknown state on qubit 0 (Alice)
    # Let's create |+⟩ state as example
    qc.h(0)
    
    # Step 2: Create entangled pair between qubits 1 and 2
    qc.h(1)
    qc.cx(1, 2)
    
    # Step 3: Alice performs Bell measurement
    qc.cx(0, 1)
    qc.h(0)
    qc.measure(0, 0)
    qc.measure(1, 1)
    
    # Step 4: Bob applies corrections based on Alice's results
    qc.cx(1, 2)   # Conditional X gate
    qc.cz(0, 2)   # Conditional Z gate
    
    # Step 5: Measure Bob's qubit
    qc.measure(2, 2)
    
    return qc

# Run teleportation protocol
teleport_circuit = quantum_teleportation()

print("Quantum Teleportation Circuit:")
print(teleport_circuit.draw())

# Simulate multiple times to see the protocol working
backend = Aer.get_backend('qasm_simulator')
job = execute(teleport_circuit, backend, shots=1000)
result = job.result()
counts = result.get_counts()

print(f"\\nTeleportation Results: {counts}")
print("\\nThe final qubit (rightmost bit) should show the teleported state!")""",

        "Grover's Algorithm": """# Grover's search algorithm for 2 qubits
from qiskit import QuantumCircuit, execute, Aer
import numpy as np

def grovers_algorithm(marked_item):
    \"\"\"
    Grover's algorithm to find a marked item in unsorted database
    marked_item: the item we're looking for (0, 1, 2, or 3)
    \"\"\"
    
    # Create circuit with 2 qubits
    qc = QuantumCircuit(2, 2)
    
    # Step 1: Initialize superposition
    qc.h([0, 1])
    
    # Step 2: Apply oracle (marks the target item)
    if marked_item == 0:  # |00⟩
        qc.cz(0, 1)
        qc.z(0)
        qc.z(1)
    elif marked_item == 1:  # |01⟩
        qc.x(0)
        qc.cz(0, 1)
        qc.z(1)
        qc.x(0)
    elif marked_item == 2:  # |10⟩
        qc.x(1)
        qc.cz(0, 1)
        qc.z(0)
        qc.x(1)
    elif marked_item == 3:  # |11⟩
        qc.cz(0, 1)
    
    # Step 3: Apply diffusion operator
    qc.h([0, 1])
    qc.z([0, 1])
    qc.cz(0, 1)
    qc.h([0, 1])
    
    # Step 4: Measure
    qc.measure_all()
    
    return qc

# Search for item 2 (binary: 10)
target = 2
grover_circuit = grovers_algorithm(target)

print(f"Grover's Algorithm searching for item {target} (binary: {target:02b})")
print("\\nCircuit:")
print(grover_circuit.draw())

# Simulate
backend = Aer.get_backend('qasm_simulator')
job = execute(grover_circuit, backend, shots=1000)
result = job.result()
counts = result.get_counts()

print(f"\\nSearch Results: {counts}")

# Analyze results
max_count = max(counts.values())
for state, count in counts.items():
    if count == max_count:
        found_item = int(state, 2)
        print(f"\\nFound item: {found_item} (binary: {state})")
        if found_item == target:
            print("✅ Search successful!")
        else:
            print("❌ Search failed")""",

        "Random Number Generator": """# Quantum random number generator
from qiskit import QuantumCircuit, execute, Aer
import numpy as np

def quantum_random_int(num_bits):
    \"\"\"Generate a random integer using quantum superposition\"\"\"
    
    # Create circuit with specified number of qubits
    qc = QuantumCircuit(num_bits, num_bits)
    
    # Put all qubits in superposition
    for i in range(num_bits):
        qc.h(i)
    
    # Measure all qubits
    qc.measure_all()
    
    return qc

def run_quantum_rng(num_bits, num_samples=10):
    \"\"\"Run the quantum RNG multiple times\"\"\"
    
    qc = quantum_random_int(num_bits)
    backend = Aer.get_backend('qasm_simulator')
    
    print(f"Quantum Random Number Generator ({num_bits} bits)")
    print("=" * 40)
    print("Sample | Binary | Decimal")
    print("-------|--------|--------")
    
    random_numbers = []
    
    for i in range(num_samples):
        job = execute(qc, backend, shots=1)
        result = job.result()
        counts = result.get_counts()
        
        # Get the random bit string
        random_bits = list(counts.keys())[0]
        random_decimal = int(random_bits, 2)
        random_numbers.append(random_decimal)
        
        print(f"  {i+1:2d}   |  {random_bits}  |   {random_decimal:2d}")
    
    # Statistics
    print(f"\\nStatistics:")
    print(f"Range: 0 to {2**num_bits - 1}")
    print(f"Average: {np.mean(random_numbers):.2f}")
    print(f"Unique values: {len(set(random_numbers))}/{num_samples}")
    
    return random_numbers

# Generate 4-bit random numbers
random_nums = run_quantum_rng(4, 15)

print("\\nThese are truly random numbers generated by quantum mechanics!")"""
    }
    
    if st.button("Load Example", use_container_width=True):
        if 'selected_example' not in st.session_state:
            st.session_state.selected_example = examples[example_choice]
        else:
            st.session_state.selected_example = examples[example_choice]
    
    st.markdown("---")
    
    # Quick reference
    st.markdown("## 📖 Quick Reference")
    
    with st.expander("Common Quantum Gates"):
        st.markdown("""
        ```python
        # Single qubit gates
        qc.h(0)      # Hadamard
        qc.x(0)      # Pauli-X (NOT)
        qc.y(0)      # Pauli-Y
        qc.z(0)      # Pauli-Z
        qc.s(0)      # S gate
        qc.t(0)      # T gate
        
        # Rotation gates
        qc.rx(angle, 0)  # X rotation
        qc.ry(angle, 0)  # Y rotation
        qc.rz(angle, 0)  # Z rotation
        
        # Multi-qubit gates
        qc.cx(0, 1)     # CNOT
        qc.cz(0, 1)     # Controlled-Z
        qc.swap(0, 1)   # SWAP
        qc.ccx(0, 1, 2) # Toffoli
        ```
        """)
    
    with st.expander("Circuit Operations"):
        st.markdown("""
        ```python
        # Create circuit
        qc = QuantumCircuit(n_qubits, n_bits)
        
        # Add measurement
        qc.measure(qubit, bit)
        qc.measure_all()
        
        # Simulate
        backend = Aer.get_backend('qasm_simulator')
        job = execute(qc, backend, shots=1000)
        result = job.result()
        counts = result.get_counts()
        
        # Visualize
        print(qc.draw())
        ```
        """)
    
    with st.expander("Useful Libraries"):
        st.markdown("""
        ```python
        # Already imported for you:
        import numpy as np
        import matplotlib.pyplot as plt
        import plotly.graph_objects as go
        from qiskit import QuantumCircuit, execute, Aer
        from qiskit.visualization import plot_histogram
        ```
        """)

# Main content area with tabs
tab1, tab2, tab3, tab4 = st.tabs(["💻 Code Editor", "📊 Results", "🎓 Tutorials", "📝 Challenges"])

with tab1:
    st.markdown("""
    <div class="code-playground">
    <h3>🚀 Interactive Quantum Code Editor</h3>
    <p>Write your quantum computing code below. All necessary libraries are already imported!</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Code editor
    default_code = """# Welcome to the Quantum Code Playground!
from qiskit import QuantumCircuit, execute, Aer

# Create your first quantum circuit
qc = QuantumCircuit(1, 1)

# Add gates here
qc.h(0)  # Hadamard gate for superposition

# Add measurement
qc.measure(0, 0)

# Draw the circuit
print("Your quantum circuit:")
print(qc.draw())

# Simulate the circuit
backend = Aer.get_backend('qasm_simulator')
job = execute(qc, backend, shots=1000)
result = job.result()
counts = result.get_counts()

print(f"\\nResults: {counts}")

# Try modifying the code above!
# Hint: Add more gates or change the number of qubits"""
    
    # Use the example if one was selected
    if 'selected_example' in st.session_state:
        code_content = st.session_state.selected_example
    else:
        code_content = default_code
    
    code = st_ace(
        value=code_content,
        language='python',
        theme='monokai',
        key="quantum_code_editor",
        height=400,
        auto_update=False,
        font_size=14,
        tab_size=4,
        wrap=False,
        annotations=None
    )
    
    # Execution controls
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("▶️ Run Code", use_container_width=True):
            st.session_state.execute_code = True
    
    with col2:
        if st.button("🗑️ Clear Output", use_container_width=True):
            st.session_state.code_output = ""
    
    with col3:
        if st.button("💾 Save to History", use_container_width=True):
            if code.strip():
                st.session_state.execution_history.append({
                    'code': code,
                    'timestamp': str(np.datetime64('now'))
                })
                st.success("Code saved to history!")
    
    # Execute code if button was pressed
    if hasattr(st.session_state, 'execute_code') and st.session_state.execute_code:
        st.session_state.execute_code = False
        
        # Capture stdout
        old_stdout = sys.stdout
        sys.stdout = captured_output = StringIO()
        
        try:
            # Create a safe execution environment
            exec_globals = {
                '__builtins__': __builtins__,
                'np': np,
                'numpy': np,
                'plt': plt,
                'matplotlib': plt,
                'go': go,
                'plotly': go,
                'QuantumCircuit': QuantumCircuit,
                'execute': execute,
                'Aer': Aer,
                'transpile': transpile,
                'plot_histogram': plot_histogram,
                'circuit_drawer': circuit_drawer,
                'print': print,
                'range': range,
                'len': len,
                'str': str,
                'int': int,
                'float': float,
                'list': list,
                'dict': dict,
                'sum': sum,
                'max': max,
                'min': min,
                'enumerate': enumerate,
                'zip': zip
            }
            
            # Execute the code
            exec(code, exec_globals)
            
            # Get the output
            output = captured_output.getvalue()
            st.session_state.code_output = output if output else "Code executed successfully (no output)"
            
        except Exception as e:
            error_msg = f"Error: {str(e)}\n\nTraceback:\n{traceback.format_exc()}"
            st.session_state.code_output = error_msg
        
        finally:
            # Restore stdout
            sys.stdout = old_stdout

with tab2:
    st.header("📊 Execution Results")
    
    if st.session_state.code_output:
        # Check if it's an error
        if st.session_state.code_output.startswith("Error:"):
            st.markdown(f"""
            <div class="error-box">
            <h4>❌ Execution Error</h4>
            <pre>{st.session_state.code_output}</pre>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="output-box">
            <h4>✅ Output</h4>
            <pre>{st.session_state.code_output}</pre>
            </div>
            """, unsafe_allow_html=True)
        
        # Additional visualization capabilities
        st.markdown("---")
        st.markdown("### 📈 Additional Visualizations")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📊 Show Matplotlib Plots"):
                st.info("Matplotlib plots will appear automatically when you use plt.show() in your code!")
        
        with col2:
            if st.button("🔄 Clear All Results"):
                st.session_state.code_output = ""
                st.rerun()
    
    else:
        st.info("🚀 Run some code in the editor to see results here!")
    
    # Code execution history
    if st.session_state.execution_history:
        st.markdown("---")
        st.markdown("### 📚 Execution History")
        
        for i, entry in enumerate(reversed(st.session_state.execution_history[-5:])):  # Show last 5
            with st.expander(f"Code {len(st.session_state.execution_history) - i} - {entry['timestamp']}"):
                st.code(entry['code'], language='python')
                if st.button(f"Load Code {len(st.session_state.execution_history) - i}", key=f"load_history_{i}"):
                    st.session_state.selected_example = entry['code']
                    st.rerun()

with tab3:
    st.header("🎓 Quantum Programming Tutorials")
    
    tutorial_choice = st.selectbox(
        "Choose a tutorial:",
        [
            "Basic Circuit Construction",
            "Understanding Measurements",
            "Working with Superposition",
            "Creating Entanglement",
            "Quantum Gates Deep Dive",
            "Building Quantum Algorithms",
            "Error Handling and Debugging"
        ]
    )
    
    if tutorial_choice == "Basic Circuit Construction":
        st.markdown("""
        <div class="tutorial-box">
        <h3>📝 Tutorial: Basic Circuit Construction</h3>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        ### Step 1: Creating a Quantum Circuit
        
        Every quantum program starts with creating a circuit:
        
        ```python
        from qiskit import QuantumCircuit
        
        # Create circuit with 2 qubits and 2 classical bits
        qc = QuantumCircuit(2, 2)
        ```
        
        ### Step 2: Adding Gates
        
        Add quantum gates to manipulate qubits:
        
        ```python
        qc.h(0)      # Hadamard on qubit 0
        qc.cx(0, 1)  # CNOT from qubit 0 to 1
        ```
        
        ### Step 3: Measurement
        
        Measure quantum states to get classical results:
        
        ```python
        qc.measure(0, 0)  # Measure qubit 0 into bit 0
        qc.measure(1, 1)  # Measure qubit 1 into bit 1
        # Or use: qc.measure_all()
        ```
        
        ### Step 4: Visualization
        
        View your circuit:
        
        ```python
        print(qc.draw())
        ```
        
        ### 🏗️ Try It Yourself!
        Copy this template and modify it:
        """)
        
        tutorial_code = """# Basic quantum circuit template
from qiskit import QuantumCircuit, execute, Aer

# Step 1: Create circuit
qc = QuantumCircuit(2, 2)

# Step 2: Add gates (modify these!)
qc.h(0)        # Try changing this gate
qc.cx(0, 1)    # Try different target qubits

# Step 3: Add measurements
qc.measure_all()

# Step 4: Visualize
print("Circuit diagram:")
print(qc.draw())

# Step 5: Simulate
backend = Aer.get_backend('qasm_simulator')
job = execute(qc, backend, shots=1000)
result = job.result()
counts = result.get_counts()

print(f"\\nResults: {counts}")

# Experiment: Try adding more gates!"""
        
        if st.button("Load Tutorial Code"):
            st.session_state.selected_example = tutorial_code
            st.success("Tutorial code loaded! Switch to the Code Editor tab.")
    
    elif tutorial_choice == "Understanding Measurements":
        st.markdown("""
        <div class="tutorial-box">
        <h3>📏 Tutorial: Understanding Quantum Measurements</h3>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        ### The Measurement Postulate
        
        Measurement in quantum mechanics is probabilistic:
        - **Before measurement**: Qubit can be in superposition
        - **After measurement**: Qubit collapses to definite state (0 or 1)
        - **Probability**: Determined by the quantum state amplitudes
        
        ### Types of Measurements
        
        1. **Computational Basis Measurement** (most common)
        2. **Pauli Measurements** (X, Y, Z basis)
        3. **Custom Basis Measurements**
        
        ### Measurement in Code
        
        ```python
        # Add measurement to circuit
        qc.measure(qubit_index, classical_bit_index)
        
        # Measure all qubits
        qc.measure_all()
        
        # Run simulation to get results
        job = execute(qc, backend, shots=1000)
        counts = result.get_counts()
        ```
        
        ### Key Points:
        - More shots = more accurate statistics
        - Each shot gives one measurement outcome
        - Quantum randomness is fundamental, not just lack of knowledge
        """)
    
    elif tutorial_choice == "Working with Superposition":
        st.markdown("""
        <div class="tutorial-box">
        <h3>🌊 Tutorial: Quantum Superposition</h3>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        ### Creating Superposition
        
        The Hadamard gate creates equal superposition:
        
        ```python
        qc.h(0)  # |0⟩ → (|0⟩ + |1⟩)/√2
        ```
        
        ### Custom Superposition States
        
        Use rotation gates for custom superposition:
        
        ```python
        import numpy as np
        
        # Create |ψ⟩ = cos(θ/2)|0⟩ + sin(θ/2)|1⟩
        theta = np.pi/3  # 60 degrees
        qc.ry(theta, 0)
        ```
        
        ### Superposition Properties
        
        - **Coherence**: Superposition maintains phase relationships
        - **Interference**: Amplitudes can add or cancel
        - **Measurement**: Destroys superposition (collapse)
        
        ### Common Superposition States
        
        ```python
        # |+⟩ state: (|0⟩ + |1⟩)/√2
        qc.h(0)
        
        # |-⟩ state: (|0⟩ - |1⟩)/√2  
        qc.x(0)
        qc.h(0)
        
        # |i⟩ state: (|0⟩ + i|1⟩)/√2
        qc.h(0)
        qc.s(0)
        ```
        """)

with tab4:
    st.header("📝 Quantum Programming Challenges")
    
    challenge_level = st.selectbox(
        "Choose difficulty:",
        ["Beginner", "Intermediate", "Advanced"]
    )
    
    if challenge_level == "Beginner":
        st.markdown("""
        ### 🌱 Beginner Challenges
        
        #### Challenge 1: Create Your First Bell State
        **Goal**: Create the Bell state |Φ+⟩ = (|00⟩ + |11⟩)/√2
        
        **Hints**: 
        - Start with 2 qubits
        - Apply H gate to first qubit
        - Apply CNOT gate
        
        **Expected Result**: Only |00⟩ and |11⟩ should appear in measurements
        """)
        
        challenge1_code = """# Challenge 1: Bell State Creation
from qiskit import QuantumCircuit, execute, Aer

# TODO: Create a circuit with 2 qubits and 2 classical bits


# TODO: Apply Hadamard gate to first qubit


# TODO: Apply CNOT gate from first to second qubit


# TODO: Add measurements


# Simulate and check results
backend = Aer.get_backend('qasm_simulator')
job = execute(qc, backend, shots=1000)
result = job.result()
counts = result.get_counts()

print("Circuit:")
print(qc.draw())
print(f"\\nResults: {counts}")

# Check if you succeeded
success = ('00' in counts and '11' in counts and 
          '01' not in counts and '10' not in counts)
print(f"\\nSuccess: {success}")"""
        
        if st.button("Load Challenge 1"):
            st.session_state.selected_example = challenge1_code
    
    elif challenge_level == "Intermediate":
        st.markdown("""
        ### 🎯 Intermediate Challenges
        
        #### Challenge 1: Implement Grover's Algorithm
        **Goal**: Find a marked item in a 4-item database
        
        **Requirements**:
        - 2 qubits for 4 items
        - Oracle function to mark target
        - Diffusion operator for amplification
        
        #### Challenge 2: Quantum Random Walk
        **Goal**: Implement a quantum random walk on a line
        
        **Hints**:
        - Use position and coin qubits
        - Apply coin flip operation
        - Apply conditional shift operation
        """)
    
    elif challenge_level == "Advanced":
        st.markdown("""
        ### 🚀 Advanced Challenges
        
        #### Challenge 1: Quantum Error Correction
        **Goal**: Implement the 3-qubit bit-flip code
        
        **Requirements**:
        - Encode 1 logical qubit into 3 physical qubits
        - Introduce errors
        - Detect and correct errors
        
        #### Challenge 2: Variational Quantum Eigensolver (VQE)
        **Goal**: Find ground state energy of H₂ molecule
        
        **Requirements**:
        - Parameterized quantum circuit
        - Classical optimization loop
        - Expectation value calculation
        """)
    
    # Challenge submission and verification
    st.markdown("---")
    st.markdown("### 🏆 Challenge Verification")
    
    verification_code = st.text_area(
        "Paste your solution code here for verification:",
        height=100,
        placeholder="# Paste your challenge solution here..."
    )
    
    if st.button("🔍 Verify Solution"):
        if verification_code.strip():
            st.info("Solution verification feature coming soon! For now, run your code in the editor to test it.")
        else:
            st.warning("Please paste your solution code first.")

# Footer with helpful tips
st.markdown("---")
st.markdown("""
### 💡 Programming Tips

1. **Start Small**: Begin with single-qubit circuits before building complex algorithms
2. **Visualize**: Always draw your circuits to understand the flow
3. **Test Frequently**: Run small pieces of code to catch errors early
4. **Use Comments**: Document your quantum circuits for clarity
5. **Experiment**: Try different gate combinations and see what happens!

### 🔗 Useful Resources
- [Qiskit Documentation](https://qiskit.org/documentation/)
- [Quantum Computing Fundamentals](https://qiskit.org/textbook/)
- [Quantum Algorithm Zoo](https://quantumalgorithmzoo.org/)

**Remember**: Quantum programming is different from classical programming - embrace the probabilistic nature!
""")

# Navigation
col1, col2, col3 = st.columns(3)
with col1:
    if st.button("← Circuit Builder", use_container_width=True):
        st.switch_page("pages/06_Circuit_Builder.py")
with col2:
    if st.button("⚗️ Quantum Simulator", use_container_width=True):
        st.switch_page("pages/08_Quantum_Simulator.py")
with col3:
    if st.button("🏠 Home", use_container_width=True):
        st.switch_page("app.py")