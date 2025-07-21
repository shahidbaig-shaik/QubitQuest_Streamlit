import streamlit as st

from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit.visualization import plot_histogram
import matplotlib.pyplot as plt

st.set_page_config(page_title="Entanglement")

st.sidebar.title("QubitQuest")
st.sidebar.markdown("### 🧠  Multi-Qubit Concepts")

st.markdown("## 🔗 **Entanglement**")
st.write("""
**Entanglement** links qubits so their states are *correlated no matter the distance*.  
Measuring one instantly tells you about the other.

A famous example is the **Bell state**:
""")

st.latex(r"""
|\Phi^+\rangle = \frac{1}{\sqrt{2}}(|00\rangle + |11\rangle)
""")

st.write("Einstein called it _'spooky action at a distance.'_")

# Default Qiskit example code for entanglement
default_code = """from qiskit import QuantumCircuit

qc = QuantumCircuit(2)
qc.h(0)
qc.cx(0, 1)
circuit = qc  # Required for rendering
"""

st.markdown("### 🧪 Try it Yourself:")
code = st.text_area("Edit Qiskit Code Here", value=default_code, height=250)

if st.button("🔵 Run & Show Circuit:"):
    try:
        exec_globals = {}
        exec(code, exec_globals)
    except Exception as e:
        st.error(f"⚠️ Error in your code:\n\n{e}")

st.title("🔗 Entanglement")

st.markdown("""
Entanglement links qubits so their states are *correlated no matter the distance*.

- Measuring one instantly tells you about the other.
- Famous Example: The Bell state  
  $$
  |\\Phi^+\\rangle = \\frac{1}{\\sqrt{2}}(|00\\rangle + |11\\rangle)
  $$

Einstein called it “spooky action at a distance.”
""")

