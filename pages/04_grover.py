import streamlit as st
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit

st.title("🔍 Grover's Algorithm")

# --- THEORY SECTION ---
st.header("📖 What is Grover's Algorithm?")
st.markdown("""
Grover’s algorithm is a **quantum search algorithm** that finds a target item in an unsorted database of \( N \) items in only \( O(\sqrt{N}) \) time, compared to \( O(N) \) in classical search.

It uses:
- **Superposition** to explore all states simultaneously.
- **Oracle** to mark the correct solution.
- **Diffusion operator** to amplify the correct answer.
""")

# --- FORMULA SECTION ---
st.subheader("🧠 Grover Iteration Formula")
st.latex(r"G = D \cdot O")
st.markdown("""
Where:  
- \( O \): Oracle operator (flips sign of solution)  
- \( D \): Diffusion (inversion about the mean)  
- \( G \): Grover operator (applied ~√N times)
""")

# --- QISKIT CODE EDITOR ---
st.header("🧪 Qiskit Code Editor")

default_code = """from qiskit import QuantumCircuit

qc = QuantumCircuit(2)
qc.h([0, 1])
qc.x([0, 1])
qc.cz(0, 1)
qc.x([0, 1])
qc.h([0, 1])
qc.measure_all()

qc.draw('mpl')
"""

code = st.text_area("Your Qiskit Code", value=default_code, height=280)

if st.button("▶️ Render Circuit Only"):
    try:
        local_vars = {}
        exec(code, {}, local_vars)

        qc = local_vars.get("qc")
        if qc is None or not isinstance(qc, QuantumCircuit):
            st.error("No valid QuantumCircuit 'qc' found in the code.")
        else:
            st.success("✅ Circuit rendered successfully.")
            fig = qc.draw("mpl")
            st.pyplot(fig)

    except Exception as e:
        st.error(f"⚠️ Error while rendering your circuit:\n\n{e}")
