import streamlit as st
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit

st.set_page_config(page_title="Superposition", page_icon="🌀")
st.title("🌀 Superposition")

st.markdown("""
Superposition means a qubit can be in a mix of both $|0⟩$ and $|1⟩$ until measured.

- It's like a coin spinning in the air — both heads and tails at once.
- Mathematically:
  $$
  |\\psi\\rangle = \\alpha|0\\rangle + \\beta|1\\rangle
  $$
  where $|\\alpha|^2 + |\\beta|^2 = 1$
""")

# --- Example Circuit
example_code = '''from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit

qreg_q = QuantumRegister(1, 'q')
creg_c = ClassicalRegister(1, 'c')
circuit = QuantumCircuit(qreg_q, creg_c)

circuit.h(qreg_q[0])
circuit.measure(qreg_q[0], creg_c[0])'''

# --- Show Code Preview
st.markdown("### 📋 Example Superposition Code")
st.code(example_code, language="python")

# --- Insert Example Button
if st.button("📥 Copy Example to Editor"):
    st.session_state.code = example_code

# --- Code Editor
st.markdown("### 💻 Qiskit Code Editor")
code_input = st.text_area("Edit or paste your code below", st.session_state.get("code", ""), height=250)

# --- Render Button
if st.button("🚀 Render Circuit"):
    local_vars = {}
    try:
        exec(code_input, {}, local_vars)
        qc = local_vars.get("circuit")

        if isinstance(qc, QuantumCircuit):
            st.subheader("🔧 Quantum Circuit Output")
            st.code(qc.draw(output="text"), language="text")
        else:
            st.error("⚠️ Please define a QuantumCircuit named `circuit`.")
    except Exception as e:
        st.error(f"❌ Error: {e}")
