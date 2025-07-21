import streamlit as st
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit.visualization import matplotlib
import matplotlib.pyplot as plt

st.set_page_config(page_title="Hadamard Gate")

st.sidebar.title("QubitQuest")
st.sidebar.markdown("## 🟧 Single-Qubit Gates")

# Add links to other gate pages here

st.markdown("## 🌀 **Hadamard Gate (H)**")
st.write(
    "The **Hadamard gate** creates a superposition of states. It transforms "
    "the basis states |0⟩ and |1⟩ into equal superpositions:"
)

st.latex(r"""
H|0\rangle = \frac{|0\rangle + |1\rangle}{\sqrt{2}}, \quad
H|1\rangle = \frac{|0\rangle - |1\rangle}{\sqrt{2}}
""")

st.markdown("It is represented by the matrix:")
st.latex(r"""
H = \frac{1}{\sqrt{2}} \begin{bmatrix}
1 & 1 \\
1 & -1
\end{bmatrix}
""")

st.markdown("### 🧪 Example Code: Hadamard on |0⟩")

default_code = '''from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit

qreg_q = QuantumRegister(1, 'q')
creg_c = ClassicalRegister(1, 'c')
circuit = QuantumCircuit(qreg_q, creg_c)

circuit.h(qreg_q[0])
circuit.measure(qreg_q[0], creg_c[0])
circuit.draw(output='mpl')
'''

code = st.text_area("📝 Edit Qiskit Code Here", value=default_code, height=250)

if st.button("🚀 Run & Show Circuit"):
    try:
        exec_globals = {}
        exec(code, {"QuantumRegister": QuantumRegister,
                    "ClassicalRegister": ClassicalRegister,
                    "QuantumCircuit": QuantumCircuit,
                    "plt": plt}, exec_globals)
        circuit = exec_globals.get("circuit", None)

        if circuit:
            st.pyplot(circuit.draw(output="mpl"))
        else:
            st.warning("⚠️ Circuit not found. Make sure your code defines a `circuit` object.")
    except Exception as e:
        st.error(f"❌ Error while executing your code:\n\n{e}")
