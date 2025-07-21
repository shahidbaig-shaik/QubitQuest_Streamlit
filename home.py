import streamlit as st



st.markdown("## Welcome to QubitQuest ")
# Continue with your main page...

st.markdown("""
Enter the quantum realm where real particles, real math, and real code power your quest for fundamental understanding.

### Why “Fundamental” Matters
QubitQuest uses **trapped-ion qubits** because they let you directly harness the *actual particles and interactions* at the heart of quantum theory:

- **Single Atomic Ions**: Each qubit is one `¹⁷¹Yb⁺` ion—no emergent circuits, just fundamental particles.  
- **Laser-Driven Transitions**: We manipulate real electronic energy levels (|0⟩ ↔ |1⟩) via focused laser beams—textbook quantum mechanics in action.  
- **Direct Quantum Interactions**: Entanglement and gates arise from Coulomb coupling and photon exchange, not engineered Josephson junctions.

We surface these first-principles in every lesson, so you learn quantum computing by directly engaging the same particles your equations describe.

### What is QubitQuest?
QubitQuest guides you through a curated selection of fundamental quantum computing algorithms. Each lesson builds your understanding by aligning the mathematics, code, and hardware:

- Start with single-qubit basics (superposition, Hadamard, Pauli gates).  
- Progress to entanglement and multi-qubit states (Bell, GHZ, W).  
- Dive into core algorithms: Grover’s search, Quantum Fourier Transform, Phase Estimation, and more.

Practice by writing and running code directly on real IonQ hardware, reinforcing quantum mechanics from the ground up.
""")
if st.button("🚀 Begin QubitQuest"):
    st.switch_page("pages/02_Superposition.py")
    st.sidebar.page_link("pages/02_Superposition.py", label="Superposition")
    st.sidebar.page_link("pages/03_Entanglement.py", label="Entanglement")
    st.sidebar.page_link("pages/04_grover.py", label="Grover")
    st.sidebar.page_link("pages/05_Hadamard.py", label="Hadamard")

    # or whatever your page file is named
