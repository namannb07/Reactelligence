# 🧪 Reactelligence – AI-Powered Chemistry Lab

Reactelligence is an **AI-driven molecular analysis tool** built with **Streamlit**, **RDKit**, and **ChemBERTa**.  
It allows chemists, researchers, and students to analyze molecules, predict chemical properties, assess drug-likeness, compare compounds, and even explore reactions — all in an interactive web app.

---

## 🚀 Features

### 🔍 Single Molecule Analysis
- Enter a **SMILES** string to:
  - View molecular structure
  - Calculate basic chemical properties (Molecular weight, LogP, TPSA, H-bond donors/acceptors, rotatable bonds, rings)
  - Predict AI-enhanced properties using **ChemBERTa**:
    - Solubility Score
    - Drug-likeness
    - Bioavailability
    - Toxicity Risk
  - Lipinski’s Rule of Five assessment
  - Property radar chart visualization

### ⚗️ Reaction Analysis
- Input multiple reactant and product SMILES
- Analyze basic properties of each molecule
- Estimate **reaction feasibility**

### 📊 Batch Processing
- Upload a CSV file with a `SMILES` column
- Automatically analyze **all molecules** in bulk
- Download results as CSV

### 🔀 Property Comparison
- Compare two molecules side-by-side
- See differences in structure and properties

### 📜 Analysis History
- Keep track of previously analyzed molecules

---

## 🛠 Tech Stack

- **[Streamlit](https://streamlit.io/)** – Web app framework
- **[RDKit](https://www.rdkit.org/)** – Chemical informatics
- **[PyTorch](https://pytorch.org/)** – AI computations
- **[Transformers](https://huggingface.co/transformers/)** – ChemBERTa models
- **[Plotly](https://plotly.com/)** & **Matplotlib** – Data visualizations
- **[Pandas](https://pandas.pydata.org/)** & **NumPy** – Data processing

---

## 📦 Installation

### 1️⃣ Clone the repository
```bash
git clone https://github.com/your-username/reactelligence.git
cd reactelligence

2️⃣ Create and activate a virtual environment

python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

3️⃣ Install dependencies

pip install -r requirements.txt

4️⃣ Run the app

streamlit run app.py


---

🌐 Deployment on Streamlit Cloud

1. Push your code to a GitHub repository


2. Make sure you include:

app.py

requirements.txt (with compatible versions)

runtime.txt (to pin Python version, e.g., python-3.10)



3. Go to Streamlit Cloud


4. Deploy your app from the GitHub repo




---

📄 Example SMILES

Compound	SMILES

Aspirin	CC(=O)Oc1ccccc1C(=O)O
Caffeine	CN1C=NC2=C1C(=O)N(C(=O)N2C)C
Ibuprofen	CC(C)Cc1ccc(cc1)[C@@H](C)C(=O)O
Paracetamol	CC(=O)Nc1ccc(O)cc1
Penicillin	CC1([C@@H](N2[C@H](S1)[C@@H](C2=O)NC(=O)Cc3ccccc3)C(=O)O)C



---

📷 Screenshots

Main UI



Molecule Analysis




---

⚠️ Notes

Large ChemBERTa models may take a while to load on first run.

For Streamlit Cloud, RDKit requires Python ≤ 3.11 — use a runtime.txt with python-3.10.

Predictions are AI-assisted estimates and should not replace professional chemical analysis.



---

📜 License

This project is licensed under the MIT License – feel free to use, modify, and share.


---

❤️ Credits

RDKit

DeepChem / ChemBERTa Models

Streamlit

PyTorch

Plotly
