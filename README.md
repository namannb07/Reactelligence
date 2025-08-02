# 🧪 Reactelligence - AI Chemistry Lab

Reactelligence is an **AI-powered chemistry platform** that integrates **RDKit**, **Hugging Face models**, and **Streamlit** to provide **reaction prediction**, **molecule generation**, and **molecular property analysis** in an interactive web app.
It supports **multilingual input**, visualizes molecules in **2D** and **graph form**, and calculates key chemical descriptors for research and educational purposes.

---

## 📌 Features

### 1. **Reaction Prediction**

* Predicts reaction products from reactants' SMILES strings.
* Uses **ChemFormer** model from Hugging Face (`HenryNguyen5/ChemFormer-ZINC`).
* Fallback rules for basic organic reactions when the model isn't available.
* Displays predicted products with **2D molecular structure** and **property tables**.

### 2. **Molecule Generation**

* Generates novel molecules from textual prompts.
* Can design drug-like molecules (e.g., pain relief, fever treatment).
* Applies **Lipinski's Rule of Five** for drug-likeness evaluation.
* Visualizes generated molecules in 2D and shows molecular properties.

### 3. **Molecular Property Analysis**

* Calculates:

  * Molecular Formula
  * Molecular Weight
  * LogP
  * Topological Polar Surface Area (TPSA)
  * Hydrogen Bond Donors/Acceptors
  * Rotatable Bonds
  * Aromatic Rings
  * Ring Count
  * Formal Charge
  * Atom/Bond Count
* Generates **radar charts** for property comparison.
* Creates **molecular graphs** using NetworkX for topology visualization.

### 4. **Multilingual Input Support**

* Accepts input in multiple languages (English, Hindi, Spanish, French, German, Chinese, Japanese, Korean).
* Uses **Google Translate API** to process non-English queries.

### 5. **Visual Analysis**

* **2D Molecule Rendering** with RDKit.
* **Graph Visualization** with NetworkX + Matplotlib.
* **Property Radar Charts** with Plotly.

---

## 🛠️ Tech Stack

**Languages:**

* Python 3.x

**Libraries & Frameworks:**

* **Streamlit** → Interactive web interface
* **RDKit** → Chemical informatics and molecular analysis
* **Transformers (Hugging Face)** → Reaction prediction & molecule generation
* **Googletrans** → Multilingual translation
* **Matplotlib / Plotly / NetworkX** → Visualization
* **NumPy / Pandas** → Data processing

---

## 📂 Project Structure

```
Reactelligence/
│
├── app.py                # Main Streamlit application (this script)
├── requirements.txt      # List of dependencies
└── README.md             # Project documentation
```

---

## 📥 Installation & Setup

### 1️⃣ Clone the repository

```bash
git clone https://github.com/yourusername/reactelligence.git
cd reactelligence
```

### 2️⃣ Create a virtual environment (optional but recommended)

```bash
python -m venv venv
source venv/bin/activate  # On Linux/Mac
venv\Scripts\activate     # On Windows
```

### 3️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

**Example `requirements.txt`:**

```
streamlit
pandas
numpy
rdkit-pypi
matplotlib
networkx
plotly
transformers
googletrans==4.0.0-rc1
requests
```

---

## 🚀 Running the App

```bash
streamlit run app.py
```

The app will open in your browser at:
**[http://localhost:8501](http://localhost:8501)**

---

## 💡 Usage Guide

### **Reaction Prediction**

1. Enter a query like:

   ```
   Predict the reaction: CCO + O2
   ```
2. Select **Reaction Prediction** intent.
3. View:

   * Reactant structures
   * Predicted products
   * Product properties

### **Molecule Generation**

1. Enter a prompt like:

   ```
   Generate a pain relief molecule
   ```
2. The app:

   * Generates SMILES
   * Displays 2D structure
   * Shows chemical properties & drug-likeness score

### **Molecular Analysis**

1. Enter a SMILES string or name:

   ```
   CC(=O)Oc1ccccc1C(=O)O
   ```
2. View:

   * Properties
   * 2D molecular diagram
   * Molecular graph
   * Radar chart of properties

---

## 🌍 Language Support

Supported input languages:

* Auto-detect
* English
* Hindi
* Spanish
* French
* German
* Chinese
* Japanese
* Korean

---

## ⚠️ Limitations

* The **ChemFormer** model may require internet access to download from Hugging Face.
* Google Translate API can fail if rate limits are exceeded.
* Predictions are for **educational purposes only** and not a substitute for laboratory validation.

---

## 📜 License

This project is licensed under the **MIT License**.
Feel free to use and modify for **research and educational purposes**.

---

## 🙌 Acknowledgements

* **RDKit** for cheminformatics tools.
* **Hugging Face** for ChemFormer model.
* **Streamlit** for making web apps easy.
* **Google Translate API** for multilingual support.

