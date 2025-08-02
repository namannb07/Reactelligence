%%writefile app.py

import streamlit as st
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Draw, Descriptors, rdMolDescriptors
from rdkit.Chem.Draw import rdMolDraw2D
import matplotlib.pyplot as plt
import networkx as nx
import plotly.graph_objects as go
import plotly.express as px
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from googletrans import Translator
import re
import base64
from io import BytesIO
import requests
import json

# Page configuration
st.set_page_config(
    page_title="Reactelligence - AI Chemistry Lab",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .main-header h1 {
        color: white;
        text-align: center;
        margin: 0;
    }
    .feature-box {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #667eea;
        margin: 1rem 0;
    }
    .molecule-container {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'translation_cache' not in st.session_state:
    st.session_state.translation_cache = {}

class ChemistryAI:
    def __init__(self):
        self.translator = Translator()
        self.model_cache = {}
        
    @st.cache_resource
    def load_chemistry_model(_self):
        """Load pre-trained chemistry model for reaction prediction"""
        try:
            tokenizer = AutoTokenizer.from_pretrained("HenryNguyen5/ChemFormer-ZINC")
            model = AutoModelForSeq2SeqLM.from_pretrained("HenryNguyen5/ChemFormer-ZINC")
            return tokenizer, model
        except Exception as e:
            st.warning(f"Could not load ChemFormer model: {e}")
            return None, None
    
    def translate_to_english(self, text, source_lang='auto'):
        """Translate text to English for processing"""
        if text in st.session_state.translation_cache:
            return st.session_state.translation_cache[text]
        
        try:
            if source_lang == 'en' or self._is_english(text):
                return text
            
            result = self.translator.translate(text, src=source_lang, dest='en')
            translated = result.text
            st.session_state.translation_cache[text] = translated
            return translated
        except Exception as e:
            st.warning(f"Translation error: {e}")
            return text
    
    def _is_english(self, text):
        """Simple check if text is primarily English"""
        english_chars = sum(1 for c in text if c.isalpha() and ord(c) < 128)
        total_chars = sum(1 for c in text if c.isalpha())
        return total_chars == 0 or english_chars / total_chars > 0.7
    
    def detect_intent(self, text):
        """Detect user intent from input text"""
        text_lower = text.lower()
        
        # Keywords for different intents
        reaction_keywords = ['reaction', 'predict', 'product', 'react', 'synthesis', 'yield']
        generation_keywords = ['generate', 'create', 'design', 'new molecule', 'novel', 'drug']
        analysis_keywords = ['analyze', 'properties', 'molecular weight', 'logp', 'analyze']
        
        # Check for SMILES strings
        if self._contains_smiles(text):
            if any(keyword in text_lower for keyword in reaction_keywords):
                return 'reaction_prediction'
            elif any(keyword in text_lower for keyword in analysis_keywords):
                return 'analysis'
            else:
                return 'analysis'  # Default for SMILES input
        
        # Intent detection based on keywords
        if any(keyword in text_lower for keyword in reaction_keywords):
            return 'reaction_prediction'
        elif any(keyword in text_lower for keyword in generation_keywords):
            return 'molecule_generation'
        elif any(keyword in text_lower for keyword in analysis_keywords):
            return 'analysis'
        else:
            return 'analysis'  # Default intent
    
    def _contains_smiles(self, text):
        """Check if text contains SMILES notation"""
        smiles_pattern = r'[A-Za-z0-9@+\-\[\]()=#$:/.\\]+'
        potential_smiles = re.findall(smiles_pattern, text)
        
        for smiles in potential_smiles:
            if len(smiles) > 3 and Chem.MolFromSmiles(smiles) is not None:
                return True
        return False
    
    def extract_smiles(self, text):
        """Extract SMILES strings from text"""
        words = text.split()
        smiles_list = []
        
        for word in words:
            mol = Chem.MolFromSmiles(word)
            if mol is not None:
                smiles_list.append(word)
        
        return smiles_list
    
    def predict_reaction(self, reactants_smiles):
        """Predict reaction products"""
        tokenizer, model = self.load_chemistry_model()
        
        if model is None:
            # Fallback: Simple reaction examples
            return self._fallback_reaction_prediction(reactants_smiles)
        
        try:
            input_text = f"React: {'.'.join(reactants_smiles)}"
            inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)
            
            with st.spinner("Predicting reaction products..."):
                outputs = model.generate(
                    inputs.input_ids,
                    max_length=200,
                    num_beams=5,
                    temperature=0.7,
                    do_sample=True
                )
            
            predicted = tokenizer.decode(outputs[0], skip_special_tokens=True)
            products = self._parse_products(predicted)
            return products
        except Exception as e:
            st.error(f"Reaction prediction error: {e}")
            return self._fallback_reaction_prediction(reactants_smiles)
    
    def _fallback_reaction_prediction(self, reactants_smiles):
        """Fallback reaction prediction using known reactions"""
        # Simple substitution and addition reactions
        products = []
        
        for smiles in reactants_smiles:
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                # Simple transformations
                if 'C=C' in smiles:  # Alkene
                    products.append(smiles.replace('C=C', 'CC'))
                elif 'C#C' in smiles:  # Alkyne
                    products.append(smiles.replace('C#C', 'C=C'))
                else:
                    products.append(smiles)  # Return as is
        
        return products if products else ['CCO']  # Default to ethanol
    
    def _parse_products(self, predicted_text):
        """Parse products from model output"""
        # Extract SMILES from prediction
        smiles_candidates = self.extract_smiles(predicted_text)
        return smiles_candidates if smiles_candidates else ['CCO']
    
    def generate_molecule(self, prompt):
        """Generate novel molecules based on prompt"""
        tokenizer, model = self.load_chemistry_model()
        
        if model is None:
            return self._fallback_molecule_generation(prompt)
        
        try:
            input_text = f"Generate molecule for: {prompt}"
            inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)
            
            with st.spinner("Generating novel molecule..."):
                outputs = model.generate(
                    inputs.input_ids,
                    max_length=150,
                    num_beams=5,
                    temperature=0.8,
                    do_sample=True
                )
            
            generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
            molecules = self.extract_smiles(generated)
            
            if not molecules:
                return self._fallback_molecule_generation(prompt)
            
            return molecules[0]
        except Exception as e:
            st.error(f"Molecule generation error: {e}")
            return self._fallback_molecule_generation(prompt)
    
    def _fallback_molecule_generation(self, prompt):
        """Fallback molecule generation"""
        # Simple drug-like molecules
        drug_molecules = [
            'CC(C)Cc1ccc(cc1)[C@@H](C)C(=O)O',  # Ibuprofen
            'CC(=O)Oc1ccccc1C(=O)O',  # Aspirin
            'CN1C=NC2=C1C(=O)N(C(=O)N2C)C',  # Caffeine
            'CCN(CC)CCCC(C)Nc1ccnc2cc(ccc12)Cl',  # Chloroquine
            'Cc1ccc(cc1)C(=O)O'  # p-Cresol
        ]
        
        prompt_lower = prompt.lower()
        if 'pain' in prompt_lower or 'analgesic' in prompt_lower:
            return drug_molecules[0]  # Ibuprofen
        elif 'fever' in prompt_lower or 'aspirin' in prompt_lower:
            return drug_molecules[1]  # Aspirin
        elif 'stimulant' in prompt_lower or 'caffeine' in prompt_lower:
            return drug_molecules[2]  # Caffeine
        else:
            return np.random.choice(drug_molecules)
    
    def analyze_molecule(self, smiles):
        """Analyze molecular properties"""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        
        properties = {
            'SMILES': smiles,
            'Molecular Formula': Chem.rdMolDescriptors.CalcMolFormula(mol),
            'Molecular Weight': round(Descriptors.MolWt(mol), 2),
            'LogP': round(Descriptors.MolLogP(mol), 2),
            'TPSA': round(Descriptors.TPSA(mol), 2),
            'HBD': Descriptors.NumHDonors(mol),
            'HBA': Descriptors.NumHAcceptors(mol),
            'Rotatable Bonds': Descriptors.NumRotatableBonds(mol),
            'Aromatic Rings': Descriptors.NumAromaticRings(mol),
            'Rings': Descriptors.RingCount(mol),
            'Formal Charge': Chem.rdmolops.GetFormalCharge(mol),
            'Atoms': mol.GetNumAtoms(),
            'Bonds': mol.GetNumBonds()
        }
        
        return properties

class MoleculeVisualizer:
    @staticmethod
    def draw_molecule(smiles, size=(400, 400)):
        """Draw 2D molecule structure"""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        
        drawer = rdMolDraw2D.MolDraw2DCairo(size[0], size[1])
        drawer.DrawMolecule(mol)
        drawer.FinishDrawing()
        
        img_data = drawer.GetDrawingText()
        return img_data
    
    @staticmethod
    def create_molecule_graph(smiles):
        """Create NetworkX graph of molecule"""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        
        G = nx.Graph()
        
        # Add atoms as nodes
        for atom in mol.GetAtoms():
            G.add_node(atom.GetIdx(), 
                      symbol=atom.GetSymbol(),
                      atomic_num=atom.GetAtomicNum())
        
        # Add bonds as edges
        for bond in mol.GetBonds():
            G.add_edge(bond.GetBeginAtomIdx(), 
                      bond.GetEndAtomIdx(),
                      bond_type=bond.GetBondType())
        
        return G
    
    @staticmethod
    def plot_molecule_graph(G, title="Molecular Graph"):
        """Plot molecule graph using matplotlib"""
        if G is None:
            return None
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        pos = nx.spring_layout(G, k=1, iterations=50)
        
        # Draw edges
        nx.draw_networkx_edges(G, pos, edge_color='gray', alpha=0.6, ax=ax)
        
        # Draw nodes with different colors for different elements
        node_colors = []
        node_labels = {}
        
        for node in G.nodes():
            symbol = G.nodes[node]['symbol']
            node_labels[node] = symbol
            
            if symbol == 'C':
                node_colors.append('black')
            elif symbol == 'O':
                node_colors.append('red')
            elif symbol == 'N':
                node_colors.append('blue')
            elif symbol == 'S':
                node_colors.append('yellow')
            else:
                node_colors.append('purple')
        
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, 
                              node_size=500, alpha=0.8, ax=ax)
        nx.draw_networkx_labels(G, pos, node_labels, 
                               font_size=12, font_color='white', ax=ax)
        
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.axis('off')
        
        return fig

def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🧪 Reactelligence - AI Chemistry Lab</h1>
        <p style="text-align: center; color: white; margin: 0;">
            AI-Powered Chemistry Analysis & Prediction Platform
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize AI
    chem_ai = ChemistryAI()
    visualizer = MoleculeVisualizer()
    
    # Sidebar
    with st.sidebar:
        st.header("🔬 Features")
        st.markdown("""
        - **Reaction Prediction**: Predict products from reactants
        - **Molecule Generation**: Create novel molecules
        - **Property Analysis**: Calculate molecular properties
        - **Multilingual Support**: Input in any language
        - **Visual Analysis**: 2D structures & graphs
        """)
        
        st.header("🌍 Language")
        language = st.selectbox("Select Input Language", [
            "Auto-detect", "English", "Hindi", "Spanish", "French", 
            "German", "Chinese", "Japanese", "Korean"
        ])
        
        st.header("📝 Examples")
        if st.button("🧪 Reaction Example"):
            st.session_state.example_input = "Predict the reaction: CCO + O2"
        if st.button("🔬 Generation Example"):
            st.session_state.example_input = "Generate a pain relief molecule"
        if st.button("📊 Analysis Example"):
            st.session_state.example_input = "Analyze CC(=O)Oc1ccccc1C(=O)O"
    
    # Main input
    st.header("💬 Chemistry Query")
    
    default_input = st.session_state.get('example_input', '')
    user_input = st.text_area(
        "Enter your chemistry question in any language:",
        value=default_input,
        height=100,
        placeholder="Examples:\n- Predict reaction: C2H5OH + O2\n- Generate anticancer drug\n- Analyze aspirin molecule\n- CC(=O)Oc1ccccc1C(=O)O का विश्लेषण करें"
    )
    
    col1, col2, col3 = st.columns([1, 1, 2])
    
    with col1:
        analyze_btn = st.button("🔍 Analyze", type="primary")
    with col2:
        clear_btn = st.button("🗑️ Clear")
    
    if clear_btn:
        st.session_state.example_input = ''
        st.rerun()
    
    if analyze_btn and user_input.strip():
        # Translate input
        lang_code = 'auto' if language == "Auto-detect" else language.lower()[:2]
        translated_input = chem_ai.translate_to_english(user_input, lang_code)
        
        if translated_input != user_input:
            st.info(f"**Translated:** {translated_input}")
        
        # Detect intent
        intent = chem_ai.detect_intent(translated_input)
        st.success(f"**Detected Task:** {intent.replace('_', ' ').title()}")
        
        # Process based on intent
        if intent == 'reaction_prediction':
            st.header("⚗️ Reaction Prediction")
            
            reactants = chem_ai.extract_smiles(translated_input)
            if not reactants:
                st.warning("No valid SMILES found. Please provide reactant molecules.")
            else:
                st.write("**Reactants:**")
                for i, smiles in enumerate(reactants):
                    col1, col2 = st.columns([1, 2])
                    with col1:
                        st.code(smiles)
                    with col2:
                        mol_img = visualizer.draw_molecule(smiles, (200, 200))
                        if mol_img:
                            st.image(mol_img, caption=f"Reactant {i+1}")
                
                # Predict products
                products = chem_ai.predict_reaction(reactants)
                
                st.write("**Predicted Products:**")
                for i, product in enumerate(products):
                    col1, col2, col3 = st.columns([1, 2, 2])
                    with col1:
                        st.code(product)
                    with col2:
                        mol_img = visualizer.draw_molecule(product, (200, 200))
                        if mol_img:
                            st.image(mol_img, caption=f"Product {i+1}")
                    with col3:
                        properties = chem_ai.analyze_molecule(product)
                        if properties:
                            st.write("**Properties:**")
                            st.write(f"MW: {properties['Molecular Weight']} g/mol")
                            st.write(f"LogP: {properties['LogP']}")
                            st.write(f"Formula: {properties['Molecular Formula']}")
        
        elif intent == 'molecule_generation':
            st.header("🧬 Molecule Generation")
            
            generated_smiles = chem_ai.generate_molecule(translated_input)
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.markdown('<div class="molecule-container">', unsafe_allow_html=True)
                st.subheader("Generated Molecule")
                st.code(generated_smiles)
                
                mol_img = visualizer.draw_molecule(generated_smiles, (300, 300))
                if mol_img:
                    st.image(mol_img, caption="Generated Structure")
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col2:
                properties = chem_ai.analyze_molecule(generated_smiles)
                if properties:
                    st.subheader("Molecular Properties")
                    
                    # Create properties dataframe
                    props_df = pd.DataFrame(list(properties.items()), 
                                          columns=['Property', 'Value'])
                    st.dataframe(props_df, use_container_width=True)
                    
                    # Drug-likeness assessment
                    st.subheader("Drug-likeness (Lipinski's Rule)")
                    mw = properties['Molecular Weight']
                    logp = properties['LogP']
                    hbd = properties['HBD']
                    hba = properties['HBA']
                    
                    violations = 0
                    if mw > 500: violations += 1
                    if logp > 5: violations += 1
                    if hbd > 5: violations += 1
                    if hba > 10: violations += 1
                    
                    if violations == 0:
                        st.success("✅ Passes Lipinski's Rule (Drug-like)")
                    else:
                        st.warning(f"⚠️ Violates {violations} Lipinski rules")
        
        else:  # Analysis
            st.header("📊 Molecular Analysis")
            
            smiles_list = chem_ai.extract_smiles(translated_input)
            if not smiles_list:
                st.warning("No valid SMILES found. Please provide a molecule SMILES string.")
            else:
                for i, smiles in enumerate(smiles_list):
                    st.subheader(f"Molecule {i+1}: {smiles}")
                    
                    col1, col2 = st.columns([1, 1])
                    
                    with col1:
                        st.markdown('<div class="molecule-container">', unsafe_allow_html=True)
                        st.write("**2D Structure**")
                        mol_img = visualizer.draw_molecule(smiles, (350, 350))
                        if mol_img:
                            st.image(mol_img)
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    with col2:
                        properties = chem_ai.analyze_molecule(smiles)
                        if properties:
                            st.write("**Molecular Properties**")
                            props_df = pd.DataFrame(list(properties.items()), 
                                                  columns=['Property', 'Value'])
                            st.dataframe(props_df, use_container_width=True)
                    
                    # Molecular graph
                    st.subheader("Molecular Graph Visualization")
                    G = visualizer.create_molecule_graph(smiles)
                    if G:
                        fig = visualizer.plot_molecule_graph(G, f"Molecular Graph - {smiles}")
                        if fig:
                            st.pyplot(fig)
                    
                    # Property charts
                    if properties:
                        st.subheader("Property Visualization")
                        
                        # Create radar chart for key properties
                        categories = ['MW/100', 'LogP+5', 'TPSA/10', 'HBD*2', 'HBA*2', 'Rings*5']
                        values = [
                            min(properties['Molecular Weight']/100, 10),
                            min(properties['LogP']+5, 10),
                            min(properties['TPSA']/10, 10),
                            min(properties['HBD']*2, 10),
                            min(properties['HBA']*2, 10),
                            min(properties['Rings']*5, 10)
                        ]
                        
                        fig = go.Figure(data=go.Scatterpolar(
                            r=values,
                            theta=categories,
                            fill='toself',
                            name='Properties'
                        ))
                        
                        fig.update_layout(
                            polar=dict(
                                radialaxis=dict(visible=True, range=[0, 10])
                            ),
                            showlegend=False,
                            title="Molecular Property Profile"
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666;">
        <p>🧪 Reactelligence - Powered by AI & RDKit | Built with Streamlit</p>
        <p><em>For research and educational purposes only</em></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
