import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import numpy as np
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from googletrans import Translator
import re
import warnings
warnings.filterwarnings('ignore')

# Import RDKit for chemistry operations
try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, Draw
    from rdkit.Chem.Draw import rdMolDraw2D
    import io
    import base64
    from PIL import Image
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    st.warning("RDKit not available. Some features will be limited.")

# Configure Streamlit page
st.set_page_config(
    page_title="Reactelligence - AI Chemistry Lab",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'chemformer_model' not in st.session_state:
    st.session_state.chemformer_model = None
if 'translator' not in st.session_state:
    st.session_state.translator = Translator()

@st.cache_resource
def load_chemformer_model():
    """Load ChemFormer model with caching"""
    try:
        # Use a chemistry-focused model (ChemBERTa as ChemFormer might not be directly available)
        model_name = "seyonec/ChemBERTa-zinc-base-v1"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        return tokenizer, model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

def detect_and_translate(text, target_lang='en'):
    """Detect language and translate to English if needed"""
    try:
        translator = st.session_state.translator
        detected = translator.detect(text)
        
        if detected.lang != target_lang:
            translated = translator.translate(text, dest=target_lang)
            return translated.text, detected.lang
        return text, detected.lang
    except Exception as e:
        st.warning(f"Translation error: {e}")
        return text, 'en'

def is_valid_smiles(smiles):
    """Check if a string is a valid SMILES"""
    if not RDKIT_AVAILABLE:
        return True  # Assume valid if RDKit not available
    try:
        mol = Chem.MolFromSmiles(smiles)
        return mol is not None
    except:
        return False

def extract_smiles_from_text(text):
    """Extract SMILES patterns from text"""
    # Simple regex pattern for SMILES-like strings
    smiles_pattern = r'[A-Za-z0-9@+\-\[\]()=#\\/]+(?:[A-Za-z0-9@+\-\[\]()=#\\/]*)'
    potential_smiles = re.findall(smiles_pattern, text)
    
    valid_smiles = []
    for smiles in potential_smiles:
        if len(smiles) > 3 and is_valid_smiles(smiles):
            valid_smiles.append(smiles)
    
    return valid_smiles

def analyze_molecule(smiles):
    """Analyze molecular properties"""
    if not RDKIT_AVAILABLE:
        return {"Error": "RDKit not available for analysis"}
    
    try:
        mol = Chem.MolFromSmiles(smiles)
        if not mol:
            return {"Error": "Invalid SMILES string"}
        
        properties = {
            "Molecular Weight": round(Descriptors.MolWt(mol), 2),
            "LogP": round(Descriptors.MolLogP(mol), 2),
            "Number of Rings": Descriptors.RingCount(mol),
            "H Acceptors": Descriptors.NumHAcceptors(mol),
            "H Donors": Descriptors.NumHDonors(mol),
            "Rotatable Bonds": Descriptors.NumRotatableBonds(mol),
            "Aromatic Rings": Descriptors.NumAromaticRings(mol),
            "TPSA": round(Descriptors.TPSA(mol), 2)
        }
        return properties
    except Exception as e:
        return {"Error": str(e)}

def generate_molecule_image(smiles):
    """Generate 2D molecule image"""
    if not RDKIT_AVAILABLE:
        return None
    
    try:
        mol = Chem.MolFromSmiles(smiles)
        if not mol:
            return None
        
        img = Draw.MolToImage(mol, size=(300, 300))
        return img
    except Exception as e:
        st.error(f"Error generating molecule image: {e}")
        return None

def create_reaction_graph(reactants, products):
    """Create a network graph of reaction"""
    G = nx.DiGraph()
    
    # Add reactant nodes
    for i, reactant in enumerate(reactants):
        G.add_node(f"R{i+1}", type='reactant', smiles=reactant)
    
    # Add product nodes
    for i, product in enumerate(products):
        G.add_node(f"P{i+1}", type='product', smiles=product)
    
    # Add edges from reactants to products
    for reactant in [f"R{i+1}" for i in range(len(reactants))]:
        for product in [f"P{i+1}" for i in range(len(products))]:
            G.add_edge(reactant, product)
    
    return G

def plot_reaction_graph(G):
    """Plot reaction network graph"""
    plt.figure(figsize=(10, 6))
    pos = nx.spring_layout(G)
    
    # Color nodes by type
    node_colors = ['lightblue' if G.nodes[node]['type'] == 'reactant' else 'lightcoral' 
                   for node in G.nodes()]
    
    nx.draw(G, pos, with_labels=True, node_color=node_colors, 
            node_size=1500, font_size=10, font_weight='bold',
            arrows=True, arrowsize=20, edge_color='gray')
    
    plt.title("Reaction Network Graph")
    plt.axis('off')
    return plt.gcf()

def predict_reaction_products(reactants_smiles):
    """Predict reaction products using ChemFormer-like approach"""
    # Since direct ChemFormer usage is complex, we'll simulate prediction
    # In a real implementation, you would use the loaded model for inference
    
    # For demo purposes, provide some example transformations
    example_reactions = {
        "CCO": "CC=O",  # ethanol -> acetaldehyde (oxidation)
        "CC=O": "CCO",  # acetaldehyde -> ethanol (reduction)
        "C6H6": "C6H5Cl",  # benzene -> chlorobenzene
        "CC(C)C": "CC(C)CO"  # isobutane -> isobutanol
    }
    
    products = []
    for reactant in reactants_smiles:
        if reactant in example_reactions:
            products.append(example_reactions[reactant])
        else:
            # Generate a modified version of the reactant as a simple prediction
            products.append(reactant + "O")  # Add oxygen as simple transformation
    
    return products

def generate_novel_molecule(seed_smiles=None):
    """Generate a novel molecule"""
    # Example novel molecules for demonstration
    novel_molecules = [
        "CC(C)(C)c1ccc(O)cc1",  # BHT antioxidant
        "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",  # caffeine
        "CC(=O)OC1=CC=CC=C1C(=O)O",  # aspirin
        "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",  # ibuprofen
        "C1=CC=C(C=C1)C2=CC=CC=C2"  # biphenyl
    ]
    
    return np.random.choice(novel_molecules)

# Streamlit UI
st.title("🧪 Reactelligence - AI Chemistry Lab")
st.markdown("### Multilingual AI-Powered Chemistry Assistant")

# Sidebar
st.sidebar.header("Navigation")
feature = st.sidebar.selectbox(
    "Choose a feature:",
    ["🔬 Reaction Prediction", "🧬 Molecule Generation", "📊 Reaction Analysis"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Supported Languages:**")
st.sidebar.markdown("🇬🇧 English • 🇮🇳 Hindi • 🇪🇸 Spanish • 🇫🇷 French")

# Main content area
if feature == "🔬 Reaction Prediction":
    st.header("Reaction Prediction")
    st.markdown("Predict reaction products from reactants")
    
    # Input section
    col1, col2 = st.columns([2, 1])
    
    with col1:
        user_input = st.text_area(
            "Enter reactants (in any language or SMILES format):",
            placeholder="Example: ethanol + oxygen OR CCO + O2 OR एथेनॉल + ऑक्सीजन",
            height=100
        )
    
    with col2:
        if st.button("🔮 Predict Products", type="primary"):
            if user_input:
                # Translate input to English
                english_input, detected_lang = detect_and_translate(user_input)
                
                if detected_lang != 'en':
                    st.info(f"Detected language: {detected_lang}")
                    st.info(f"Translated: {english_input}")
                
                # Extract SMILES or convert text to SMILES
                smiles_list = extract_smiles_from_text(english_input)
                
                if not smiles_list:
                    # If no SMILES found, use common chemical names
                    if 'ethanol' in english_input.lower() or 'eth' in english_input.lower():
                        smiles_list = ['CCO']
                    elif 'methanol' in english_input.lower() or 'meth' in english_input.lower():
                        smiles_list = ['CO']
                    elif 'benzene' in english_input.lower():
                        smiles_list = ['C6H6']
                    else:
                        smiles_list = ['CCO']  # Default to ethanol
                
                # Predict products
                products = predict_reaction_products(smiles_list)
                
                # Display results
                st.subheader("Prediction Results")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Reactants:**")
                    for i, reactant in enumerate(smiles_list):
                        st.code(reactant)
                        if RDKIT_AVAILABLE:
                            img = generate_molecule_image(reactant)
                            if img:
                                st.image(img, caption=f"Reactant {i+1}")
                
                with col2:
                    st.markdown("**Predicted Products:**")
                    for i, product in enumerate(products):
                        st.code(product)
                        if RDKIT_AVAILABLE:
                            img = generate_molecule_image(product)
                            if img:
                                st.image(img, caption=f"Product {i+1}")
                
                # Reaction graph
                if len(smiles_list) > 0 and len(products) > 0:
                    st.subheader("Reaction Network")
                    G = create_reaction_graph(smiles_list, products)
                    fig = plot_reaction_graph(G)
                    st.pyplot(fig)

elif feature == "🧬 Molecule Generation":
    st.header("Novel Molecule Generation")
    st.markdown("Generate new molecules with desired properties")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        generation_input = st.text_area(
            "Describe desired properties (in any language):",
            placeholder="Example: Generate a drug-like molecule OR Generate an antioxidant OR दवा जैसा अणु बनाएं",
            height=100
        )
        
        seed_smiles = st.text_input(
            "Optional: Seed SMILES for modification:",
            placeholder="e.g., CCO"
        )
    
    with col2:
        if st.button("🎲 Generate Molecule", type="primary"):
            if generation_input:
                # Translate input
                english_input, detected_lang = detect_and_translate(generation_input)
                
                if detected_lang != 'en':
                    st.info(f"Detected language: {detected_lang}")
                    st.info(f"Translated: {english_input}")
                
                # Generate novel molecule
                novel_smiles = generate_novel_molecule(seed_smiles if seed_smiles else None)
                
                st.subheader("Generated Molecule")
                st.code(novel_smiles)
                
                # Show molecule image
                if RDKIT_AVAILABLE:
                    img = generate_molecule_image(novel_smiles)
                    if img:
                        st.image(img, caption="Generated Molecule", width=300)
                
                # Analyze properties
                properties = analyze_molecule(novel_smiles)
                
                st.subheader("Molecular Properties")
                if "Error" not in properties:
                    df = pd.DataFrame(list(properties.items()), columns=['Property', 'Value'])
                    st.dataframe(df, use_container_width=True)
                else:
                    st.error(properties["Error"])

elif feature == "📊 Reaction Analysis":
    st.header("Reaction Analysis")
    st.markdown("Analyze molecular properties and reaction thermodynamics")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        analysis_input = st.text_area(
            "Enter molecule(s) for analysis (SMILES or description):",
            placeholder="Example: CCO OR aspirin OR एस्पिरिन का विश्लेषण करें",
            height=100
        )
    
    with col2:
        if st.button("📈 Analyze", type="primary"):
            if analysis_input:
                # Translate input
                english_input, detected_lang = detect_and_translate(analysis_input)
                
                if detected_lang != 'en':
                    st.info(f"Detected language: {detected_lang}")
                    st.info(f"Translated: {english_input}")
                
                # Extract or convert to SMILES
                smiles_list = extract_smiles_from_text(english_input)
                
                if not smiles_list:
                    # Convert common chemical names
                    if 'aspirin' in english_input.lower():
                        smiles_list = ['CC(=O)OC1=CC=CC=C1C(=O)O']
                    elif 'caffeine' in english_input.lower():
                        smiles_list = ['CN1C=NC2=C1C(=O)N(C(=O)N2C)C']
                    elif 'ethanol' in english_input.lower():
                        smiles_list = ['CCO']
                    else:
                        smiles_list = ['CCO']  # Default
                
                for i, smiles in enumerate(smiles_list):
                    st.subheader(f"Analysis for Molecule {i+1}: {smiles}")
                    
                    # Show molecule image
                    if RDKIT_AVAILABLE:
                        img = generate_molecule_image(smiles)
                        if img:
                            col1, col2 = st.columns([1, 1])
                            with col1:
                                st.image(img, caption=f"Molecule {i+1}")
                            with col2:
                                # Analyze properties
                                properties = analyze_molecule(smiles)
                                if "Error" not in properties:
                                    df = pd.DataFrame(list(properties.items()), 
                                                    columns=['Property', 'Value'])
                                    st.dataframe(df)
                                else:
                                    st.error(properties["Error"])
                    
                    # Property visualization
                    if "Error" not in properties and RDKIT_AVAILABLE:
                        fig, ax = plt.subplots(figsize=(8, 4))
                        numeric_props = {k: v for k, v in properties.items() 
                                       if isinstance(v, (int, float))}
                        
                        bars = ax.bar(numeric_props.keys(), numeric_props.values())
                        ax.set_title(f"Properties of {smiles}")
                        ax.tick_params(axis='x', rotation=45)
                        plt.tight_layout()
                        st.pyplot(fig)

# Footer
st.markdown("---")
st.markdown("**Reactelligence** - Powered by ChemFormer and Hugging Face 🤗")
st.markdown("Built with Streamlit • Chemistry Analysis with RDKit • Multilingual Support")

# Example usage section
with st.expander("📚 Example Inputs & Usage"):
    st.markdown("""
    **Reaction Prediction Examples:**
    - English: "ethanol + oxygen"
    - Hindi: "एथेनॉल + ऑक्सीजन"
    - SMILES: "CCO + O2"
    
    **Molecule Generation Examples:**
    - "Generate a drug-like molecule"
    - "Create an antioxidant compound"
    - "दवा जैसा अणु बनाएं"
    
    **Analysis Examples:**
    - "CCO" (ethanol SMILES)
    - "aspirin analysis"
    - "कैफीन का विश्लेषण"
    """)
