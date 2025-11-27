import streamlit as st
import pandas as pd
import numpy as np
import ast
import os
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Reviewer Matcher", page_icon="🎓", layout="wide")

# --- 1. LOAD AI MODEL ---
@st.cache_resource
def load_model():
    # This downloads the model (~400MB) to your computer once
    return SentenceTransformer('allenai/specter')

# --- HELPER: GENERATE DUMMY DATA ---
def generate_dummy_data():
    """Generates synthetic experts so the app works without external files."""
    model = SentenceTransformer('allenai/specter')
    
    experts = [
        ("Dr. A. Smith (NLP)", "natural language processing bert transformers llm text analytics"),
        ("Dr. B. Johnson (Vision)", "computer vision image object detection cnn yolo medical imaging"),
        ("Dr. C. Lee (Bio-Tech)", "biology genetics crispr dna sequencing protein folding"),
        ("Dr. D. Patel (Security)", "cybersecurity encryption malware network security cryptography"),
        ("Dr. E. Robot (Robotics)", "robotics kinematics motion planning sensors actuators")
    ]
    
    data = []
    for name, keywords in experts:
        vector = model.encode([keywords])[0]
        data.append({
            'reviewer_id': name,
            'paper_count': 12, 
            'profile_embedding': vector.tolist()
        })
    
    return pd.DataFrame(data)

# --- 2. LOAD DATA (SMART PATH FIX) ---
@st.cache_data
def load_data():
    # CRITICAL FIX: Get the absolute path of the folder where THIS script lives
    # This ensures Python looks in the right folder, not C:\Users\...
    current_folder = os.path.dirname(os.path.abspath(__file__))
    csv_file = os.path.join(current_folder, 'reviewer_profiles.csv')
    
    df = None
    
    # METHOD A: Try to load from CSV
    if os.path.exists(csv_file):
        try:
            df = pd.read_csv(csv_file)
            
            # Clean up the embeddings (Convert string -> array)
            df['profile_embedding'] = df['profile_embedding'].apply(
                lambda x: np.array(ast.literal_eval(x)) if isinstance(x, str) else x
            )
            
            # Validation: Ensure it actually has data
            if df.empty:
                raise ValueError("CSV is empty")
                
        except Exception as e:
            # If CSV is corrupted or empty, we print error and switch to dummy
            print(f"DEBUG: CSV load failed ({e}). Switching to dummy data.")
            df = None 

    # METHOD B: Fallback to Fake Data (If Method A failed)
    if df is None:
        st.warning(f"⚠️ Could not load file at: {csv_file}")
        st.info("🛠️ Using Synthetic Expert Database instead.")
        df = generate_dummy_data()

    return df

# Initialize
with st.spinner("Initializing System..."):
    model = load_model()
    df_reviewers = load_data()

# --- 3. MATCHING ENGINE ---
def get_recommendations(title, abstract, top_k):
    text = title + " " + abstract
    
    # 1. Vectorize Input (Reshape to 2D matrix)
    input_vec = model.encode([text])[0].reshape(1, -1)
    
    # 2. Vectorize Reviewers
    # Ensure all embeddings are valid numpy arrays before stacking
    valid_embeddings = [np.array(e) for e in df_reviewers['profile_embedding']]
    reviewer_matrix = np.vstack(valid_embeddings)
    
    # 3. Calculate Similarity
    scores = cosine_similarity(input_vec, reviewer_matrix)[0]
    
    results = df_reviewers.copy()
    results['similarity_score'] = scores
    return results.sort_values('similarity_score', ascending=False).head(top_k)

# --- 4. USER INTERFACE ---
st.title("🎓 Semantic Reviewer Recommendation System")
st.markdown("Automated assignment using **SciBERT / SPECTER** embeddings.")

if df_reviewers is None:
    st.error("Critical System Failure.")
    st.stop()

# Sidebar
with st.sidebar:
    st.header("Admin Panel")
    st.success(f"✅ System Online")
    st.info(f"Loaded {len(df_reviewers)} Expert Profiles")
    top_k = st.slider("Candidates to Retrieve", 1, 5, 3)

# Main Area
col1, col2 = st.columns(2)

with col1:
    st.subheader("📄 New Submission")
    title = st.text_input("Paper Title", "Attention Is All You Need")
    abstract = st.text_area("Abstract", "We propose a new simple network architecture based solely on attention mechanisms, dispensing with recurrence and convolutions entirely.", height=200)
    btn = st.button("Find Reviewers", type="primary")

with col2:
    st.subheader("🔍 Recommended Experts")
    if btn:
        with st.spinner("Analyzing semantics..."):
            recs = get_recommendations(title, abstract, top_k)
            
            for i, row in recs.iterrows():
                score = row['similarity_score']
                
                # Visual Logic
                if score > 0.65:
                    color = "#d4edda" # Green
                    border = "green"
                    label = "High Confidence"
                elif score > 0.5:
                    color = "#fff3cd" # Yellow
                    border = "orange"
                    label = "Potential Match"
                else:
                    color = "#f8d7da" # Red
                    border = "red"
                    label = "Low Confidence"
                
                st.markdown(f"""
                <div style="background-color:{color}; padding:15px; border-radius:10px; border-left: 5px solid {border}; margin-bottom:10px;">
                    <h4 style="margin:0; color:#333;">👤 {row['reviewer_id']}</h4>
                    <p style="margin:5px 0;"><b>{label}</b> (Score: {score:.1%})</p>
                </div>
                """, unsafe_allow_html=True)