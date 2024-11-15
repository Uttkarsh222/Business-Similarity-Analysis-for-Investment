import streamlit as st
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set Streamlit page layout to "wide"
st.set_page_config(page_title="Company Similarity Viewer", layout="wide")

# Custom CSS for styling and increasing width
st.markdown("""
    <style>
    /* Increase overall container width */
    .main .block-container {
        max-width: 85%;
        padding-top: 1rem;
        padding-bottom: 1rem;
        padding-left: 3rem;
        padding-right: 3rem;
    }
    /* Header Styling */
    .header-container {
        background-color: #4A90E2;
        padding: 20px;
        border-radius: 8px;
        margin-bottom: 20px;
    }
    .main-title { 
        font-size: 2.5rem; 
        font-weight: bold; 
        color: #FFFFFF; 
        margin: 0; 
    }
    .subheader { 
        font-size: 1.1rem; 
        color: #F0F2F6; 
        margin-top: 5px; 
    }
    /* Sidebar Styling */
    .sidebar .sidebar-content { background-color: #F0F2F6; padding: 20px; border-radius: 10px; }
    /* Button Styling */
    .stButton > button { 
        background-color: #4CAF50; 
        color: white; 
        font-size: 1rem; 
        border-radius: 10px; 
        padding: 0.5rem 1rem; 
    }
    /* Card Styling for Similar Companies */
    .company-card {
        background-color: #F9F9F9;
        padding: 15px;
        border-radius: 8px;
        margin-bottom: 10px;
        border: 1px solid #e0e0e0;
    }
    .company-card h4 { 
        font-size: 1.2rem; 
        color: #4A90E2; 
        margin: 0; 
    }
    .company-card p { 
        font-size: 1rem; 
        color: #6E6E6E; 
    }
    /* Footer Styling */
    .footer {
        text-align: center; 
        padding-top: 20px; 
        font-size: 0.9rem; 
        color: #888;
    }
    </style>
    """, unsafe_allow_html=True)

# Header
st.markdown("""
<div class="header-container">
    <h1 class="main-title">🔍 Company Similarity Viewer</h1>
    <p class="subheader">Discover and compare companies based on their descriptions. This tool uses TF-IDF and cosine similarity to find the most similar companies.</p>
</div>
""", unsafe_allow_html=True)

# Sidebar for settings and visualizations
st.sidebar.header("⚙️ Settings & Visualizations")

# Load raw data for missing data visualization (before processing)
@st.cache_data
def load_raw_data():
    raw_data = pd.read_excel('innovius_case_study_data.xlsx', engine='openpyxl', na_values=['NA', 'null', 'missing', 'N/A', 'NaN'])
    return raw_data

# Load preprocessed data and models for similarity calculations
@st.cache_data
def load_processed_data():
    with open('data.pkl', 'rb') as f:
        data = pickle.load(f)
    return data

@st.cache_resource
def load_models():
    with open('vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    with open('svd.pkl', 'rb') as f:
        svd = pickle.load(f)
    with open('nearest_neighbors.pkl', 'rb') as f:
        nearest_neighbors = pickle.load(f)
    with open('tfidf_reduced.pkl', 'rb') as f:
        tfidf_reduced = pickle.load(f)
    return vectorizer, svd, nearest_neighbors, tfidf_reduced

# Load raw and processed data
raw_data = load_raw_data()
data = load_processed_data()
vectorizer, svd, nearest_neighbors, tfidf_reduced = load_models()

# Function to retrieve top-N similar companies
def get_similar_companies(company_name, top_n=5):
    idx = data[data['Name'] == company_name].index[0]
    distances, indices = nearest_neighbors.kneighbors([tfidf_reduced[idx]], n_neighbors=top_n)
    results = [(data['Name'].iloc[i], 1 - dist) for i, dist in zip(indices[0], distances[0])]
    return results

# Enhanced Missing Data Visualization (using raw data)
if st.sidebar.button("📊 Show Missing Data Visualization"):
    def visualize_missing(data, title='Missing Data Before Processing'):
        missing_data = data.isnull().sum()
        missing_data = missing_data[missing_data > 0]  # Only show columns with missing values

        plt.figure(figsize=(12, 6))
        sns.barplot(x=missing_data.index, y=missing_data.values, palette='coolwarm')
        plt.title(title, fontsize=16)
        plt.xlabel('Columns', fontsize=14)
        plt.ylabel('Number of Missing Values', fontsize=14)
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(plt)
    
    visualize_missing(raw_data)

# Dropdown for company selection
company_name = st.selectbox("🔹 Select a Company:", data['Name'].unique())

# Slider for number of similar companies
top_n = st.sidebar.slider("🔢 Number of similar companies to display:", 1, 10, 5)

# Display Results with Enhanced Formatting and Cards for Similar Companies
if st.button("Show Similar Companies"):
    st.write(f"## 🏢 Top {top_n} Similar Companies to '{company_name}':")
    similar_companies = get_similar_companies(company_name, top_n=top_n)
    
    # Display each similar company in a "card" style div for better UI
    for comp, score in similar_companies:
        st.markdown(f"""
        <div class="company-card">
            <h4>{comp} - Similarity Score: {score:.4f}</h4>
            <p><strong>Description:</strong> {data[data['Name'] == comp].iloc[0]['Description']}</p>
            <p><strong>Top Level Category:</strong> {data[data['Name'] == comp].iloc[0]['Top Level Category']}</p>
            <p><strong>Secondary Category:</strong> {data[data['Name'] == comp].iloc[0]['Secondary Category']}</p>
            <p><strong>Employee Count:</strong> {data[data['Name'] == comp].iloc[0]['Employee Count']}</p>
        </div>
        """, unsafe_allow_html=True)

# Footer
st.markdown("""
<div class="footer">
    <hr style="margin-top: 2rem; margin-bottom: 1rem; border: none; border-top: 1px solid #ddd;" />
    <p style="text-align: center; font-size: 0.9rem; color: #888;">
        Created by <strong>Uttkarsh Bharadia</strong> | Connect with me on 
        <a href="https://github.com/Uttkarsh222?tab=repositories" target="_blank" style="text-decoration: none;">
            <img src="https://img.icons8.com/ios-glyphs/30/4A90E2/github.png" style="vertical-align: middle;" />
            GitHub
        </a> and 
        <a href="https://www.linkedin.com/in/uttkarsh-bharadia/" target="_blank" style="text-decoration: none;">
            <img src="https://img.icons8.com/ios-filled/30/0077B5/linkedin.png" style="vertical-align: middle;" />
            LinkedIn
        </a>
    </p>
</div>
""", unsafe_allow_html=True)
