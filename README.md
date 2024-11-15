# Business Similarity Analysis for Investment

This project provides a solution for computing and visualizing similarity scores between companies based on their descriptions. By analyzing business descriptions, this tool helps uncover insights into market trends, competition, and potential investment opportunities. The application is built using Python, and the interactive user interface is developed with Streamlit.

## Project Overview

This case study involves multiple data science and data engineering tasks:
- **Data Cleaning & Preprocessing**: Handling missing values and preparing the text data for analysis.
- **Feature Extraction**: Converting text descriptions into a numerical format using TF-IDF (Term Frequency-Inverse Document Frequency) and dimensionality reduction with SVD (Singular Value Decomposition).
- **Similarity Calculation**: Calculating similarity scores between company descriptions using cosine similarity.
- **Scalability**: Designing a scalable data pipeline to efficiently process and update similarity scores as new companies are added.
- **Visualization**: Building an interactive Streamlit application that allows users to view and analyze similarity scores.

## Table of Contents
- [Project Overview]
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Data Pipeline Architecture](#data-pipeline-architecture)
- [Technologies Used](#technologies-used)
- [Contributors](#contributors)


## Features

- **Missing Data Visualization**: Analyze missing data before processing to understand data quality.
- **Data Cleaning**: Fill missing values and clean text fields for accurate analysis.
- **Interactive Company Similarity Viewer**: Select a company and view the most similar companies based on description similarity.
- **Scalable Data Pipeline Design**: Outline for a pipeline that can process new data entries efficiently.

## Installation

1. **Clone the repository**:
bash:

   git clone https://github.com/Uttkarsh222/Business-Similarity-Analysis-for-Investment.git
   cd Business-Similarity-Analysis-for-Investment

Create a virtual environment (recommended):

bash:

    python -m venv env
    source env/bin/activate  # On Windows use `env\Scripts\activate`
    Install dependencies:

bash:

    pip install -r requirements.txt
    Add Data Files:

Place your data file (innovius_case_study_data.xlsx) in the root directory.
Run the App:

bash:

    streamlit run app.py

Usage :

Open the app in your browser (usually at http://localhost:8501).
In the sidebar, view missing data visualization and configure settings.
Select a company to view similar companies, and adjust the number of similar companies displayed.
Click "Show Similar Companies" to view detailed information and similarity scores for the top matches.

Data Pipeline Architecture:

The proposed data pipeline handles the following steps:
Data Ingestion: New entries are added to the raw data source.
Data Preprocessing: Clean and prepare the new data, handling missing values and other issues.
Feature Extraction: Transform text descriptions into TF-IDF vectors and reduce dimensions with SVD.
Similarity Calculation: Compute similarity scores between companies using cosine similarity.
Storage & Retrieval: Store processed data for fast retrieval in the Streamlit application.

Technologies Used
Python: Main programming language for data processing and application logic.
Streamlit: For building an interactive web interface.
Scikit-learn: Used for TF-IDF vectorization, dimensionality reduction, and nearest neighbors similarity calculations.
Pandas: Data manipulation and cleaning.
Matplotlib & Seaborn: Visualization libraries for data analysis and missing data visualization.


Contributors
- Uttkarsh Bharadia


