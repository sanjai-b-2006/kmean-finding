import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Function to find the optimal number of clusters using the Elbow Method by analyzing slopes
def find_optimal_k(data, max_k):
    wcss = []
    
    for k in range(1, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(data)
        wcss.append(kmeans.inertia_)
    
    # Calculate the slopes between consecutive points
    slopes = np.diff(wcss)
    
    # Calculate the changes in slopes (second derivative)
    slope_changes = np.diff(slopes)
    
    # Find the index of the maximum change in slope (most significant elbow point)
    elbow_index = np.argmax(np.abs(slope_changes)) + 2 
    
    return elbow_index

# Streamlit app
st.title("Elbow Method for Optimal k")

uploaded_file = st.file_uploader("Upload your dataset (CSV file)", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    
    # Automatically select only the numeric columns
    numeric_data = data.select_dtypes(include=[np.number])
    
    # Input for max_k
    max_k = st.number_input("Enter the maximum number of clusters (max_k):", min_value=2, max_value=20, value=10)
    
    if st.button("Find Optimal k"):
        data_array = numeric_data.values
        optimal_k = find_optimal_k(data_array, max_k)
        st.write(f"The optimal number of clusters is: {optimal_k}")
