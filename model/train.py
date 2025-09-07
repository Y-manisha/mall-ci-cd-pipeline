# train.py

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import joblib

# Load dataset
df = pd.read_csv("data/Mall_Customers.csv")

# Select two features
X = df[["Annual Income (k$)", "Spending Score (1-100)"]]

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train KMeans with 5 clusters
kmeans = KMeans(n_clusters=5, random_state=42)
kmeans.fit(X_scaled)

# Save the model 
model = joblib.dump(model, "Mall_Customer_model.pkl")
