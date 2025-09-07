# evaluate.py 

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import joblib

import os
BASE_DIR = os.path.dirname(os.path.dirname(__file__)) # go one level updata_path = os.path.join(BASE_DIR, 'data', 'Mall_Customer.csv')

# Load dataset
df = pd.read_csv("data/Mall_Customers.csv")
X = df[["Annual Income (k$)", "Spending Score (1-100)"]]

# Load saved model
BASE_DIR = os.path.dirname(os.path.dirname(__file__)) # go one level updata_path = os.path.join(BASE_DIR, 'Mall_Customer_model.pkl')
model = joblib.load(Mall_Customer_path)

# Scale the data
X_scaled = scaler.transform(X)

# Predict clusters
labels = kmeans.predict(X_scaled)

# Evaluate with silhouette score
score = silhouette_score(X_scaled, labels)
print(f"Silhouette Score: {score:.4f}")

