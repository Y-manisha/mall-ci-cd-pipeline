# train.py

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import joblib
import os 

BASE_DIR = os.path.dirname(os.path.dirname(__file__)) # go one level 
data_path = os.path.join(BASE_DIR, 'data', 'Mall_Customers.csv')

# Load dataset
df = pd.read_csv(data_path)

# Select two features
X = df[["Annual Income (k$)", "Spending Score (1-100)"]]

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train KMeans with 5 clusters
model = KMeans(n_clusters=5, random_state=42)
model.fit(X_scaled)

# Save the model 
model = joblib.dump(model, "Mall_Customers_model.pkl")
