import pandas as pd
from sklearn.cluster import KMeans

# Load your dataset (replace 'your_dataset.csv' with the actual path to your CSV file)
df = pd.read_csv('c:/Users/karth/Downloads/Diet-Recommendation-System-main/split_file_1.csv')

# Select features for K-means clustering
features_kmeans = df[['FiberContent', 'Calories']]

# Determine the optimal number of clusters using the Elbow Method
inertia = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)  # Explicitly set n_init
    kmeans.fit(features_kmeans)
    inertia.append(kmeans.inertia_)

# Choose the optimal number of clusters (k) based on the Elbow Method
optimal_k = 3  # Adjust this based on the plot or other considerations

# Apply k-means clustering with the chosen number of clusters
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)  # Explicitly set n_init
df['Cluster'] = kmeans.fit_predict(features_kmeans)

# Display the clustered data
print(df[['RecipeId', 'Name', 'Cluster']])
