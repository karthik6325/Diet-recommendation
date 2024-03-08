import pandas as pd
from sklearn.cluster import KMeans

def kmeans_clustering(df, optimal_k=3):
    print(df.columns)
    features_kmeans = df[
        ['Calories', 'FatContent', 'SodiumContent', 'CarbohydrateContent', 'FiberContent', 'SugarContent','ProteinContent']]

    # Determine the optimal number of clusters using the Elbow Method
    inertia = []
    for k in range(1, 11):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)  # Explicitly set n_init
        kmeans.fit(features_kmeans)
        inertia.append(kmeans.inertia_)

    # Apply k-means clustering with the chosen number of clusters
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)  # Explicitly set n_init
    df['Cluster'] = kmeans.fit_predict(features_kmeans)

    return df
