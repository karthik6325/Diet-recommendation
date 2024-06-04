import pandas as pd
from sklearn.cluster import KMeans

def kmeans_clustering(df, optimal_k=3, food_timing=None):
    features_kmeans = df[
        ['CarbohydrateContent', 'FiberContent']]

    # Determine the optimal number of clusters using the Elbow Method
    inertia = []
    for k in range(1, 11):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)  # Explicitly set n_init
        kmeans.fit(features_kmeans)
        inertia.append(kmeans.inertia_)

    # Apply k-means clustering with the chosen number of clusters
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)  # Explicitly set n_init
    df['Cluster'] = kmeans.fit_predict(features_kmeans)

    # Filter the DataFrame based on the assigned food timing cluster
    if food_timing is not None:
        matching_df = df[df['Cluster'] == food_timing]
    else:
        matching_df = df

    return matching_df

if __name__ == '__main__':
    # Load the original dataset
    df = pd.read_csv('./split_file_1.csv')

    # Step 1: Clustering
    food_timing_cluster = 2  # You can set this to 0, 1, or 2
    clustering_df = kmeans_clustering(df, optimal_k=3, food_timing=food_timing_cluster)

    print(clustering_df[['RecipeId','CarbohydrateContent', 'FiberContent','ProteinContent']])