import pandas as pd
from sklearn.cluster import KMeans

def kmeans_clustering(df, optimal_k=3, food_timing=None):
    features_kmeans = df[
        ['Calories', 'FatContent', 'SodiumContent', 'CarbohydrateContent', 'FiberContent', 'SugarContent','ProteinContent']]

    # Apply k-means clustering with the chosen number of clusters
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)  # Explicitly set n_init
    df['Cluster'] = kmeans.fit_predict(features_kmeans)

    # Filter the DataFrame based on the assigned food timing cluster
    if food_timing is not None:
        matching_df = df[df['Cluster'] == food_timing]
    else:
        matching_df = df

    return matching_df


# if __name__ == '__main__':
#     # Load the original dataset
#     df = pd.read_csv('./split_file_1.csv')

#     # Step 1: Clustering
#     food_timing_cluster = 2  # You can set this to 0, 1, or 2
#     clustering_df = kmeans_clustering(df, optimal_k=3, food_timing=food_timing_cluster)

#     print(clustering_df)


import matplotlib.pyplot as plt
import matplotlib.cm as cm
# import numpy as np
# import pandas as pd
from sklearn.metrics import silhouette_samples, silhouette_score
# from sklearn.cluster import KMeans

def plot_silhouette_analysis(df, optimal_k):
    features_kmeans = df[['Calories', 'FatContent', 'SodiumContent', 'CarbohydrateContent', 'FiberContent', 'SugarContent', 'ProteinContent']]
    
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(features_kmeans)

    silhouette_avg = silhouette_score(features_kmeans, cluster_labels)
    print(f'Silhouette Score for k={optimal_k}: {silhouette_avg:.2f}')

    sample_silhouette_values = silhouette_samples(features_kmeans, cluster_labels)

    y_lower = 10
    plt.figure(figsize=(10, 7))
    for i in range(optimal_k):
        ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]
        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.nipy_spectral(float(i) / optimal_k)
        plt.fill_betweenx(np.arange(y_lower, y_upper), 0, ith_cluster_silhouette_values, facecolor=color, edgecolor=color, alpha=0.7)

        plt.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
        y_lower = y_upper + 10

    plt.axvline(x=silhouette_avg, color="red", linestyle="--")
    plt.xlabel("Silhouette coefficient values")
    plt.ylabel("Cluster label")
    plt.title("Silhouette plot for the various clusters")
    plt.show()

# if __name__ == '__main__':
#     df = pd.read_csv('./split_file_1.csv')
#     optimal_k = 3  # Assume optimal_k is determined
#     plot_silhouette_analysis(df, optimal_k)

