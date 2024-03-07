import pandas as pd
from sklearn.cluster import KMeans

def kmeans_clustering():
    # Load the dataset
    dataset_path='./split_file_1.csv'
    optimal_k=3
    df = pd.read_csv(dataset_path)

    # Select features for K-means clustering
    features_kmeans = df[['Calories', 'FatContent', 'SodiumContent', 'CarbohydrateContent', 'FiberContent', 'SugarContent', 'ProteinContent']]

    # Determine the optimal number of clusters using the Elbow Method
    inertia = []
    for k in range(1, 11):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)  # Explicitly set n_init
        kmeans.fit(features_kmeans)
        inertia.append(kmeans.inertia_)

    # Apply k-means clustering with the chosen number of clusters
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)  # Explicitly set n_init
    df['Cluster'] = kmeans.fit_predict(features_kmeans)

    # Display the clustered data
    result_df = df[['RecipeId', 'Name', 'CookTime', 'PrepTime', 'TotalTime', 'RecipeIngredientParts',
                    'Calories', 'FatContent', 'SaturatedFatContent', 'CholesterolContent', 'SodiumContent',
                    'CarbohydrateContent', 'FiberContent', 'SugarContent', 'ProteinContent', 'RecipeInstructions', 'Cluster']]
    
    return result_df


result = kmeans_clustering()
print(result)
