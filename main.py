from clustering import kmeans_clustering
from knn import train_knn_model, predict_matching_recipes
from disease_recommendation import recommend_recipes_for_disease
import pandas as pd

if __name__ == '__main__':
    # Load the original dataset
    df = pd.read_csv('./split_file_1.csv')

    # Step 1: Disease Recommendation
    disease = 'hypertension'
    disease_recommendation_df = recommend_recipes_for_disease(df, disease)

    # Create a mapping of RecipeId to its index in the original DataFrame
    recipe_id_mapping = {recipe_id: idx for idx, recipe_id in enumerate(df['RecipeId'])}

    # Step 2: Clustering
    clustering_df = kmeans_clustering(disease_recommendation_df)

    # Map the Cluster column back to the original order
    clustering_df['Cluster'] = clustering_df['RecipeId'].map(recipe_id_mapping)

    # Step 3: KNN
    knn_model, label_encoder, scaler = train_knn_model(clustering_df)

    # Step 4: Predict Matching Recipes
    approx_calories = 500
    approx_protein = 20
    approx_carbohydrate = 30
    approx_fat = 15
    matching_recipes_df = predict_matching_recipes(knn_model, label_encoder, scaler,
                                                   approx_calories, approx_protein, approx_carbohydrate, approx_fat, clustering_df)

    # Map the RecipeId back to the original order
    matching_recipes_df['RecipeId'] = matching_recipes_df['RecipeId'].map(recipe_id_mapping)

    # Sort the final DataFrame by the original order
    matching_recipes_df = matching_recipes_df.sort_values(by='RecipeId')

    print(matching_recipes_df)
