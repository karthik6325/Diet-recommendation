import pandas as pd
from clustering import kmeans_clustering
from knn_weight_loss import train_knn_model, predict_matching_recipes
from disease_recommendation import recommend_recipes_for_disease

def match_recipe_ids(disease_df, matching_df, original_df):
    # Create a mapping of RecipeId to its index in the original DataFrame
    recipe_id_mapping = {recipe_id: idx for idx, recipe_id in enumerate(original_df['RecipeId'])}

    # Map the RecipeId back to the original order in both DataFrames
    disease_df['OriginalIndex'] = disease_df['RecipeId'].map(recipe_id_mapping)
    
    # Add 'Similarity' column to matching_df
    matching_df = matching_df.merge(disease_df[['RecipeId', 'Similarity']], on='RecipeId', how='left')
    matching_df['OriginalIndex'] = matching_df['RecipeId'].map(recipe_id_mapping)

    # Sort DataFrames by the order of similarity
    disease_df_sorted = disease_df.sort_values(by='Similarity', ascending=False)
    matching_df_sorted = matching_df.sort_values(by='Similarity', ascending=False)

    # Extract the RecipeId in the order of similarity
    disease_recipe_ids = disease_df_sorted['RecipeId'].tolist()

    # Match RecipeIds
    matched_recipe_ids = []
    for recipe_id in disease_recipe_ids:
        matching_row = matching_df_sorted[matching_df_sorted['RecipeId'] == recipe_id].head(1)
        if not matching_row.empty:
            matched_recipe_ids.append(matching_row['RecipeId'].iloc[0])

    # Filter the original DataFrame based on matched RecipeIds
    matched_df = original_df[original_df['RecipeId'].isin(matched_recipe_ids)]

    return matched_df

if __name__ == '__main__':
    # Load the original dataset
    df = pd.read_csv('./split_file_1.csv')

    # Step 1: Disease Recommendation
    disease = 'hypertension'
    disease_recommendation_df = recommend_recipes_for_disease(df, disease)
    food_timing=2

    # Step 2: Clustering
    clustering_df = kmeans_clustering(df,3,food_timing)

    # Step 3: KNN
    knn_model, label_encoder, scaler = train_knn_model(clustering_df)

    # Step 4: Predict Matching Recipes
    approx_calories = 500
    approx_protein = 20
    approx_carbohydrate = 30
    approx_fat = 15
    matching_recipes_df = predict_matching_recipes(knn_model, label_encoder, scaler,
                                                   approx_calories, approx_protein, approx_carbohydrate, approx_fat, clustering_df)

    # Step 5: Match Recipe IDs
    matched_df = match_recipe_ids(disease_recommendation_df, matching_recipes_df, df)

    print(matched_df)
