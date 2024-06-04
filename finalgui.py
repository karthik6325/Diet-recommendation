
from tabulate import tabulate
import pandas as pd
from clustering import kmeans_clustering
from knn_weight_loss import recipe_prediction_function_weight_loss
from knn_weight_gain import recipe_prediction_function_weight_gain
from knn_healthy import recipe_prediction_function_healthy
from disease_recommendation import recommend_recipes_for_disease

def calculate_calories_for_weight_gain(weight, age, height, desired_gain_kg, num_days, activity_factor, gender):
    
    if gender == 1:  # Assuming 1 for male, 0 for female
        bmr = 10 * weight + 6.25 * height - 5 * age + 5
    else:
        bmr = 10 * weight + 6.25 * height - 5 * age - 161
      
    # Calculate Total Daily Energy Expenditure (TDEE)
    tdee = bmr * activity_factor
    
    # Calculate total additional calories needed for desired weight gain
    additional_calories = (desired_gain_kg * 7700) / num_days
    
    # Calculate total daily calories needed for weight gain
    total_calories = tdee + additional_calories
    
    return total_calories


def calculate_calories_for_weight_loss(weight, age, height, desired_loss_kg, num_days, activity_factor, gender):
    # Calculate Basal Metabolic Rate (BMR) using Mifflin-St Jeor Equation
    if gender == 1:  # Assuming 1 for male, 0 for female
        bmr = 10 * weight + 6.25 * height - 5 * age + 5
    else:
        bmr = 10 * weight + 6.25 * height - 5 * age - 161
        
    # Calculate Total Daily Energy Expenditure (TDEE)
    tdee = bmr * activity_factor
    
    # Calculate total calories needed for weight loss
    total_calories = tdee - (desired_loss_kg * 7700) / num_days
    
    if total_calories >= 0:
        return total_calories
    
    # Adjust total calories if it's negative
    total_calories = max(100, 500 + (total_calories * 0.5))
            
    return total_calories

def calculate_calories_for_healthy(weight, age, height, activity_level,gender):
    
     # Calculate Basal Metabolic Rate (BMR) using Mifflin-St Jeor Equation
    if gender == 1:  # Assuming 1 for male, 0 for female
        bmr = 10 * weight + 6.25 * height - 5 * age + 5
    else:
        bmr = 10 * weight + 6.25 * height - 5 * age - 161
        

    if activity_level == 1:
        activity_factor = 1.2
    elif activity_level == 2:
        activity_factor = 1.375
    elif activity_level == 3:
        activity_factor = 1.55
    elif activity_level == 4:
        activity_factor = 1.725
    elif activity_level == 5:
        activity_factor = 1.9
    
    total_calories = bmr * activity_factor    
    return total_calories//3


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

def Weight_Loss(age,weight,height,food_timing,disease,desired_loss_kg,num_days,activity_level):
    # Load the original dataset
    df = pd.read_csv('./split_file_1.csv')

    # Step 1: Disease Recommendation
    disease_recommendation_df = recommend_recipes_for_disease(df, disease)
      # Calculate Total Daily Energy Expenditure (TDEE) based on activity level
    if activity_level == 1:
        activity_factor = 1.2
    elif activity_level == 2:
        activity_factor = 1.375
    elif activity_level == 3:
        activity_factor = 1.55
    elif activity_level == 4:
        activity_factor = 1.725
    elif activity_level == 5:
        activity_factor = 1.9
    else:
     raise ValueError("Invalid activity level. Choose from:1-> sedentary, 2->lightly active,3-> moderately active,4-> very active,5-> extremely active")

    # Step 2: Clustering
    clustering_df = kmeans_clustering(df,3,food_timing)

    # Step 3: KNN
    cal_intake=calculate_calories_for_weight_loss(weight, age, height, desired_loss_kg, num_days, activity_factor)
    cal_protein=(cal_intake)//4
    cal_carb=(cal_intake)//4
    cal_fat=(cal_intake)//9
    approx_calories = cal_intake-cal_carb-cal_protein-cal_fat
    approx_protein = cal_protein
    approx_carbohydrate = cal_carb
    approx_fat = cal_fat
    matching_recipes_df = recipe_prediction_function_weight_loss(clustering_df, approx_calories, approx_protein, approx_carbohydrate, approx_fat)

    # Step 4: Match Recipe IDs
    matched_df = match_recipe_ids(disease_recommendation_df, matching_recipes_df, df)
    
     # Calculate the acceptable range for approx_calories
    lower_bound = approx_calories * 0.85
    upper_bound = approx_calories * 0.15
    
    # Filter matched_df to only contain values within 15% of the range of approx_calories
    filtered_matched_df = matched_df[(matched_df['calories'] >= lower_bound) & (matched_df['calories'] <= upper_bound)]
    
    return filtered_matched_df
            

def Weight_Gain(age,weight,height,food_timing,disease,desired_gain_kg,num_days,activity_level):
    # Load the original dataset
    df = pd.read_csv('./split_file_1.csv')

    # Step 1: Disease Recommendation
    disease_recommendation_df = recommend_recipes_for_disease(df, disease)
     # Calculate Total Daily Energy Expenditure (TDEE) based on activity level
    if activity_level == 1:
        activity_factor = 1.2
    elif activity_level == 2:
        activity_factor = 1.375
    elif activity_level == 3:
        activity_factor = 1.55
    elif activity_level == 4:
        activity_factor = 1.725
    elif activity_level == 5:
        activity_factor = 1.9
    else:
     raise ValueError("Invalid activity level. Choose from:1-> sedentary, 2->lightly active,3-> moderately active,4-> very active,5-> extremely active")

    # Step 2: Clustering
    clustering_df = kmeans_clustering(df,3,food_timing)

    # Step 3: KNN
    cal_intake=calculate_calories_for_weight_gain(weight, age, height, desired_gain_kg, num_days, activity_factor)
    #convert calorie to protein
    cal_protein=(cal_intake)//4
    # Step 4: Predict Matching Recipes
    approx_calories = cal_intake-cal_protein
    approx_protein = cal_protein
    matching_recipes_df = recipe_prediction_function_weight_gain(clustering_df, approx_calories, approx_protein)

    # Step 5: Match Recipe IDs
    matched_df = match_recipe_ids(disease_recommendation_df, matching_recipes_df, df)

     # Calculate the acceptable range for approx_calories
    lower_bound = approx_calories * 0.85
    upper_bound = approx_calories * 0.15
    
    # Filter matched_df to only contain values within 15% of the range of approx_calories
    filtered_matched_df = matched_df[(matched_df['calories'] >= lower_bound) & (matched_df['calories'] <= upper_bound)]
    
    return filtered_matched_df
                 

def Healthy(age,weight,height,food_timing,disease,activity_level):
    df = pd.read_csv('./split_file_1.csv')

    # Step 1:Disease Recommendation
    disease_recommendation_df=recommend_recipes_for_disease(df, disease)

    # Step 2:Clustering
    clustering_df=kmeans_clustering(df,3,food_timing)

    # Step 3:KNN
    approx_calories = calculate_calories_for_healthy(weight, age, height,activity_level)
    
    matching_recipes_df = recipe_prediction_function_healthy(clustering_df, approx_calories)

    # Step 4:Match Recipe IDs
    matched_df=match_recipe_ids(disease_recommendation_df, matching_recipes_df, df)
    
     # Calculate the acceptable range for approx_calories
    lower_bound = approx_calories * 0.85
    upper_bound = approx_calories * 0.15
    
    # Filter matched_df to only contain values within 15% of the range of approx_calories
    filtered_matched_df = matched_df[(matched_df['calories'] >= lower_bound) & (matched_df['calories'] <= upper_bound)]
    
    return filtered_matched_df




