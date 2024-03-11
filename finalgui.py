
from tabulate import tabulate
import pandas as pd
from clustering import kmeans_clustering
from knn_weight_loss import train_knn_model_weightloss, predict_matching_recipes_weightloss
from knn_weight_gain import train_knn_model_weightgain, predict_matching_recipes_weightgain
from knn_healthy import train_knn_model_healthy, predict_matching_recipes_healthy
from disease_recommendation import recommend_recipes_for_disease

def calculate_calories_for_weight_gain(weight, age, height, desired_gain_kg, num_days, activity_factor):
    # Calculate Basal Metabolic Rate (BMR) using Mifflin-St Jeor Equation
    bmr = 10*weight+6.25*height-5*age+5

    tdee = bmr * activity_factor
    
    # Calculate total calories needed for weight gain
    total_calories = tdee + (desired_gain_kg * 7700) / num_days
    
    return total_calories//3


def calculate_calories_for_weight_loss(weight, age, height, desired_loss_kg, num_days, activity_factor):
    # Calculate Basal Metabolic Rate (BMR) using Mifflin-St Jeor Equation
    bmr =10*weight+6.25*height-5*age+5
    
    tdee =bmr*activity_factor
    
    # Calculate total calories needed for weight loss
    total_calories =tdee-(desired_loss_kg * 7700)/num_days
    if total_calories < 0:
        if total_calories < -900:
            total_calories = 50 + (total_calories + 900) * (50 / 100)  
        elif total_calories < -800:
            total_calories = 100 + (total_calories + 800) * (50 / 100)  
        elif total_calories < -700:
            total_calories = 150 + (total_calories + 700) * (50 / 100)
        elif total_calories < -600:
            total_calories = 200 + (total_calories + 600) * (50 / 100)  
        elif total_calories < -500:
            total_calories = 250 + (total_calories + 500) * (50 / 100)  
        elif total_calories <-400:
            total_calories = 300 + (total_calories + 400) * (50 / 100)  
        elif total_calories < -300:
            total_calories = 350 + (total_calories + 300) * (50 / 100) 
        elif total_calories < -200:
            total_calories = 400 + (total_calories + 200) * (50 / 100) 
        elif total_calories < -100:
            total_calories = 450 + (total_calories + 100) * (50 / 100)  
        else:
            total_calories =0
            
    return total_calories//3

def calculate_calories_for_healthy(weight, age, height):
    # Calculate Basal Metabolic Rate (BMR) using Mifflin-St Jeor Equation
    bmr = 10 * weight + 6.25 * height - 5 *age+5
    
    tdee = bmr * activity_factor    
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
    knn_model, label_encoder, scaler = train_knn_model_weightloss(clustering_df)
     
    # Step 4: Predict Matching Recipes
    cal_intake=calculate_calories_for_weight_loss(weight, age, height, desired_loss_kg, num_days, activity_factor)
    cal_protein=(cal_intake)//4
    cal_carb=(cal_intake)//4
    cal_fat=(cal_intake)//9
    approx_calories = cal_intake-cal_carb-cal_protein-cal_fat
    approx_protein = cal_protein
    approx_carbohydrate = cal_carb
    approx_fat = cal_fat
    matching_recipes_df = predict_matching_recipes_weightloss(knn_model, label_encoder, scaler,
                                                   approx_calories, approx_protein, approx_carbohydrate, approx_fat, clustering_df)

    # Step 5: Match Recipe IDs
    matched_df = match_recipe_ids(disease_recommendation_df, matching_recipes_df, df)

    print(matched_df)
            

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
    knn_model, label_encoder, scaler = train_knn_model_weightgain(clustering_df)
    cal_intake=calculate_calories_for_weight_gain(weight, age, height, desired_gain_kg, num_days, activity_factor)
    #convert calorie to protein
    cal_protein=(cal_intake)//4
    # Step 4: Predict Matching Recipes
    approx_calories = cal_intake-cal_protein
    approx_protein = cal_protein
    matching_recipes_df = predict_matching_recipes_weightgain(knn_model, label_encoder, scaler,
                                                   approx_calories, approx_protein, clustering_df)

    # Step 5: Match Recipe IDs
    matched_df = match_recipe_ids(disease_recommendation_df, matching_recipes_df, df)

    print(matched_df)
                 

def Healthy(age,weight,height,food_timing,disease):
    df = pd.read_csv('./split_file_1.csv')

    # Step 1:Disease Recommendation
    disease_recommendation_df=recommend_recipes_for_disease(df, disease)

    # Step 2:Clustering
    clustering_df=kmeans_clustering(df,3,food_timing)

    # Step 3:KNN
    knn_model,label_encoder,scaler = train_knn_model_healthy(clustering_df)

    # Step 4:Predict Matching Recipes
    approx_calories = calculate_calories_for_healthy(weight, age, height)
    
    matching_recipes_df = predict_matching_recipes_healthy(knn_model, label_encoder, scaler,
                                                   approx_calories, clustering_df)

    # Step 5:Match Recipe IDs
    matched_df=match_recipe_ids(disease_recommendation_df, matching_recipes_df, df)
    print(matched_df)




