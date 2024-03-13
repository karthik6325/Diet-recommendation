import pandas as pd
import spacy
import numpy as np
from tqdm import tqdm
import concurrent.futures
from tabulate import tabulate


disease_ingredients_mapping = {
    'heart_disease': ['Salmon', 'Oats', 'Berries', 'Nuts', 'Olive oil', 'Leafy green vegetables'],
    'hypertension': ['Bananas', 'Berries', 'Oats', 'Leafy green vegetables', 'Beets', 'Low-fat dairy products'],
    'obesity': ['Lean proteins', 'Whole grains', 'Fruits and vegetables', 'Legumes', 'Nuts and seeds', 'Greek yogurt'],
    'diabetes': ['Whole grains', 'Leafy green vegetables', 'Berries', 'Beans and lentils', 'Nuts and seeds', 'Fatty fish'],
    'kidney_disease': ['Cauliflower', 'Berries', 'Cabbage', 'Apples', 'Fish', 'Egg whites'],
    'rickets': ['Fatty fish', 'Milk and dairy products', 'Egg yolks', 'Cheese', 'Fortified cereals', 'Fortified orange juice'],
    'scurvy': ['Citrus fruits', 'Strawberries', 'Kiwi', 'Bell peppers', 'Broccoli', 'Tomatoes'],
    'anemia': ['Iron-rich foods', 'Beans and lentils', 'Tofu', 'Spinach', 'Broccoli', 'Fortified cereals'],
    'goitre': ['Iodized salt', 'Seafood', 'Dairy products', 'Eggs', 'Iodine-rich fruits and vegetables', 'Cranberries'],
    'eye_disease': ['Carrots', 'Sweet potatoes', 'Spinach', 'Kale', 'Bell peppers', 'Eggs'],
    'low_blood_pressure': ['Hydration', 'Small, frequent meals', 'Salt', 'Lean proteins', 'Healthy fats', 'Whole grains'],
    'thyroid': ['Iodized salt', 'Seafood', 'Dairy products', 'Eggs', 'Nuts and seeds', 'Whole grains'],
    'cholera': ['Rehydration solutions', 'Boiled or treated water', 'Oral rehydration salts', 'Bananas', 'Rice', 'Applesauce'],
    'malnutrition': ['Nutrient-dense foods', 'Whole grains', 'Lean proteins', 'Fruits and vegetables', 'Nuts and seeds', 'Dairy products'],
}

ingredients_to_avoid_mapping = {
    'heart_disease': ['Processed meats', 'Trans fats', 'Excessive salt', 'Sugary beverages', 'Processed snacks'],
    'hypertension': ['Salt', 'Canned soups', 'Processed deli meats', 'Pickles and olives', 'Frozen meals'],
    'obesity': ['Sugary beverages', 'Processed snacks and sweets', 'Fast food', 'Highly processed foods', 'Fried foods'],
    'diabetes': ['Highly processed carbohydrates', 'Sugary beverages', 'Sweets and candies', 'White bread', 'Fried foods'],
    'kidney_disease': ['High-potassium foods', 'High-phosphorus foods', 'Processed and cured meats', 'Cola', 'Packaged and processed foods'],
    'rickets': ['Excessive caffeine', 'Alcohol', 'Processed and fried foods', 'Sodas and sugary drinks', 'High-phosphorus foods'],
    'scurvy': [],
    'anemia': ['Calcium-rich foods', 'Tea and coffee', 'High-fiber foods'],
    'goitre': ['Excessive consumption of raw cruciferous vegetables', 'Non-iodized salt', 'Unpasteurized dairy products'],
    'eye_disease': ['Highly processed foods', 'Sugary beverages', 'Trans fats', 'Excessive alcohol', 'Smoking'],
    'low_blood_pressure': ['Excessive caffeine', 'High-sugar foods', 'Processed snacks', 'Alcohol', 'Dehydration'],
    'thyroid': ['Excessive consumption of raw cruciferous vegetables', 'Unpasteurized dairy products', 'Processed foods', 'Soy-based products', 'Excessive caffeine'],
    'cholera': ['Raw or undercooked seafood', 'Raw fruits and vegetables', 'Unpasteurized dairy products', 'Caffeine and alcohol'],
    'malnutrition': ['Processed and refined foods', 'Sugary beverages', 'Highly processed snacks', 'Excessive caffeine', 'Alcohol'],
}

nlp = spacy.load('en_core_web_sm')

def calculate_similarity_and_filter(row, disease):
    recipe_id = row['RecipeId']
    name = row['Name']
    cook_time = row['CookTime']
    prep_time = row['PrepTime']
    total_time = row['TotalTime']
    recipe_ingredient_parts = row['RecipeIngredientParts']
    calories = row['Calories']
    fat_content = row['FatContent']
    saturated_fat_content = row['SaturatedFatContent']
    cholesterol_content = row['CholesterolContent']
    sodium_content = row['SodiumContent']
    carbohydrate_content = row['CarbohydrateContent']
    fiber_content = row['FiberContent']
    sugar_content = row['SugarContent']
    protein_content = row['ProteinContent']
    recipe_instructions = row['RecipeInstructions']

    # Use recommended ingredients for the given disease
    disease_ingredients = disease_ingredients_mapping.get(disease, [])
    avoid_ingredients = ingredients_to_avoid_mapping.get(disease, [])

    # Process the ingredients with spaCy
    recipe_embeddings = nlp(recipe_ingredient_parts).vector
    disease_ingredient_embeddings = nlp(' '.join(disease_ingredients)).vector

    # Calculate the Euclidean norm using numpy.linalg.norm
    similarity = np.dot(recipe_embeddings, disease_ingredient_embeddings) / (
            np.linalg.norm(recipe_embeddings) * np.linalg.norm(disease_ingredient_embeddings))

    # Remove recipes with ingredients to avoid
    for avoid_ingredient in avoid_ingredients:
        if avoid_ingredient.lower() in recipe_ingredient_parts.lower():
            similarity = 0  # Set similarity to zero if ingredients to avoid are found
            break

    # Include only specific columns in the result
    result = {
        'RecipeId': recipe_id,
        'Name': name,
        'CookTime': cook_time,
        'PrepTime': prep_time,
        'TotalTime': total_time,
        'RecipeIngredientParts': recipe_ingredient_parts,
        'Calories': calories,
        'FatContent': fat_content,
        'SaturatedFatContent': saturated_fat_content,
        'CholesterolContent': cholesterol_content,
        'SodiumContent': sodium_content,
        'CarbohydrateContent': carbohydrate_content,
        'FiberContent': fiber_content,
        'SugarContent': sugar_content,
        'ProteinContent': protein_content,
        'RecipeInstructions': recipe_instructions,
        'Disease': disease,
        'RecommendedIngredients': disease_ingredients,
        'AvoidIngredients': avoid_ingredients,
        'Similarity': similarity  # Include the 'Similarity' column
    }

    return result

def process_chunk_wrapper(chunk, disease):
    return [calculate_similarity_and_filter(row, disease) for _, row in chunk.iterrows()]

def recommend_recipes_for_disease(df, disease):
    # Convert DataFrame to chunks
    chunk_size = 20
    chunks = [df[i:i + chunk_size] for i in range(0, df.shape[0], chunk_size)]

    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = list(
            tqdm(executor.map(process_chunk_wrapper, chunks, [disease] * len(chunks)), total=len(chunks)))

    # Combine the results from all chunks into a single DataFrame
    result_df = pd.concat([pd.DataFrame(chunk_result) for chunk_result in results])

    # Rank and sort the DataFrame
    result_df_sorted = result_df.sort_values(by='Similarity', ascending=False)

    # Return only the top 100 results
    return result_df_sorted.head(100)

# if __name__ == '__main__':
#     # Example usage for hypertension
#     df = pd.read_csv('./split_file_1.csv')
#     result_df = recommend_recipes_for_disease(df, 'hypertension')
#     print(result_df)
