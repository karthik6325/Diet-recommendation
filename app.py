import pandas as pd
import spacy
import numpy as np
from tqdm import tqdm
import concurrent.futures
from tabulate import tabulate

# Step 1: Read the dataset in chunks
chunk_size = 100
chunks = pd.read_csv('c:/Users/karth/Downloads/Diet-Recommendation-System-main/split_file_1.csv', chunksize=chunk_size)

# Step 2: Load a spaCy model
nlp = spacy.load('en_core_web_lg')

# Step 3: Define disease-related ingredients and ingredients to avoid
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

# Step 4: Create an empty DataFrame to store results
result_df = pd.DataFrame(columns=['RecipeId', 'Name', 'Similarity', 'RecipeIngredientParts'])

# Step 5: Define a function for calculating similarity and filtering out ingredients to avoid
def calculate_similarity_and_filter(row, disease):
    recipe_id = row['RecipeId']
    name = row['Name']
    recipe_ingredient_parts = row['RecipeIngredientParts']

    # Use recommended ingredients for the given disease
    disease_ingredients = disease_ingredients_mapping.get(disease, [])
    avoid_ingredients = ingredients_to_avoid_mapping.get(disease, [])

    # Process the ingredients with spaCy
    recipe_embeddings = nlp(recipe_ingredient_parts).vector
    disease_ingredient_embeddings = nlp(' '.join(disease_ingredients)).vector

    # Calculate the Euclidean norm using numpy.linalg.norm
    similarity = np.dot(recipe_embeddings, disease_ingredient_embeddings) / (np.linalg.norm(recipe_embeddings) * np.linalg.norm(disease_ingredient_embeddings))
    
    # Remove recipes with ingredients to avoid
    for avoid_ingredient in avoid_ingredients:
        if avoid_ingredient.lower() in recipe_ingredient_parts.lower():
            similarity = 0  # Set similarity to zero if ingredients to avoid are found
            break
    
    return recipe_id, name, similarity, recipe_ingredient_parts

# Define a wrapper function for the multiprocessing environment
def process_chunk_wrapper(chunk, disease):
    return [calculate_similarity_and_filter(row, disease) for _, row in chunk.iterrows()]

def recommend_recipes_for_disease(disease):
    # Convert chunks to a list to avoid exhaustion
    chunk_list = list(chunks)
    
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = list(tqdm(executor.map(process_chunk_wrapper, chunk_list, [disease]*len(chunk_list)), total=len(chunk_list)))

    # Step 7: Flatten the list of tuples
    flattened_results = [item for sublist in results for item in sublist]

    # Step 8: Add results to the DataFrame
    result_df = pd.DataFrame(flattened_results, columns=['RecipeId', 'Name', 'Similarity', 'RecipeIngredientParts'])

    # Step 9: Rank and sort the DataFrame
    result_df_sorted = result_df.sort_values(by='Similarity', ascending=False)

    # Step 10: Display top matches
    top_n = 10
    top_matches = result_df_sorted.head(top_n)
    print(tabulate(top_matches[['RecipeId', 'Name', 'Similarity', 'RecipeIngredientParts']], headers='keys', tablefmt='pretty'))

if __name__ == '__main__':
    # Example usage for heart disease
    recommend_recipes_for_disease('hypertension')
