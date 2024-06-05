import pandas as pd
import spacy
import numpy as np
from tqdm import tqdm
import concurrent.futures
from tabulate import tabulate


disease_ingredients_mapping = {
    'hypertension': ["Leafy greens (spinach, kale)", "Berries (strawberries, blueberries)", "Oats", "Bananas", "Beets", "Garlic", "Fatty fish (salmon, mackerel)", "Seeds (flaxseeds, chia seeds)", "Olive oil", "Tomatoes", "Dark chocolate", "Pomegranates", "Citrus fruits (oranges, lemons)", "Nuts (almonds, walnuts)", "Legumes (lentils, chickpeas)", "Avocados", "Sweet potatoes", "Quinoa", "Whole grains", "Low-fat dairy products", "Herbal teas", "Bell peppers"],
    'diabetes': ["Leafy greens (spinach, kale)", "Broccoli", "Carrots", "Berries (blueberries, strawberries)", "Citrus fruits (oranges, grapefruits)", "Tomatoes", "Avocados", "Nuts (almonds, walnuts)", "Seeds (chia seeds, flaxseeds)", "Whole grains (quinoa, oats)", "Beans (black beans, lentils)", "Sweet potatoes", "Greek yogurt (plain, unsweetened)", "Fish (salmon, mackerel)", "Tofu", "Eggs", "Chicken breast (skinless)", "Olive oil", "Apple cider vinegar", "Cinnamon"],
    'arthritis':["Fatty fish (salmon, mackerel, sardines)", "Flaxseeds", "Chia seeds", "Walnuts", "Berries (strawberries, blueberries)", "Citrus fruits", "Dark leafy greens (spinach, kale)", "Broccoli", "Bell peppers", "Oats", "Brown rice", "Quinoa", "Whole wheat", "Almonds", "Olive oil", "Green tea", "Turmeric", "Ginger", "Garlic"],
    'rickets': ["Fortified milk", "Cheese", "Yogurt", "Egg yolks", "Fatty fish (salmon, mackerel, sardines)", "Cod liver oil", "Fortified cereals", "Fortified orange juice", "Beef liver", "Mushrooms (exposed to sunlight)", "Spinach", "Kale", "Collard greens", "Broccoli", "Okra", "Almonds", "Soy milk", "Tofu", "Chia seeds", "Sunflower seeds", "Pumpkin seeds", "Black-eyed peas", "White beans"],
    'scurvy': ["Citrus fruits (oranges, lemons, limes, grapefruits)", "Strawberries", "Kiwi", "Guava", "Papaya", "Pineapple", "Mango", "Broccoli", "Brussels sprouts", "Red and green bell peppers", "Tomatoes", "Spinach", "Cabbage", "Cauliflower", "Potatoes"],
    'anemia': ["Red meat (beef, lamb)", "Organ meats (liver)", "Poultry (chicken, turkey)", "Fish (salmon, tuna)", "Shellfish (oysters, clams)", "Beans (kidney beans, chickpeas)", "Lentils", "Spinach", "Kale", "Swiss chard", "Beet greens", "Broccoli", "Brussels sprouts", "Tomatoes", "Bell peppers", "Citrus fruits (oranges, grapefruits)", "Strawberries", "Kiwi", "Guava", "Papaya"],
    'goitre': ["Iodized salt", "Seaweed", "Fish (cod, tuna)", "Shellfish (shrimp, scallops)", "Dairy products (milk, cheese, yogurt)", "Eggs", "Turkey", "Chicken", "Beef liver", "Fortified bread", "Fortified cereals", "Potatoes", "Cranberries", "Strawberries", "Pineapple", "Beans", "Lentils", "Nuts", "Spinach", "Garlic", "Onions", "Pumpkin seeds"],
    'low_blood_pressure': ["Salt", "Olives", "Pickles", "Broth-based soups", "Cheese", "Tuna", "Anchovies", "Smoked fish", "Green leafy vegetables", "Beets", "Carrots", "Celery", "Whole grains", "Nuts", "Seeds", "Eggs", "Lean meats", "Beans", "Lentils", "Bananas", "Avocados", "Potatoes"],
    'thyroid': ["Iodized salt", "Seafood (fish, shrimp, scallops)", "Seaweed (kelp, nori)", "Dairy products (milk, yogurt, cheese)", "Eggs", "Turkey", "Chicken", "Beef liver", "Brazil nuts", "Sunflower seeds", "Oats", "Whole grains", "Beans (navy beans, chickpeas)", "Spinach", "Bananas", "Berries (blueberries, strawberries)", "Pumpkin", "Sweet potatoes", "Broccoli", "Zinc-rich foods (pumpkin seeds, beef, lentils)"],
    'cholera': ["Bananas","Coconut water", "Rice", "Boiled potatoes", "Carrots", "Broth-based soups", "Yogurt", "Plain crackers", "Oatmeal", "Steamed chicken", "Boiled eggs", "White bread (toasted)", "Applesauce", "Herbal teas (ginger, chamomile)", "Soft-cooked vegetables", "Boiled plantains"],
}

ingredients_to_avoid_mapping = {
    'hypertension': ["Salt", "Processed foods", "Canned soups", "Pickles", "Processed meats (bacon, sausage)", "Fast food", "Snack foods (chips, pretzels)", "Sugary drinks", "Alcohol", "Red meat (in excess)", "Full-fat dairy products", "Butter", "Cream", "Fried foods", "Pastries", "Candy", "Refined grains (white bread, white rice)", "Instant noodles", "Frozen meals", "Excessive coffee", "Energy drinks"],
    'diabetes': ["Sugary drinks", "Candy", "Pastries", "White bread", "White rice", "Potato chips", "French fries", "Sugary cereals", "Full-fat dairy products", "Processed meats (bacon, sausage)", "Fast food", "Fried foods", "High-sugar fruits (grapes, bananas)", "Sweetened yogurt", "Packaged snacks", "Honey", "Maple syrup", "Agave nectar", "High-sugar sauces (barbecue sauce, ketchup)", "Regular pasta"],
    'arthritis': ["White bread", "Sugary drinks", "Red meat", "Full-fat dairy products", "Fried foods", "Margarine", "Shortening", "Processed foods with partially hydrogenated oils", "Corn oil", "Sunflower oil", "Soybean oil", "Fast food", "Frozen meals", "Snack foods", "Excessive alcohol", "Processed foods", "Canned soups", "Certain cheeses", "Artificial sweeteners (e.g., aspartame)"],
    'rickets': ["Soda", "Candy", "Pastries", "Sugary drinks", "Processed foods", "Fast food", "High-sodium foods", "Caffeinated beverages", "Alcohol", "Refined grains (white bread, white rice)", "Snack foods", "Fried foods", "Excessive red meat", "High-fat dairy products", "Butter", "Cream", "Ice cream", "Margarine", "Shortening", "Sugary cereals", "Chips", "Instant noodles"],
    'scurvy': ["Processed foods", "Fast food", "Candies", "Pastries", "Sugary drinks", "Refined grains (white bread, white rice)", "Snack foods", "Red meats (in excess)", "Full-fat dairy products", "Fried foods", "Alcohol (in excess)", "Canned foods"],
    'anemia': ["Coffee", "Tea", "Calcium-rich foods (dairy products, fortified plant-based milk)", "Foods high in oxalates (spinach, rhubarb)", "High-fiber foods (whole grains, bran)", "Sugary foods", "Soda", "Alcohol", "Processed foods", "Fast food", "Uncooked beans", "Raw eggs (in excess)", "Unripe bananas", "Excessive intake of zinc supplements", "Excessive intake of calcium supplements"],
    'goitre': ["Cabbage", "Broccoli", "Cauliflower", "Brussels sprouts", "Kale", "Turnips", "Bok choy", "Soy products (tofu, soy milk)", "Peanuts", "Millet", "Mustard greens", "Radishes", "Peaches", "Strawberries (in large amounts)", "Spinach (in large amounts)", "Sweet potatoes", "Cassava", "Processed foods", "Sugary foods", "Excessive caffeine", "Alcohol", "Highly processed grains"],
    'low_blood_pressure': ["Alcohol", "Sugary foods", "Candy", "Pastries", "Refined grains (white bread, white rice)", "Fast food", "Processed foods", "High-fiber foods (in excess)", "High-potassium foods (in excess)", "Leafy greens (in excess)", "Caffeine (in excess)", "Soda", "Sugary drinks", "Instant noodles", "Frozen meals", "Greasy foods", "Butter", "Cream", "Lard", "Excessive amounts of dairy", "Excessive amounts of fruit juices"],
    'thyroid': ["Soy products (tofu, soy milk)", "Cruciferous vegetables (broccoli, cauliflower, kale)", "Brussels sprouts", "Cabbage", "Turnips", "Mustard greens", "Peaches", "Strawberries (in large amounts)", "Spinach (in large amounts)", "Cassava", "Peanuts", "Linseed", "Millet", "Pears", "Radishes", "Corn", "Lima beans", "Sweet potatoes (in large amounts)", "Sorghum", "Foods high in gluten"],
    'cholera': ["Caffeinated beverages", "Alcohol", "Sugary drinks", "Spicy foods", "Fried foods", "High-fiber foods", "Dairy products (if lactose intolerant)", "Raw vegetables", "Raw fruits (except bananas)", "Greasy foods", "Beans", "Legumes", "Nuts", "Seeds", "Whole grains", "Fatty meats", "Heavy creams", "Butter", "Processed snacks"],
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
