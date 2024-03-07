import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score

def categorize_calories(calories):
    if calories > 500:
        return 'High Calorie'
    elif 300 <= calories <= 500:
        return 'Medium Calorie'
    else:
        return 'Low Calorie'

def train_knn_model(df):
    df['CalorieCategory'] = df['Calories'].apply(categorize_calories)
    X = df[['Calories', 'ProteinContent', 'CarbohydrateContent', 'FatContent']]  # Add 'ProteinContent', 'CarbohydrateContent', and 'FatContent' as additional features
    y = df['CalorieCategory']

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42)

    knn_model = KNeighborsClassifier(n_neighbors=3)
    knn_model.fit(X_train, y_train)

    y_pred = knn_model.predict(X_test)

    y_pred_decoded = label_encoder.inverse_transform(y_pred)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy * 100:.2f}%")

    return knn_model, label_encoder, scaler

def predict_matching_recipes(knn_model, label_encoder, scaler, approx_calories, approx_protein, approx_carbohydrate, approx_fat, df):
    approx_features = [[approx_calories, approx_protein, approx_carbohydrate, approx_fat]]
    approx_features_scaled = scaler.transform(approx_features)

    predicted_encoded_label = knn_model.predict(approx_features_scaled)
    predicted_calorie_category = label_encoder.inverse_transform(predicted_encoded_label)[0]

    # Filter the dataframe based on the predicted calorie category
    filtered_df = df[df['CalorieCategory'] == predicted_calorie_category]

    # Print RecipeId, food name, and food ingredients for the matching foods
    print("\nMatching Recipes:")
    for index, row in filtered_df.iterrows():
        print(f"RecipeId: {row['RecipeId']}")
        print(f"Food Name: {row['Name']}")
        print(f"Food Ingredients: {row['RecipeIngredientParts']}\n")
        print(f"Calories: {row['Calories']}")
        print(f"ProteinContent: {row['ProteinContent']}")
        print(f"CarbohydrateContent: {row['CarbohydrateContent']}")
        print(f"FatContent: {row['FatContent']}\n")
        print(f"CalorieCategory: {row['CalorieCategory']}\n")

# Load your dataset
df = pd.read_csv('c:/Users/karth/Downloads/Diet-Recommendation-System-main/split_file_1.csv')

# Train the KNN model and obtain the label encoder
knn_model, label_encoder, scaler = train_knn_model(df)

# Example: Predict calorie category based on approximate calories, protein, carbohydrate, and fat
approx_calories_value = 500
approx_protein_value = 20
approx_carbohydrate_value = 30
approx_fat_value = 15

predict_matching_recipes(knn_model, label_encoder, scaler, approx_calories_value, approx_protein_value, approx_carbohydrate_value, approx_fat_value, df)
