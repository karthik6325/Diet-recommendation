# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.preprocessing import LabelEncoder, StandardScaler
# from sklearn.metrics import accuracy_score

# def categorize_calories(calories):
#     if calories > 500:
#         return 'High Calorie'
#     elif 300 <= calories <= 500:
#         return 'Medium Calorie'
#     else:
#         return 'Low Calorie'

# def train_knn_model_healthy(df):
#     df['CalorieCategory'] = df['Calories'].apply(categorize_calories)
#     X = df[['Calories']]
#     y = df['CalorieCategory']

#     label_encoder = LabelEncoder()
#     y_encoded = label_encoder.fit_transform(y)

#     scaler = StandardScaler()
#     X_scaled = scaler.fit_transform(X)

#     X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42)

#     knn_model = KNeighborsClassifier(n_neighbors=3)
#     knn_model.fit(X_train, y_train)

#     y_pred = knn_model.predict(X_test)

#     y_pred_decoded = label_encoder.inverse_transform(y_pred)

#     accuracy = accuracy_score(y_test, y_pred)
#     print(f"Accuracy: {accuracy * 100:.2f}%")

#     return knn_model, label_encoder, scaler

# def predict_matching_recipes_healthy(knn_model, label_encoder, scaler, approx_calories, input_df):
#     approx_features = [[approx_calories]]
#     approx_features_scaled = scaler.transform(approx_features)

#     predicted_encoded_label = knn_model.predict(approx_features_scaled)
#     predicted_calorie_category = label_encoder.inverse_transform(predicted_encoded_label)[0]

#     # Filter the dataframe based on the predicted calorie category
#     filtered_df = input_df[input_df['CalorieCategory'] == predicted_calorie_category]

#     return filtered_df

# def recipe_prediction_function(input_df, approx_calories):
#     # Train the KNN model and obtain the label encoder
#     knn_model, label_encoder, scaler = train_knn_model_healthy(input_df)

#     # Predict matching recipes based on input values
#     result_df = predict_matching_recipes_healthy(knn_model, label_encoder, scaler, approx_calories, input_df)

#     return result_df

# # Example usage:
# # Replace 'your_dataset.csv' with the actual path to your CSV file
# # Load your DataFrame

# df = pd.read_csv('./split_file_1.csv')

# # Set input values for prediction
# approx_calories_value = 500

# # Call the function
# result = recipe_prediction_function(input_df=df, approx_calories=approx_calories_value)

# # Print the result
# print(result)



import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

def recipe_prediction_function_healthy(input_df, approx_calories):
    # Standardize the data
    scaler = StandardScaler()
    input_df_scaled = scaler.fit_transform(input_df[['Calories']])
    approx_features_scaled = scaler.transform([[approx_calories]])

    # Perform KNN clustering
    knn_model = NearestNeighbors(n_neighbors=100)
    knn_model.fit(input_df_scaled)

    # Find nearest neighbors
    distances, indices = knn_model.kneighbors(approx_features_scaled)

    # Get the indices of nearest neighbors
    neighbor_indices = indices.flatten()

    # Get the recipes corresponding to nearest neighbors
    result_df = input_df.iloc[neighbor_indices]

    return result_df

# if __name__ == '__main__':
#     # Example usage:
#     # Replace 'your_dataset.csv' with the actual path to your CSV file
#     # Load your DataFrame
#     df = pd.read_csv('./split_file_1.csv')

#     # Set input values for prediction
#     approx_calories_value = 500

#     # Call the function
#     result = recipe_prediction_function(input_df=df, approx_calories=approx_calories_value)

#     # Plot the results
#     plt.figure(figsize=(8, 6))
#     sns.scatterplot(x='Calories', y='Calories', data=df, label='All Recipes')  # Using 'Calories' for both axes
#     sns.scatterplot(x='Calories', y='Calories', data=result, color='red', label='Predicted Recipes')  # Using 'Calories' for both axes
#     plt.scatter(approx_calories_value, approx_calories_value, color='green', label='Approximate Calories', marker='X', s=100)
#     plt.title('Predicted Recipes Based on Calories')
#     plt.xlabel('Calories')
#     plt.ylabel('Calories')
#     plt.legend()
#     plt.show()


