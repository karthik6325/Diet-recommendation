# routes.py
from flask import request, jsonify
from finalgui import Weight_Loss, Weight_Gain, Healthy

def weight_loss_endpoint():
    data = request.json

    # Extracting data from the JSON request
    age = int(data['Age'])
    weight = float(data['Weight'])
    height = float(data['Height'])
    food_timing = int(data['Food_Timing'])
    disease = data['Disease']
    desired_loss_kg = float(data['Desired_Loss_Kg'])
    num_days = int(data['Num_Days'])
    activity_level = int(data['Activity_Level'])
    
    # Call the Weight_Loss function with the extracted data
    suggested_food_items_df = Weight_Loss(age, weight, height, food_timing, disease, desired_loss_kg, num_days, activity_level)
    
    # Convert DataFrame to dictionary
    suggested_food_items_dict = suggested_food_items_df.to_dict(orient='records')

    # Return the suggested food items as JSON response
    return jsonify({'suggested_food_items': suggested_food_items_dict})

def weight_gain_endpoint():
    data = request.json

    # Extracting data from the JSON request
    age = int(data['Age'])
    weight = float(data['Weight'])
    height = float(data['Height'])
    food_timing = int(data['Food_Timing'])
    disease = data['Disease']
    desired_gain_kg = float(data['Desired_Gain_Kg'])
    num_days = int(data['Num_Days'])
    activity_level = int(data['Activity_Level'])

    # Call the Weight_Gain function with the extracted data
    suggested_food_items_df = Weight_Gain(age, weight, height, food_timing, disease, desired_gain_kg, num_days, activity_level)
    
    # Convert DataFrame to dictionary
    suggested_food_items_dict = suggested_food_items_df.to_dict(orient='records')

    # Return the suggested food items as JSON response
    return jsonify({'suggested_food_items': suggested_food_items_dict})

def healthy_endpoint():
    data = request.json

    # Extracting data from the JSON request
    age = int(data['Age'])
    weight = float(data['Weight'])
    height = float(data['Height'])
    food_timing = int(data['Food_Timing'])
    disease = data['Disease']

    # Call the Healthy function with the extracted data
    suggested_food_items_df = Healthy(age, weight, height, food_timing, disease)
    
    # Convert DataFrame to dictionary
    suggested_food_items_dict = suggested_food_items_df.to_dict(orient='records')

    # Return the suggested food items as JSON response
    return jsonify({'suggested_food_items': suggested_food_items_dict})
