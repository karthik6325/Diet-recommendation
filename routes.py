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
    diet_type = data['Diet_preference']

    # Call the Weight_Loss function with the extracted data
    suggested_food_items = Weight_Loss(age, weight, height, food_timing, disease, diet_type)

    # Return the suggested food items as JSON response
    return jsonify({'suggested_food_items': suggested_food_items})

def weight_gain_endpoint():
    data = request.json

    # Extracting data from the JSON request
    age = int(data['Age'])
    weight = float(data['Weight'])
    height = float(data['Height'])
    food_timing = int(data['Food_Timing'])
    disease = data['Disease']
    diet_type = data['Diet_preference']

    # Call the Weight_Loss function with the extracted data
    suggested_food_items = Weight_Gain(age, weight, height, food_timing, disease, diet_type)

    # Return the suggested food items as JSON response
    return jsonify({'suggested_food_items': suggested_food_items})


def healthy_endpoint():
    data = request.json

    # Extracting data from the JSON request
    age = int(data['Age'])
    weight = float(data['Weight'])
    height = float(data['Height'])
    food_timing = int(data['Food_Timing'])
    disease = data['Disease']
    diet_type = data['Diet_preference']

    # Call the Weight_Loss function with the extracted data
    suggested_food_items = Healthy(age, weight, height, food_timing, disease, diet_type)

    # Return the suggested food items as JSON response
    return jsonify({'suggested_food_items': suggested_food_items})


