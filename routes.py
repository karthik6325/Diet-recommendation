# routes.py
from flask import request, jsonify
from finalgui import Weight_Loss, Weight_Gain, Healthy

import requests
from PIL import Image
from io import BytesIO

import requests

def fetch_image_url(query, api_key, cx):
    search_url = "https://www.googleapis.com/customsearch/v1"
    params = {
        'q': query,
        'key': api_key,
        'cx': cx,
        'searchType': 'image',
    }

    response = requests.get(search_url, params=params)
    data = response.json()

    if 'items' in data and data['items']:
        return data['items'][0]['link']
    else:
        # print(f"No image found for query: {query}")
        return "https://i0.wp.com/wonkywonderful.com/wp-content/uploads/2020/08/spinach-tomato-pasta-sauce-recipe-4.jpg?ssl=1"


# Replace 'YOUR_GOOGLE_API_KEY' and 'YOUR_CX' with your actual Google API key and Custom Search Engine ID
YOUR_GOOGLE_API_KEY = ''
CX = ''


def process_food_items(food_items):
    # Iterate through the food items and add image links using Google Custom Search
    for item in food_items:
        item['image_url'] = fetch_image_url(item['Name'], YOUR_GOOGLE_API_KEY, CX)

    return food_items

def weight_loss_endpoint():
    data = request.json

    # Extracting data from the JSON request
    age = int(data['Age'])
    weight = float(data['Weight'])
    height = float(data['Height'])
    food_timing = int(data['Food_Timing'])
    disease = data['Disease']
    desired_loss_kg = float(data['WeightLossField'])
    num_days = int(data['WeightLossGoal'])
    activity_level = int(data['Activity_level'])
    
    # Call the Weight_Loss function with the extracted data
    suggested_food_items_df = Weight_Loss(age, weight, height, food_timing, disease, desired_loss_kg, num_days, activity_level)
    
    # Convert DataFrame to dictionary
    suggested_food_items_dict = suggested_food_items_df.to_dict(orient='records')

    processed_food_items = process_food_items(suggested_food_items_dict)

    return jsonify({'suggested_food_items': processed_food_items})

def weight_gain_endpoint():
    data = request.json
    print(data)
    # Extracting data from the JSON request
    age = int(data['Age'])
    weight = float(data['Weight'])
    height = float(data['Height'])
    food_timing = int(data['Food_Timing'])
    disease = data['Disease']
    desired_gain_kg = float(data['WeightGainField'])
    num_days = int(data['WeightGainGoal'])
    activity_level = int(data['Activity_level'])

    # Call the Weight_Gain function with the extracted data
    suggested_food_items_df = Weight_Gain(age, weight, height, food_timing, disease, desired_gain_kg, num_days, activity_level)
    
    # Convert DataFrame to dictionary
    suggested_food_items_dict = suggested_food_items_df.to_dict(orient='records')

    processed_food_items = process_food_items(suggested_food_items_dict)

    return jsonify({'suggested_food_items': processed_food_items})

def healthy_endpoint():
    data = request.json

    # Extracting data from the JSON request
    age = int(data['Age'])
    weight = float(data['Weight'])
    height = float(data['Height'])
    food_timing = int(data['Food_Timing'])
    disease = data['Disease']
    activity_level = int(data['Activity_level'])

    # Call the Healthy function with the extracted data
    suggested_food_items_df = Healthy(age, weight, height, food_timing, disease,activity_level)
    
    # Convert DataFrame to dictionary
    suggested_food_items_dict = suggested_food_items_df.to_dict(orient='records')

    processed_food_items = process_food_items(suggested_food_items_dict)

    return jsonify({'suggested_food_items': processed_food_items})
