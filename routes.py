# routes.py
from flask import request, jsonify
from finalgui import Weight_Loss, Weight_Gain, Healthy
import threading  # Import threading module

# Create a lock object
lock = threading.Lock()

import requests

import requests

def fetch_image_url(query, api_key, cx):
    search_url = "https://www.googleapis.com/customsearch/v1"
    params = {
        'q': f"{query} dish",
        'key': api_key,
        'cx': cx,
        'searchType': 'image',
    }

    response = requests.get(search_url, params=params)
    data = response.json()

    if 'items' in data and data['items']:
        return data['items'][0]['link']
    else:
        print(f"No image found for query: {query}")
        return "https://i0.wp.com/wonkywonderful.com/wp-content/uploads/2020/08/spinach-tomato-pasta-sauce-recipe-4.jpg?ssl=1"


# Replace 'YOUR_GOOGLE_API_KEY' and 'YOUR_CX' with your actual Google API key and Custom Search Engine ID
YOUR_GOOGLE_API_KEY = 'AIzaSyCLd5wiefeb1Z6YytNW88FbrBrPjeO14BM'
CX = 'b6aad2c5c59f548f5'


def process_food_items(food_items):
    # Iterate through the food items and add image links using Google Custom Search
    for item in food_items:
        item['image_url'] = fetch_image_url(item['Name'], YOUR_GOOGLE_API_KEY, CX)

    return food_items

def weight_loss_endpoint():
    data = request.json
    global lock
    lock.acquire()
    # Extracting data from the JSON request
    try:
        print(data)
        age = int(data['Age'])
        weight = float(data['Weight'])
        height = float(data['Height'])
        gender = int(data['Gender'])
        food_timing = int(data['Food_Timing'])
        disease = data['Disease']
        desired_loss_kg = float(data['WeightLossField'])
        num_days = int(data['WeightLossGoal'])
        activity_level = int(data['Activity_level'])
        
        # Call the Weight_Loss function with the extracted data
        suggested_food_items_df = Weight_Loss(age, weight, height, food_timing, disease, desired_loss_kg, num_days, activity_level, gender)
        
        # Convert DataFrame to dictionary
        suggested_food_items_dict = suggested_food_items_df.to_dict(orient='records')

        processed_food_items = process_food_items(suggested_food_items_dict)

        return jsonify({'suggested_food_items': processed_food_items})
    finally:
        # Release the lock
        lock.release()

def weight_gain_endpoint():
    data = request.json
    global lock
    lock.acquire()
    # Extracting data from the JSON request
    try:
    # Extracting data from the JSON request
        age = int(data['Age'])
        weight = float(data['Weight'])
        height = float(data['Height'])
        gender = int(data['Gender'])
        food_timing = int(data['Food_Timing'])
        disease = data['Disease']
        desired_gain_kg = float(data['WeightGainField'])
        num_days = int(data['WeightGainGoal'])
        activity_level = int(data['Activity_level'])

        # Call the Weight_Gain function with the extracted data
        suggested_food_items_df = Weight_Gain(age, weight, height, food_timing, disease, desired_gain_kg, num_days, activity_level, gender)
        
        # Convert DataFrame to dictionary
        suggested_food_items_dict = suggested_food_items_df.to_dict(orient='records')

        processed_food_items = process_food_items(suggested_food_items_dict)

        return jsonify({'suggested_food_items': processed_food_items})
    
    finally:
        # Release the lock
        lock.release()

def healthy_endpoint():
    data = request.json
    global lock
    lock.acquire()
    # Extracting data from the JSON request
    try:
        age = int(data['Age'])
        weight = float(data['Weight'])
        height = float(data['Height'])
        gender = int(data['Gender'])
        food_timing = int(data['Food_Timing'])
        disease = data['Disease']
        activity_level = int(data['Activity_level'])

        # Call the Healthy function with the extracted data
        suggested_food_items_df = Healthy(age, weight, height, food_timing, disease,activity_level, gender)
        
        # Convert DataFrame to dictionary
        suggested_food_items_dict = suggested_food_items_df.to_dict(orient='records')

        processed_food_items = process_food_items(suggested_food_items_dict)

        return jsonify({'suggested_food_items': processed_food_items})
    
    finally:
        # Release the lock
        lock.release()
