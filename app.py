from flask import Flask, render_template, jsonify, request
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import folium
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.gridspec as gridspec
import random
import io
import base64
import os
import time

# Import functions from solution.py
from solution import create_traffic_prediction_dashboard, get_congestion_level, create_traffic_map, get_color_for_score

app = Flask(__name__, static_folder='.')

@app.route('/')
def index():
    """Serve the main HTML page"""
    return render_template('traffic_predict.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Run the traffic prediction model and return results"""
    # Get parameters from the request
    location = request.json.get('location', 'all')
    time_period = request.json.get('time_period', 'all')
    
    # Log the request parameters
    print(f"Prediction requested for location: {location}, time period: {time_period}")
    
    # Run the prediction (simulate steps)
    response_data = {"steps": []}
    
    # Step 1: Initialize
    response_data["steps"].append({"message": "Initializing TrafficSense AI..."})
    time.sleep(0.8)  # Simulate processing time
    
    # Step 2: Load data
    response_data["steps"].append({"message": "Loading historical traffic patterns..."})
    time.sleep(1.0)
    
    # Step 3: Analyze weather
    response_data["steps"].append({"message": "Analyzing weather forecasts..."})
    time.sleep(0.8)
    
    # Step 4: Process road data
    response_data["steps"].append({"message": "Processing road sensor data..."})
    time.sleep(0.8)
    
    # Step 5: Check events
    response_data["steps"].append({"message": "Checking scheduled events..."})
    time.sleep(0.8)
    
    # Step 6: Run AI model
    response_data["steps"].append({"message": "Running DeepTraffic GPT-7 neural network..."})
    time.sleep(1.2)
    
    # Step 7: Generate maps
    response_data["steps"].append({"message": "Generating prediction maps..."})
    time.sleep(0.8)
    
    # Actually run the prediction function
    create_traffic_prediction_dashboard()
    
    # Return the results
    response_data["status"] = "success"
    response_data["message"] = "Traffic prediction complete"
    response_data["image_path"] = "winnipeg_traffic_prediction.png"
    response_data["map_path"] = "winnipeg_traffic_prediction_map.html"
    response_data["update_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    return jsonify(response_data)

if __name__ == '__main__':
    # Create templates directory if it doesn't exist
    if not os.path.exists('templates'):
        os.makedirs('templates')
    
    # Copy the HTML file to templates directory with explicit UTF-8 encoding
    with open('traffic_predict.html', 'r', encoding='utf-8') as src:
        html_content = src.read()
    
    with open('templates/traffic_predict.html', 'w', encoding='utf-8') as dest:
        dest.write(html_content)
    
    # Run the Flask app
    app.run(debug=True) 