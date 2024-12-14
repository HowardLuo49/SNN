from flask import Flask, request, jsonify, render_template, url_for
from werkzeug.utils import secure_filename
import tensorflow as tf
from PIL import Image
import numpy as np
import os
import requests

# Initialize Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

# Load the trained model
model = tf.keras.models.load_model("models/test_model_5.h5")

# Define image dimensions
IMG_HEIGHT = 360
IMG_WIDTH = 640

def preprocess_image(image_path):
    # Open the image file
    image = Image.open(image_path).convert("RGB")
    
    # Find the smaller dimension and scale it
    scale = min(image.width / IMG_WIDTH, image.height / IMG_HEIGHT)
    new_width = int(image.width / scale)
    new_height = int(image.height / scale)
    image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    # Calculate crop box to center the image
    left = (new_width - IMG_WIDTH) // 2
    top = (new_height - IMG_HEIGHT) // 2
    right = left + IMG_WIDTH
    bottom = top + IMG_HEIGHT
    
    # Crop the centered part of the image
    image = image.crop((left, top, right, bottom))
    
    # Convert to NumPy array and normalize
    img_array = np.array(image) / 255.0  # Normalize to [0, 1]
    
    # Add batch dimension
    return np.expand_dims(img_array, axis=0)

@app.route('/')
def index():
    # Render the HTML template for uploading images
    return render_template('index.html')

def fetch_anime_details(title):
    """Fetch anime details from MAL API based on the title."""
    client_id = "b8f73ea8501865336729757a491210d6"  # Replace with your MAL Client ID
    title = ''.join(title[:-2])
    url = f"https://api.myanimelist.net/v2/anime?q={title}&limit=1"

    headers = {
        "X-MAL-CLIENT-ID": client_id
    }

    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        data = response.json()
        if "data" in data and data["data"]:
            anime = data["data"][0]["node"]
            return {
                "title": anime.get("title"),
                "image_url": anime.get("main_picture", {}).get("large"),
                "details_url": f"https://myanimelist.net/anime/{anime.get('id')}",
            }
    return {
        "title": "Not added yet",
        "image_url": "https://via.placeholder.com/300?text=No+Image",
        "details_url": "#"
    }

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files or request.files['file'].filename == '':
        return render_template('error.html', message="No file selected. Please choose a file to upload.")

    # Save uploaded file
    file = request.files['file']
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    file.save(filepath)

    # Preprocess the image
    input_data = preprocess_image(filepath)

    # Perform inference
    predictions = model.predict(input_data)[0]  # Flatten the predictions
    sorted_indices = np.argsort(predictions)[::-1]  # Sort indices by descending probabilities

    # Extract top 5 candidates and fetch their MAL details
    top_candidates = []
    for idx in sorted_indices[:5]:  # Top 5 predictions
        # Input: Paste your titles here as a multi-line string
        titles_file = "static/titles.txt"

        # Read the file and store titles in a list
        with open(titles_file, "r") as file:
            titles_list = [line.strip() for line in file if line.strip()]

        candidate_file = titles_list[idx]
        candidate_name = os.path.splitext(candidate_file)[0]  # Remove file extension
        probability = predictions[idx]
        anime_details = fetch_anime_details(candidate_name)  # Fetch details from MAL
        top_candidates.append({
            "name": candidate_name[:-2],
            "file": candidate_file,
            "probability": probability,
            "details": anime_details,
        })

    # Optional: Clean up uploaded file
    os.remove(filepath)

    # Render the template and pass the top 5 candidates
    return render_template('result.html', top_candidates=top_candidates)




if __name__ == '__main__':
    app.run(debug=True)
