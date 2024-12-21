from flask import Flask, request, render_template, redirect, url_for
from werkzeug.utils import secure_filename
from shlex import quote
import tensorflow as tf
from PIL import Image
import numpy as np
import os
import requests
import random
import logging

# Initialize Flask app
app = Flask(__name__)

logging.basicConfig(
    filename='access.log',          # Log file name
    level=logging.INFO,             # Log level
    format='%(asctime)s %(message)s' # Log format
)

@app.errorhandler(Exception)
def handle_exception(error):
    return render_template('error_generic.html', error_message=str(error)), 500

app.config['UPLOAD_FOLDER'] = 'uploads'

# Load the trained model
model = tf.keras.models.load_model("models/epoch-05_189.keras")

# Define image dimensions
IMG_HEIGHT = 360
IMG_WIDTH = 640

with open("static/titles.txt", "r") as file:
    titles_list = [line.strip() for line in file if line.strip()]


@app.route('/play_3/result', methods=['POST'])
def play_3_result():
    # Get data from the form
    selected_frame = request.form.get('selected_frame')  # User's selected frame
    correct_title = request.form.get('correct_title')    # Correct anime title
    frames = request.form.getlist('frames')             # All frame paths

    # Determine the correct frame (no need to adjust paths for this comparison)
    correct_frame = next((frame for frame in frames if correct_title in frame), None)

    # Neural network's prediction
    frame_probabilities = []
    for frame in frames:
        # Preprocess each frame
        frame_path = os.path.join(app.root_path, 'static', frame)  # Ensure correct path
        input_data = preprocess_image(frame_path)
        prediction = model.predict(input_data)[0]
        title_index = titles_list.index(correct_title)
        frame_probabilities.append((frame, prediction[title_index]))

    # Find the frame with the highest probability
    neural_network_prediction = max(frame_probabilities, key=lambda x: x[1])[0]

    # No need to modify the paths here as they are already relative to 'static/'
    return render_template(
        'game_result_3.html',
        correct_title=correct_title,
        correct_frame=correct_frame,
        selected_frame=selected_frame,
        nn_frame=neural_network_prediction,
        user_correct=(selected_frame == correct_frame),
        nn_correct=(neural_network_prediction == correct_frame)
    )

@app.route('/play_3', methods=['GET'])
def play_3():
    sample_frames_dir = os.path.join(app.root_path, "static/sample_frames")

    # Load the list of titles from `titles.txt`
    titles_file = "static/titles.txt"
    with open(titles_file, "r") as file:
        titles_list = [line.strip() for line in file if line.strip()]

    # Select a random anime title
    correct_title = random.choice(titles_list)

    # Find a correct frame for the selected title
    correct_dir = os.path.join(sample_frames_dir, correct_title)
    correct_frame = random.choice(os.listdir(correct_dir))

    # Select 5 incorrect frames from other titles
    incorrect_titles = [title for title in titles_list if title != correct_title]
    incorrect_frames = []
    for title in random.sample(incorrect_titles, 5):
        incorrect_dir = os.path.join(sample_frames_dir, title)
        incorrect_frame = random.choice(os.listdir(incorrect_dir))
        incorrect_frames.append(f"sample_frames/{title}/{incorrect_frame}")

    # Combine correct frame and incorrect frames, then shuffle
    all_frames = [f"sample_frames/{correct_title}/{correct_frame}"] + incorrect_frames
    random.shuffle(all_frames)

    return render_template('play_3.html', correct_title=correct_title, frames=all_frames)

@app.route('/play_2', methods=['GET'])
def play_2():
    sample_frames_dir = os.path.join(app.root_path, "static/sample_frames")

    # Select 6 distinct anime
    anime_folders = random.sample(os.listdir(sample_frames_dir), 6)

    # Select one image pair from each anime
    grid_images = []
    for anime in anime_folders:
        anime_dir = os.path.join(sample_frames_dir, anime)
        selected_images = random.sample(os.listdir(anime_dir), 2)  # Select 2 images
        grid_images.extend([f"sample_frames/{anime}/{img}" for img in selected_images])

    # Shuffle the grid
    random.shuffle(grid_images)

    return render_template('play_2.html', grid_images=grid_images)

@app.route('/play_2/result', methods=['GET'])
def play_2_result():
    time_taken = request.args.get('time', 0)
    return render_template('game_result_2.html', time_taken=time_taken)

@app.route('/play_1', methods=['GET'])
def play_1():
    # Randomly select an anime and an image
    anime = random.choice(titles_list)
    sample_frames_dir = os.path.join(app.static_folder, "sample_frames")
    anime_dir = os.path.join(sample_frames_dir, anime)
    # print(f"Anime directory path: {anime_dir}")
    image_file = random.choice(os.listdir(anime_dir))
    image_path = f"sample_frames/{anime}/{image_file}"

    cleaned_titles_list = [title[:-2] for title in titles_list]
    
    # Render the play page
    return render_template('play_1.html', image_path=image_path, titles_list=cleaned_titles_list, correct_title=anime[:-2])

@app.route('/play_1/result_1', methods=['POST'])
def play_result_1():
    # Get form data
    user_guess = request.form.get('user_guess')
    correct_title = request.form.get('correct_title')
    image_path = request.form.get('image_path')

    # Run the neural network on the image
    full_image_path = os.path.join(app.root_path, "static", image_path)
    input_data = preprocess_image(full_image_path)
    predictions = model.predict(input_data)[0]  # Flatten the predictions
    predicted_index = np.argmax(predictions)
    nn_guess = titles_list[predicted_index][:-2]

    # Determine outcomes
    user_correct = user_guess == correct_title
    nn_correct = nn_guess == correct_title

    # Render the result page
    return render_template('game_result_1.html', image_path=image_path, user_correct=user_correct,
                           nn_correct=nn_correct, correct_title=correct_title, user_guess=user_guess, nn_guess=nn_guess)

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
        return render_template('error_upload.html', message="No file selected. Please choose a file to upload.")

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
        
        # candidate_name = os.path.splitext(candidate_file)[0]  # Remove file extension
        candidate_name = titles_list[idx]

        probability = predictions[idx]
        anime_details = fetch_anime_details(candidate_name)  # Fetch details from MAL
        top_candidates.append({
            "name": candidate_name[:-2],
            "file": candidate_file,
            "probability": probability,
            "details": anime_details,
        })

    # Optional: Clean up uploaded file
    # os.remove(filepath)

    # Render the template and pass the top 5 candidates
    return render_template('result.html', top_candidates=top_candidates)

@app.before_request
def log_request_info():
    # Gather client information
    ip_address = request.remote_addr
    url = request.url
    method = request.method
    user_agent = request.headers.get('User-Agent')
    
    # Log the access details
    logging.info(f"IP: {ip_address}, Method: {method}, URL: {url}, User-Agent: {user_agent}")



if __name__ == '__main__':
    app.run( host="0.0.0.0", port=8080, debug=True)
