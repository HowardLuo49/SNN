from flask import Flask, request, render_template, redirect, url_for, session, flash
from werkzeug.utils import secure_filename
from shlex import quote
import tensorflow as tf
from PIL import Image
import numpy as np
import os
import requests
import random
import logging

app = Flask(__name__)
app.secret_key = 'Redacted'

logging.basicConfig(
    filename='access.log',
    level=logging.INFO,
    format='%(asctime)s %(message)s'
)

app.config['UPLOAD_FOLDER'] = 'uploads'

model = tf.keras.models.load_model("models/epoch-09-resnet50_189.keras")

IMG_HEIGHT = 360
IMG_WIDTH = 640

with open("static/titles.txt", "r") as file:
    titles_list = [line.strip() for line in file if line.strip()]
    cleaned_titles_list = [title[:-2] for title in titles_list]




@app.errorhandler(Exception)
def handle_exception(error):
    return render_template('error_generic.html', error_message=str(error)), 500

@app.errorhandler(413)
def handle_large_file_error(error):
    return render_template('error_generic.html', error_message="The uploaded file is too large. Please try again with a smaller file."), 413



@app.route('/select_anime', methods=['GET'])
def select_anime():    
    return render_template('select_anime.html', all_anime=titles_list, selected_anime=session.get('selected_anime', []))

@app.route('/save_anime_selection', methods=['POST'])
def save_anime_selection():
    session['selected_anime'] = request.form.getlist('anime')
    return redirect(url_for('index'))



@app.route('/play_3/result', methods=['POST'])
def play_3_result():
    selected_frame = request.form.get('selected_frame') 
    correct_title = request.form.get('correct_title') 
    frames = request.form.getlist('frames')

    correct_frame = next((frame for frame in frames if correct_title in frame), None)

    # Neural network processing
    frame_probabilities = []
    for frame in frames:
        frame_path = os.path.join(app.root_path, 'static', frame)
        input_data = preprocess_image(frame_path)
        prediction = model.predict(input_data)[0]
        title_index = titles_list.index(correct_title)
        frame_probabilities.append((frame, prediction[title_index]))
    neural_network_prediction = max(frame_probabilities, key=lambda x: x[1])[0]

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

    # Select a random anime and sample a frame
    selected_anime = session.get('selected_anime', [])
    if not selected_anime:
        selected_anime = titles_list
    correct_title = random.choice(selected_anime)
    correct_dir = os.path.join(sample_frames_dir, correct_title)
    correct_frame = random.choice(os.listdir(correct_dir))

    # Game requires minimum amount of shows to work
    if len(selected_anime) < 6:
        return render_template('error_insufficient_anime.html', message="Not enough anime selected.")

    # Select 5 frames from other anime
    incorrect_titles = [title for title in selected_anime if title != correct_title]
    incorrect_frames = []
    for title in random.sample(incorrect_titles, 5):
        incorrect_dir = os.path.join(sample_frames_dir, title)
        incorrect_frame = random.choice(os.listdir(incorrect_dir))
        incorrect_frames.append(f"sample_frames/{title}/{incorrect_frame}")

    all_frames = [f"sample_frames/{correct_title}/{correct_frame}"] + incorrect_frames
    random.shuffle(all_frames)
    return render_template('play_3.html', correct_title=correct_title, frames=all_frames)

@app.route('/play_2', methods=['GET'])
def play_2():
    sample_frames_dir = os.path.join(app.root_path, "static/sample_frames")

    # Available Anime
    selected_anime = session.get('selected_anime', [])
    if not selected_anime:
        selected_anime = titles_list
    available_anime = [anime for anime in os.listdir(sample_frames_dir) if anime in selected_anime]

    # Game requires minimum amount of shows to work
    if len(available_anime) < 6:
        return render_template('error_insufficient_anime.html', message="Not enough anime selected.")

    # Select 6 distinct anime
    anime_folders = random.sample(available_anime, 6)

    # Select one image pair from each anime
    grid_images = []
    for anime in anime_folders:
        anime_dir = os.path.join(sample_frames_dir, anime)
        selected_images = random.sample(os.listdir(anime_dir), 2)
        grid_images.extend([f"sample_frames/{anime}/{img}" for img in selected_images])

    random.shuffle(grid_images)
    return render_template('play_2.html', grid_images=grid_images)

@app.route('/play_2/result', methods=['GET'])
def play_2_result():
    time_taken = request.args.get('time', 0)
    return render_template('game_result_2.html', time_taken=time_taken)

@app.route('/play_1', methods=['GET'])
def play_1():
    # Select a random anime and sample a frame
    selected_anime = session.get('selected_anime', [])
    if not selected_anime:
        selected_anime = titles_list

    anime = random.choice(selected_anime)
    sample_frames_dir = os.path.join(app.static_folder, "sample_frames")
    anime_dir = os.path.join(sample_frames_dir, anime)
    image_file = random.choice(os.listdir(anime_dir))
    image_path = f"sample_frames/{anime}/{image_file}"

    cleaned_selected_titles_list = [title[:-2] for title in selected_anime]
    
    return render_template('play_1.html', image_path=image_path, titles_list=cleaned_selected_titles_list, correct_title=anime[:-2])

@app.route('/play_1/result_1', methods=['POST'])
def play_result_1():
    user_guess = request.form.get('user_guess')
    correct_title = request.form.get('correct_title')
    image_path = request.form.get('image_path')

    # Neural Network Processing
    full_image_path = os.path.join(app.root_path, "static", image_path)
    input_data = preprocess_image(full_image_path)
    predictions = model.predict(input_data)[0]
    predicted_index = np.argmax(predictions)
    nn_guess = titles_list[predicted_index][:-2]

    # Outcomes
    user_correct = user_guess == correct_title
    nn_correct = nn_guess == correct_title

    return render_template('game_result_1.html', image_path=image_path, user_correct=user_correct, nn_correct=nn_correct, correct_title=correct_title, user_guess=user_guess, nn_guess=nn_guess)

def preprocess_image(image_path):
    image = Image.open(image_path).convert("RGB")
    
    # Scaling down & cropping image
    scale = min(image.width / IMG_WIDTH, image.height / IMG_HEIGHT)
    new_width = int(image.width / scale)
    new_height = int(image.height / scale)
    image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    left = (new_width - IMG_WIDTH) // 2
    top = (new_height - IMG_HEIGHT) // 2
    right = left + IMG_WIDTH
    bottom = top + IMG_HEIGHT
    image = image.crop((left, top, right, bottom))
    
    img_array = np.array(image) / 255.0 
    return np.expand_dims(img_array, axis=0)

@app.route('/')
def index():
    selected_anime = session.get('selected_anime', [])
    return render_template('index.html', selected_anime=selected_anime)

def fetch_anime_details(title):
    # MAL API
    client_id = "Redacted"
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

    file = request.files['file']
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    file.save(filepath)

    input_data = preprocess_image(filepath)

    # Neural Network predicting
    predictions = model.predict(input_data)[0]
    sorted_indices = np.argsort(predictions)[::-1]

    # Extract top 5 results
    top_candidates = []
    for idx in sorted_indices[:5]:
        candidate_name = titles_list[idx]

        probability = predictions[idx]
        anime_details = fetch_anime_details(candidate_name)  # Fetch details from MAL
        top_candidates.append({
            "name": candidate_name[:-2],
            "file": candidate_name,
            "probability": probability,
            "details": anime_details,
        })

    # Clean up uploaded file if desired
    # os.remove(filepath)

    return render_template('result.html', top_candidates=top_candidates)

@app.before_request
def log_request_info():
    # Log user information
    ip_address = request.remote_addr
    url = request.url
    method = request.method
    user_agent = request.headers.get('User-Agent')    
    logging.info(f"IP: {ip_address}, Method: {method}, URL: {url}, User-Agent: {user_agent}")



if __name__ == '__main__':
    app.run(host="0.0.0.0", port=80, debug=True)
