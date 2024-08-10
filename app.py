from flask import Flask, render_template, request, redirect, url_for, flash
import cv2
import numpy as np
import dlib
from keras.models import load_model
import os
from player_data import player_name

app = Flask(__name__, static_url_path='/static', static_folder='static', template_folder='templates')
app.secret_key = b'_5#y2L"F4Q8z\n\xec]/'

model = load_model("soccer_face_model.h5")

def preprocess_image(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(src=image, dsize=(100, 100))
    image = np.array(image)
    image = image.reshape((1, 100, 100, 1))
    image = image / 255.0
    return image

detector = dlib.get_frontal_face_detector()

uploaded_image = None

@app.route('/', methods=['GET', 'POST'])
def index():
    my_variable = "some-class"
    global uploaded_image
    if request.method == 'POST':
        file = request.files['file']
        if file:
            uploaded_image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
            if uploaded_image.shape[0] < 300 or uploaded_image.shape[1] < 300:
                flash('Vui lòng chọn ảnh cần nhận diện', 'error')
                uploaded_image = None
                return redirect(url_for('index'))
            return redirect(url_for('recognize'))
    return render_template('index.html', uploaded_image=uploaded_image, player_info=None, my_variable=my_variable)

@app.route('/recognize', methods=['GET'])
def recognize():
    global uploaded_image
    if uploaded_image is None:
        return redirect(url_for('index'))

    gray = cv2.cvtColor(uploaded_image, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    player_info = []
    cropped_faces = []

    if len(faces) == 0:
        flash('Không phát hiện cầu thủ trong ảnh.', 'error')
        return redirect(url_for('index'))

    for i, face in enumerate(faces):
        x, y, w, h = face.left(), face.top(), face.width(), face.height()
        cropped_face = uploaded_image[y:y + h, x:x + w]
        if cropped_face.shape[0] == 0 or cropped_face.shape[1] == 0 or cropped_face.shape[2] == 0:
            continue
        cropped_faces.append(cropped_face)  # Store the cropped face
        predicted_player_info = predict_player(cropped_face)
        player_info.append(predicted_player_info)

    for i, cropped_face in enumerate(cropped_faces):
        cv2.imwrite(f'static/output_{i}.jpg', cropped_face)

    return render_template('recognize.html', player_info=player_info, player_name=player_name)  # Thêm player_name ở đây

@app.route('/about', methods=['GET'])
def about():
    return render_template('about.html', player_name=player_name)

@app.route('/contact',methods=['GET'])
def contact():
    return render_template('contact.html')

@app.route('/home',methods=['GET'])
def home():
    return render_template('home.html')

def predict_player(image):
    test_image = preprocess_image(image)
    predictions = model.predict(test_image)
    predicted_label = np.argmax(predictions)
    predicted_player = player_name.get(predicted_label, "Unknown")
    return predicted_player

if __name__ == '__main__':
    app.run(debug=True)
