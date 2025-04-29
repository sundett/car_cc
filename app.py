from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from ultralytics import YOLO

app = Flask(__name__)

# Папка для загрузки файлов
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# 📦 Модельдерді жүктеу
yolo_model = YOLO("best.pt")
cnn_model = load_model("model_retrain.h5")

# 🔤 Label map (0-9 + A-Z)
label_map = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] + \
            [chr(i) for i in range(ord('A'), ord('Z') + 1)]

# 🎨 Символды алдын-ала өңдеу
def preprocess_symbol(img):
    img = cv2.resize(img, (28, 28))
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=-1)      # (28,28,1)
    img = np.repeat(img, 3, axis=-1)        # (28,28,3)
    img = np.expand_dims(img, axis=0)       # (1,28,28,3)
    return img

# 🔍 Нөмірді тану функциясы
def recognize_plate(img_path):
    image = cv2.imread(img_path)
    results = yolo_model(image)[0]

    recognized_text = ''

    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        plate_img = image[y1:y2, x1:x2]

        gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY_INV, 11, 2)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        boxes = []
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            if h > 10 and h < plate_img.shape[0] * 0.9 and w > 5:
                boxes.append((x, y, w, h))

        boxes = sorted(boxes, key=lambda b: b[0])

        for (x, y, w, h) in boxes:
            char = thresh[y:y+h, x:x+w]
            char_img = preprocess_symbol(char)

            pred = cnn_model.predict(char_img)
            class_id = np.argmax(pred)

            if class_id < len(label_map):
                recognized_text += label_map[class_id]

    return recognized_text

# 🌐 Басты бет
@app.route('/')
def index():
    return render_template('index.html')  # index.html – шаблон

# 📤 Сурет жүктеу және тану
@app.route('/upload', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    result = recognize_plate(filepath)
    return jsonify({'recognized_number': result})

# 🧪 Жүктеу папкасы жоқ болса – жасаймыз
if __name__ == '__main__':
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    app.run(debug=True)
