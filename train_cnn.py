from ultralytics import YOLO
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

yolo_model = YOLO('best.pt')
classifier = load_model('model_retrain.h5')

def preprocess_char(char_img):
    img = cv2.resize(char_img, (28, 28))
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=0)  
    return img

def split_characters(plate_img):
    gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    chars = []
    for cnt in sorted(contours, key=lambda c: cv2.boundingRect(c)[0]):
        x, y, w, h = cv2.boundingRect(cnt)
        if w > 5 and h > 10:
            char = plate_img[y:y+h, x:x+w]
            chars.append(char)
    return chars

def recognize_plate(image_path):
    image = cv2.imread(image_path)
    results = yolo_model(image)
    
    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()

        for box in boxes:
            x1, y1, x2, y2 = map(int, box)
            plate_img = image[y1:y2, x1:x2]
            chars = split_characters(plate_img)

            recognized = ''
            for char_img in chars:
                input_char = preprocess_char(char_img)
                pred = classifier.predict(input_char)
                label = np.argmax(pred)
                recognized += str(label)

            print("номер:", recognized)

recognize_plate('11.jpeg')