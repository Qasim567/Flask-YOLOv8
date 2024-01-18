from flask import Flask, request, jsonify
from PIL import Image
import numpy as np
from ultralytics import YOLO

app = Flask(__name__)

model = YOLO("./YOLO-Weights/cookie.pt")
classNames = ['Akhrot Cookie', 'Almond Cookie', 'Besan Khatai', 'Black Currant Cookie', 'Cake Rusk', 'Check Biscuit', 'Cheese Cookie', 'Choclate Chip Cookie', 'Choclate Cookie', 'Choclate Stick', 'Cocunut Cookie', 'Dil Jam Cookie', 'Fruit Cookie', 'Fudge Cookie', 'Ginger Cookie', 'Nan Khatai', 'Peanut Cookie', 'Pista Cookie', 'Stawbery Biscuit', 'Zeera Biscuit']

def predict_image(image):
    try:
        # Resize the image to fit the YOLO model input size (optional based on model requirements)
        input_size = (608, 608)
        img = image.resize(input_size)
        img_array = np.array(img)

        # Perform YOLO prediction
        results = model(img_array)

        # Process results and extract class name and confidence score
        predictions = []
        for r in results:
            for box in r.boxes:
                conf = float(box.conf)
                cls = int(box.cls)
                class_name = classNames[cls]
                predictions.append({'class': class_name, 'confidence': conf})

        return predictions
    except Exception as e:
        print("Error during prediction:", str(e))
        return []

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the image from the request
        image_file = request.files['image']
        image = Image.open(image_file.stream)

        # Perform prediction
        predictions = predict_image(image)

        # Return predictions as JSON
        return jsonify(predictions)
    except Exception as e:
        print("Error:", str(e))
        return jsonify({'error': 'An error occurred during prediction.'})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
