from flask import Flask, request, jsonify, render_template
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)

# Load the pre-trained model
model = MobileNetV2(weights='imagenet')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    file = request.files['file']
    if file:
        # Save the file
        filepath = os.path.join('uploads', file.filename)
        file.save(filepath)

        # Load the image for prediction
        img = image.load_img(filepath, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)

        # Make a prediction
        predictions = model.predict(img_array)
        results = decode_predictions(predictions, top=1)[0][0]

        # Respond with the result
        response = {
            'message': f'Prediction: {results[1]} with probability {results[2]:.2f}'
        }
        return jsonify(response)

    return jsonify({'message': 'No file uploaded!'})

if __name__ == '__main__':
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    app.run(debug=True)
