from flask import Flask, request, jsonify
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)

# Load pre-trained MobileNet model
model = MobileNetV2(weights='imagenet')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'message': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'message': 'No selected file'}), 400

    # Save the file to process
    filepath = os.path.join('uploads', file.filename)
    file.save(filepath)

    # Process the image
    img = image.load_img(filepath, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    # Predict the image
    predictions = model.predict(img_array)
    decoded_preds = decode_predictions(predictions, top=3)[0]

    # Format the response
    result = []
    for i, pred in enumerate(decoded_preds):
        result.append(f"{i+1}. {pred[1]}: {round(pred[2] * 100, 2)}% confidence")

    return jsonify({'message': 'Image processed. Top predictions:\n' + '\n'.join(result)})

if __name__ == '__main__':
    app.run(debug=True)
