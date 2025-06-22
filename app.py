from flask import Flask, render_template, request, jsonify
from PIL import Image
import re, base64, io, numpy as np
import joblib
from skimage.transform import resize
import matplotlib.pyplot as plt

app = Flask(__name__)
model = joblib.load('digit_model.pkl')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    image_data = re.sub('^data:image/.+;base64,', '', data['image'])
    image = Image.open(io.BytesIO(base64.b64decode(image_data))).convert('L')  # Convert to grayscale

    # Resize to 8x8
    image = image.resize((8, 8), resample=Image.LANCZOS)
    image = np.array(image)

    # Invert and normalize
    image = 255 - image  # Invert: black ink becomes white background
    image = (image / 255.0) * 16  # Scale to 0–16 as expected by sklearn digits
    image = image.clip(0, 16)  # Just in case

    image = image.reshape(1, -1)


    prediction = model.predict(image)[0]
    return jsonify({'prediction': int(prediction)})

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)