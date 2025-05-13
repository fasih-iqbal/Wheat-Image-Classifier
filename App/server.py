from flask import Flask, request, jsonify
import cv2
import numpy as np
from skimage import measure, feature
from skimage.filters import threshold_otsu
from skimage.segmentation import watershed
from skimage.morphology import label
import joblib
import os
from werkzeug.utils import secure_filename
from flask_cors import CORS  # Add CORS support


app = Flask(__name__)
CORS(app)  # Enable CORS for all routes
app.config['UPLOAD_FOLDER'] = 'uploads'

# Load the pre-trained model
try:
    model = joblib.load('wheat_classifier_model.joblib')
    print("Model loaded successfully")
except Exception as e:
    print(f"Error loading model: {e}")
    exit(1)

categories = ['akbar', 'faislabad', 'galaxy', 'Saher']

# Feature Extraction Function


def extract_features(image):
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Preprocessing: Denoise and threshold
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(
            blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Feature 1: Spike Length
        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)
            spike_length = max(w, h)
        else:
            spike_length = 0

        # Feature 2: Number of Grains
        markers = np.zeros_like(gray)
        markers[gray < threshold_otsu(gray)] = 1
        markers[gray > threshold_otsu(gray)] = 2
        labels_ws = watershed(-gray, markers, mask=thresh)
        num_grains = len(np.unique(labels_ws)) - 1

        # Feature 3: Awn Length and Feature 4: Awn Density
        edges = cv2.Canny(gray, 100, 200)
        awns = np.sum(edges > 0)
        contours_awn, _ = cv2.findContours(
            edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        awn_length = max([cv2.arcLength(c, False)
                         for c in contours_awn], default=0)

        # Feature 5: Color Intensity (mean hue)
        hue = hsv[:, :, 0]
        mean_hue = np.mean(hue[hue > 0]) if np.any(hue > 0) else 0

        # Feature 6: Grain Size Mean
        grain_areas = []
        for region in measure.regionprops(label(labels_ws)):
            grain_areas.append(region.area)
        grain_size_mean = np.mean(grain_areas) if grain_areas else 0

        # Feature 7: Texture Contrast
        glcm = feature.graycomatrix(
            gray, [5], [0], levels=256, normed=True, symmetric=True)
        texture_contrast = feature.graycoprops(glcm, 'contrast')[0, 0]

        # Feature 8: Grain Density
        grain_density = num_grains / spike_length if spike_length > 0 else 0

        return [spike_length, num_grains, awn_length, awns, mean_hue, grain_size_mean, texture_contrast, grain_density]
    except Exception as e:
        print(f"Error in feature extraction: {e}")
        return [0] * 8

# API endpoint to predict wheat category


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        print("No file part in request")
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        print("No selected file")
        return jsonify({'error': 'No selected file'}), 400

    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        file.save(file_path)
        print(f"Image saved to {file_path}")

        # Read and process the image
        image = cv2.imread(file_path)
        if image is None:
            print(f"Failed to load image from {file_path}")
            return jsonify({'error': 'Failed to load image'}), 500
        print("Image loaded successfully")

        # Extract features
        features = extract_features(image)
        print("Extracted features:", features)
        if len(features) != 8:
            print("Feature length mismatch")
            return jsonify({'error': 'Feature extraction failed'}), 500
        features_array = np.array(features).reshape(1, -1)

        # Predict category
        try:
            prediction = model.predict(features_array)
            predicted_category = categories[prediction[0]]
            print(f"Predicted category: {predicted_category}")
            os.remove(file_path)
            return jsonify({'category': predicted_category}), 200, {'Content-Type': 'application/json'}
        except Exception as e:
            print(f"Prediction error: {e}")
            return jsonify({'error': 'Prediction failed'}), 500


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
