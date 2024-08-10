from flask import Flask, request, jsonify, send_from_directory
import joblib
import numpy as np
import os

app = Flask(__name__)

# Load the saved model and accuracy
try:
    model_path = os.path.join(os.path.dirname(__file__), 'svm_model.pkl')
    model = joblib.load(model_path)
except Exception as e:
    print(f"Error loading model: {e}")

accuracy_path = os.path.join(os.path.dirname(__file__), 'accuracy.txt')
with open(accuracy_path, 'r') as f:
    accuracy_str = f.read().strip()  # Read and strip any surrounding whitespace
    accuracy_value = float(accuracy_str[-4:])  # Extract and convert the accuracy value
    accuracy_percentage = accuracy_value * 100  # Convert to percentage

classes = ['Iris Setosa', 'Iris Versicolor', 'Iris Virginica']
image_paths = {
    'Iris Setosa': 'static/setosa.jpg',
    'Iris Versicolor': 'static/versicolor.jpg',
    'Iris Virginica': 'static/virginica.jpg'
}

@app.route('/')
def home():
    return send_from_directory('', 'index.html')

@app.route('/styles.css')
def styles():
    return send_from_directory('', 'styles.css')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json  # Expecting JSON data
    sample = np.array(data['features']).reshape(1, -1)  # Reshape to a single sample
    prediction = model.predict(sample)
    predicted_class = classes[prediction[0]]
    image_url = f'/static/{image_paths[predicted_class].split("/")[-1]}'  # Format for URL
    return jsonify({'prediction': predicted_class, 'accuracy': f'{accuracy_percentage:.2f}%', 'image_url': image_url})

if __name__ == '__main__':
    app.run(debug=True)
