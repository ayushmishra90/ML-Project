# README

## Project Overview

This project consists of two primary tasks:

1. **House Price Prediction Model**
2. **Iris Flower Classification with SVM and Flask Deployment**

## Task 1: House Price Prediction Model

### Overview

Developed a machine learning model to predict house prices using various features. The model was built and evaluated using the following steps:

### Steps

1. **Data Loading:**
   - Loaded the dataset from `Housing.csv`.
   - Explored the data, including statistical summaries and visualizations.

2. **Data Preprocessing:**
   - Handled missing values and encoded categorical variables.
   - Analyzed the impact of various features on house prices.

3. **Model Building:**
   - Used linear regression for prediction.
   - Split data into training and testing sets.
   - Trained and evaluated the model.

4. **Results:**
   - Mean Absolute Error: `1127483.35`
   - Mean Squared Error: `2292721545725.36`
   - R² Score: `0.55`

5. **Visualizations:**
   - Plotted feature distributions and price distributions to understand data characteristics.

### How to Run

1. Ensure you have the required libraries: `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`.
2. Place `Housing.csv` in the project directory.
3. Run the script to train and evaluate the model.

## Task 2: Iris Flower Classification with SVM and Flask Deployment

### Overview

Deployed an SVM model for Iris flower classification using Flask, providing a web interface for predictions.

### Steps

1. **Model Training:**
   - Loaded the Iris dataset.
   - Trained an SVM model and saved it as `svm_model.pkl`.
   - Calculated and saved the model's accuracy in `accuracy.txt`.

2. **Flask Application:**
   - Created a Flask app to serve the model.
   - Implemented routes for serving HTML, CSS, and prediction endpoints.
   - The `/predict` endpoint takes feature data, returns the predicted class, accuracy, and image URL.

### How to Run

1. Ensure you have the required libraries: `Flask`, `joblib`, `numpy`.
2. Place `svm_model.pkl` and `accuracy.txt` in the project directory.
3. Save the Flask app script as `app.py`.
4. Run the Flask app: `python app.py`.
5. Access the web interface at `http://127.0.0.1:5000/`.

### Files

- `app.py`: Flask application for serving the model.
- `svm_model.pkl`: Saved SVM model.
- `accuracy.txt`: Model accuracy.
- `index.html` and `styles.css`: Web interface files.
- `static/`: Contains images for Iris flower types.
