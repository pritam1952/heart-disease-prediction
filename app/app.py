import os
import numpy as np
import pandas as pd
import pickle
from flask import Flask, request, render_template

app = Flask(__name__)

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR    = os.path.dirname(BASE_DIR)
RESULTS_DIR = os.path.join(ROOT_DIR, 'results')

# Lazy loading — load once on first request
_model  = None
_scaler = None

def get_model():
    global _model
    if _model is None:
        from tensorflow.keras.models import load_model
        _model = load_model(os.path.join(RESULTS_DIR, 'best_model.keras'))
    return _model

def get_scaler():
    global _scaler
    if _scaler is None:
        with open(os.path.join(RESULTS_DIR, 'scaler.pkl'), 'rb') as f:
            _scaler = pickle.load(f)
    return _scaler

FEATURE_COLUMNS = [
    'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs',
    'restecg', 'thalach', 'exang', 'oldpeak',
    'slope', 'ca', 'thal',
    'age_group', 'high_chol', 'high_bp', 'hr_ratio'
]


def preprocess_input(form):
    age      = float(form['age'])
    trestbps = float(form['trestbps'])
    chol     = float(form['chol'])
    thalach  = float(form['thalach'])

    age_group = 0 if age < 40 else 1 if age < 50 else 2 if age < 60 else 3
    high_chol = 1 if chol     >= 240 else 0
    high_bp   = 1 if trestbps >= 140 else 0
    hr_ratio  = round(thalach / (220 - age), 3)

    features = pd.DataFrame([[
        age,
        float(form['sex']),
        float(form['cp']),
        trestbps,
        chol,
        float(form['fbs']),
        float(form['restecg']),
        thalach,
        float(form['exang']),
        float(form['oldpeak']),
        float(form['slope']),
        float(form['ca']),
        float(form['thal']),
        age_group, high_chol, high_bp, hr_ratio
    ]], columns=FEATURE_COLUMNS)

    return get_scaler().transform(features)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    features    = preprocess_input(request.form)
    probability = float(get_model().predict(features).flatten()[0])
    prediction  = int(probability >= 0.5)

    result = {
        'label'      : 'Heart Disease Detected' if prediction == 1 else 'No Heart Disease',
        'probability': round(probability * 100, 2),
        'risk'       : 'High Risk'   if probability >= 0.7
                  else 'Medium Risk' if probability >= 0.4
                  else 'Low Risk',
        'prediction' : prediction
    }

    return render_template('result.html', result=result, form=request.form)


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)