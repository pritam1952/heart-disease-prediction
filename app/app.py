import os
import pickle
import pandas as pd
from flask import Flask, request, render_template
from tensorflow.keras.models import load_model

# ------------------ Flask app ------------------
app = Flask(__name__)

# ------------------ Paths ------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, 'results', 'best_model.keras')
PREPROCESSOR_PATH = os.path.join(BASE_DIR, 'results', 'preprocessor.pkl')

# ------------------ Load model & preprocessor ------------------
try:
    model = load_model(MODEL_PATH)
except Exception as e:
    raise FileNotFoundError(f"Model not found at {MODEL_PATH}") from e

try:
    with open(PREPROCESSOR_PATH, 'rb') as f:
        preprocessor = pickle.load(f)
except Exception as e:
    raise FileNotFoundError(f"Preprocessor not found at {PREPROCESSOR_PATH}") from e

# ------------------ Feature columns ------------------
FEATURE_COLUMNS = [
    'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs',
    'restecg', 'thalach', 'exang', 'oldpeak',
    'slope', 'ca', 'thal',
    'age_group', 'high_chol', 'high_bp', 'hr_ratio'
]

# ------------------ Preprocessing function ------------------
def preprocess_input(form):
    # Safely get form values
    def safe_float(key, default=0):
        try:
            return float(form.get(key, default))
        except ValueError:
            return default

    age      = safe_float('age')
    trestbps = safe_float('trestbps')
    chol     = safe_float('chol')
    thalach  = safe_float('thalach')

    # Derived features
    age_group = 0 if age < 40 else 1 if age < 50 else 2 if age < 60 else 3
    high_chol = 1 if chol >= 240 else 0
    high_bp   = 1 if trestbps >= 140 else 0
    hr_ratio  = round(thalach / (220 - age), 3) if (220 - age) != 0 else 0

    # Create DataFrame for preprocessor
    features = pd.DataFrame([[
        age,
        safe_float('sex'),
        safe_float('cp'),
        trestbps,
        chol,
        safe_float('fbs'),
        safe_float('restecg'),
        thalach,
        safe_float('exang'),
        safe_float('oldpeak'),
        safe_float('slope'),
        safe_float('ca'),
        safe_float('thal'),
        age_group,
        high_chol,
        high_bp,
        hr_ratio
    ]], columns=FEATURE_COLUMNS)

    # Transform features using preprocessor
    return preprocessor.transform(features)

# ------------------ Routes ------------------
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        features = preprocess_input(request.form)
        probability = float(model.predict(features).flatten()[0])
        prediction = int(probability >= 0.5)

        result = {
            'label'      : 'Heart Disease Detected' if prediction == 1 else 'No Heart Disease',
            'probability': round(probability * 100, 2),
            'risk'       : 'High Risk'   if probability >= 0.7
                            else 'Medium Risk' if probability >= 0.4
                            else 'Low Risk',
            'prediction' : prediction
        }

        return render_template('result.html', result=result, form=request.form)
    except Exception as e:
        return f"Error processing input: {e}", 400

# ------------------ Run ------------------
if __name__ == '__main__':
    app.run(debug=True)