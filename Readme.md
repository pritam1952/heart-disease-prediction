# ❤️ Heart Disease Detection using ANN

A machine learning web application that predicts heart disease risk using an Artificial Neural Network (ANN) trained on the Cleveland Heart Disease dataset.

---

## 🖥️ Live Demo

> Deploy karne ke baad apni URL yahan daalo
> `https://heart-disease-ann.onrender.com`

---

## 📁 Project Structure

```
heart-disease-ann/
├── app/
│   ├── app.py
│   ├── templates/
│   │   ├── index.html
│   │   └── result.html
│   └── static/
│       └── style.css
├── data/
│   └── heart.csv
├── notebooks/
│   └── eda.ipynb
├── src/
│   ├── preprocess.py
│   ├── model.py
│   └── evaluate.py
├── results/
│   ├── best_model.keras
│   ├── scaler.pkl
│   └── ...
├── requirements.txt
├── Procfile
├── .gitignore
└── README.md
```

---

## 📊 Dataset

| Property | Detail |
|----------|--------|
| Source | Cleveland Heart Disease Dataset (UCI) |
| Samples | 303 patients |
| Features | 13 original + 4 engineered |
| Target | 1 = Heart Disease, 0 = No Disease |

**Features used:**
`age`, `sex`, `cp`, `trestbps`, `chol`, `fbs`, `restecg`, `thalach`, `exang`, `oldpeak`, `slope`, `ca`, `thal`

**Engineered features:**
`age_group`, `high_chol`, `high_bp`, `hr_ratio`

---

## 🧠 ANN Architecture

```
Input  Layer  →  17 neurons
Hidden Layer1 →  64 neurons  (ReLU + BatchNorm + Dropout 0.3)
Hidden Layer2 →  32 neurons  (ReLU + BatchNorm + Dropout 0.2)
Hidden Layer3 →  16 neurons  (ReLU + Dropout 0.1)
Output Layer  →   1 neuron   (Sigmoid)
```

| Setting | Value |
|---------|-------|
| Optimizer | Adam (lr=0.001) |
| Loss | Binary Crossentropy |
| Epochs | 200 (EarlyStopping) |
| Batch Size | 32 |

---

## ⚙️ Setup & Installation

### 1. Clone the repository
```bash
git clone https://github.com/pritam1952/heart-disease-ann.git
cd heart-disease-ann
```

### 2. Create virtual environment
```bash
python -m venv ml_env
ml_env\Scripts\activate       # Windows
source ml_env/bin/activate    # Mac/Linux
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Add dataset
Download `heart.csv` from [Kaggle](https://www.kaggle.com/datasets/cherngs/heart-disease-cleveland-uci) and place it in `data/` folder.

---

## 🚀 How to Run

### Train the model
```bash
cd src
python preprocess.py
python model.py
python evaluate.py
```

### Run the Flask app
```bash
cd app
python app.py
```

Open browser: **http://127.0.0.1:5000**

---

## 📈 Model Performance

| Metric | Score |
|--------|-------|
| Accuracy | ~87% |
| Precision | ~85% |
| Recall | ~89% |
| F1 Score | ~87% |
| AUC-ROC | ~92% |

> Note: Results may vary slightly due to random seed.

---

## 🌐 Deployment (Render)

1. Push code to GitHub
2. Go to [render.com](https://render.com)
3. New → Web Service → Connect GitHub repo
4. Set build command: `pip install -r requirements.txt`
5. Set start command: `gunicorn app.app:app`
6. Deploy!

---

## 🛠️ Tech Stack

| Category | Tools |
|----------|-------|
| Language | Python 3 |
| ML Framework | TensorFlow / Keras |
| Data Processing | Pandas, NumPy, Scikit-learn |
| Visualization | Matplotlib, Seaborn |
| Web Framework | Flask |
| Deployment | Render |

---

## 📋 Requirements

```
flask
numpy
pandas
scikit-learn
tensorflow
gunicorn
matplotlib
seaborn
jupyter
notebook
```

---

## 👤 Author

**Pritam Kumar**
- GitHub: [@pritam1952](https://github.com/pritam1952)

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).