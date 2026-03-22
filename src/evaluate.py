import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import load_model
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    roc_curve, classification_report
)
import os

# ------------------------------
# Paths
# ------------------------------
RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)
METRICS_FILE = os.path.join(RESULTS_DIR, 'metrics.txt')

# ------------------------------
# STEP 1: Load Model & Data
# ------------------------------
def load_model_and_data():
    model  = load_model(os.path.join(RESULTS_DIR, 'best_model.keras'))
    X_test = np.load(os.path.join(RESULTS_DIR, 'X_test.npy'))
    y_test = np.load(os.path.join(RESULTS_DIR, 'y_test.npy'))
    return model, X_test, y_test

# ------------------------------
# STEP 2: Predictions
# ------------------------------
def get_predictions(model, X_test, threshold=0.5):
    y_prob = model.predict(X_test).flatten()
    y_pred = (y_prob >= threshold).astype(int)
    return y_prob, y_pred

# ------------------------------
# STEP 3: Metrics
# ------------------------------
def print_and_save_metrics(y_test, y_pred, y_prob):
    metrics_str = (
        f"{'='*45}\n"
        f"        MODEL EVALUATION RESULTS\n"
        f"{'='*45}\n"
        f"  Accuracy  : {accuracy_score(y_test, y_pred)*100:.2f}%\n"
        f"  Precision : {precision_score(y_test, y_pred)*100:.2f}%\n"
        f"  Recall    : {recall_score(y_test, y_pred)*100:.2f}%\n"
        f"  F1 Score  : {f1_score(y_test, y_pred)*100:.2f}%\n"
        f"  AUC-ROC   : {roc_auc_score(y_test, y_prob)*100:.2f}%\n"
        f"{'='*45}\n\n"
        f"Classification Report:\n"
        f"{classification_report(y_test, y_pred, target_names=['No Disease', 'Heart Disease'])}\n"
    )

    print(metrics_str)

    # Save to file
    with open(METRICS_FILE, 'w') as f:
        f.write(metrics_str)

# ------------------------------
# STEP 4: Confusion Matrix
# ------------------------------
def plot_confusion_matrix(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(7, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['No Disease', 'Heart Disease'],
                yticklabels=['No Disease', 'Heart Disease'],
                linewidths=1, linecolor='white',
                annot_kws={'size': 14, 'weight': 'bold'})
    plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
    plt.ylabel('Actual', fontsize=12)
    plt.xlabel('Predicted', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'confusion_matrix.png'), dpi=150, bbox_inches='tight')
    plt.show()

# ------------------------------
# STEP 5: ROC Curve
# ------------------------------
def plot_roc_curve(y_test, y_prob):
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc = roc_auc_score(y_test, y_prob)
    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, color='#e74c3c', linewidth=2.5,
             label=f'ANN  (AUC = {auc:.3f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--', linewidth=1.5, label='Random Classifier')
    plt.fill_between(fpr, tpr, alpha=0.08, color='#e74c3c')
    plt.title('ROC Curve', fontsize=14, fontweight='bold')
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'roc_curve.png'), dpi=150, bbox_inches='tight')
    plt.show()

# ------------------------------
# STEP 6: Prediction Distribution
# ------------------------------
def plot_prediction_distribution(y_prob, y_test):
    plt.figure(figsize=(9, 5))
    plt.hist(y_prob[y_test == 0], bins=25, alpha=0.6,
             color='#2ecc71', label='No Disease', edgecolor='black')
    plt.hist(y_prob[y_test == 1], bins=25, alpha=0.6,
             color='#e74c3c', label='Heart Disease', edgecolor='black')
    plt.axvline(x=0.5, color='black', linestyle='--',
                linewidth=1.5, label='Threshold = 0.5')
    plt.title('Prediction Probability Distribution', fontsize=13, fontweight='bold')
    plt.xlabel('Predicted Probability', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'prediction_distribution.png'), dpi=150, bbox_inches='tight')
    plt.show()

# ------------------------------
# MAIN PIPELINE
# ------------------------------
def evaluate_pipeline():
    model, X_test, y_test  = load_model_and_data()
    y_prob, y_pred          = get_predictions(model, X_test)

    print_and_save_metrics(y_test, y_pred, y_prob)
    plot_confusion_matrix(y_test, y_pred)
    plot_roc_curve(y_test, y_prob)
    plot_prediction_distribution(y_prob, y_test)
    print(f"\nAll metrics saved to: {METRICS_FILE}")

if __name__ == "__main__":
    evaluate_pipeline()