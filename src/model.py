import numpy as np
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

# ------------------------------
# Paths
# ------------------------------
RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

# ------------------------------
# STEP 1: Load Processed Data
# ------------------------------
def load_processed_data():
    X_train = np.load(os.path.join(RESULTS_DIR, 'X_train.npy'))
    X_test  = np.load(os.path.join(RESULTS_DIR, 'X_test.npy'))
    y_train = np.load(os.path.join(RESULTS_DIR, 'y_train.npy'))
    y_test  = np.load(os.path.join(RESULTS_DIR, 'y_test.npy'))
    return X_train, X_test, y_train, y_test

# ------------------------------
# STEP 2: Build ANN Model
# ------------------------------
def build_model(input_dim):
    model = Sequential([
        Dense(64, activation='relu', input_dim=input_dim),
        BatchNormalization(),
        Dropout(0.3),

        Dense(32, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),

        Dense(16, activation='relu'),
        Dropout(0.1),

        Dense(1, activation='sigmoid')  # Binary classification
    ])

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model

# ------------------------------
# STEP 3: Callbacks
# ------------------------------
def get_callbacks():
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=20,
        restore_best_weights=True
    )

    checkpoint = ModelCheckpoint(
        filepath=os.path.join(RESULTS_DIR, 'best_model.keras'),
        monitor='val_accuracy',
        save_best_only=True
    )

    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=10,
        min_lr=1e-6
    )

    return [early_stop, checkpoint, reduce_lr]

# ------------------------------
# STEP 4: Train Model
# ------------------------------
def train_model(model, X_train, y_train, callbacks):
    history = model.fit(
        X_train, y_train,
        epochs=200,
        batch_size=32,
        validation_split=0.2,
        callbacks=callbacks,
        verbose=1
    )
    return history

# ------------------------------
# STEP 5: Plot Training History
# ------------------------------
def plot_history(history):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Accuracy
    axes[0].plot(history.history['accuracy'], label='Train Accuracy', color='#2ecc71', linewidth=2)
    axes[0].plot(history.history['val_accuracy'], label='Val Accuracy', color='#e74c3c', linewidth=2)
    axes[0].set_title('Model Accuracy', fontsize=13, fontweight='bold')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Loss
    axes[1].plot(history.history['loss'], label='Train Loss', color='#2ecc71', linewidth=2)
    axes[1].plot(history.history['val_loss'], label='Val Loss', color='#e74c3c', linewidth=2)
    axes[1].set_title('Model Loss', fontsize=13, fontweight='bold')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.suptitle('Training History', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'training_history.png'), dpi=150, bbox_inches='tight')
    plt.show()

# ------------------------------
# STEP 6: Save Final Model
# ------------------------------
def save_model(model):
    model.save(os.path.join(RESULTS_DIR, 'ann_model.keras'))

# ------------------------------
# STEP 7: Evaluate on Test Set
# ------------------------------
def evaluate_model(model, X_test, y_test):
    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nTest Loss: {loss:.4f}, Test Accuracy: {acc:.4f}")

    y_pred = (model.predict(X_test) > 0.5).astype(int)
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# ------------------------------
# MAIN PIPELINE
# ------------------------------
def model_pipeline():
    X_train, X_test, y_train, y_test = load_processed_data()

    model     = build_model(input_dim=X_train.shape[1])
    callbacks = get_callbacks()
    history   = train_model(model, X_train, y_train, callbacks)

    plot_history(history)
    save_model(model)
    evaluate_model(model, X_test, y_test)

    return model, history, X_test, y_test

if __name__ == "__main__":
    model_pipeline()