# train.py
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from classifier import build_1d_cnn
from autoencoder import build_autoencoder
import joblib
import os

def train_classifier(X, y, model_path='fra_classifier.h5', epochs=20, batch_size=32):
    """
    X: (N, L) float array, y: integer labels (N,)
    """
    X = X[..., None]
    y_cat = to_categorical(y, num_classes=np.max(y)+1)
    X_train, X_val, y_train, y_val = train_test_split(X, y_cat, test_size=0.2, random_state=42)
    model = build_1d_cnn(input_len=X.shape[1], n_classes=y_cat.shape[1])
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs, batch_size=batch_size)
    model.save(model_path)
    print("Saved classifier to", model_path)
    return model

def train_autoencoder(X_normal, model_path='fra_autoencoder.h5', epochs=50, batch_size=32):
    """
    X_normal: (N_normal, L) only healthy signatures
    """
    Xn = X_normal[..., None]
    ae = build_autoencoder(input_len=Xn.shape[1])
    ae.fit(Xn, Xn, epochs=epochs, batch_size=batch_size, validation_split=0.1)
    ae.save(model_path)
    print("Saved autoencoder to", model_path)
    return ae

if __name__ == "__main__":
    # placeholder quick test using synthetic data generator if available
    if os.path.exists('synthetic.npy'):
        data = np.load('synthetic.npy', allow_pickle=True)
        X = np.stack([d['mag'] for d in data])
        y = np.array([d['label'] for d in data])
        train_classifier(X, y)
        # save simple scaler
        joblib.dump({'dummy':'ok'}, 'meta.joblib')
    else:
        print("No dataset found. Create dataset or run synthetic_data.py to generate samples.")
