# classifier.py
import tensorflow as tf
from tensorflow.keras import layers, models

def build_1d_cnn(input_len=1024, n_classes=4):
    inp = layers.Input(shape=(input_len,1), name='inp')
    x = layers.Conv1D(32, 11, activation='relu', padding='same')(inp)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Conv1D(64, 9, activation='relu', padding='same')(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Conv1D(128, 7, activation='relu', padding='same')(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(128, activation='relu')(x)
    out_class = layers.Dense(n_classes, activation='softmax', name='class_out')(x)
    model = models.Model(inputs=inp, outputs=out_class)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model
