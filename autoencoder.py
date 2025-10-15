# autoencoder.py
from tensorflow.keras import layers, models

def build_autoencoder(input_len=1024, latent_dim=64):
    inp = layers.Input(shape=(input_len,1))
    x = layers.Conv1D(32, 7, activation='relu', padding='same')(inp)
    x = layers.MaxPooling1D(2, padding='same')(x)
    x = layers.Conv1D(64, 5, activation='relu', padding='same')(x)
    x = layers.MaxPooling1D(2, padding='same')(x)
    x = layers.Flatten()(x)
    z = layers.Dense(latent_dim, activation='relu')(x)

    x = layers.Dense((input_len//4)*64, activation='relu')(z)
    x = layers.Reshape((input_len//4,64))(x)
    x = layers.UpSampling1D(2)(x)
    x = layers.Conv1D(32,5, activation='relu', padding='same')(x)
    x = layers.UpSampling1D(2)(x)
    out = layers.Conv1D(1,7, activation='linear', padding='same')(x)
    model = models.Model(inputs=inp, outputs=out)
    model.compile(optimizer='adam', loss='mse')
    return model
