import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, RepeatVector, TimeDistributed, Dense
from tensorflow.keras.optimizers import Adam

TIMESTEPS = 100
FEATURES = 3

def build_model():
    inputs = Input(shape=(TIMESTEPS, FEATURES))

    encoded = LSTM(32, return_sequences=False)(inputs)
    decoded = RepeatVector(TIMESTEPS)(encoded)
    decoded = LSTM(32, return_sequences=True)(decoded)
    outputs = TimeDistributed(Dense(FEATURES))(decoded)

    model = Model(inputs, outputs)
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    return model

def generate_normal_data(samples=1500):
    hr = np.random.normal(75, 4, samples)
    spo2 = np.random.normal(98, 1, samples)
    bp = np.random.normal(120, 5, samples)

    data = np.stack([hr, spo2, bp], axis=1)
    return data

def train_model(model):
    data = generate_normal_data()

    sequences = []
    for i in range(len(data) - TIMESTEPS):
        sequences.append(data[i:i+TIMESTEPS])

    sequences = np.array(sequences)

    model.fit(sequences, sequences,
              epochs=10,
              batch_size=32,
              verbose=1)

    # Calculate threshold from training reconstruction error
    recon = model.predict(sequences, verbose=0)
    mse = np.mean(np.square(sequences - recon), axis=(1,2))
    threshold = np.mean(mse) + 3*np.std(mse)

    return model, threshold
