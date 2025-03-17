import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam


def build_model(vocab_size, seq_length):
    model = Sequential()
    
    # Embedding Layer
    model.add(Embedding(input_dim=vocab_size, output_dim=100, input_length=seq_length))
    
    # LSTM Layers
    model.add(LSTM(128, return_sequences=True))
    model.add(Dropout(0.3))
    model.add(LSTM(128))     
    
    # Dense Layer for Prediction
    model.add(Dense(vocab_size, activation='softmax'))

    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=Adam(learning_rate=0.001), 
                  metrics=['accuracy'])
    
    return model

def train_model(model, X, y, batch_size=64, epochs=30):
    history = model.fit(X, y, batch_size=batch_size, epochs=epochs, validation_split=0.2)
    return history


if __name__ == "__main__":
    X = np.load("data/X.npy")
    y = np.load("data/y.npy")
    
    seq_length = X.shape[1]
    vocab_size = 14410

    model = build_model(vocab_size, seq_length)
    model.summary()
    
    history = train_model(model, X, y, batch_size=64, epochs=30)

    model.save("models/lstm_model.h5")
    print("Model training completed... Saved to 'models/lstm_model.h5'")
