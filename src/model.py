import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# Define LSTM Model
def build_model(vocab_size, seq_length):
    model = Sequential()
    
    # Embedding Layer
    model.add(Embedding(input_dim=vocab_size, output_dim=100, input_length=seq_length))
    
    # LSTM Layers
    model.add(LSTM(128, return_sequences=True))
    model.add(Dropout(0.3))  # Dropout for regularization
    model.add(LSTM(128))     
    
    # Dense Layer for Prediction
    model.add(Dense(vocab_size, activation='softmax'))
    
    # Compile the model
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=Adam(learning_rate=0.001), 
                  metrics=['accuracy'])
    
    return model

# Training Process
def train_model(model, X, y, batch_size=64, epochs=30):
    history = model.fit(X, y, batch_size=batch_size, epochs=epochs, validation_split=0.2)
    return history

# Main Execution
if __name__ == "__main__":
    # Load processed data
    X = np.load("data/X.npy")
    y = np.load("data/y.npy")
    
    # Hyperparameters
    seq_length = X.shape[1]  # Sequence Length (50)
    vocab_size = 14410       # From data preparation output

    # Build and Train Model
    model = build_model(vocab_size, seq_length)
    model.summary()  # Display model architecture
    
    history = train_model(model, X, y, batch_size=64, epochs=30)

    # Save the trained model
    model.save("models/lstm_model.h5")
    print("✅ Model training complete! Model saved to 'models/lstm_model.h5'")
