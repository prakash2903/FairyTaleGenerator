import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle


def load_cleaned_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
    return text


def create_sequences(text, seq_length=50):
    tokens = text.split()
    sequences = []
    
    for i in range(seq_length, len(tokens)):
        seq = tokens[i-seq_length:i+1]  # +1 ensures target word is included
        sequences.append(seq)
    
    return sequences


def tokenize_and_encode(sequences):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(sequences)
    
    encoded_sequences = tokenizer.texts_to_sequences(sequences)
    vocab_size = len(tokenizer.word_index) + 1  # +1 for padding token

    return encoded_sequences, tokenizer, vocab_size


def pad_sequences_data(sequences, seq_length):
    sequences = np.array(sequences)
    X, y = sequences[:, :-1], sequences[:, -1]
    X_padded = pad_sequences(X, maxlen=seq_length, padding='pre')
    y = np.array(y)
    
    return X_padded, y


def save_tokenizer(tokenizer, file_path):
    with open(file_path, 'wb') as file:
        pickle.dump(tokenizer, file)


if __name__ == "__main__":
    input_file = "data/cleaned_fairy_tales.txt"
    tokenizer_file = "models/tokenizer.pkl"
    seq_length = 50

    text_data = load_cleaned_data(input_file)
    sequences = create_sequences(text_data, seq_length)

    encoded_sequences, tokenizer, vocab_size = tokenize_and_encode(sequences)
    X, y = pad_sequences_data(encoded_sequences, seq_length)

    np.save("data/X.npy", X)
    np.save("data/y.npy", y)
    save_tokenizer(tokenizer, tokenizer_file)

    print(f"Data preparation complete!")
    print(f"Vocabulary Size: {vocab_size}")
    print(f"Total Sequences Created: {len(sequences)}")
    print(f"Feature Shape (X): {X.shape}")
    print(f"Target Shape (y): {y.shape}")
