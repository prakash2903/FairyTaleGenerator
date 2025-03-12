import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.translate.bleu_score import sentence_bleu

# Load Model and Tokenizer
model = load_model("models/lstm_model.h5")

with open("models/tokenizer.pkl", 'rb') as file:
    tokenizer = pickle.load(file)

# Generate Text Function
def generate_text(seed_text, max_words=100):
    result = seed_text.split()

    for _ in range(max_words):
        encoded = tokenizer.texts_to_sequences([seed_text])[0]
        encoded = pad_sequences([encoded], maxlen=50, padding='pre')

        # Predict next word
        predicted_word_id = np.argmax(model.predict(encoded, verbose=0))
        
        # Map predicted ID to word
        for word, index in tokenizer.word_index.items():
            if index == predicted_word_id:
                next_word = word
                break
        
        # Append predicted word to result
        result.append(next_word)
        seed_text += ' ' + next_word
    
    return ' '.join(result)

# BLEU Score Calculation
def calculate_bleu_score(reference_text, generated_text):
    reference_tokens = [reference_text.split()]
    generated_tokens = generated_text.split()
    score = sentence_bleu(reference_tokens, generated_tokens)
    return score

# Main Evaluation
if __name__ == "__main__":
    # Example starting sequence
    starting_sequence = "once upon a time in a faraway kingdom"
    
    # Generate sample story
    generated_story = generate_text(starting_sequence, max_words=100)
    print("\n📜 Generated Story:\n")
    print(generated_story)

    # BLEU Score Evaluation
    with open("data/cleaned_fairy_tales.txt", 'r', encoding='utf-8') as file:
        original_text = file.read().split('\n')[0]  # Sample original text for BLEU comparison

    bleu_score = calculate_bleu_score(original_text, generated_story)
    print(f"\n🔍 BLEU Score: {bleu_score:.4f}")
