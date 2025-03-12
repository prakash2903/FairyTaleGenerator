import random
import re
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.translate.bleu_score import sentence_bleu
from rouge import Rouge

# Load Model and Tokenizer
model = load_model("models/lstm_model.h5")

with open("models/tokenizer.pkl", 'rb') as file:
    tokenizer = pickle.load(file)

# Enhanced Text Generation with Sampling Strategy
def generate_text(seed_text, max_words=100, temperature=0.7, top_k=10):
    result = seed_text.split()

    for _ in range(max_words):
        encoded = tokenizer.texts_to_sequences([seed_text])[0]
        encoded = pad_sequences([encoded], maxlen=50, padding='pre')

        # Predict next word probabilities
        predicted_probs = model.predict(encoded, verbose=0)[0]

        # Apply Temperature Scaling
        predicted_probs = np.log(predicted_probs + 1e-8) / temperature
        predicted_probs = np.exp(predicted_probs) / np.sum(np.exp(predicted_probs))

        # Top-k Sampling
        top_k_indices = np.argsort(predicted_probs)[-top_k:]
        top_k_probs = predicted_probs[top_k_indices]
        top_k_probs /= np.sum(top_k_probs)  # Normalize probabilities

        # Randomly choose from top-k predictions
        predicted_word_id = np.random.choice(top_k_indices, p=top_k_probs)

        # Map predicted ID to word
        next_word = None
        for word, index in tokenizer.word_index.items():
            if index == predicted_word_id:
                next_word = word
                break

        if not next_word:
            break  # Stop if no valid word found

        # Append predicted word to result
        result.append(next_word)
        seed_text += ' ' + next_word

    return ' '.join(result)

# Improved BLEU Score Calculation (Longer Reference Sequences)
def calculate_bleu_score(reference_text, generated_text):
    # Extract actual story content (skip metadata and table of contents)
    story_start = re.search(r'(?i)contents(.*?)a story', reference_text, re.DOTALL)
    if story_start:
        reference_text = reference_text[story_start.end():]  # Extract text after contents section

    # Use longer reference text (5+ lines)
    reference_sentences = reference_text.split('.')[:5]  # Longer reference for better evaluation
    reference_tokens = [sentence.strip().split() for sentence in reference_sentences]

    # Tokenize generated text
    generated_tokens = generated_text.split()

    # BLEU Score Calculation
    score = sentence_bleu(reference_tokens, generated_tokens)
    return score

# ROUGE Score Calculation
def calculate_rouge_score(reference_text, generated_text):
    rouge = Rouge()
    try:
        scores = rouge.get_scores(generated_text, reference_text[:500])  # First 500 characters for coherence
        return scores[0]['rouge-l']['f']  # F1 score for best alignment
    except Exception:
        return 0.0  # Handle scoring errors safely

# Main Evaluation
if __name__ == "__main__":
    # Example starting sequence
    starting_sequence = "once upon a time in a faraway kingdom"

    # Generate sample story
    generated_story = generate_text(starting_sequence, max_words=100, temperature=0.7, top_k=10)
    print("\n📜 Generated Story:\n")
    print(generated_story)

    # BLEU & ROUGE Score Evaluation
    with open("data/cleaned_fairy_tales.txt", 'r', encoding='utf-8') as file:
        reference_text = file.read()

    bleu_score = calculate_bleu_score(reference_text, generated_story)
    rouge_score = calculate_rouge_score(reference_text, generated_story)

    print(f"\n🔍 BLEU Score: {bleu_score:.4f}")
    print(f"🔍 ROUGE Score: {rouge_score:.4f}")
