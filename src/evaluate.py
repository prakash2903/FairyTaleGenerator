import random
import re
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.translate.bleu_score import sentence_bleu
from rouge import Rouge


model = load_model("models/lstm_model.h5")

with open("models/tokenizer.pkl", 'rb') as file:
    tokenizer = pickle.load(file)


def generate_text(seed_text, max_words=100, temperature=0.7, top_k=10):
    result = seed_text.split()

    for _ in range(max_words):
        encoded = tokenizer.texts_to_sequences([seed_text])[0]
        encoded = pad_sequences([encoded], maxlen=50, padding='pre')

        # Predicting next word probabilities
        predicted_probs = model.predict(encoded, verbose=0)[0]

        # Applying Temperature Scaling
        predicted_probs = np.log(predicted_probs + 1e-8) / temperature
        predicted_probs = np.exp(predicted_probs) / np.sum(np.exp(predicted_probs))

        # Top-k Sampling
        top_k_indices = np.argsort(predicted_probs)[-top_k:]
        top_k_probs = predicted_probs[top_k_indices]
        
        # Normalize probabilities
        top_k_probs /= np.sum(top_k_probs)  

        # Randomly choose from top-k predictions
        predicted_word_id = np.random.choice(top_k_indices, p=top_k_probs)

        # Map predicted ID to word
        next_word = None
        for word, index in tokenizer.word_index.items():
            if index == predicted_word_id:
                next_word = word
                break

        if not next_word:
            # Stop if no valid word found
            break  

        # Appending predicted word to result
        result.append(next_word)
        seed_text += ' ' + next_word

    return ' '.join(result)


def calculate_bleu_score(reference_text, generated_text):
    # Skipping Metadata if exists.
    story_start = re.search(r'(?i)contents(.*?)a story', reference_text, re.DOTALL)
    if story_start:
        reference_text = reference_text[story_start.end():]

    reference_sentences = reference_text.split('.')
    
    sample_size = min(20, len(reference_sentences))
    reference_sentences = random.sample(reference_sentences, sample_size)
    
    reference_tokens = [sentence.strip().split() for sentence in reference_sentences]
    generated_tokens = generated_text.split()

    score = sentence_bleu(reference_tokens, generated_tokens)
    return score


def calculate_rouge_score(reference_text, generated_text):
    rouge = Rouge()
    try:
        sampled_reference_text = reference_text[1000:2000]
        scores = rouge.get_scores(generated_text, sampled_reference_text)
        return scores[0]['rouge-l']['f']
    except Exception:
        return 0.0
    
from nltk.translate.meteor_score import meteor_score

def calculate_meteor_score(reference_text, generated_text):
    reference_sentences = reference_text.split('.')[:10]
    reference_tokens = [sentence.strip().split() for sentence in reference_sentences]

    generated_tokens = generated_text.split()
    
    score = meteor_score(reference_tokens, generated_tokens)
    return score

    
if __name__ == "__main__":
    starting_sequence = "once upon a time"

    generated_story = generate_text(starting_sequence, max_words=75, temperature=0.7, top_k=10)
    print("\nGenerated Story:\n")
    print(generated_story)

    with open("data/cleaned_fairy_tales.txt", 'r', encoding='utf-8') as file:
        reference_text = file.read()

    bleu_score = calculate_bleu_score(reference_text, generated_story)
    rouge_score = calculate_rouge_score(reference_text, generated_story)
    meteor_score = calculate_meteor_score(reference_text, generated_story)

    print(f"\nBLEU Score: {bleu_score:.4f}")
    print(f"ROUGE Score: {rouge_score:.4f}")
    print(f"Meteor Score: {meteor_score:.4f}")
