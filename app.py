import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.meteor_score import meteor_score
from rouge import Rouge


model = load_model("models/lstm_model.h5")

with open("models/tokenizer.pkl", 'rb') as file:
    tokenizer = pickle.load(file)

def top_p_sampling(predicted_probs, top_p=0.9):
    sorted_indices = np.argsort(predicted_probs)[::-1]
    sorted_probs = np.sort(predicted_probs)[::-1]
    
    cumulative_probs = np.cumsum(sorted_probs)
    cutoff_index = np.where(cumulative_probs > top_p)[0][0]

    top_p_indices = sorted_indices[:cutoff_index + 1]
    top_p_probs = sorted_probs[:cutoff_index + 1]

    top_p_probs /= np.sum(top_p_probs)
    predicted_word_id = np.random.choice(top_p_indices, p=top_p_probs)

    return predicted_word_id


def generate_text(seed_text, max_words=100, temperature=0.7, top_p=0.9):
    result = seed_text.split()

    for _ in range(max_words):
        encoded = tokenizer.texts_to_sequences([seed_text])[0]
        encoded = pad_sequences([encoded], maxlen=70, padding='pre')

        predicted_probs = model.predict(encoded, verbose=0)[0]

        predicted_probs = np.clip(predicted_probs, 1e-8, 1.0)
        predicted_probs = np.log(predicted_probs) / temperature
        predicted_probs = np.exp(predicted_probs) / np.sum(np.exp(predicted_probs))

        predicted_word_id = top_p_sampling(predicted_probs, top_p=top_p)

        next_word = None
        for word, index in tokenizer.word_index.items():
            if index == predicted_word_id:
                next_word = word
                break

        if not next_word:
            break

        result.append(next_word)
        seed_text += ' ' + next_word

    return ' '.join(result)

def calculate_bleu_score(reference_text, generated_text):
    reference_sentences = reference_text.split('.')[:20]
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

def calculate_meteor_score(reference_text, generated_text):
    reference_sentences = reference_text.split('.')[:10]
    reference_tokens = [sentence.strip().split() for sentence in reference_sentences]

    generated_tokens = generated_text.split()
    
    score = meteor_score(reference_tokens, generated_tokens)
    return score

# Streamlit UI
st.title("📚 FairyTale Story Generator")
st.write("Powered by LSTM and sampling techniques.")

starting_sequence = st.text_input("Enter a starting sentence:", "once upon a time")
max_words = st.slider("Number of words in the story:", 50, 300, 100)
temperature = st.slider("Creativity (Temperature):", 0.2, 1.2, 0.7)
top_p = st.slider("Top-p Sampling (Higher = More Creativity):", 0.5, 1.0, 0.9)

if st.button("✨ Generate Story"):
    with st.spinner("Generating your story..."):
        generated_story = generate_text(starting_sequence, max_words, temperature, top_p)
        st.subheader("📜 Generated Story:")
        st.write(generated_story)

        with open("data/cleaned_fairy_tales.txt", 'r', encoding='utf-8') as file:
            reference_text = file.read()

        bleu_score = calculate_bleu_score(reference_text, generated_story)
        rouge_score = calculate_rouge_score(reference_text, generated_story)
        meteor_score = calculate_meteor_score(reference_text, generated_story)

        st.success("✅ Story Generated Successfully!")
        st.markdown(f"**🔍 BLEU Score:** {bleu_score:.4f}")
        st.markdown(f"**🔍 ROUGE Score:** {rouge_score:.4f}")
        st.markdown(f"**🔍 METEOR Score:** {meteor_score:.4f}")

st.markdown("---")
st.markdown("💬 By **21PT17** | TensorFlow")

