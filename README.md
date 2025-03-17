# 📚 FairyTale Story Generator

Create Fairy tale stories using an **LSTM-based language model** enhanced with **Top-p Sampling**, **Temperature Control**.

---

## 🚀 Project Overview
This project builds a powerful **LSTM model** using TensorFlow/Keras to generate imaginative fairy tales inspired by the works of **Hans Christian Andersen**. The project uses text generation techniques to produce diverse and coherent narratives.

---

## ✨ Features
✅ Interactive **Streamlit Web App** for seamless user experience.  
✅ Generate creative and engaging stories using a custom **LSTM Model**.  
✅ Tune creativity using **Temperature Control** and **Top-p Sampling**.  
✅ Evaluate generated text quality with **BLEU** and **ROUGE** scores.

---

## 🛠️ Installation Guide

### 1. Create a Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Download the Dataset
- Download **"Fairy Tales of Hans Christian Andersen"** from [Project Gutenberg](https://www.gutenberg.org/) in **Plain Text UTF-8** format.
- Save the file as:  
```
data/cleaned_fairy_tales.txt
```

### 4. Setup the Model
If not trained yet, run the following command to train the model:
```bash
python src/model.py
```

---

## 🚀 Usage Instructions

### 1. Run the Streamlit Web App
```bash
streamlit run app.py
```

### 2. Open in Your Browser
- Navigate to **`http://localhost:8501`**.

### 3. Using the Web App
- Enter a **starting sentence** for your story.
- Choose the **word count** for the generated text.
- Adjust **Temperature** (lower = more predictable, higher = more creative).
- Tune **Top-p Sampling** to control the randomness of word selection.
- Click the **"✨ Generate Story"** button to see the magic unfold! 🧙‍♂️

---

## 📊 Evaluation
The model evaluates text quality using:

- **BLEU Score** — Measures exact sequence overlap (ideal for structured text).
- **ROUGE Score** — Measures content overlap for creative text.
- **METEOR Score** — Evaluates text similarity using synonyms and word order flexibility.

Both scores are shown after generating a story to help assess performance.

---

## 🛠️ Model Details
- **Architecture:** LSTM-based sequence model with Bidirectional LSTM layers.  
- **LSTM Units:** 256  
- **Sampling Strategy:** Combination of **Top-p Sampling** and **Temperature Control**.  
- **Training Dataset:** Based on **Hans Christian Andersen's Fairy Tales** (~374,000 tokens).  

---

## 🌟 Future Improvements
- The evaluation scores are still LOW, need to improve.
- Integrate **transformer models** like GPT for improved text generation.  
- Develop a **"Story Themes"** feature (e.g., Adventure, Romance, Mystery).  

---
