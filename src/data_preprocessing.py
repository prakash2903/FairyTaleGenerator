import re
import string
import numpy as np

# Read the dataset
def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
    return text

# Clean the text
def clean_text(text):
    # Step 1: Remove Gutenberg headers and footers
    text = re.sub(r'\n+', ' ', text)  # Remove excessive newlines
    text = re.sub(r'(?s)\*\*\*.*?START OF THIS PROJECT GUTENBERG EBOOK.*?\*\*\*', '', text)
    text = re.sub(r'(?s)\*\*\*.*?END OF THIS PROJECT GUTENBERG EBOOK.*?\*\*\*', '', text)

    # Step 2: Normalize text
    text = text.lower()                         # Lowercase the text
    text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
    text = re.sub(r'\d+', '', text)             # Remove numbers
    text = re.sub(r'\s+', ' ', text).strip()    # Remove extra spaces

    return text

# Tokenize the text
def tokenize_text(text):
    tokens = text.split()  # Simple word-level tokenization
    return tokens

# Save processed data
def save_cleaned_data(tokens, file_path):
    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(' '.join(tokens))

# Main preprocessing flow
if __name__ == "__main__":
    # File paths
    input_file = "data/fairy_tales.txt"
    output_file = "data/cleaned_fairy_tales.txt"

    # Load and process data
    raw_text = load_data(input_file)
    cleaned_text = clean_text(raw_text)
    tokens = tokenize_text(cleaned_text)

    # Save cleaned text
    save_cleaned_data(tokens, output_file)
    
    print(f"✅ Preprocessing complete! Cleaned data saved to: {output_file}")
    print(f"📝 Total tokens (words): {len(tokens)}")
