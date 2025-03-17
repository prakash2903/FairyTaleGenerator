import re
import string


def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
    return text


def clean_text(text):
    # Removing Gutenberg headers, footers and excessive newlines
    text = re.sub(r'\n+', ' ', text)
    text = re.sub(r'(?s)\*\*\*.*?START OF THIS PROJECT GUTENBERG EBOOK.*?\*\*\*', '', text)
    text = re.sub(r'(?s)\*\*\*.*?END OF THIS PROJECT GUTENBERG EBOOK.*?\*\*\*', '', text)

    # Normalizing text (Lower casing, Removing punctuations, numbers and extra spaces)
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()

    return text


def tokenize_text(text):
    tokens = text.split()
    return tokens


def save_cleaned_data(tokens, file_path):
    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(' '.join(tokens))


if __name__ == "__main__":
    input_file = "data/fairy_tales.txt"
    output_file = "data/cleaned_fairy_tales.txt"

    raw_text = load_data(input_file)
    cleaned_text = clean_text(raw_text)
    tokens = tokenize_text(cleaned_text)

    save_cleaned_data(tokens, output_file)
    
    print(f"Preprocessing complete! Cleaned data saved to: {output_file}")
    print(f"Total tokens (words): {len(tokens)}")
