import re
import nltk
from nltk.tokenize import sent_tokenize


nltk.download('punkt')

def clean_text(text):
    """
    Clean input text by removing non-ASCII characters and excess whitespace.
    """
    text = re.sub(r'\s+', ' ', text)  
    text = re.sub(r'[^\x00-\x7F]+', '', text) 
    return text.strip()

def chunk_text(text, max_words=500):
    """
    Breaks cleaned text into chunks of maximum `max_words` length while preserving sentence boundaries.
    """
    text = clean_text(text)
    sentences = sent_tokenize(text)
    chunks = []
    current_chunk = []
    current_length = 0

    for sentence in sentences:
        words = sentence.split()
        if current_length + len(words) <= max_words:
            current_chunk.append(sentence)
            current_length += len(words)
        else:
            chunks.append(" ".join(current_chunk))
            current_chunk = [sentence]
            current_length = len(words)

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks
