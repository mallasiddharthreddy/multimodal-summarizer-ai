# nlp_tasks.py
from transformers import pipeline
from nltk.tokenize import word_tokenize
import nltk
from summa import keywords as summa_keywords

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

# Initialize Hugging Face pipelines
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
ner_pipeline = pipeline("ner", model="dslim/bert-base-NER", grouped_entities=True)
sentiment_pipeline = pipeline("sentiment-analysis")

topics = ["governance", "legal", "environment", "infrastructure", "finance", "education", "transparency", "agenda"]

# Generate abstractive summary
def generate_abstractive_summary(text):
    return summarizer(text, max_length=60, min_length=15, do_sample=False)[0]['summary_text']

# Generate extractive keywords using TextRank
def generate_extractive_keywords(text):
    return summa_keywords.keywords(text, ratio=0.4).replace('\n', '. ')

# Classify topic of text
def classify_topic(text):
    result = classifier(text, candidate_labels=topics)
    return result['labels'][0], round(result['scores'][0], 3)

# Named Entity Recognition
def extract_named_entities(text):
    ner = ner_pipeline(text)
    return ", ".join(set([ent["word"] for ent in ner])) if ner else "None"

# Sentiment Analysis
def analyze_sentiment(text):
    result = sentiment_pipeline(text)[0]
    return result['label'], round(result['score'], 3)

# Keyword Extraction using TF-IDF
from sklearn.feature_extraction.text import TfidfVectorizer

def extract_keywords_tfidf(text, top_n=5):
    tokens = word_tokenize(text)
    words = list(set([w.lower() for w in tokens if w.isalpha()]))
    tfidf = TfidfVectorizer()
    tfidf_matrix = tfidf.fit_transform([" ".join(words)])
    scores = zip(tfidf.get_feature_names_out(), tfidf.idf_)
    sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)
    return ", ".join([w for w, s in sorted_scores[:top_n]])