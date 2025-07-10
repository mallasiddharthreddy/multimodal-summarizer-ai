
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import bert_score

smoothie = SmoothingFunction().method4

def compute_metrics(reference, hypothesis):
    
    metrics = {}

    # BLEU Score
    metrics["BLEU"] = round(sentence_bleu([reference.split()], hypothesis.split(), smoothing_function=smoothie), 4)

    # TF-IDF Cosine Similarity
    tfidf = TfidfVectorizer().fit_transform([reference, hypothesis])
    metrics["Cosine Similarity"] = round(cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0], 4)

    return metrics

def compute_bertscore(summary_list, reference_list):
    
    _, _, f1 = bert_score.score(summary_list, reference_list, lang='en', verbose=False)
    return f1.numpy().round(4).tolist()
