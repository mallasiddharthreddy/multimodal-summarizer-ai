# evaluation/metrics.py

from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import bert_score

# Initialize ROUGE scorer and BLEU smoothing
scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
smoothie = SmoothingFunction().method4

def compute_metrics(reference, hypothesis):
    """
    Compute ROUGE-1, ROUGE-L, BLEU, and Cosine Similarity between reference and hypothesis texts.
    """
    metrics = {}

    # ROUGE Scores
    rouge_scores = scorer.score(reference, hypothesis)
    metrics["ROUGE-1"] = round(rouge_scores['rouge1'].fmeasure, 4)
    metrics["ROUGE-L"] = round(rouge_scores['rougeL'].fmeasure, 4)

    # BLEU Score
    metrics["BLEU"] = round(sentence_bleu([reference.split()], hypothesis.split(), smoothing_function=smoothie), 4)

    # TF-IDF Cosine Similarity
    tfidf = TfidfVectorizer().fit_transform([reference, hypothesis])
    metrics["Cosine Similarity"] = round(cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0], 4)

    return metrics

def compute_bertscore(summary_list, reference_list):
    """
    Compute BERTScore (F1) for a list of summaries vs. references.
    Returns a list of scores (1 per pair).
    """
    _, _, f1 = bert_score.score(summary_list, reference_list, lang='en', verbose=False)
    return f1.numpy().round(4).tolist()
