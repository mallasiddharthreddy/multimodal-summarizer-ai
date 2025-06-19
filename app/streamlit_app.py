# NOTE: Streamlit and Whisper must be installed in your local environment. Run this app locally with:
# pip install streamlit openai-whisper transformers sklearn rouge-score bert-score matplotlib torch fpdf python-docx textblob

import pandas as pd
import tempfile
import os
import base64
import matplotlib.pyplot as plt
from fpdf import FPDF
from textblob import TextBlob

try:
    import torch
except ModuleNotFoundError:
    raise ImportError("The 'torch' module is not installed. Please run `pip install torch` in your terminal.")

try:
    from transformers import pipeline
except ModuleNotFoundError:
    raise ImportError("The 'transformers' module is not installed. Please run `pip install transformers` in your terminal.")

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
except ModuleNotFoundError:
    raise ImportError("The 'sklearn' module is not installed. Please run `pip install scikit-learn` in your terminal.")

try:
    from rouge_score import rouge_scorer
except ModuleNotFoundError:
    raise ImportError("The 'rouge-score' module is not installed. Please run `pip install rouge-score` in your terminal.")

try:
    from bert_score import score as bertscore
except ModuleNotFoundError:
    raise ImportError("The 'bert-score' module is not installed. Please run `pip install bert-score` in your terminal.")

try:
    import streamlit as st
except ModuleNotFoundError:
    raise ImportError("The 'streamlit' module is not installed. Run `pip install streamlit` in your terminal and run this script locally.")

try:
    import whisper
except ModuleNotFoundError:
    raise ImportError("The 'whisper' module is not installed. Install via `pip install openai-whisper`.")

try:
    import docx
except ModuleNotFoundError:
    raise ImportError("The 'python-docx' module is not installed. Run `pip install python-docx`.")

# Loading summarization models 
st.session_state.setdefault("bart", pipeline("summarization", model="facebook/bart-large-cnn"))
st.session_state.setdefault("pegasus", pipeline("summarization", model="google/pegasus-xsum"))
st.session_state.setdefault("t5", pipeline("summarization", model="t5-small"))


def chunk_text(text, max_tokens=1000):
    words = text.split()
    for i in range(0, len(words), max_tokens):
        yield " ".join(words[i:i + max_tokens])

def get_summary(text, model_choice, max_len):
    summary_chunks = []
    for chunk in chunk_text(text):
        if model_choice == "BART":
            s = st.session_state.bart(chunk, max_length=max_len, min_length=80, do_sample=False)[0]['summary_text']
        elif model_choice == "PEGASUS":
            s = st.session_state.pegasus(chunk, max_length=max_len, min_length=80, do_sample=False)[0]['summary_text']
        else:
            s = st.session_state.t5("summarize: " + chunk, max_length=max_len, min_length=80, do_sample=False)[0]['summary_text']
        summary_chunks.append(s)
    return "\n".join(summary_chunks)

# Evaluation Metrics 
def get_scores(original, summary):
    rouge = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    r_scores = rouge.score(original, summary)

    tfidf = TfidfVectorizer().fit_transform([original, summary])
    cos_sim = cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]

    P, R, F1 = bertscore([summary], [original], lang="en", verbose=False)

    return {
        "ROUGE-1": round(r_scores['rouge1'].fmeasure, 4),
        "ROUGE-L": round(r_scores['rougeL'].fmeasure, 4),
        "Cosine Similarity": round(cos_sim, 4),
        "BERTScore": round(F1[0].item(), 4)
    }

# Sentiment Analysis 
def get_sentiment(text):
    blob = TextBlob(text)
    return blob.sentiment.polarity


st.title("AI Audio/Text Summarizer and Evaluation App")
st.write("Upload audio, document, or text input. Choose a summarization model and export a PDF report with evaluation metrics.")


input_mode = st.radio("Choose input type:", ["Audio", "Text", "Document"])

transcript_text = ""
sentiment_score = None
if input_mode == "Audio":
    file = st.file_uploader("Upload an audio file (.mp3 or .wav)", type=["mp3", "wav"])
    if file and st.button("Transcribe Audio"):
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(file.read())
            tmp_path = tmp_file.name

        st.info("Transcribing audio with Whisper...")
        audio_model = whisper.load_model("medium")
        transcription = audio_model.transcribe(tmp_path)
        transcript_text = transcription["text"]
        st.session_state.transcript_text = transcript_text
        st.session_state.sentiment_score = get_sentiment(transcript_text)

if input_mode == "Text":
    transcript_text = st.text_area("Paste your text here:")
    st.session_state.transcript_text = transcript_text

elif input_mode == "Document":
    doc_file = st.file_uploader("Upload a .docx file", type=["docx"])
    if doc_file is not None:
        doc = docx.Document(doc_file)
        transcript_text = "\n".join([para.text for para in doc.paragraphs])
        st.text_area("Document Content", transcript_text, height=200)
        st.session_state.transcript_text = transcript_text


model_choice = st.selectbox("Choose a summarization model", ["BART", "PEGASUS", "T5"])


max_len = st.slider("Summary length (max tokens)", 80, 400, 150)


compare_models = st.checkbox("Compare all models")


if input_mode == "Audio" and "transcript_text" in st.session_state:
    st.subheader("Transcription")
    st.text_area("Full Transcript", st.session_state.transcript_text, height=200)
    st.subheader("Overall Sentiment")
    score = st.session_state.sentiment_score
    if score > 0.2:
        st.success(f"Positive ({round(score, 2)})")
    elif score < -0.2:
        st.error(f"Negative ({round(score, 2)})")
    else:
        st.warning(f"Neutral ({round(score, 2)})")

#Process Summary 
if "transcript_text" in st.session_state and st.session_state.transcript_text and st.button("Generate Summary"):
    st.info(f"Generating summary using {model_choice} model...")
    summary = get_summary(st.session_state.transcript_text, model_choice, max_len)

    st.subheader("Generated Summary")
    st.text_area("Summary", summary, height=150)

    scores = get_scores(st.session_state.transcript_text, summary)
    st.subheader("Evaluation Metrics")
    st.write(scores)

    # Export as PDF 
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    safe_text = st.session_state.transcript_text.encode("latin-1", "replace").decode("latin-1")
    safe_summary = summary.encode("latin-1", "replace").decode("latin-1")
    safe_scores = str(scores).encode("latin-1", "replace").decode("latin-1")
    pdf.multi_cell(0, 10, f"Transcript:\n{safe_text}\n\nSummary ({model_choice}):\n{safe_summary}\n\nEvaluation Metrics:\n{safe_scores}")
    pdf_path = os.path.join(tempfile.gettempdir(), "summary_report.pdf")
    pdf.output(pdf_path)

    with open(pdf_path, "rb") as f:
        b64_pdf = base64.b64encode(f.read()).decode()
        pdf_link = f'<a href="data:application/pdf;base64,{b64_pdf}" download="summary_report.pdf">📄 Download PDF Summary Report</a>'
        st.markdown(pdf_link, unsafe_allow_html=True)

# Model Comparison
if compare_models and "transcript_text" in st.session_state and st.session_state.transcript_text:
    st.info("Generating summaries for all models...")
    results = {}
    for model in ["BART", "PEGASUS", "T5"]:
        sm = get_summary(st.session_state.transcript_text, model, max_len)
        sc = get_scores(st.session_state.transcript_text, sm)
        results[model] = {"Summary": sm, **sc}

    df = pd.DataFrame(results).T
    st.subheader("Model Comparison")
    st.dataframe(df)

    st.subheader("Metric Comparison Chart")
    chart_data = df.drop(columns="Summary")
    st.bar_chart(chart_data)
