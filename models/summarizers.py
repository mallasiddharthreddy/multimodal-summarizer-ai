# summarizers.py
from transformers import pipeline
import torch

# Set device based on GPU availability
device = 0 if torch.cuda.is_available() else -1

# Load all summarization models
bart_summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=device)
pegasus_summarizer = pipeline("summarization", model="google/pegasus-xsum", device=device)
t5_summarizer = pipeline("summarization", model="t5-base", device=device)

def generate_summary(text, model_name="bart", min_length=15, max_length=60):
    if model_name.lower() == "bart":
        return bart_summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)[0]['summary_text']
    elif model_name.lower() == "pegasus":
        return pegasus_summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)[0]['summary_text']
    elif model_name.lower() == "t5":
        return t5_summarizer("summarize: " + text, max_length=max_length, min_length=min_length, do_sample=False)[0]['summary_text']
    else:
        raise ValueError("Model name must be one of: 'bart', 'pegasus', or 't5'")

def load_summarizers():
    return {
        "bart": bart_summarizer,
        "pegasus": pegasus_summarizer,
        "t5": t5_summarizer
    }
