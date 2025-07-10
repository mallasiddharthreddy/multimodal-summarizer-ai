
# Multimodal Summarizer AI

Audio, Text, and Document Summarization with Transformer Models + Evaluation UI


# Overview

This project enables users to summarize audio files, raw text, and .docx documents using state-of-the-art transformer models (BART (Fine-tuned), PEGASUS, T5). The Streamlit interface supports summary generation, model comparison, sentiment analysis, and evaluation metric visualization. A downloadable PDF report is also generated.


# Key Features

✅ Supports Audio, Text, and Document inputs

✅ Uses Whisper for audio transcription

✅ Offers BART (Fine-tuned), PEGASUS, and T5 summarizers

✅ Generates summary reports as downloadable PDFs

✅ Computes Cosine Similarity and BERTScore

✅ Includes sentiment analysis of input

✅ Supports side-by-side model comparison and metric visualization



# Models Used

| Model             | Task Type    | Source                              |
| ----------------- | ------------ | ----------------------------------- |
| BART (Fine-tuned) | Abstractive  | `models/bart-meetingbank-finetuned` |
| PEGASUS           | Abstractive  | `google/pegasus-xsum`               |
| T5                | Abstractive  | `t5-small`                          |
| Whisper           | Audio → Text | `openai/whisper`                    |




# App Interface

### Home – Upload Options
![Home Interface](images/ui_streamlit.png)

### Uploading an Audio File
![Uploading Audio File](images/uploaded_audio.png)

### Transcription and Sentiment Analysis
![Transcription + Sentiment](images/transcription_sentiment.png)

### Model Comparison and Metric Visualization (clicking on compare all models toggle from above)
![Comparison & Chart](images/model_comparison_chart.png)

### Choosing a Summarization Model based on metrics above
![Choose Model Dropdown](images/choose_model_dropdown.png)

### Generated Summary, Metrics, and PDF Export
![Summary + Metrics + PDF](images/summary_metrics_pdflink.png)


# Sample Summary Report
  
View a full example of the summary and evaluation for an uploaded audio file using the BART model:  
👉 [Download Summary Report (PDF)](reports/summary_report-1.pdf)


# How It Works

1) Input: User uploads audio (.wav, .mp3), text, or document (.docx)

2) Processing:

- Audio is transcribed using Whisper

- Text is split into chunks (if lengthy)

3) Summarization: Selected model generates summaries

4) Evaluation: Two key metrics are computed:

- Cosine Similarity (TF-IDF)

- BERTScore (F1)

5) Sentiment Analysis: Performed using TextBlob

6) Export: PDF report is generated for download



## Evaluation & Benchmarking

Metrics auto-generated in:

```bash
  results/summary_outputs.csv
  results/model_metrics.csv
```
For graphing:

```bash
  python3 -m evaluation.visualizations
```


## Project Structure


```bash

multimodal-summarizer-ai/
├── app/
│   └── summary_app.py
├── core/
│   └── nlp_tasks.py
├── data/                #data/ not included in repo for efficient cloning
│   └── chunks.json
├── evaluation/
│   ├── metrics.py
│   ├── run_metrics.py
│   └── visualizations.py
├── models/
│   ├── summarizers.py
│   └── bart-meetingbank-finetuned/
│       ├── config.json
│       ├── generation_config.json
│       ├── merges.txt
│       ├── model.safetensors
│       ├── special_tokens_map.json
│       ├── tokenizer_config.json
│       └── vocab.json
├── results/             #results/ not included in repo(will be auto-generated depending on data/ used)
│   ├── summary_outputs.csv
│   └── model_metrics.csv
├── utils/
│   ├── preprocessing.py
│   └── audio_handler.py
├── reports/
│   └── sample_summary_report.pdf
├── images/
│   └── .png (screenshots)
├── requirements.txt
├── README.md
└── .gitignore

```

The fine-tuned BART model (~500MB) can be downloaded from [Google Drive](https://drive.google.com/drive/folders/1xgx6-y8fumi1zS4DdcBTkkb-RUrLlG6c?usp=drive_link).  
Place it inside: `models/bart-meetingbank-finetuned/`



## Setup & Installation

✅ Clone Repo and Create Virtual Environment

```bash
git clone https://github.com/your-username/ multimodal-summarizer-ai.git
cd multimodal-summarizer-ai
python3 -m venv .venv
source .venv/bin/activate

```
✅ Install Requirements

```bash
  pip install -r requirements.txt
```

✅ Run the App

```bash
  streamlit run app/summary_app.py

```

# Future Work (Planned for Project + Research Phase)

2) Training custom summarizers for specialized domains

3) Adding Named Entity Recognition, keyword extraction, and more advanced analytics

# Acknowledgements

1) https://huggingface.co/docs/transformers/index

2) https://github.com/openai/whisper

3) https://github.com/Tiiiger/bert_score

4) https://github.com/google-research/google-research/tree/master/rouge

5) https://streamlit.io/


# License

This project is licensed under the MIT License.




