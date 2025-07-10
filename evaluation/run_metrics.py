import json
import pandas as pd
from evaluation.metrics import compute_metrics, compute_bertscore
from models.summarizers import generate_summary
from utils.preprocessing import chunk_text

with open("data/chunks.json", "r") as f:
    chunks = json.load(f)

models = ["bart", "pegasus", "t5"]
model_labels = {
    "bart": "BART (Fine-tuned)",
    "pegasus": "PEGASUS",
    "t5": "T5"
}
all_results = []

# Summarize and evaluate
for chunk in chunks:
    reference = chunk["text"]
    for model in models:
        try:
            summary = generate_summary(reference, model_name=model)
            scores = compute_metrics(reference, summary)
            all_results.append({
                "Chunk": reference,
                "Model": model_labels[model],
                "Summary": summary,
                **scores
            })
        except Exception as e:
            print(f"Error with model {model}: {e}")

df = pd.DataFrame(all_results)
df.to_csv("results/summary_outputs.csv", index=False)

model_metrics = df.groupby("Model")[["BLEU", "Cosine Similarity"]].mean().reset_index()
model_metrics.to_csv("results/model_metrics.csv", index=False)

print("Summary and metrics generated successfully!")
