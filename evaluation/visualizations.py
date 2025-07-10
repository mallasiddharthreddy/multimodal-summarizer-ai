import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_bar_comparison(df_scores):
    df_scores.set_index("Chunk")[[
        "BART (Fine-tuned)_SummaryScore",
        "PEGASUS_SummaryScore",
        "T5_SummaryScore"
    ]].plot(kind="bar", figsize=(10, 6))

    plt.title("Model Comparison per Chunk")
    plt.ylabel("Average Summary Score")
    plt.xticks(rotation=0)
    plt.ylim(0, 1.1)
    plt.legend(title="Model")
    plt.grid(axis="y")
    plt.tight_layout()
    plt.show()


def plot_heatmap(df_metrics):
    heatmap_data = df_metrics[[
        "BART (Fine-tuned)_ROUGE-1", "PEGASUS_ROUGE-1", "T5_ROUGE-1",
        "BART (Fine-tuned)_ROUGE-L", "PEGASUS_ROUGE-L", "T5_ROUGE-L",
        "BART (Fine-tuned)_BLEU", "PEGASUS_BLEU", "T5_BLEU",
        "BART (Fine-tuned)_Cosine Similarity", "PEGASUS_Cosine Similarity", "T5_Cosine Similarity",
        "BART (Fine-tuned)_BERTScore", "PEGASUS_BERTScore", "T5_BERTScore"
    ]]

    plt.figure(figsize=(12, 8))
    sns.heatmap(heatmap_data, annot=True, cmap="YlGnBu", linewidths=0.5)
    plt.title("Metric Scores Heatmap Across Models")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()
