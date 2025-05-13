import json
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)
pd.set_option("display.float_format", lambda x: "%.4f" % x)

with open("/home/e4user/r2r-eval/results/last/evaluation_results_1024.json", "r") as f:
    base_results = json.load(f)

with open(
    "/home/e4user/r2r-eval/results/last/evaluation_results_enriched_1024.json", "r"
) as f:
    enriched_results = json.load(f)


def calculate_f1(precision, recall):
    if precision + recall == 0:
        return 0
    return 2 * (precision * recall) / (precision + recall)


data = []
for chunker in base_results.keys():
    for retriever in base_results[chunker].keys():

        base = base_results[chunker][retriever]
        base_f1 = calculate_f1(base["context_precision"], base["context_recall"])

        enriched = enriched_results[chunker].get(retriever, {})
        if enriched:
            enriched_f1 = calculate_f1(
                enriched["context_precision"], enriched["context_recall"]
            )
        else:
            enriched_f1 = None

        # Calculate differences
        precision_diff = (
            enriched.get("context_precision", 0) - base["context_precision"]
        )
        recall_diff = enriched.get("context_recall", 0) - base["context_recall"]

        data.append(
            {
                "Chunker": chunker,
                "Retriever": retriever,
                "Base F1": base_f1,
                "Base Precision": base["context_precision"],
                "Base Recall": base["context_recall"],
                "Base Faithfulness": base["faithfulness"],
                "Enriched F1": enriched_f1,
                "Enriched Precision": enriched.get("context_precision", None),
                "Enriched Recall": enriched.get("context_recall", None),
                "Enriched Faithfulness": enriched.get("faithfulness", None),
                "Precision Difference": precision_diff,
                "Recall Difference": recall_diff,
                "F1 Difference": (
                    enriched_f1 - base_f1 if enriched_f1 is not None else None
                ),
            }
        )

df = pd.DataFrame(data)


def add_arrow(value):
    if value > 0:
        return f"↑ {value:.4f}"
    else:
        return f"↓ {value:.4f}"


diff_df = df.copy()
diff_df["Precision Change"] = diff_df["Precision Difference"].apply(add_arrow)
diff_df["Recall Change"] = diff_df["Recall Difference"].apply(add_arrow)

fig_table = go.Figure(
    data=[
        go.Table(
            header=dict(
                values=[
                    "Chunker",
                    "Retriever",
                    "Base Precision",
                    "Enriched Precision",
                    "Precision Change",
                    "Base Recall",
                    "Enriched Recall",
                    "Recall Change",
                ],
                fill_color="paleturquoise",
                align="left",
            ),
            cells=dict(
                values=[
                    diff_df["Chunker"],
                    diff_df["Retriever"],
                    diff_df["Base Precision"].round(4),
                    diff_df["Enriched Precision"].round(4),
                    diff_df["Precision Change"],
                    diff_df["Base Recall"].round(4),
                    diff_df["Enriched Recall"].round(4),
                    diff_df["Recall Change"],
                ],
                fill_color="lavender",
                align="left",
            ),
        )
    ]
)
fig_table.update_layout(title="Precision and Recall Changes")
fig_table.show()

print("\n=== Average Changes by Retriever ===")
retriever_changes = (
    df.groupby("Retriever")
    .agg(
        {
            "Precision Difference": ["mean", "min", "max"],
            "Recall Difference": ["mean", "min", "max"],
        }
    )
    .round(4)
)

print("\nPrecision Changes:")
print(retriever_changes["Precision Difference"])
print("\nRecall Changes:")
print(retriever_changes["Recall Difference"])

print("\n=== Average Changes by Chunker ===")
chunker_changes = (
    df.groupby("Chunker")
    .agg(
        {
            "Precision Difference": ["mean", "min", "max"],
            "Recall Difference": ["mean", "min", "max"],
        }
    )
    .round(4)
)

print("\nPrecision Changes:")
print(chunker_changes["Precision Difference"])
print("\nRecall Changes:")
print(chunker_changes["Recall Difference"])

print("\n=== Notable Changes ===")
best_precision = df.loc[df["Precision Difference"].idxmax()]
worst_precision = df.loc[df["Precision Difference"].idxmin()]
best_recall = df.loc[df["Recall Difference"].idxmax()]
worst_recall = df.loc[df["Recall Difference"].idxmin()]

print("\nBest Precision Improvement:")
print(f"Chunker: {best_precision['Chunker']}")
print(f"Retriever: {best_precision['Retriever']}")
print(f"Change: {best_precision['Precision Difference']:.4f}")

print("\nWorst Precision Change:")
print(f"Chunker: {worst_precision['Chunker']}")
print(f"Retriever: {worst_precision['Retriever']}")
print(f"Change: {worst_precision['Precision Difference']:.4f}")

print("\nBest Recall Improvement:")
print(f"Chunker: {best_recall['Chunker']}")
print(f"Retriever: {best_recall['Retriever']}")
print(f"Change: {best_recall['Recall Difference']:.4f}")

print("\nWorst Recall Change:")
print(f"Chunker: {worst_recall['Chunker']}")
print(f"Retriever: {worst_recall['Retriever']}")
print(f"Change: {worst_recall['Recall Difference']:.4f}")
