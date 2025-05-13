import plotly.graph_objects as go
import json
import numpy as np
from plotly.subplots import make_subplots
import plotly.express as px
import pandas as pd

retrieval_approaches = {
    "semantic_search": "evaluation_results_semantic_search.json",
    "rag_fusion": "evaluation_results_rag_fusion.json",
    "hybrid_search": "evaluation_results_hybrid_search.json",
    "hyde": "evaluation_results_hyde.json",
}

data = {}
for approach, filename in retrieval_approaches.items():
    with open(f"/home/e4user/r2r-eval/results/gemma4b/{filename}", "r") as f:
        data[approach] = json.load(f)

# Get list of strategies and metrics (should be the same across all files)
strategies = list(data["semantic_search"].keys())
metrics = list(data["semantic_search"]["character"].keys())

# Create a dataframe for easier plotting with plotly express
df_rows = []
for approach, results in data.items():
    for strategy in strategies:
        for metric, value in results[strategy].items():
            df_rows.append(
                {
                    "approach": approach.replace("_", " ").title(),
                    "strategy": (
                        "SDPM"
                        if strategy == "Semantic_Double_Pass_Merging"
                        else strategy
                    ),
                    "metric": metric,
                    "value": value,
                }
            )
df = pd.DataFrame(df_rows)

# Compute F1 scores for all strategies
print("\nF1 Scores for All Strategies:")
f1_scores = {}

for strategy in strategies:
    print(f"\n{strategy} Strategy:")
    f1_scores[strategy] = {}

    for approach, results in data.items():
        precision = results[strategy]["context_precision"]
        recall = results[strategy]["context_recall"]
        # Calculate F1 score: 2 * (precision * recall) / (precision + recall)
        if precision + recall > 0:  # Avoid division by zero
            f1 = 2 * (precision * recall) / (precision + recall)
        else:
            f1 = 0
        f1_scores[strategy][approach] = f1
        print(f"{approach.replace('_', ' ').title()}: {f1:.4f}")

# Create bar charts to visualize F1 scores for each strategy
for strategy in strategies:
    fig = go.Figure()

    approaches = list(f1_scores[strategy].keys())
    scores = list(f1_scores[strategy].values())

    fig.add_trace(
        go.Bar(
            x=[approach.replace("_", " ").title() for approach in approaches],
            y=scores,
            text=[f"{score:.4f}" for score in scores],
            textposition="outside",
        )
    )

    # Find the best approach for this strategy
    best_approach = max(approaches, key=lambda a: f1_scores[strategy][a])

    strategy_display = (
        "SDPM" if strategy == "Semantic_Double_Pass_Merging" else strategy
    )

    fig.update_layout(
        title=f"F1 Scores for {strategy_display} Strategy (Precision-Recall Balance)",
        xaxis_title="Retrieval Approach",
        yaxis_title="F1 Score",
        yaxis=dict(range=[0, 1]),
        width=800,
        height=600,  # Increased height to accommodate annotation
        margin=dict(b=120),  # Increased bottom margin for annotation
        showlegend=False,
        annotations=[
            dict(
                x=0.5,
                y=-0.2,  # Increased distance from bottom of plot
                xref="paper",
                yref="paper",
                text=f"Best approach: {best_approach.replace('_', ' ').title()} (F1={f1_scores[strategy][best_approach]:.4f})",
                showarrow=False,
                font=dict(size=12, color="green"),
            )
        ],
    )

    fig.show()
    fig = go.Figure()

    for approach, results in data.items():
        display_name = approach.replace("_", " ").title()
        values = list(results[strategy].values())
        labels = list(results[strategy].keys())

        fig.add_trace(
            go.Scatterpolar(
                r=values,
                theta=labels,
                fill="toself",
                name=display_name,
                text=[f"{v:.4f}" for v in values],
                hovertemplate="%{theta}: %{text}<br>%{r:.4f}",
            )
        )

    # Add annotations for strategy-specific insights
    strategy_insights = {
        "character": "High precision with Semantic Search",
        "agentic": "Good precision-recall balance",
        "semantic": "Best overall performance",
        "Semantic_Double_Pass_Merging": "Excellent faithfulness across all approaches",
        "recursive": "Lower recall compared to other strategies",
    }

    insight = strategy_insights.get(strategy, "")

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1],
                showticklabels=True,
                tickvals=[0.2, 0.4, 0.6, 0.8, 1.0],
            )
        ),
        showlegend=True,
        title=f"Retrieval Approach Comparison - {strategy} strategy<br><sub>{insight}</sub>",
        width=800,
        annotations=[
            dict(
                x=0.5,
                y=-0.1,
                xref="paper",
                yref="paper",
                text=f"Best approach: {max([(approach, np.mean(list(results[strategy].values()))) for approach, results in data.items()], key=lambda x: x[1])[0].replace('_', ' ').title()}",
                showarrow=False,
                font=dict(size=12),
            )
        ],
    )

    fig.show()

# 2. Create bar charts for each strategy (comparing retrieval approaches by metrics)
for strategy in strategies:
    fig = go.Figure()

    for approach, results in data.items():
        display_name = approach.replace("_", " ").title()
        values = [results[strategy][metric] for metric in metrics]

        # Add text labels with values
        fig.add_trace(
            go.Bar(
                x=metrics,
                y=values,
                name=display_name,
                text=[f"{val:.4f}" for val in values],
                textposition="outside",
            )
        )

    # Determine best approach for each metric in this strategy
    best_approaches = {}
    for metric in metrics:
        best_approach = max(data.items(), key=lambda x: x[1][strategy][metric])[0]
        best_approaches[metric] = best_approach.replace("_", " ").title()

    # Add annotations highlighting best approaches
    annotations = []
    for i, metric in enumerate(metrics):
        annotations.append(
            dict(
                x=metric,
                y=1.05,
                xref="x",
                yref="paper",
                text=f"Best: {best_approaches[metric]}",
                showarrow=False,
                font=dict(size=10),
            )
        )

    fig.update_layout(
        title=f"Retrieval Approach Comparison - {strategy} strategy",
        xaxis_title="Metric",
        yaxis_title="Score",
        yaxis=dict(range=[0, 1.1]),
        barmode="group",
        width=900,
        height=600,
        showlegend=True,
        annotations=annotations,
    )

    fig.show()

# 3. Create mean score comparison across all retrieval approaches for each strategy
mean_scores = {}
for strategy in strategies:
    mean_scores[strategy] = {}
    for approach, results in data.items():
        display_name = approach.replace("_", " ").title()
        mean_scores[strategy][display_name] = np.mean(list(results[strategy].values()))

for strategy in strategies:
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=list(mean_scores[strategy].keys()),
            y=list(mean_scores[strategy].values()),
            text=[f"{score:.4f}" for score in mean_scores[strategy].values()],
            textposition="outside",
        )
    )

    fig.update_layout(
        title=f"Mean Scores Across Retrieval Approaches - {strategy} strategy",
        xaxis_title="Retrieval Approach",
        yaxis_title="Mean Score",
        yaxis=dict(range=[0, 1]),
        width=800,
        height=500,
        showlegend=False,
    )

    fig.show()

# 4. Create a comparison of individual metrics across retrieval approaches and strategies
for metric in metrics:
    fig = make_subplots(rows=1, cols=1, subplot_titles=[f"{metric} Comparison"])

    # Calculate average values across strategies for this metric
    avg_values = {}
    for approach in retrieval_approaches.keys():
        avg_values[approach] = np.mean(
            [data[approach][strategy][metric] for strategy in strategies]
        )

    # Find best and worst approaches for this metric
    best_approach = max(avg_values.items(), key=lambda x: x[1])[0]
    worst_approach = min(avg_values.items(), key=lambda x: x[1])[0]

    # Find the best strategy for this metric
    best_strategy_by_approach = {}
    for approach in retrieval_approaches.keys():
        best_strategy = max(strategies, key=lambda s: data[approach][s][metric])
        best_strategy_by_approach[approach] = (
            "SDPM" if best_strategy == "Semantic_Double_Pass_Merging" else best_strategy
        )

    for strategy in strategies:
        values = [
            data[approach][strategy][metric] for approach in retrieval_approaches.keys()
        ]
        display_names = [
            approach.replace("_", " ").title()
            for approach in retrieval_approaches.keys()
        ]

        # Some strategies have long names, use abbreviation for display
        strategy_display = (
            "SDPM" if strategy == "Semantic_Double_Pass_Merging" else strategy
        )

        fig.add_trace(
            go.Bar(
                x=display_names,
                y=values,
                name=strategy_display,
                text=[f"{v:.4f}" for v in values],
                textposition="outside",
            ),
            row=1,
            col=1,
        )

    # Add metric descriptions and insights
    metric_insights = {
        "faithfulness": "How well responses are grounded in the retrieved context",
        "answer_relevancy": "How relevant responses are to the questions",
        "context_precision": "How much of the retrieved context is relevant",
        "context_recall": "How much relevant information was retrieved",
    }

    insight = metric_insights.get(metric, "")

    annotations = [
        dict(
            x=0.5,
            y=1.15,
            xref="paper",
            yref="paper",
            text=insight,
            showarrow=False,
            font=dict(size=12),
        ),
        dict(
            x=0.5,
            y=-0.15,
            xref="paper",
            yref="paper",
            text=f"Best approach: {best_approach.replace('_', ' ').title()} (average: {avg_values[best_approach]:.4f})",
            showarrow=False,
            font=dict(size=12, color="green"),
        ),
        dict(
            x=0.5,
            y=-0.20,
            xref="paper",
            yref="paper",
            text=f"Optimal strategy for {best_approach.replace('_', ' ').title()}: {best_strategy_by_approach[best_approach]}",
            showarrow=False,
            font=dict(size=12, color="green"),
        ),
    ]

    fig.update_layout(
        title=f"{metric} Comparison Across Retrieval Approaches",
        xaxis_title="Retrieval Approach",
        yaxis_title="Score",
        yaxis=dict(range=[0, 1.1]),
        barmode="group",
        width=1000,
        height=700,
        showlegend=True,
        annotations=annotations,
    )

    fig.show()

# 5. Create heatmaps for each metric
for metric in metrics:
    # Prepare data for heatmap
    heatmap_data = []
    for strategy in strategies:
        strategy_display = (
            "SDPM" if strategy == "Semantic_Double_Pass_Merging" else strategy
        )
        row = []
        for approach in retrieval_approaches.keys():
            row.append(data[approach][strategy][metric])
        heatmap_data.append(row)

    # Identify max and min values for annotation
    max_val = max([max(row) for row in heatmap_data])
    min_val = min([min(row) for row in heatmap_data])
    max_pos = None
    min_pos = None
    for i, row in enumerate(heatmap_data):
        for j, val in enumerate(row):
            if val == max_val:
                max_pos = (i, j)
            if val == min_val:
                min_pos = (i, j)

    # Create heatmap
    fig = go.Figure(
        data=go.Heatmap(
            z=heatmap_data,
            x=[
                approach.replace("_", " ").title()
                for approach in retrieval_approaches.keys()
            ],
            y=[
                "SDPM" if s == "Semantic_Double_Pass_Merging" else s for s in strategies
            ],
            colorscale="Viridis",
            text=[[f"{val:.4f}" for val in row] for row in heatmap_data],
            texttemplate="%{text}",
            colorbar=dict(title=metric),
        )
    )

    # Add annotations for insights based on metric
    metric_insights = {
        "faithfulness": "Shows how well responses are grounded in the retrieved context",
        "answer_relevancy": "Indicates how relevant responses are to the questions",
        "context_precision": "Measures how much of the retrieved context is relevant",
        "context_recall": "Shows how much relevant information was retrieved",
    }

    insight = metric_insights.get(metric, "")

    annotations = [
        dict(
            x=0.5,
            y=-0.15,
            xref="paper",
            yref="paper",
            text=insight,
            showarrow=False,
            font=dict(size=12),
        )
    ]

    # Add annotation for max value
    if max_pos:
        max_strategy = (
            "SDPM"
            if strategies[max_pos[0]] == "Semantic_Double_Pass_Merging"
            else strategies[max_pos[0]]
        )
        max_approach = (
            list(retrieval_approaches.keys())[max_pos[1]].replace("_", " ").title()
        )
        annotations.append(
            dict(
                x=max_approach,
                y=max_strategy,
                text="MAX",
                showarrow=True,
                arrowhead=1,
                font=dict(color="white", size=10),
                bgcolor="black",
                bordercolor="black",
            )
        )

    fig.update_layout(
        title=f"Heatmap of {metric} Across Strategies and Approaches",
        xaxis_title="Retrieval Approach",
        yaxis_title="Strategy",
        width=900,
        height=650,
        annotations=annotations,
    )

    fig.show()

# 6. Create precision-recall curves
# For each strategy, plot precision vs recall for all retrieval approaches
for strategy in strategies:
    fig = go.Figure()

    # Calculate average precision and recall across all approaches for this strategy
    avg_precision = np.mean(
        [
            data[approach][strategy]["context_precision"]
            for approach in retrieval_approaches.keys()
        ]
    )
    avg_recall = np.mean(
        [
            data[approach][strategy]["context_recall"]
            for approach in retrieval_approaches.keys()
        ]
    )

    for approach in retrieval_approaches.keys():
        display_name = approach.replace("_", " ").title()

        # Get precision and recall values for this approach and strategy
        precision = data[approach][strategy]["context_precision"]
        recall = data[approach][strategy]["context_recall"]

        # Add a scatter point for this approach
        fig.add_trace(
            go.Scatter(
                x=[recall],
                y=[precision],
                mode="markers+text",
                marker=dict(size=15),
                text=[display_name],
                textposition="top center",
                name=f"{display_name} ({precision:.4f}, {recall:.4f})",
            )
        )

    # Add a line connecting the origin to (1,1) for reference
    x_ref = np.linspace(0, 1, 100)
    y_ref = x_ref
    fig.add_trace(
        go.Scatter(
            x=x_ref,
            y=y_ref,
            mode="lines",
            line=dict(dash="dash", color="gray"),
            name="Random Performance",
            showlegend=False,
        )
    )

    # Add annotation for insight
    annotations = [
        dict(
            x=0.5,
            y=0.03,
            xref="paper",
            yref="paper",
            text=f"Average: Precision={avg_precision:.4f}, Recall={avg_recall:.4f}",
            showarrow=False,
            font=dict(size=12),
        )
    ]

    # Identify best approach for this strategy based on F1 score (harmonic mean of precision and recall)
    best_approach = max(
        retrieval_approaches.keys(),
        key=lambda a: 2
        * data[a][strategy]["context_precision"]
        * data[a][strategy]["context_recall"]
        / (
            data[a][strategy]["context_precision"] + data[a][strategy]["context_recall"]
        ),
    )

    annotations.append(
        dict(
            x=0.5,
            y=0.0,
            xref="paper",
            yref="paper",
            text=f"Best precision-recall balance: {best_approach.replace('_', ' ').title()}",
            showarrow=False,
            font=dict(size=12, color="green"),
        )
    )

    fig.update_layout(
        title=f"Precision-Recall Visualization for {strategy} Strategy",
        xaxis_title="Recall",
        yaxis_title="Precision",
        xaxis=dict(range=[0, 1]),
        yaxis=dict(range=[0, 1]),
        width=900,
        height=650,
        showlegend=True,
        annotations=annotations,
    )

    fig.show()

# 12. Create combined precision-recall plot with multiple strategies
# For each retrieval approach, plot precision vs recall for all strategies
for approach in retrieval_approaches.keys():
    display_name = approach.replace("_", " ").title()
    fig = go.Figure()

    for strategy in strategies:
        strategy_display = (
            "SDPM" if strategy == "Semantic_Double_Pass_Merging" else strategy
        )

        # Get precision and recall values for this approach and strategy
        precision = data[approach][strategy]["context_precision"]
        recall = data[approach][strategy]["context_recall"]

        # Add a scatter point for this strategy
        fig.add_trace(
            go.Scatter(
                x=[recall],
                y=[precision],
                mode="markers+text",
                marker=dict(size=15),
                text=[strategy_display],
                textposition="top center",
                name=strategy_display,
            )
        )

    # Add a line connecting the origin to (1,1) for reference
    x_ref = np.linspace(0, 1, 100)
    y_ref = x_ref
    fig.add_trace(
        go.Scatter(
            x=x_ref,
            y=y_ref,
            mode="lines",
            line=dict(dash="dash", color="gray"),
            name="Random Performance",
            showlegend=False,
        )
    )

    fig.update_layout(
        title=f"Precision-Recall Visualization for {display_name} Approach",
        xaxis_title="Recall",
        yaxis_title="Precision",
        xaxis=dict(range=[0, 1]),
        yaxis=dict(range=[0, 1]),
        width=800,
        height=600,
        showlegend=True,
    )

    fig.show()

# 13. Create a comprehensive precision-recall plot with all data points
fig = go.Figure()

# Calculate performance quartiles
all_precisions = []
all_recalls = []
for approach in retrieval_approaches.keys():
    for strategy in strategies:
        all_precisions.append(data[approach][strategy]["context_precision"])
        all_recalls.append(data[approach][strategy]["context_recall"])

median_precision = np.median(all_precisions)
median_recall = np.median(all_recalls)

# Draw reference lines for median values
fig.add_shape(
    type="line",
    x0=median_recall,
    y0=0,
    x1=median_recall,
    y1=1,
    line=dict(color="gray", dash="dash"),
    name="Median Recall",
)

fig.add_shape(
    type="line",
    x0=0,
    y0=median_precision,
    x1=1,
    y1=median_precision,
    line=dict(color="gray", dash="dash"),
    name="Median Precision",
)

for approach in retrieval_approaches.keys():
    approach_display = approach.replace("_", " ").title()

    # Collect precision and recall for all strategies for this approach
    recalls = []
    precisions = []
    strategy_names = []

    for strategy in strategies:
        strategy_display = (
            "SDPM" if strategy == "Semantic_Double_Pass_Merging" else strategy
        )
        recalls.append(data[approach][strategy]["context_recall"])
        precisions.append(data[approach][strategy]["context_precision"])
        strategy_names.append(strategy_display)

    # Calculate average for this approach
    avg_precision = np.mean(precisions)
    avg_recall = np.mean(recalls)

    # Add a scatter trace for this approach with hover text showing strategy
    fig.add_trace(
        go.Scatter(
            x=recalls,
            y=precisions,
            mode="markers",
            marker=dict(size=12),
            name=f"{approach_display} (Avg P={avg_precision:.3f}, R={avg_recall:.3f})",
            text=strategy_names,
            hovertemplate="<b>%{text}</b><br>"
            + "Recall: %{x:.4f}<br>"
            + "Precision: %{y:.4f}<br>",
        )
    )

# Add a line connecting the origin to (1,1) for reference
x_ref = np.linspace(0, 1, 100)
y_ref = x_ref
fig.add_trace(
    go.Scatter(
        x=x_ref,
        y=y_ref,
        mode="lines",
        line=dict(dash="dash", color="gray"),
        name="Random Performance",
        showlegend=False,
    )
)

# Add annotations for quadrants
annotations = [
    dict(
        x=0.25,
        y=0.85,
        xref="paper",
        yref="paper",
        text="High Precision<br>Low Recall",
        showarrow=False,
        font=dict(size=10),
    ),
    dict(
        x=0.75,
        y=0.85,
        xref="paper",
        yref="paper",
        text="High Precision<br>High Recall<br>(Optimal)",
        showarrow=False,
        font=dict(size=10, color="green"),
    ),
    dict(
        x=0.25,
        y=0.15,
        xref="paper",
        yref="paper",
        text="Low Precision<br>Low Recall<br>(Suboptimal)",
        showarrow=False,
        font=dict(size=10, color="red"),
    ),
    dict(
        x=0.75,
        y=0.15,
        xref="paper",
        yref="paper",
        text="Low Precision<br>High Recall",
        showarrow=False,
        font=dict(size=10),
    ),
    dict(
        x=0.5,
        y=-0.1,
        xref="paper",
        yref="paper",
        text="Semantic Search shows the best balance between precision and recall",
        showarrow=False,
        font=dict(size=12, color="black"),
    ),
]

fig.update_layout(
    title="Comprehensive Precision-Recall Visualization",
    xaxis_title="Recall",
    yaxis_title="Precision",
    xaxis=dict(range=[0, 1]),
    yaxis=dict(range=[0, 1]),
    width=1000,
    height=800,
    showlegend=True,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    annotations=annotations,
)

fig.show()
