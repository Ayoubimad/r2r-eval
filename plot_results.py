import json
import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from tabulate import tabulate

# Configuration
RESULTS_DIR = "/home/e4user/r2r-eval/results/last"
OUTPUT_PRECISION = 3  # Decimal places for F1 scores


def load_evaluation_data():
    """
    Load all evaluation result files from the results directory.
    Returns a DataFrame with all the data.
    """

    # Helper function to extract info from filenames
    def extract_info_from_filename(filename):
        parts = filename.split("_")
        retrieval_approach = "_".join(parts[2:4])
        if retrieval_approach == "semantic_search":
            retrieval_approach = "Semantic Search"
        elif retrieval_approach == "rag_fusion":
            retrieval_approach = "RAG Fusion"
        elif retrieval_approach == "hyde_chunk":
            retrieval_approach = "HyDE"
        else:
            retrieval_approach = "Hybrid Search"
        chunk_size = int(parts[-1].split(".")[0])
        return retrieval_approach, chunk_size

    # Collect data from all files
    all_data = []
    chunk_sizes = []

    for filename in os.listdir(RESULTS_DIR):
        if filename.startswith("evaluation_results_") and filename.endswith(".json"):
            retrieval_approach, chunk_size = extract_info_from_filename(filename)
            if chunk_size not in chunk_sizes:
                chunk_sizes.append(chunk_size)

            with open(os.path.join(RESULTS_DIR, filename), "r") as f:
                file_data = json.load(f)

            for strategy, metrics in file_data.items():
                for metric, value in metrics.items():
                    all_data.append(
                        {
                            "retrieval_approach": retrieval_approach,
                            "chunk_size": chunk_size,
                            "chunk_strategy": (
                                "SDPM"
                                if strategy == "Semantic_Double_Pass_Merging"
                                else strategy
                            ),
                            "metric": metric,
                            "value": value,
                        }
                    )

    return pd.DataFrame(all_data), sorted(chunk_sizes)


def calculate_f1_scores(df):
    """
    Calculate F1 scores for each configuration based on precision and recall.
    Returns a DataFrame with F1 scores.
    """
    summary_data = []

    for approach in df["retrieval_approach"].unique():
        approach_df = df[df["retrieval_approach"] == approach]

        for chunk_size in approach_df["chunk_size"].unique():
            chunk_df = approach_df[approach_df["chunk_size"] == chunk_size]

            for strategy in chunk_df["chunk_strategy"].unique():
                strategy_df = chunk_df[chunk_df["chunk_strategy"] == strategy]

                # Get precision and recall values
                precision = strategy_df[strategy_df["metric"] == "context_precision"][
                    "value"
                ].values[0]
                recall = strategy_df[strategy_df["metric"] == "context_recall"][
                    "value"
                ].values[0]

                # Calculate F1 score
                f1_score = (
                    2 * (precision * recall) / (precision + recall)
                    if (precision + recall) > 0
                    else 0
                )

                # Create readable display name for retrieval approach
                approach_name = approach

                summary_data.append(
                    {
                        "retrieval_approach": approach_name,
                        "chunk_size": chunk_size,
                        "chunk_strategy": strategy,
                        "f1_score": f1_score,
                    }
                )

    return pd.DataFrame(summary_data)


def find_best_configuration(summary_df):
    """
    Find and print the overall best configuration.
    """
    # Sort by F1 score (highest first)
    top_configs = summary_df.sort_values("f1_score", ascending=False)

    # Get the best configuration
    best_config = top_configs.iloc[0]

    print(f"\nBest Configuration:")
    print(f"Strategy: {best_config['chunk_strategy']}")
    print(f"Retrieval Approach: {best_config['retrieval_approach']}")
    print(f"Chunk Size: {best_config['chunk_size']}")
    print(f"F1 Score: {best_config['f1_score']:.{OUTPUT_PRECISION}f}")


def display_tables(summary_df):
    """
    Display all result tables in the terminal.
    """
    # Table 1: For each chunk size, show best chunking strategy and retrieval approach
    print(
        f"\n=== Tabella 1: Per ogni Chunk Size → Migliore Chunking Strategy e Retrieval Approach ==="
    )
    best_by_chunk = []

    for chunk in sorted(summary_df["chunk_size"].unique()):
        chunk_data = summary_df[summary_df["chunk_size"] == chunk]
        best = chunk_data.loc[chunk_data["f1_score"].idxmax()]
        best_by_chunk.append(
            {
                "Chunk Size": chunk,
                "Best Chunking Strategy": best["chunk_strategy"],
                "Best Retrieval Approach": best["retrieval_approach"],
                "F1 Score": f"{best['f1_score']:.{OUTPUT_PRECISION}f}",
            }
        )

    chunk_df = pd.DataFrame(best_by_chunk)
    print(tabulate(chunk_df, headers="keys", tablefmt="grid", showindex=False))

    # Table 2: For each chunking strategy, show best chunk size and retrieval approach
    print(
        f"\n=== Tabella 2: Per ogni Chunking Strategy → Migliore Chunk Size e Retrieval Approach ==="
    )
    best_by_strategy = []

    for strategy in sorted(summary_df["chunk_strategy"].unique()):
        strategy_data = summary_df[summary_df["chunk_strategy"] == strategy]
        best = strategy_data.loc[strategy_data["f1_score"].idxmax()]
        best_by_strategy.append(
            {
                "Chunking Strategy": strategy,
                "Best Chunk Size": best["chunk_size"],
                "Retrieval Approach": best["retrieval_approach"],
                "F1 Score": f"{best['f1_score']:.{OUTPUT_PRECISION}f}",
            }
        )

    strategy_df = pd.DataFrame(best_by_strategy)
    print(tabulate(strategy_df, headers="keys", tablefmt="grid", showindex=False))

    # Table 3: Top 10 configurations by F1 score
    print(f"\n=== Tabella 3: Top 10 Configurazioni per F1 Score ===")
    top_configs = summary_df.sort_values("f1_score", ascending=False).head(10)
    top_configs = top_configs.rename(
        columns={
            "chunk_strategy": "Chunking Strategy",
            "retrieval_approach": "Retrieval Approach",
            "chunk_size": "Chunk Size",
            "f1_score": "F1 Score",
        }
    )
    top_configs["F1 Score"] = top_configs["F1 Score"].apply(
        lambda x: f"{x:.{OUTPUT_PRECISION}f}"
    )
    print(
        tabulate(
            top_configs[
                ["Chunking Strategy", "Retrieval Approach", "Chunk Size", "F1 Score"]
            ],
            headers="keys",
            tablefmt="grid",
            showindex=False,
        )
    )

    # Table 4: For each retrieval approach, show best chunk size and chunking strategy
    print(
        f"\n=== Tabella 4: Per ogni Retrieval Approach → Migliore Chunk Size e Chunking Strategy ===="
    )
    best_by_retrieval = []

    for approach in sorted(summary_df["retrieval_approach"].unique()):
        approach_data = summary_df[summary_df["retrieval_approach"] == approach]
        best = approach_data.loc[approach_data["f1_score"].idxmax()]
        best_by_retrieval.append(
            {
                "Retrieval Approach": approach,
                "Best Chunk Size": best["chunk_size"],
                "Best Chunking Strategy": best["chunk_strategy"],
                "F1 Score": f"{best['f1_score']:.{OUTPUT_PRECISION}f}",
            }
        )

    retrieval_df = pd.DataFrame(best_by_retrieval)
    print(tabulate(retrieval_df, headers="keys", tablefmt="grid", showindex=False))


def create_plotly_tables(summary_df):
    """
    Create interactive Plotly tables for all result tables.
    """
    # Table 1: Best chunking strategy and retrieval approach for each chunk size
    best_by_chunk = []

    for chunk in sorted(summary_df["chunk_size"].unique()):
        chunk_data = summary_df[summary_df["chunk_size"] == chunk]
        best = chunk_data.loc[chunk_data["f1_score"].idxmax()]
        best_by_chunk.append(
            {
                "Chunk Size": chunk,
                "Best Chunking Strategy": best["chunk_strategy"],
                "Best Retrieval Approach": best["retrieval_approach"],
                "F1 Score": round(best["f1_score"], OUTPUT_PRECISION),
            }
        )

    chunk_df = pd.DataFrame(best_by_chunk)

    # Create Plotly table for chunk size
    fig_chunk = go.Figure(
        data=[
            go.Table(
                header=dict(
                    values=list(chunk_df.columns),
                    fill_color="paleturquoise",
                    align="left",
                    font=dict(size=14),
                ),
                cells=dict(
                    values=[chunk_df[col] for col in chunk_df.columns],
                    fill_color="lavender",
                    align="left",
                    font=dict(size=12),
                ),
            )
        ]
    )

    fig_chunk.update_layout(
        title="Per ogni Chunk Size → Migliore Chunking Strategy e Retrieval Approach",
        height=400,
        width=900,
    )

    fig_chunk.show()

    # Table 2: Best chunk size and retrieval approach for each chunking strategy
    best_by_strategy = []

    for strategy in sorted(summary_df["chunk_strategy"].unique()):
        strategy_data = summary_df[summary_df["chunk_strategy"] == strategy]
        best = strategy_data.loc[strategy_data["f1_score"].idxmax()]
        best_by_strategy.append(
            {
                "Chunking Strategy": strategy,
                "Best Chunk Size": best["chunk_size"],
                "Retrieval Approach": best["retrieval_approach"],
                "F1 Score": round(best["f1_score"], OUTPUT_PRECISION),
            }
        )

    strategy_df = pd.DataFrame(best_by_strategy)

    # Create Plotly table for strategy
    fig_strategy = go.Figure(
        data=[
            go.Table(
                header=dict(
                    values=list(strategy_df.columns),
                    fill_color="paleturquoise",
                    align="left",
                    font=dict(size=14),
                ),
                cells=dict(
                    values=[strategy_df[col] for col in strategy_df.columns],
                    fill_color="lavender",
                    align="left",
                    font=dict(size=12),
                ),
            )
        ]
    )

    fig_strategy.update_layout(
        title="Per ogni Chunking Strategy → Migliore Chunk Size e Retrieval Approach",
        height=400,
        width=900,
    )

    fig_strategy.show()

    # Table 3: Top 10 configurations by F1 score
    top_configs = summary_df.sort_values("f1_score", ascending=False).head(10)
    top_configs = top_configs.rename(
        columns={
            "chunk_strategy": "Chunking Strategy",
            "retrieval_approach": "Retrieval Approach",
            "chunk_size": "Chunk Size",
            "f1_score": "F1 Score",
        }
    )
    top_configs["F1 Score"] = top_configs["F1 Score"].apply(
        lambda x: round(x, OUTPUT_PRECISION)
    )

    # Create Plotly table for top configurations
    fig_top = go.Figure(
        data=[
            go.Table(
                header=dict(
                    values=list(top_configs.columns),
                    fill_color="paleturquoise",
                    align="left",
                    font=dict(size=14),
                ),
                cells=dict(
                    values=[top_configs[col] for col in top_configs.columns],
                    fill_color="lavender",
                    align="left",
                    font=dict(size=12),
                ),
            )
        ]
    )

    fig_top.update_layout(
        title="Top 10 Configurazioni per F1 Score", height=500, width=900
    )

    fig_top.show()

    # Table 4: Best chunk size and chunking strategy for each retrieval approach
    best_by_retrieval = []

    for approach in sorted(summary_df["retrieval_approach"].unique()):
        approach_data = summary_df[summary_df["retrieval_approach"] == approach]
        best = approach_data.loc[approach_data["f1_score"].idxmax()]
        best_by_retrieval.append(
            {
                "Retrieval Approach": approach,
                "Best Chunk Size": best["chunk_size"],
                "Best Chunking Strategy": best["chunk_strategy"],
                "F1 Score": round(best["f1_score"], OUTPUT_PRECISION),
            }
        )

    retrieval_df = pd.DataFrame(best_by_retrieval)

    # Create Plotly table for retrieval approach
    fig_retrieval = go.Figure(
        data=[
            go.Table(
                header=dict(
                    values=list(retrieval_df.columns),
                    fill_color="paleturquoise",
                    align="left",
                    font=dict(size=14),
                ),
                cells=dict(
                    values=[retrieval_df[col] for col in retrieval_df.columns],
                    fill_color="lavender",
                    align="left",
                    font=dict(size=12),
                ),
            )
        ]
    )

    fig_retrieval.update_layout(
        title="Per ogni Retrieval Approach → Migliore Chunk Size e Chunking Strategy",
        height=300,
        width=900,
    )

    fig_retrieval.show()


def create_top_configs_plot(summary_df, top_x=25):
    """
    Create a table of the top X configurations by F1 score instead of a bar chart.
    """
    # Create a copy of the data and sort by F1 score
    top_configs = summary_df.sort_values("f1_score", ascending=False)

    # Get only the top X configurations
    top_configs = top_configs.head(top_x)

    # Prepare table data
    table_df = top_configs[
        ["chunk_strategy", "retrieval_approach", "chunk_size", "f1_score"]
    ]
    table_df = table_df.rename(
        columns={
            "chunk_strategy": "Chunking Strategy",
            "retrieval_approach": "Retrieval Approach",
            "chunk_size": "Chunk Size",
            "f1_score": "F1 Score",
        }
    )

    # Round F1 scores to specified precision
    table_df["F1 Score"] = table_df["F1 Score"].apply(
        lambda x: round(x, OUTPUT_PRECISION)
    )

    # Create Plotly table
    fig = go.Figure(
        data=[
            go.Table(
                header=dict(
                    values=list(table_df.columns),
                    fill_color="paleturquoise",
                    align="left",
                    font=dict(size=14),
                ),
                cells=dict(
                    values=[table_df[col] for col in table_df.columns],
                    fill_color="lavender",
                    align="left",
                    font=dict(size=12),
                ),
            )
        ]
    )

    fig.update_layout(
        title=f"Top {top_x} Configurazioni per F1 Score", height=800, width=1200
    )

    fig.show()


def create_retrieval_comparison(summary_df):
    """
    Create a comparison of different retrieval approaches for each chunk size,
    highlighting where RAG Fusion or HyDE approach the performance of Semantic Search.
    """
    # Get all unique chunk sizes and retrieval approaches
    chunk_sizes = sorted(summary_df["chunk_size"].unique())
    retrieval_approaches = sorted(summary_df["retrieval_approach"].unique())

    # Create a dataframe to store the best F1 score for each retrieval approach at each chunk size
    comparison_data = []

    for chunk_size in chunk_sizes:
        chunk_data = summary_df[summary_df["chunk_size"] == chunk_size]
        row_data = {"Chunk Size": chunk_size}

        # For each retrieval approach, find the best chunking strategy at this chunk size
        for approach in retrieval_approaches:
            approach_data = chunk_data[chunk_data["retrieval_approach"] == approach]
            if not approach_data.empty:
                best_score = approach_data["f1_score"].max()
                best_strategy = approach_data.loc[approach_data["f1_score"].idxmax()][
                    "chunk_strategy"
                ]
                row_data[f"{approach}"] = best_score
                row_data[f"{approach} Strategy"] = best_strategy
            else:
                row_data[f"{approach}"] = None
                row_data[f"{approach} Strategy"] = None

        comparison_data.append(row_data)

    comparison_df = pd.DataFrame(comparison_data)

    # Calculate performance gap between Semantic Search and other approaches
    if "Semantic Search" in retrieval_approaches:
        for approach in retrieval_approaches:
            if approach != "Semantic Search":
                comparison_df[f"{approach} vs Semantic Search"] = (
                    comparison_df[f"{approach}"] / comparison_df["Semantic Search"]
                )

    # Display the comparison table
    print(
        "\n=== Tabella 5: Confronto tra Approcci di Retrieval per ogni Chunk Size ==="
    )
    print(
        "OSSERVAZIONE: Semantic Search domina in generale, ma RAG Fusion è competitivo con chunk piccoli e HyDE migliora con chunk grandi"
    )
    display_df = comparison_df[["Chunk Size"] + [a for a in retrieval_approaches]]
    display_df = display_df.round(OUTPUT_PRECISION)
    print(tabulate(display_df, headers="keys", tablefmt="grid", showindex=False))

    # Display the relative performance table
    print("\n=== Tabella 6: Performance Relativa rispetto a Semantic Search ===")
    print(
        "SPIEGAZIONE: Questa tabella mostra la percentuale di efficacia di ogni approccio rispetto a Semantic Search."
    )
    print(
        "Ad esempio, con chunk size 128, RAG Fusion raggiunge il 79.5% dell'efficacia di Semantic Search"
    )
    print(
        "(calcolato come: F1 score di RAG Fusion / F1 score di Semantic Search * 100)"
    )
    print(
        "OSSERVAZIONE: RAG Fusion raggiunge il suo picco relativo con chunk piccoli, mentre HyDE è più vicino a Semantic Search con chunk grandi"
    )
    relative_cols = ["Chunk Size"] + [
        f"{a} vs Semantic Search"
        for a in retrieval_approaches
        if a != "Semantic Search"
    ]
    relative_df = comparison_df[relative_cols].round(OUTPUT_PRECISION)

    # Format percentage
    for col in relative_cols[1:]:
        relative_df[col] = relative_df[col].apply(
            lambda x: f"{x:.1%}" if pd.notnull(x) else "N/A"
        )

    print(tabulate(relative_df, headers="keys", tablefmt="grid", showindex=False))

    # Add annotations to highlight key observations
    rag_fusion_small_chunks = []
    hyde_large_chunks = []

    if (
        "RAG Fusion" in retrieval_approaches
        and "Semantic Search" in retrieval_approaches
    ):
        for i, chunk_size in enumerate(chunk_sizes):
            if chunk_size <= 256:  # Small chunks
                rag_fusion_small_chunks.append(
                    f"Con chunk size {chunk_size}, RAG Fusion raggiunge il {comparison_df.iloc[i]['RAG Fusion vs Semantic Search']*100:.1f}% delle performance di Semantic Search"
                )

    if "HyDE" in retrieval_approaches and "Semantic Search" in retrieval_approaches:
        for i, chunk_size in enumerate(chunk_sizes):
            if chunk_size >= 4096:  # Large chunks
                hyde_large_chunks.append(
                    f"Con chunk size {chunk_size}, HyDE raggiunge il {comparison_df.iloc[i]['HyDE vs Semantic Search']*100:.1f}% delle performance di Semantic Search"
                )

    if rag_fusion_small_chunks:
        print("\nRAG FUSION CON CHUNK PICCOLI:")
        for observation in rag_fusion_small_chunks:
            print(f"- {observation}")

    if hyde_large_chunks:
        print("\nHyDE CON CHUNK GRANDI:")
        for observation in hyde_large_chunks:
            print(f"- {observation}")

    # Create a Plotly table with title highlighting the key insight
    fig = go.Figure(
        data=[
            go.Table(
                header=dict(
                    values=list(display_df.columns),
                    fill_color="paleturquoise",
                    align="center",
                    font=dict(size=14),
                ),
                cells=dict(
                    values=[display_df[col] for col in display_df.columns],
                    fill_color="lavender",
                    align="center",
                    font=dict(size=12),
                ),
            )
        ]
    )

    fig.update_layout(
        title={
            "text": "F1 Score per ogni Approccio di Retrieval e Chunk Size<br><sup>Semantic Search domina in generale, ma gli altri approcci sono competitivi in specifici ranges di chunk size</sup>",
            "y": 0.95,
            "x": 0.5,
            "xanchor": "center",
            "yanchor": "top",
        },
        height=600,
        width=1200,
    )

    fig.show()

    # Create a relative performance table with key insight in the title
    fig_relative = go.Figure(
        data=[
            go.Table(
                header=dict(
                    values=list(relative_df.columns),
                    fill_color="paleturquoise",
                    align="center",
                    font=dict(size=14),
                ),
                cells=dict(
                    values=[relative_df[col] for col in relative_df.columns],
                    fill_color="lavender",
                    align="center",
                    font=dict(size=12),
                ),
            )
        ]
    )

    fig_relative.update_layout(
        title={
            "text": "Performance Relativa rispetto a Semantic Search<br><sup>Quanto un approccio si avvicina a Semantic Search (F1 score approccio / F1 score Semantic Search * 100%)</sup><br><sup>RAG Fusion si avvicina con chunk piccoli, mentre HyDE si avvicina con chunk grandi</sup>",
            "y": 0.95,
            "x": 0.5,
            "xanchor": "center",
            "yanchor": "top",
        },
        height=600,
        width=1200,
    )

    fig_relative.show()

    # Create a line chart to visualize the trend with annotations outside the chart
    fig_line = go.Figure()

    # Add traces for each retrieval approach with different colors and line styles
    colors = ["#636EFA", "#EF553B", "#00CC96", "#AB63FA"]
    for i, approach in enumerate(retrieval_approaches):
        fig_line.add_trace(
            go.Scatter(
                x=comparison_df["Chunk Size"],
                y=comparison_df[approach],
                mode="lines+markers",
                name=approach,
                marker=dict(size=10),
                line=dict(width=3, color=colors[i % len(colors)]),
            )
        )

    # Create observations text to add outside the chart
    observations = "<br>• RAG Fusion è più competitivo con chunk piccoli (128-256)"
    observations += "<br>• HyDE migliora relativamente con chunk grandi (4096-8192)"
    observations += "<br>• Semantic Search domina in tutto l'intervallo di chunk size"

    # Add the layout with the observations text outside the chart
    fig_line.update_layout(
        title={
            "text": "F1 Score per Approccio di Retrieval e Chunk Size<br><sup>"
            + observations
            + "</sup>",
            "y": 0.95,
            "x": 0.5,
            "xanchor": "center",
            "yanchor": "top",
        },
        xaxis_title="Chunk Size",
        yaxis_title="F1 Score",
        xaxis=dict(type="category"),
        height=600,
        width=1200,
        legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99, font=dict(size=14)),
    )

    fig_line.show()


def analyze_results():
    """
    Main function to analyze results and display tables/plots.
    """
    # Load and prepare data
    df, chunk_sizes = load_evaluation_data()

    # Calculate F1 scores for all configurations
    summary_df = calculate_f1_scores(df)

    # Find and print the best configuration
    find_best_configuration(summary_df)

    # Display text tables in terminal
    display_tables(summary_df)

    # Create interactive Plotly tables
    create_plotly_tables(summary_df)

    # Create bar chart with top X configurations
    create_top_configs_plot(summary_df, top_x=100)

    # Create retrieval approach comparison
    create_retrieval_comparison(summary_df)


if __name__ == "__main__":
    analyze_results()
