import json
import glob
import os

"""
results_dir = "/home/e4user/r2r-eval/results/last"
pattern = os.path.join(results_dir, "evaluation_results_*_chunk_size_1024.json")

merged = {}


def extract_search_name(filename):
    return filename.split("evaluation_results_")[1].split("_chunk_size_")[0]


for filepath in glob.glob(pattern):
    search_name = extract_search_name(os.path.basename(filepath))
    with open(filepath, "r") as f:
        data = json.load(f)
    for chunker, metrics in data.items():
        if chunker not in merged:
            merged[chunker] = {}
        merged[chunker][search_name] = metrics

with open(os.path.join(results_dir, "evaluation_results_1024.json"), "x") as f:
    json.dump(merged, f, indent=2, ensure_ascii=False)

"""
import pandas as pd
import json

df = pd.read_csv("/home/e4user/r2r-eval/datasets/ragas_testset_dell_servers.csv")

print(df.columns)

data = {
    "user_input": df["user_input"].tolist(),
    "reference": df["reference"].tolist(),
    "reference_contexts": [[context] for context in df["reference_contexts"].tolist()],
}

with open("/home/e4user/r2r-eval/datasets/ragas_testset_dell_servers.json", "w") as f:
    json.dump(data, f, indent=2, ensure_ascii=False)
