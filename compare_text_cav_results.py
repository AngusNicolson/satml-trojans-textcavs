
from pathlib import Path

import pandas as pd
import numpy as np

from utils import TARGET_CLASS_NAMES


def main():
    layer = "avgpool"
    concept_names = "tulu_4bit_00_cleaned"
    csv_name = f"ordered_by_directional_derivative_concepts_for_each_class_{concept_names}.csv"

    outdir = Path(f'models/{layer}_cycle_aligner')
    trojan_outdir = Path(f'models/{layer}_cycle_aligner_trojan')
    explanation_outdir = Path("output/explanations")
    explanation_outdir.mkdir(exist_ok=True, parents=True)

    df = pd.read_csv(outdir / csv_name)
    trojan_df = pd.read_csv(trojan_outdir / csv_name)

    df = df.drop("Unnamed: 0", axis=1)
    trojan_df = trojan_df.drop("Unnamed: 0", axis=1)

    concepts = list(df["bullfrog"].unique())
    concepts = [c for c in concepts if c is not np.nan]
    concepts.sort()

    explanations = get_explanations(df, trojan_df)
    explanations.to_csv(explanation_outdir / f"text_cav_explanations_{concept_names}.csv")

    print("Done!")


def get_explanations(df, trojan_df, n=10):
    explanations = pd.DataFrame()
    for target in TARGET_CLASS_NAMES:
        trojan_scores = trojan_df[[target, f"{target}_vals"]].reset_index()
        trojan_scores = trojan_scores.rename(
            columns={"index": "trojan_score", f"{target}_vals": "trojan_val"},
        )

        normal_scores = df[[target, f"{target}_vals"]].reset_index()
        normal_scores = normal_scores.rename(
            columns={"index": "normal_score", f"{target}_vals": "normal_val"}
        )

        scores = pd.merge(normal_scores, trojan_scores)
        scores["val_diff"] = scores["trojan_val"] - scores["normal_val"]
        scores["val_diff_norm"] = scores["val_diff"] - scores["val_diff"].min()
        scores["val_diff_norm"] = 2 * scores["val_diff_norm"] / scores["val_diff_norm"].max()
        scores["val_diff_norm"] = scores["val_diff_norm"] - 1
        scores["score"] = (scores["trojan_score"].max() - scores["trojan_score"]) * scores["val_diff_norm"]
        scores = scores.sort_values(by="score", ascending=False)
        scores = scores.iloc[:n, :]
        scores = scores[[target]].reset_index(drop=True)

        explanations = pd.concat([explanations, scores], axis=1)
    return explanations


if __name__ == "__main__":
    main()
