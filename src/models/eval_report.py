"""P9 evaluation report - same metrics code path as the gate, rendered.

Run:  python -m src.models.eval_report
Output: docs/eval_report.md + docs/figures/*.png (Clarivance palette).
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.models.evaluate import MODEL_NAME, load_test_data, recall_at_k

TEAL, NAVY, AMBER, RED, SLATE = "#00B4A6", "#0D1F3C", "#F5A623", "#E03E3E", "#F4F6F9"
FIG_DIR = Path("docs/figures")


def score_frame(model, test: pd.DataFrame) -> pd.Series:
    cols = ["rms_value", "rolling_mean_1h", "rolling_std_1h", "deviation_from_baseline"]
    return pd.Series(model.predict_proba(test[cols])[:, 1], index=test.index)


def slice_by_life_stage(test: pd.DataFrame, scores: pd.Series) -> pd.DataFrame:
    """Recall@10% per life-stage third (early/mid/late) of the test bearing."""
    test = test.sort_values("timestamp").copy()
    test["stage"] = pd.qcut(range(len(test)), 3, labels=["early", "mid", "late"])

    rows = []
    for stage in ["early", "mid", "late"]:
        subset = test[test["stage"] == stage]
        subset_scores = scores.loc[subset.index]
        rows.append(
            {
                "stage": stage,
                "rows": len(subset),
                "anomalies": int(subset["is_anomaly"].sum()),
                "recall_at_10pct": recall_at_k(subset["is_anomaly"], subset_scores, 0.10),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    import mlflow
    from azure.ai.ml import MLClient
    from azure.identity import DefaultAzureCredential

    from src.config.settings import settings

    ml_client = MLClient(
        DefaultAzureCredential(),
        settings.subscription_id,
        settings.resource_group,
        settings.workspace_name,
    )
    mlflow.set_tracking_uri(ml_client.workspaces.get(settings.workspace_name).mlflow_tracking_uri)
    versions = sorted(int(m.version) for m in ml_client.models.list(name=MODEL_NAME))
    latest = versions[-1]

    test = load_test_data()
    model = mlflow.lightgbm.load_model(f"models:/{MODEL_NAME}/{latest}")
    scores = score_frame(model, test)
    overall = recall_at_k(test["is_anomaly"], scores, 0.10)
    slices = slice_by_life_stage(test, scores)

    # Fig 1: score distribution, anomalies vs normal
    fig, ax = plt.subplots(figsize=(8, 4.5), facecolor=SLATE)
    ax.hist(scores[test["is_anomaly"] == 0], bins=40, color=TEAL, alpha=0.8, label="normal")
    ax.hist(scores[test["is_anomaly"] == 1], bins=40, color=RED, alpha=0.8, label="anomaly")
    ax.set_yscale("log")
    ax.set_title("Score distribution - held-out bearing", color=NAVY, fontweight="bold")
    ax.legend()
    fig.savefig(FIG_DIR / "score_distribution.png", dpi=150, bbox_inches="tight")

    # Fig 2: recall by life stage
    fig, ax = plt.subplots(figsize=(8, 4.5), facecolor=SLATE)
    ax.bar(slices["stage"], slices["recall_at_10pct"], color=[TEAL, AMBER, NAVY])
    ax.axhline(0.90, color=RED, linestyle="--", label="gate floor 0.90")
    ax.set_title("Recall@10% by life stage", color=NAVY, fontweight="bold")
    ax.legend()
    fig.savefig(FIG_DIR / "recall_by_stage.png", dpi=150, bbox_inches="tight")

    md = [
        "# P9 Evaluation Report",
        f"\n**Model:** `{MODEL_NAME}` v{latest} · **Registry versions:** {versions}",
        f"\n**Overall recall@10% (held-out bearing):** {overall:.3f} (gate floor 0.90)",
        "\n## Slice evaluation - life stages\n",
        slices.to_markdown(index=False),
        "\n## Figures\n",
        "![](figures/score_distribution.png)\n",
        "![](figures/recall_by_stage.png)\n",
        "\n*Same metric code path as the CI gate (`src/models/evaluate.py`). "
        "Caveat: label-feature affinity (3-sigma labels vs deviation features)"
        " documented in the model card.*",
    ]
    Path("docs/eval_report.md").write_text("\n".join(md), encoding="utf-8")
    print(f"report written: overall recall {overall:.3f}; slices:\n{slices}")


if __name__ == "__main__":
    main()
