import json
from pathlib import Path
import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ANALYSIS_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_PATH = ANALYSIS_ROOT / "Outputs" / "Accuracies by window size" / "Heatmaps"

TYPE_OF_PLOT = {
    0: "First position",
    1: "Second position",
    2: "Third position",
    3: "Fourth position",
    4: "Stimulus Identity",
}

OFFSETS = [0.25, 0.38, 0.5, 0.63, 0.75]


def load_accuracy_grid(animal: str, offset: float, type_idx: int):
    json_path = ANALYSIS_ROOT / f"accuracies_by_window_{animal}_{offset}.json"
    if not json_path.exists():
        raise FileNotFoundError(f"Missing expected file: {json_path}")

    with open(json_path, "r") as handle:
        raw_data = json.load(handle)

    data_by_window = {}
    for window_key, accuracy_values in raw_data.items():
        window_val = round(float(window_key), 3)
        data_by_window[window_val] = float(accuracy_values[type_idx])

    return pd.DataFrame([data_by_window], index=[offset]).T


def make_heatmap(animal: str, type_idx: int):
    grid_parts = []
    missing_offsets = []
    summary_rows = []

    for offset in OFFSETS:
        try:
            part = load_accuracy_grid(animal, offset, type_idx)
            part.columns = [offset]
            grid_parts.append(part)
        except FileNotFoundError:
            missing_offsets.append(offset)

    if not grid_parts:
        raise FileNotFoundError(f"No JSON files found for {animal} and decoding type {type_idx}")

    grid = pd.concat(grid_parts, axis=1)
    grid = grid.sort_index(axis=0)
    grid = grid.sort_index(axis=1)

    flat_grid = grid.stack().reset_index()
    flat_grid.columns = ["window_size", "offset", "accuracy"]
    best_row = flat_grid.loc[flat_grid["accuracy"].idxmax()]
    summary_rows.append({
        "animal": animal,
        "decoding_type": TYPE_OF_PLOT[type_idx],
        "best_offset": float(best_row["offset"]),
        "best_window_size": float(best_row["window_size"]),
        "best_accuracy": float(best_row["accuracy"]),
    })

    fig, ax = plt.subplots(figsize=(8.5, 7.5))
    image = ax.imshow(grid.to_numpy(), cmap="viridis", aspect="auto", vmin=0.0, vmax=1.0)

    window_labels = [f"{value:.2f}" for value in grid.index.tolist()]
    offset_labels = [f"{value:.2f}" for value in grid.columns.tolist()]

    ax.set_xticks(range(len(offset_labels)))
    ax.set_xticklabels(offset_labels, rotation=45, ha="right")
    ax.set_yticks(range(len(window_labels)))
    ax.set_yticklabels(window_labels)

    ax.set_xlabel("Offset")
    ax.set_ylabel("Window size")
    ax.set_title(f"{animal} decoding accuracy heatmap\n{TYPE_OF_PLOT[type_idx]}")
    fig.colorbar(image, ax=ax, label="Decoding accuracy")

    for spine in ax.spines.values():
        spine.set_visible(False)

    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    output_file = OUTPUT_PATH / f"Stage5_{animal}_{TYPE_OF_PLOT[type_idx].lower().replace(' ', '_')}_heatmap.png"
    fig.tight_layout()
    fig.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close(fig)

    if missing_offsets:
        print(f"Skipped offsets {missing_offsets} for {animal} because no matching JSON file was found.")

    return summary_rows


if __name__ == "__main__":
    all_summary_rows = []
    for animal in ["JK01", "JK02", "JK03", "JK04"]:
        for type_idx in range(5):
            try:
                all_summary_rows.extend(make_heatmap(animal, type_idx))
            except FileNotFoundError as exc:
                print(exc)

    summary_df = pd.DataFrame(all_summary_rows)
    summary_df.to_csv(OUTPUT_PATH / "best_offset_window_summary.csv", index=False)
    print(summary_df)
