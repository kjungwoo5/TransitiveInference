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
        try:
            data_by_window[window_val] = float(accuracy_values[type_idx])
        except (TypeError, IndexError, ValueError):
            data_by_window[window_val] = np.nan

    series = pd.Series(data_by_window)
    series.index = series.index.astype(float)
    return series.to_frame(name=offset)


def make_heatmap(animal: str):
    grid_parts = []
    summary_rows = []

    for key, value in enumerate(TYPE_OF_PLOT.values()):
        df = pd.read_json(ANALYSIS_ROOT / f'accuracies_by_window_{animal}_from_next.json').transpose()
        part = df[key].to_frame()
        part.columns = [value]
        grid_parts.append(part)

    grid = pd.concat(grid_parts, axis=1)
    grid = grid.sort_index(axis=0)
    #grid = grid.sort_index(axis=1)
    grid = grid.astype(float)
    
    # flat_grid = grid.stack().reset_index()
    # flat_grid.columns = ["window_size", "type_of_test", "accuracy"]
    # best_row = flat_grid.loc[flat_grid["accuracy"].idxmax()]
    # summary_rows.append({
    #     "animal": animal,
    #     "decoding_type": TYPE_OF_PLOT[type_idx],
    #     "best_first_window_size": float(best_row["offset"]),
    #     "best_first_accuracy": float(best_row["accuracy"]),
    #     "best_second_window": float(best_row["window_size"]),
    #     "best_accuracy": float(best_row["accuracy"]),
    # })


    fig, ax = plt.subplots(figsize=(8.5, 7.5))
    values = grid.to_numpy(dtype=float)
    values = np.where(np.isnan(values), np.nan, values)

    x_vals = np.arange(values.shape[1] + 1)
    y_vals = np.arange(values.shape[0] + 1)
    masked_values = np.ma.masked_invalid(values)
    image = ax.pcolormesh(x_vals, y_vals, masked_values, cmap="viridis", shading="auto")

    if np.isfinite(values).any():
        valid = values[np.isfinite(values)]
        vmin = float(np.min(valid))
        vmax = float(np.max(valid))
        if np.isclose(vmin, vmax):
            vmin = max(0.0, vmin - 0.01)
            vmax = min(1.0, vmax + 0.01)
        image.set_clim(vmin, vmax)

    window_labels = [f"{value:.2f}" for value in grid.index.tolist()]
    plot_labels = [value in TYPE_OF_PLOT.values()]

    ax.set_xticks(np.arange(len(plot_labels)) + 0.5)
    ax.set_xticklabels(plot_labels, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(window_labels)) + 0.5)
    ax.set_yticklabels(window_labels)

    for row_idx in range(values.shape[0]):
        for col_idx in range(values.shape[1]):
            value = values[row_idx, col_idx]
            if np.isfinite(value):
                ax.text(col_idx + 0.5, row_idx + 0.5, f"{value:.3f}", ha="center", va="center", fontsize=6, color="white")

    ax.set_xlabel("Type of plot")
    ax.set_ylabel("Window size")
    ax.set_title(f"{animal} decoding accuracy heatmap")
    fig.colorbar(image, ax=ax, label="Decoding accuracy")

    for spine in ax.spines.values():
        spine.set_visible(False)

    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    output_file = OUTPUT_PATH / f"Stage5_{animal}_heatmap.png"
    fig.tight_layout()
    fig.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return summary_rows


if __name__ == "__main__":
    all_summary_rows = []
    for animal in ["JK01", "JK02"]:
        all_summary_rows.extend(make_heatmap(animal))


    summary_df = pd.DataFrame(all_summary_rows)
    summary_df.to_csv(OUTPUT_PATH / "best_window_summary.csv", index=False)
    print(summary_df)
