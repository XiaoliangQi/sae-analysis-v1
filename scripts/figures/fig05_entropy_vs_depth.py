"""
Figure 5 — Feature entropy versus layer depth (layers 0–5).
Paper: Fig. 5, caption: "Feature entropy versus layer depth. Each gray dot is one
feature; colored lines track selected features across layers. The red dashed line
shows token vector entropy; the blue dashed line shows the average entropy of the
top-20 features."

Prerequisite data:
    entropy_comparison_resid_out_layer{0..5}_<timestamp>.pt  (one file per layer)
    (produced by: python scripts/analysis/compare_entropies_multi_layer.py --layers 0 1 2 3 4 5)

Output:
    paper/figures/entropy_vs_depth.png
"""

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import glob
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parents[2]
OUT_FILE = ROOT / "paper/figures/entropy_vs_depth.png"

LAYERS = [0, 1, 2, 3, 4, 5]
N_TRACKED = 10   # Number of colored feature lines to draw
N_TOP_AVG = 20   # Features used for the blue average line


def find_latest(layer):
    pattern = str(ROOT / f"entropy_comparison_resid_out_layer{layer}_*.pt")
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(f"No entropy_comparison file for layer {layer}. "
                                f"Run: python scripts/analysis/compare_entropies_multi_layer.py "
                                f"--layers {' '.join(map(str, LAYERS))}")
    return max(files, key=lambda f: Path(f).stat().st_mtime)


def main():
    # Load all layer data
    layer_data = {}
    for layer in LAYERS:
        path = find_latest(layer)
        layer_data[layer] = torch.load(path, map_location="cpu", weights_only=False)
        print(f"  Layer {layer}: {Path(path).name}")

    # Collect per-feature mean entropy and token vector entropy per layer
    feat_entropies = defaultdict(dict)  # layer -> feat_idx -> mean entropy
    feat_activations = defaultdict(float)  # feat_idx -> total activation (for ranking)
    token_entropies = {}  # layer -> mean token vector entropy

    for layer in LAYERS:
        batch_results = layer_data[layer]["batch_results"]
        token_ents = []
        for br in batch_results:
            for feat_idx, ent in br.get("feature_entropies", {}).items():
                if feat_idx not in feat_entropies[layer]:
                    feat_entropies[layer][feat_idx] = []
                feat_entropies[layer][feat_idx].append(ent)
                feat_activations[feat_idx] += br.get("feature_activations", {}).get(feat_idx, 0.0)
            if br.get("token_vector_entropy") is not None:
                token_ents.append(br["token_vector_entropy"])
        # Average per-feature
        feat_entropies[layer] = {k: float(np.mean(v)) for k, v in feat_entropies[layer].items()}
        token_entropies[layer] = float(np.mean(token_ents)) if token_ents else np.nan

    # Pick features to track with colored lines (most-activated across all layers)
    tracked = [fid for fid, _ in
               sorted(feat_activations.items(), key=lambda x: x[1], reverse=True)[:N_TRACKED]]

    fig, ax = plt.subplots(figsize=(7, 5))

    # Gray scatter: all features at each layer
    for layer in LAYERS:
        ents = list(feat_entropies[layer].values())
        ax.scatter([layer] * len(ents), ents, color="gray", alpha=0.2, s=10, zorder=1)

    # Colored lines: selected tracked features
    colors = plt.cm.tab10(np.linspace(0, 1, N_TRACKED))
    for i, feat_idx in enumerate(tracked):
        xs, ys = [], []
        for layer in LAYERS:
            if feat_idx in feat_entropies[layer]:
                xs.append(layer)
                ys.append(feat_entropies[layer][feat_idx])
        if xs:
            ax.plot(xs, ys, "o-", color=colors[i], linewidth=1.5, markersize=5,
                    label=f"Feature {feat_idx}", alpha=0.85, zorder=3)

    # Blue dashed: average of top-N features per layer
    avg_ents = []
    for layer in LAYERS:
        vals = sorted(feat_entropies[layer].values(), reverse=True)[:N_TOP_AVG]
        avg_ents.append(np.mean(vals) if vals else np.nan)
    ax.plot(LAYERS, avg_ents, "b--", linewidth=2, label=f"Avg top-{N_TOP_AVG} features", zorder=4)

    # Red dashed: token vector entropy
    tok_vals = [token_entropies.get(l, np.nan) for l in LAYERS]
    ax.plot(LAYERS, tok_vals, "r--", linewidth=2, label="Token vector entropy", zorder=4)

    ax.set_xlabel("Layer depth", fontsize=12)
    ax.set_ylabel("Entropy (bits)", fontsize=12)
    ax.set_title("Feature entropy vs layer depth", fontsize=13)
    ax.set_xticks(LAYERS)
    ax.set_xticklabels([f"Layer {l}" for l in LAYERS], fontsize=9)
    ax.legend(fontsize=8, loc="upper left", ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_FILE, dpi=150, bbox_inches="tight")
    print(f"Saved → {OUT_FILE}")


if __name__ == "__main__":
    main()
