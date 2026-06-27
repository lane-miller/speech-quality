"""
Analyze sweep results for a single degradation type.

Produces three outputs saved to results/analysis/{degradation}/:
  scatter_grid.png        — per-metric scatter vs. severity with baseline line
  correlation_table.csv   — Pearson r and Spearman ρ vs. severity per metric
  correlation_table.png   — same table rendered as a matplotlib figure
  metric_heatmap.png      — Spearman ρ inter-metric heatmap (incl. severity_value)

Usage:
    python analyze.py --degradation clipping
    python analyze.py --degradation reverb
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
from scipy import stats

import config


DEGRADATIONS = [
    "clipping",
    "noise_pink",
    "noise_babble",
    "noise_tonal_lf",
    "noise_tonal_hf",
    "noise_impulsive",
    "codec",
    "lowpass",
    "reverb",
]

METRICS_BASE = [
    "dnsmos_sig",
    "dnsmos_bak",
    "dnsmos_ovr",
    "nisqa",
    "clipping_rate",
    "crest_factor_db",
    "wada_snr_db",
    "spectral_flatness",
    "hf_energy_ratio",
    "pitch_confidence",
]

RESULTS_DIR = Path(config.RESULTS_DIR)
N_COLS = 4


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data(
    degradation: str, snr_filter: float | None = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    deg_path = RESULTS_DIR / f"{degradation}_results.csv"
    baseline_path = RESULTS_DIR / "baseline_results.csv"

    df = pd.read_csv(deg_path)
    df["severity_value"] = pd.to_numeric(df["severity_value"], errors="coerce")

    if snr_filter is not None and degradation == "noise_impulsive":
        df = df[df["impulse_snr_db"] == snr_filter]
        if df.empty:
            raise ValueError(
                f"No rows found for noise_impulsive with impulse_snr_db == {snr_filter}."
            )

    baseline = pd.read_csv(baseline_path)
    return df, baseline


def get_metrics(degradation: str) -> list[str]:
    metrics = METRICS_BASE.copy()
    if degradation == "reverb":
        metrics.append("c50_db")
    return metrics


def get_severity_param(df: pd.DataFrame) -> str:
    return str(df["severity_param"].dropna().iloc[0])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def to_numeric_series(s: pd.Series) -> pd.Series:
    """Convert a column to numeric, treating empty strings as NaN."""
    return pd.to_numeric(s.replace("", np.nan), errors="coerce")


def valid_pairs(
    x: pd.Series, y: pd.Series
) -> tuple[np.ndarray, np.ndarray]:
    """Return aligned arrays dropping any row where either value is NaN."""
    combined = pd.DataFrame({"x": x, "y": y}).dropna()
    return combined["x"].to_numpy(), combined["y"].to_numpy()


def correlations(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Return (Pearson r, Spearman ρ); NaN if fewer than 2 points."""
    if len(x) < 2:
        return float("nan"), float("nan")
    r, _ = stats.pearsonr(x, y)
    rho, _ = stats.spearmanr(x, y)
    return float(r), float(rho)


def baseline_means(baseline: pd.DataFrame, metrics: list[str]) -> dict[str, float]:
    means = {}
    for m in metrics:
        s = to_numeric_series(baseline[m]).dropna()
        means[m] = float(s.mean()) if len(s) > 0 else float("nan")
    return means


def compute_normalized_variance(df: pd.DataFrame, metrics: list[str]) -> dict[str, float]:
    """Mean per-level std divided by observed metric range for each metric."""
    result: dict[str, float] = {}
    for metric in metrics:
        met = to_numeric_series(df[metric])
        vals_all = met.dropna()
        if len(vals_all) == 0:
            result[metric] = float("nan")
            continue
        metric_range = float(vals_all.max() - vals_all.min())
        if metric_range < 1e-6:
            result[metric] = float("nan")
            continue
        work = pd.DataFrame(
            {"severity_value": df["severity_value"], metric: met}
        ).dropna(subset=[metric])
        level_stds: list[float] = []
        for _, group in work.groupby("severity_value"):
            vals = group[metric].dropna()
            if len(vals) == 0:
                continue
            std = float(vals.std())
            if not np.isnan(std):
                level_stds.append(std)
        mean_std = float(np.mean(level_stds)) if level_stds else float("nan")
        result[metric] = mean_std / metric_range if not np.isnan(mean_std) else float("nan")
    return result


# ---------------------------------------------------------------------------
# Output 1: Scatter subplot grid
# ---------------------------------------------------------------------------

def plot_scatter_grid(
    df: pd.DataFrame,
    baseline: pd.DataFrame,
    metrics: list[str],
    severity_param: str,
    degradation: str,
    out_path: Path,
    variance: dict[str, float],
) -> None:
    n_metrics = len(metrics)
    n_rows = (n_metrics + N_COLS - 1) // N_COLS

    fig, axes = plt.subplots(
        n_rows, N_COLS,
        figsize=(5.5 * N_COLS, 4.0 * n_rows),
    )
    axes_flat = np.array(axes).reshape(-1)

    b_means = baseline_means(baseline, metrics)

    for i, metric in enumerate(metrics):
        ax = axes_flat[i]

        sev = df["severity_value"]
        met = to_numeric_series(df[metric])
        x_arr, y_arr = valid_pairs(sev, met)

        ax.scatter(x_arr, y_arr, alpha=0.15, s=8, color="steelblue", linewidths=0)

        bm = b_means.get(metric, float("nan"))
        if not np.isnan(bm):
            ax.axhline(
                bm,
                color="crimson",
                linestyle="--",
                linewidth=1.2,
                label="baseline mean",
            )

        work = pd.DataFrame(
            {"severity_value": df["severity_value"], metric: met}
        ).dropna(subset=[metric])
        medians = work.groupby("severity_value")[metric].median()
        ax.plot(
            medians.index.to_numpy(),
            medians.to_numpy(),
            color="darkorange",
            linewidth=1.5,
            linestyle="-",
            zorder=3,
            label="median",
            marker="o",
            markersize=3,
        )

        r, rho = correlations(x_arr, y_arr)
        if not np.isnan(r):
            ann = f"r = {r:.2f}\nρ = {rho:.2f}"
        else:
            ann = "r = N/A\nρ = N/A"

        ax.text(
            0.04, 0.96,
            ann,
            transform=ax.transAxes,
            va="top", ha="left",
            fontsize=8,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.75, edgecolor="0.7"),
        )

        norm_std = variance.get(metric, float("nan"))
        if not np.isnan(norm_std):
            var_ann = f"σ_norm = {norm_std:.3f}"
        else:
            var_ann = "σ_norm = N/A"
        ax.text(
            0.96, 0.04,
            var_ann,
            transform=ax.transAxes,
            va="bottom", ha="right",
            fontsize=8,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.75, edgecolor="0.7"),
        )

        ax.set_xlabel(severity_param, fontsize=9)
        ax.set_ylabel(metric, fontsize=9)
        ax.set_title(metric, fontsize=10)
        ax.tick_params(labelsize=8)

    for j in range(n_metrics, len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle(f"{degradation}  —  metric vs. severity", fontsize=13, y=1.01)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Output 2: Severity correlation table
# ---------------------------------------------------------------------------

def build_correlation_table(
    df: pd.DataFrame,
    metrics: list[str],
    variance: dict[str, float],
) -> pd.DataFrame:
    rows: list[dict] = []

    for metric in metrics:
        sev = df["severity_value"]
        met = to_numeric_series(df[metric])
        x_arr, y_arr = valid_pairs(sev, met)
        r, rho = correlations(x_arr, y_arr)
        v = variance.get(metric, float("nan"))
        rows.append(
            {
                "metric": metric,
                "pearson_r_vs_severity": round(r, 4) if not np.isnan(r) else float("nan"),
                "spearman_rho_vs_severity": round(rho, 4) if not np.isnan(rho) else float("nan"),
                "normalized_std": round(v, 4) if not np.isnan(v) else float("nan"),
            }
        )

    # Extra row: DNSMOS OVR vs. NISQA (inter-metric, not vs. severity)
    dns = to_numeric_series(df["dnsmos_ovr"])
    nisqa = to_numeric_series(df["nisqa"])
    x_arr, y_arr = valid_pairs(dns, nisqa)
    r_cross, rho_cross = correlations(x_arr, y_arr)
    rows.append(
        {
            "metric": "DNSMOS OVR vs. NISQA",
            "pearson_r_vs_severity": round(r_cross, 4) if not np.isnan(r_cross) else float("nan"),
            "spearman_rho_vs_severity": round(rho_cross, 4) if not np.isnan(rho_cross) else float("nan"),
            "normalized_std": "",
        }
    )

    norm_vals = [
        r["normalized_std"]
        for r in rows
        if r["normalized_std"] != ""
    ]
    valid_norm = [
        v for v in norm_vals
        if isinstance(v, float) and not np.isnan(v)
    ]
    max_norm = max(valid_norm) if valid_norm else float("nan")

    for r in rows:
        if r["normalized_std"] == "":
            r["relative_normalized_std"] = ""
        else:
            ns = r["normalized_std"]
            if (
                isinstance(ns, float)
                and not np.isnan(ns)
                and not np.isnan(max_norm)
                and max_norm > 0
            ):
                r["relative_normalized_std"] = round(ns / max_norm, 4)
            else:
                r["relative_normalized_std"] = float("nan")

    return pd.DataFrame(
        rows,
        columns=[
            "metric",
            "pearson_r_vs_severity",
            "spearman_rho_vs_severity",
            "normalized_std",
            "relative_normalized_std",
        ],
    )


def save_correlation_table_csv(table: pd.DataFrame, out_path: Path) -> None:
    table.to_csv(out_path, index=False)


def save_correlation_table_png(
    table: pd.DataFrame, out_path: Path, degradation: str
) -> None:
    n_data_rows = len(table)
    fig_h = max(2.5, 0.45 + n_data_rows * 0.38)
    fig, ax = plt.subplots(figsize=(8.5, fig_h))
    ax.axis("off")

    col_labels = [
        "Metric",
        "Pearson r  (vs. severity param)",
        "Spearman ρ  (vs. severity param)",
        "Normalized std  (σ / range)",
        "Relative norm std",
    ]

    def fmt(v: object) -> str:
        if v == "":
            return ""
        if isinstance(v, float) and np.isnan(v):
            return "N/A"
        return f"{v:.4f}"

    cell_text: list[list[str]] = []
    for _, row in table.iterrows():
        cell_text.append(
            [
                str(row["metric"]),
                fmt(row["pearson_r_vs_severity"]),
                fmt(row["spearman_rho_vs_severity"]),
                fmt(row["normalized_std"]),
                fmt(row["relative_normalized_std"]),
            ]
        )

    tbl = ax.table(
        cellText=cell_text,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.auto_set_column_width([0, 1, 2, 3, 4])

    # Highlight header row
    for col_idx in range(5):
        tbl[0, col_idx].set_facecolor("#d0e4f7")
        tbl[0, col_idx].set_text_props(fontweight="bold")

    # Highlight extra DNSMOS OVR vs. NISQA row
    extra_row_idx = n_data_rows  # 1-based table rows: 0=header, 1..n_data_rows=data
    for col_idx in range(5):
        tbl[extra_row_idx, col_idx].set_facecolor("#fff3cd")

    ax.set_title(
        f"{degradation}  —  Correlation with severity\n"
        "r and ρ computed vs. severity parameter  "
        "(bottom row: DNSMOS OVR vs. NISQA inter-metric correlation)",
        fontsize=10,
        pad=10,
    )

    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Output 3: Inter-metric Spearman heatmap
# ---------------------------------------------------------------------------

def plot_heatmap(
    df: pd.DataFrame,
    metrics: list[str],
    out_path: Path,
    degradation: str,
) -> None:
    param_name = str(df["severity_param"].dropna().iloc[0])
    heatmap_cols = [param_name] + metrics

    sub = pd.DataFrame()
    for col in heatmap_cols:
        if col == param_name:
            sub[col] = df["severity_value"]
        else:
            sub[col] = to_numeric_series(df[col])

    corr = sub.corr(method="spearman")

    n = len(corr)
    cell_size = 0.72
    fig_size = max(6.0, n * cell_size + 1.5)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))

    cmap = plt.cm.RdBu_r  # type: ignore[attr-defined]
    norm = mcolors.TwoSlopeNorm(vmin=-1.0, vcenter=0.0, vmax=1.0)

    im = ax.imshow(corr.to_numpy(), cmap=cmap, norm=norm, aspect="auto")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Spearman ρ", fontsize=9)

    labels = corr.columns.tolist()
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)

    for i in range(n):
        for j in range(n):
            val = corr.to_numpy()[i, j]
            text_color = "white" if abs(val) > 0.65 else "black"
            ax.text(
                j, i,
                f"{val:.2f}",
                ha="center", va="center",
                fontsize=7,
                color=text_color,
            )

    ax.set_title(
        f"{degradation}  —  Spearman ρ inter-metric heatmap\n"
        f"(includes {param_name} as a variable)",
        fontsize=11,
        pad=14,
    )

    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze degradation sweep results and produce analysis outputs."
    )
    parser.add_argument(
        "--degradation",
        required=True,
        choices=DEGRADATIONS,
        help="Degradation type to analyze (must have a corresponding results CSV).",
    )
    parser.add_argument(
        "--snr",
        type=float,
        default=None,
        help=(
            "For noise_impulsive only: filter results to rows matching this "
            "impulse_snr_db value. Ignored for all other degradations."
        ),
    )
    args = parser.parse_args()
    degradation: str = args.degradation
    snr_filter: float | None = args.snr

    df, baseline = load_data(degradation, snr_filter=snr_filter)
    metrics = get_metrics(degradation)
    severity_param = get_severity_param(df)

    if degradation == "noise_impulsive" and snr_filter is not None:
        out_dir = RESULTS_DIR / "analysis" / f"noise_impulsive_snr{snr_filter:.0f}"
    else:
        out_dir = RESULTS_DIR / "analysis" / degradation
    out_dir.mkdir(parents=True, exist_ok=True)

    variance = compute_normalized_variance(df, metrics)

    # 1. Scatter subplot grid
    scatter_path = out_dir / "scatter_grid.png"
    plot_scatter_grid(
        df, baseline, metrics, severity_param, degradation, scatter_path, variance
    )
    print(f"Saved: {scatter_path}")

    # 2. Severity correlation table (CSV + PNG)
    table = build_correlation_table(df, metrics, variance)

    csv_path = out_dir / "correlation_table.csv"
    save_correlation_table_csv(table, csv_path)
    print(f"Saved: {csv_path}")

    png_path = out_dir / "correlation_table.png"
    save_correlation_table_png(table, png_path, degradation)
    print(f"Saved: {png_path}")

    # 3. Inter-metric Spearman heatmap
    heatmap_path = out_dir / "metric_heatmap.png"
    plot_heatmap(df, metrics, heatmap_path, degradation)
    print(f"Saved: {heatmap_path}")


if __name__ == "__main__":
    main()
