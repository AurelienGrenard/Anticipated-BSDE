"""Recreate the six paper figures from the selected checkpoints and summaries."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import torch

from .checkpoints import load_result
from .evaluation import evaluate
from .problems import get_example


PROJECT_ROOT = Path(__file__).resolve().parent
FIGURE_ROOT = PROJECT_ROOT / "figures"
RESULT_ROOT = PROJECT_ROOT / "results"
SPECS = {
    "4.3": "example_4_3",
    "5.3.1": "example_5_3_1",
    "5.3.2": "example_5_3_2",
}

PAPER_STYLE = {
    "font.size": 10.5,
    "axes.labelsize": 11.5,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
}


def plot_quantiles(key: str, slug: str, device: str) -> None:
    """Plot exact quantile lines and learned Hermite quantile markers."""
    result = load_result(
        RESULT_ROOT / "checkpoints" / f"{slug}_n20_seed42.pt", device
    )
    audit = evaluate(
        result.models,
        get_example(key),
        result.config,
        device,
        test_paths=50_000,
        seed=3_000_003,
        keep_paths=True,
    )
    levels = torch.tensor([0.05, 0.50, 0.95])
    time = audit.time_grid.numpy()
    colors = ("#D05A47", "#176B87", "#D05A47")
    labels = ("5% quantile", "median", "95% quantile")
    figure, axes = plt.subplots(1, 2, figsize=(7.2, 3.35), constrained_layout=True)

    for axis, name, learned, exact in (
        (axes[0], "Y", audit.y_prediction, audit.y_exact),
        (axes[1], "Z", audit.z_prediction, audit.z_exact),
    ):
        learned_q = torch.quantile(learned, levels, dim=0).numpy()
        exact_q = torch.quantile(exact, levels, dim=0).numpy()
        axis.fill_between(time, exact_q[0], exact_q[2], color="#176B87", alpha=0.07)
        coincident_quantiles = bool(
            torch.max(torch.abs(torch.as_tensor(exact_q[2] - exact_q[0]))) < 1e-10
        )
        for index, color in enumerate(colors):
            axis.plot(
                time,
                exact_q[index],
                color=color,
                linewidth=2.5 if index == 1 else 1.5,
            )
            axis.plot(
                time,
                learned_q[index],
                linestyle="none",
                marker="o",
                markerfacecolor="white",
                markeredgecolor=color,
                markeredgewidth=1.25,
                markersize=4.8,
                markevery=2,
            )
            if not coincident_quantiles:
                axis.annotate(
                    labels[index],
                    (time[-1], exact_q[index, -1]),
                    xytext=(5, 0),
                    textcoords="offset points",
                    color=color,
                    va="center",
                    fontsize=9.5,
                    clip_on=False,
                )
        if coincident_quantiles:
            axis.annotate(
                "5%, 50%, and 95% quantiles coincide",
                (time[-1], exact_q[1, -1]),
                xytext=(-5, 9),
                textcoords="offset points",
                color="#176B87",
                ha="right",
                va="bottom",
                fontsize=9.5,
            )
        axis.text(
            0.03,
            0.96,
            f"({name})",
            transform=axis.transAxes,
            ha="left",
            va="top",
            fontsize=11.5,
            fontweight="semibold",
        )
        axis.set(xlabel="Time $t$", ylabel=f"${name}_t$")
        axis.margins(x=0.04)
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", color="#C9D1D9", alpha=0.55, linewidth=0.7)

    figure.legend(
        handles=[
            Line2D([], [], color="#252A34", linewidth=2, label="Exact solution"),
            Line2D(
                [],
                [],
                color="#252A34",
                linestyle="none",
                marker="o",
                markerfacecolor="white",
                markeredgewidth=1.25,
                label="Hermite-3 regression",
            ),
        ],
        loc="outside lower center",
        ncol=2,
        frameon=False,
    )
    figure.savefig(
        FIGURE_ROOT / f"{slug}_y_z_quantiles.png",
        dpi=220,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(figure)


def plot_convergence(slug: str) -> None:
    """Plot the two Theorem 5.5 errors on the retained six time grids."""
    rows = sorted(
        json.loads((RESULT_ROOT / f"{slug}_reconstruction_summary.json").read_text()),
        key=lambda row: row["h"],
    )
    h = [row["h"] for row in rows]
    figure, axis = plt.subplots(figsize=(5.35, 3.7), constrained_layout=True)
    for metric, label, color, marker in (
        ("y_mse_sup", r"$\max_t\,\mathbb{E}|Y_t-\bar Y_t^h|^2$", "#176B87", "o"),
        (
            "z_mse_integrated",
            r"$\mathbb{E}\int_0^T|Z_t-\bar Z_t^h|^2\,dt$",
            "#D05A47",
            "s",
        ),
    ):
        axis.errorbar(
            h,
            [row[f"{metric}_mean"] for row in rows],
            yerr=[row[f"{metric}_std"] for row in rows],
            color=color,
            marker=marker,
            markersize=6,
            linewidth=2.2,
            capsize=3,
            label=label,
        )
    axis.set(
        xscale="log",
        yscale="log",
        xlabel="Time step $h$",
        ylabel="Squared error",
    )
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(which="major", color="#C9D1D9", alpha=0.55, linewidth=0.8)
    axis.grid(which="minor", color="#C9D1D9", alpha=0.20, linewidth=0.5)
    axis.legend(frameon=False, loc="upper left")
    figure.savefig(
        FIGURE_ROOT / f"{slug}_theorem_5_5.png",
        dpi=220,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(figure)


def main() -> None:
    """Regenerate both figures for each benchmark example."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    FIGURE_ROOT.mkdir(parents=True, exist_ok=True)
    with plt.rc_context(PAPER_STYLE):
        for key, slug in SPECS.items():
            plot_quantiles(key, slug, device)
            plot_convergence(slug)


if __name__ == "__main__":
    main()
