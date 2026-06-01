"""Attention Weight Visualization Tools.

Provides matplotlib-based visualization for:
- Channel weight bar charts per task
- Cross-modal attention heatmaps
- Multi-task sensitivity comparison
- Export to JSON/PNG

See ADR-0006 for architectural rationale.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# Optional: matplotlib may not be installed
_MATPLOTLIB_AVAILABLE = False
try:
    import matplotlib
    matplotlib.use("Agg")  # Non-interactive backend
    import matplotlib.pyplot as plt
    import matplotlib.figure
    from matplotlib.figure import Figure
    _MATPLOTLIB_AVAILABLE = True
except ImportError:
    Figure = Any  # type fallback
    logger.info("matplotlib not installed; visualization will return None.")


class AttentionVisualizer:
    """Visualization toolkit for attention weights and channel sensitivities.

    Parameters
    ----------
    figsize : tuple
        Default figure size (width, height) in inches.
    style : str
        Matplotlib style (e.g., "seaborn-v0_8-darkgrid").
    """

    DEFAULT_CHANNEL_COLORS = {
        "semantic": "#2E86AB",
        "metadata": "#A23B72",
        "topology": "#F18F01",
        "temporal": "#C73E1D",
    }

    def __init__(
        self,
        figsize: tuple = (8, 5),
        style: Optional[str] = None,
    ):
        self.figsize = figsize

        if _MATPLOTLIB_AVAILABLE and style:
            try:
                plt.style.use(style)
            except Exception:
                pass

    def plot_channel_weights(
        self,
        weights: Dict[str, float],
        task_name: str = "",
        title: Optional[str] = None,
        output_path: Optional[str] = None,
        figsize: Optional[tuple] = None,
    ) -> Optional[Figure]:
        """Plot channel weights as a horizontal bar chart.

        Parameters
        ----------
        weights : Dict[str, float]
            Channel name → weight mapping.
        task_name : str
            Task name for title.
        title : str, optional
            Custom title.
        output_path : str, optional
            If provided, save figure to this path.
        figsize : tuple, optional
            Override default figure size.

        Returns
        -------
        matplotlib.figure.Figure or None
            The figure, or None if matplotlib unavailable.
        """
        if not _MATPLOTLIB_AVAILABLE:
            return None

        channels = list(weights.keys())
        values = list(weights.values())

        fig, ax = plt.subplots(figsize=figsize or self.figsize)

        colors = [self.DEFAULT_CHANNEL_COLORS.get(c, "#888888") for c in channels]
        bars = ax.barh(channels, values, color=colors, edgecolor="white", height=0.6)

        # Add value labels on bars
        for bar, val in zip(bars, values):
            ax.text(
                val + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{val:.2f}",
                va="center", ha="left", fontsize=10, color="#333333"
            )

        ax.set_xlim(0, max(values) * 1.2)
        ax.set_xlabel("Attention Weight", fontsize=11)
        ax.set_title(
            title or f"Channel Weights — {task_name}",
            fontsize=12, fontweight="bold"
        )
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="x", alpha=0.3)

        fig.tight_layout()

        if output_path:
            fig.savefig(output_path, dpi=150, bbox_inches="tight")
            logger.info("Saved channel weights to %s", output_path)

        return fig

    def plot_attention_heatmap(
        self,
        attn_weights: np.ndarray,
        title: str = "Attention Heatmap",
        labels: Optional[List[str]] = None,
        output_path: Optional[str] = None,
        figsize: Optional[tuple] = None,
        cmap: str = "Blues",
    ) -> Optional[Figure]:
        """Plot cross-modal attention weights as a heatmap.

        Parameters
        ----------
        attn_weights : np.ndarray
            2D array of attention weights (n_heads x n_heads or n x n).
        title : str
            Heatmap title.
        labels : List[str], optional
            Tick labels for axes.
        output_path : str, optional
            Save path.
        figsize : tuple, optional
            Override default size.
        cmap : str
            Colormap name.

        Returns
        -------
        matplotlib.figure.Figure or None
        """
        if not _MATPLOTLIB_AVAILABLE:
            return None

        fig, ax = plt.subplots(figsize=figsize or self.figsize)

        im = ax.imshow(attn_weights, cmap=cmap, aspect="auto", vmin=0, vmax=1)

        # Colorbar
        fig.colorbar(im, ax=ax, label="Attention Weight")

        # Labels
        if labels:
            ax.set_xticks(np.arange(len(labels)))
            ax.set_yticks(np.arange(len(labels)))
            ax.set_xticklabels(labels, rotation=45, ha="right")
            ax.set_yticklabels(labels)

        ax.set_title(title, fontsize=12, fontweight="bold")
        fig.tight_layout()

        if output_path:
            fig.savefig(output_path, dpi=150, bbox_inches="tight")
            logger.info("Saved attention heatmap to %s", output_path)

        return fig

    def plot_sensitivity_comparison(
        self,
        reports: List[Dict[str, Any]],
        output_path: Optional[str] = None,
        figsize: Optional[tuple] = None,
    ) -> Optional[Figure]:
        """Plot sensitivity comparison across multiple tasks.

        Parameters
        ----------
        reports : List[Dict]
            List of sensitivity reports from compute_task_sensitivity_report.
        output_path : str, optional
            Save path.
        figsize : tuple, optional
            Override default size.

        Returns
        -------
        matplotlib.figure.Figure or None
        """
        if not _MATPLOTLIB_AVAILABLE:
            return None

        channels = ["semantic", "metadata", "topology", "temporal"]
        n_tasks = len(reports)
        n_channels = len(channels)

        fig, ax = plt.subplots(figsize=figsize or (10, 5))

        x = np.arange(n_channels)
        width = 0.8 / n_tasks

        for i, report in enumerate(reports):
            sens = report["channel_sensitivities"]
            values = [sens.get(c, 0.0) for c in channels]
            offset = (i - n_tasks / 2 + 0.5) * width

            bars = ax.bar(
                x + offset,
                values,
                width,
                label=report.get("task_name", f"Task {i}"),
                color=[
                    self.DEFAULT_CHANNEL_COLORS.get(c, "#888888")
                    for c in channels
                ],
                alpha=0.8,
            )

        ax.set_xticks(x)
        ax.set_xticklabels(channels, fontsize=10)
        ax.set_ylabel("Sensitivity", fontsize=11)
        ax.set_title("Channel Sensitivity Across Tasks", fontsize=12, fontweight="bold")
        ax.legend(fontsize=9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", alpha=0.3)

        fig.tight_layout()

        if output_path:
            fig.savefig(output_path, dpi=150, bbox_inches="tight")
            logger.info("Saved sensitivity comparison to %s", output_path)

        return fig

    def export_sensitivity_report(
        self,
        report: Dict[str, Any],
        output_path: str,
    ) -> None:
        """Export a sensitivity report to JSON.

        Parameters
        ----------
        report : Dict[str, Any]
            Sensitivity report dict.
        output_path : str
            JSON file path.
        """
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        logger.info("Exported sensitivity report to %s", output_path)

    def plot_channel_weight_evolution(
        self,
        weight_history: List[Dict[str, float]],
        task_name: str = "",
        output_path: Optional[str] = None,
        figsize: Optional[tuple] = None,
    ) -> Optional[Figure]:
        """Plot how channel weights evolve during training.

        Parameters
        ----------
        weight_history : List[Dict[str, float]]
            List of weight snapshots over training steps.
        task_name : str
            Task name.
        output_path : str, optional
            Save path.
        figsize : tuple, optional
            Override default size.

        Returns
        -------
        matplotlib.figure.Figure or None
        """
        if not _MATPLOTLIB_AVAILABLE or not weight_history:
            return None

        channels = list(weight_history[0].keys())
        steps = np.arange(len(weight_history))

        fig, ax = plt.subplots(figsize=figsize or (10, 5))

        for channel in channels:
            values = [w.get(channel, 0.0) for w in weight_history]
            ax.plot(
                steps,
                values,
                label=channel,
                color=self.DEFAULT_CHANNEL_COLORS.get(channel, "#888888"),
                linewidth=2,
            )

        ax.set_xlabel("Training Step", fontsize=11)
        ax.set_ylabel("Attention Weight", fontsize=11)
        ax.set_title(f"Channel Weight Evolution — {task_name}", fontsize=12, fontweight="bold")
        ax.legend(fontsize=9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(alpha=0.3)

        fig.tight_layout()

        if output_path:
            fig.savefig(output_path, dpi=150, bbox_inches="tight")
            logger.info("Saved weight evolution to %s", output_path)

        return fig
