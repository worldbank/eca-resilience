"""
report_plot_utils.py
-------------
Plotting utilities for spatial mobility analysis.
"""

# ============================================================
#  imports
# ============================================================
from typing import List, Optional, Tuple
 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib import cm
 
from bokeh.plotting import figure
from bokeh.palettes import Category20
from bokeh.models import (
    BoxAnnotation,
    ColumnDataSource,
    HoverTool,
    Label,
    Legend,
    LegendItem,
    Span,
)


# ============================================================
# 1. Matplotlib helpers
# ============================================================

def plot_map_gdf(
    gdf,
    col: str,
    log_transform: bool = True,
    figsize: tuple[int, int] = (10, 10),
    cmap: str = "Blues",
    alpha: float = 1.0,
    ax=None,
    fig=None,
    th: float = 0,
    show_cbar: bool = True,
    vmin=None,
    vmax=None,
):
    """
    Plot a choropleth map from a GeoDataFrame column.

    Parameters
    ----------
    gdf : GeoDataFrame
        Input GeoDataFrame with geometry and the target column.
    col : str
        Column name to visualise.
    log_transform : bool
        If True, applies log10(1 + x) before plotting.
    figsize : tuple[int, int]
        Figure size passed to ``plt.subplots`` when no axes are provided.
    cmap : str
        Matplotlib colormap name.
    alpha : float
        Fill opacity (0–1).
    ax : matplotlib.axes.Axes | None
        Existing axes to draw on. Created if None.
    fig : matplotlib.figure.Figure | None
        Existing figure (required when passing ``ax`` for the colorbar).
    th : float
        Rows with ``col <= th`` are dropped before plotting.
    show_cbar : bool
        Whether to add a horizontal colorbar.
    vmin, vmax : float | None
        Colorbar limits in *original* (pre-transform) space.
        Converted internally to transformed space.

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax  : matplotlib.axes.Axes
    cbar : matplotlib.colorbar.Colorbar | None
    """
    plot_gdf = gdf[gdf[col] > th].copy()

    if log_transform:
        plot_gdf["_plot_val"] = np.log10(1 + plot_gdf[col])
        label = f"log10(1 + {col})"
        transform = lambda x: np.log10(1 + x)
    else:
        plot_gdf["_plot_val"] = plot_gdf[col]
        label = col
        transform = lambda x: x

    values = plot_gdf["_plot_val"]

    if ax is None and fig is None:
        fig, ax = plt.subplots(figsize=figsize)

    vmin_plot = transform(vmin) if vmin is not None else values.min()
    vmax_plot = transform(vmax) if vmax is not None else values.max()

    plot_gdf.plot(
        column="_plot_val",
        cmap=cmap,
        legend=False,
        ax=ax,
        alpha=alpha,
        vmin=vmin_plot,
        vmax=vmax_plot,
    )

    cbar = None
    if show_cbar:
        norm = plt.Normalize(vmin=vmin_plot, vmax=vmax_plot)
        sm = cm.ScalarMappable(cmap=cmap, norm=norm)
        sm._A = []
        cbar = fig.colorbar(
            sm, ax=ax,
            fraction=0.015, pad=0.1,
            orientation="horizontal",
        )
        cbar.set_label(label)

    return fig, ax, cbar


def plot_time_series(
    df_plot: pd.DataFrame,
    axes=None,
    color: str = "k",
    plot_avg: bool = False,
    label: str = "",
):
    """
    Three-panel time series: GPS observations, active users, visited hexes.

    Parameters
    ----------
    df_plot : pd.DataFrame
        Must contain columns: ``date``, ``n_points_count``,
        ``uid_unique``, ``hex_id_unique``.
    axes : array-like of matplotlib.axes.Axes | None
        Three existing axes to draw on. Created if None.
    color : str
        Line color.
    plot_avg : bool
        If True, draws a horizontal dashed line at the column mean.
    label : str
        Legend label applied to the first panel line.

    Returns
    -------
    axes : np.ndarray of matplotlib.axes.Axes
    """
    if axes is None:
        _, axes = plt.subplots(nrows=3, ncols=1, figsize=(10, 10), sharex=False)

    panel_cfg = [
        ("n_points_count", "GPS observations"),
        ("uid_unique",      "Active users"),
        ("hex_id_unique",   "Visited hexes"),
    ]

    for ax, (col, ylabel) in zip(axes, panel_cfg):
        kw = dict(c=color, linewidth=1, marker=".")
        ax.plot(df_plot["date"], df_plot[col], label=label if col == "n_points_count" else "", **kw)
        if plot_avg:
            ax.axhline(np.mean(df_plot[col]), c=color, linewidth=1, linestyle=":")
        ax.set_ylabel(ylabel, fontweight=600)
        ax.tick_params(axis="x", labelsize=8)
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))

    return axes


# ============================================================
# 2. Bokeh helpers
# ============================================================

def _add_common_annotations(
    p,
    z_scores: pd.DataFrame,
    event_start,
    event_end,
    event_label_text: str = "Event",
    anomaly_thresholds: tuple[int, ...] = (2, 3),
    anomaly_labels: tuple[str, ...] = ("anomaly", "extreme anomaly"),
    y_event_label: float = 4.0,
):
    """
    Add shared annotations to a Bokeh figure:
    positive/negative anomaly threshold lines, event shading box, zero baseline.

    Parameters
    ----------
    p : bokeh.plotting.figure
        Target figure.
    z_scores : pd.DataFrame
        Used only to anchor label x-positions (first column timestamp).
    event_start, event_end : datetime-like
        Left/right bounds of the event shading box.
    event_label_text : str
        Text shown on the event shading label.
    anomaly_thresholds : tuple[int]
        Absolute z-score thresholds to draw horizontal lines at.
    anomaly_labels : tuple[str]
        Labels corresponding to each threshold.
    y_event_label : float
        Y position of the event label.
    """
    x_anchor = pd.to_datetime(z_scores.columns[0])

    for threshold, label_text in zip(anomaly_thresholds, anomaly_labels):
        for sign, color in [(1, "red"), (-1, "blue")]:
            p.add_layout(Span(
                location=sign * threshold,
                dimension="width",
                line_color=color,
                line_dash="dotted",
                line_width=1,
            ))
            p.add_layout(Label(
                x=x_anchor,
                y=sign * threshold,
                text=label_text,
                text_font_size="8pt",
                text_align="left",
                text_baseline="middle",
                text_font_style="bold",
            ))

    p.add_layout(BoxAnnotation(
        left=event_start,
        right=event_end,
        fill_color="orange",
        fill_alpha=0.12,
    ))
    p.add_layout(Label(
        x=event_start,
        y=y_event_label,
        text=event_label_text,
        text_color="orange",
        text_font_size="9pt",
        text_font_style="bold",
        text_align="right",
    ))

    p.add_layout(Span(
        location=0,
        dimension="width",
        line_color="black",
        line_dash="dashed",
        line_width=1,
    ))


def plot_zscore_by_spatial_feature(
    gdf_h3_landuse,
    z_scores,
    event_start,
    event_end,
    group_col=None,
    group_layers=None,
    exclude_labels=("other", "none"),
    color_map=None,
    group_label_fn=None,
    hover_group_name="Group",
    event_label_text="Event",
    title="Z-scores by Spatial Feature",
    width=770,
    height=400,
    legend_ncols=2,
):
    """
    Unified z-score time series plot grouped by any spatial feature.

    Exactly one grouping mode must be supplied:

    - ``group_col``    : categorical column in the GDF (e.g. land-use type)
    - ``group_layers`` : list of binary/count columns in the GDF (e.g. POI layers)

    Parameters
    ----------
    gdf_h3_landuse : pd.DataFrame
        GeoDataFrame with a ``hex_id`` column plus the grouping column(s).
    z_scores : pd.DataFrame
        DataFrame indexed by hex_id; columns are datetime timestamps.
    event_start, event_end : datetime-like
        Bounds of the event shading annotation.
    group_col : str | None
        Categorical column to group by (land-use mode).
    group_layers : list[str] | None
        Binary/count columns to group by (POI mode).
    exclude_labels : tuple[str]
        Category values to drop when using ``group_col`` mode.
    color_map : dict | None
        ``{group_name: color}`` overrides. Falls back to built-in land-use
        palette or Category20 for POI layers.
    group_label_fn : callable | None
        ``fn(raw_name: str) -> str`` to format legend labels.
        Defaults: ``str.title()`` for group_col; prefix-stripped title for POI.
    hover_group_name : str
        Tooltip label for the group field (e.g. "Land Use", "Layer").
    event_label_text : str
        Text on the event shading annotation.
    title : str
        Figure title.
    width, height : int
        Figure dimensions in pixels.
    legend_ncols : int
        Number of legend columns.

    Returns
    -------
    bokeh.plotting.figure
        Call ``show(p)`` in the notebook to render.

    Examples
    --------
    >>> p = plot_zscore_by_spatial_feature(
    ...     gdf_h3_landuse, z_scores,
    ...     event_start=EVENT_START, event_end=EVENT_END,
    ...     group_col="land_use",
    ...     hover_group_name="Land Use",
    ...     event_label_text="Republic Day",
    ... )
    >>> show(p)

    >>> p = plot_zscore_by_spatial_feature(
    ...     gdf_h3_landuse, z_scores,
    ...     event_start=EVENT_START, event_end=EVENT_END,
    ...     group_layers=layers_POI,
    ...     hover_group_name="Layer",
    ...     event_label_text="Republic Day",
    ... )
    >>> show(p)
    """
    if group_col is None and group_layers is None:
        raise ValueError("Provide either 'group_col' or 'group_layers'.")
    if group_col is not None and group_layers is not None:
        raise ValueError("'group_col' and 'group_layers' are mutually exclusive.")

    # ------------------------------------------------------------------
    # Build {group_name: set_of_hex_ids} — the only mode-specific step
    # ------------------------------------------------------------------
    if group_col is not None:
        gdf_filtered = gdf_h3_landuse[
            (~gdf_h3_landuse[group_col].isin(exclude_labels))
            & gdf_h3_landuse[group_col].notna()
        ]
        groups = {
            cat: set(gdf_filtered[gdf_filtered[group_col] == cat]["hex_id"])
            for cat in gdf_filtered[group_col].unique()
        }
    else:
        groups = {
            layer: set(gdf_h3_landuse[gdf_h3_landuse[layer] > 0]["hex_id"])
            for layer in group_layers
        }

    # ------------------------------------------------------------------
    # Color resolution
    # ------------------------------------------------------------------
    _default_landuse_colors = {
        "residential": "#F4A261", "commercial": "#E76F51",
        "industrial":  "#6D597A", "education":  "blue",
        "construction": "darkgrey", "farmland": "#90BE6D",
        "green": "#2A9D8F", "water": "#4A90E2", "other": "#EEEEEE",
    }
    palette = Category20[20]

    def _resolve_color(name: str, idx: int) -> str:
        if color_map and name in color_map:
            return color_map[name]
        if group_col is not None:
            return _default_landuse_colors.get(name, "grey")
        return palette[idx % 20]

    # ------------------------------------------------------------------
    # Legend label formatting
    # ------------------------------------------------------------------
    def _default_poi_label(name: str) -> str:
        return (
            name.replace("n_", "").replace("is_", "")
                .replace("way", "ways").title()
        )

    if group_label_fn is None:
        group_label_fn = str.title if group_col is not None else _default_poi_label

    # ------------------------------------------------------------------
    # Figure + lines
    # ------------------------------------------------------------------
    p = figure(
        width=width, height=height,
        x_axis_type="datetime",
        title=title,
        tools="pan,wheel_zoom,box_zoom,reset,save,hover",
        active_scroll="wheel_zoom",
    )

    legend_items = []

    for idx, (group_name, hex_ids) in enumerate(groups.items()):
        subset = z_scores[z_scores.index.isin(hex_ids)]
        if subset.empty:
            continue

        mean_series = subset.mean()
        source = ColumnDataSource(data=dict(
            x=pd.to_datetime(mean_series.index),
            y=mean_series.values,
            group=[group_name] * len(mean_series),
        ))

        r = p.line(
            "x", "y", source=source,
            line_width=2,
            color=_resolve_color(group_name, idx),
            alpha=0.9,
        )
        legend_items.append(LegendItem(label=group_label_fn(group_name), renderers=[r]))

    # ------------------------------------------------------------------
    # Hover tool
    # ------------------------------------------------------------------
    hover = p.select_one(HoverTool)
    hover.tooltips = [
        ("Date",           "@x{%F}"),
        ("Z-score",        "@y{0.00}"),
        (hover_group_name, "@group"),
    ]
    hover.formatters = {"@x": "datetime"}
    hover.mode = "vline"

    # ------------------------------------------------------------------
    # Annotations, axes, legend, grid
    # ------------------------------------------------------------------
    _add_common_annotations(p, z_scores, event_start, event_end, event_label_text)

    p.xaxis.axis_label = "Date"
    p.yaxis.axis_label = "Z-score"
    p.xaxis.axis_label_text_font_style = "bold"
    p.yaxis.axis_label_text_font_style = "bold"

    legend = Legend(
        items=legend_items,
        location="top_left",
        orientation="vertical",
        label_text_font_size="8pt",
        label_text_font_style="bold",
    )
    legend.ncols = legend_ncols
    p.add_layout(legend)
    p.legend.click_policy = "hide"

    p.grid.grid_line_alpha = 0.25

    return p
