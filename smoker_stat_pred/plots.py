import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Optional, List, Union


# ============================================================
# BASIC PLOTS
# ============================================================

def plot_scatter(df: pd.DataFrame, xcol: str, ycol: str,
                 color: Optional[str] = None,
                 title: str = "Scatter Plot"):
    fig = px.scatter(df, x=xcol, y=ycol, color=color, title=title)
    return fig


def plot_line(df: pd.DataFrame, xcol: str, ycol: str,
              color: Optional[str] = None,
              title: str = "Line Plot"):
    fig = px.line(df, x=xcol, y=ycol, color=color, title=title)
    return fig


def plot_bar(df: pd.DataFrame, xcol: str, ycol: Optional[str] = None,
             color: Optional[str] = None,
             title: str = "Bar Chart"):
    fig = px.bar(df, x=xcol, y=ycol, color=color, title=title)
    return fig


def plot_histogram(df: pd.DataFrame, xcol: str,
                   color: Optional[str] = None,
                   nbins: int = 30,
                   title: str = "Histogram"):
    fig = px.histogram(df, x=xcol, color=color, nbins=nbins, title=title)
    return fig


def plot_box(df: pd.DataFrame, ycol: str,
             xcol: Optional[str] = None,
             color: Optional[str] = None,
             title: str = "Box Plot"):
    fig = px.box(df, x=xcol, y=ycol, color=color, title=title)
    return fig


def plot_violin(df: pd.DataFrame, ycol: str,
                xcol: Optional[str] = None,
                color: Optional[str] = None,
                box: bool = True,
                title: str = "Violin Plot"):
    fig = px.violin(df, x=xcol, y=ycol, color=color, box=box, title=title)
    return fig


# ============================================================
# EXTRA PLOTS
# ============================================================

def plot_heatmap(df: pd.DataFrame,
                 xcol: str,
                 ycol: str,
                 zcol: str,
                 title: str = "Heatmap"):
    fig = px.density_heatmap(df, x=xcol, y=ycol, z=zcol, title=title)
    return fig


def plot_correlation_matrix(df: pd.DataFrame,
                            columns: Optional[List[str]] = None,
                            title: str = "Correlation Matrix"):
    if columns is None:
        columns = df.select_dtypes(include="number").columns.tolist()

    corr = df[columns].corr()

    fig = px.imshow(corr,
                    text_auto=True,
                    title=title,
                    color_continuous_scale="RdBu_r")
    return fig


def plot_density_contour(df: pd.DataFrame,
                         xcol: str, ycol: str,
                         color: Optional[str] = None,
                         title: str = "2D Density Contour"):
    fig = px.density_contour(df, x=xcol, y=ycol, color=color, title=title)
    return fig


def plot_ecdf(df: pd.DataFrame,
              xcol: str,
              color: Optional[str] = None,
              title: str = "ECDF"):
    fig = px.ecdf(df, x=xcol, color=color, title=title)
    return fig


def plot_scatter_matrix(df: pd.DataFrame,
                        columns: Optional[List[str]] = None,
                        title: str = "Scatter Matrix"):
    if columns is None:
        columns = df.select_dtypes(include="number").columns.tolist()

    fig = px.scatter_matrix(df[columns], title=title)
    return fig


# ============================================================
# DISPATCHER FUNCTION
# ============================================================

def plot(df: pd.DataFrame,
         xcol: Optional[str] = None,
         ycol: Optional[str] = None,
         plot_type: str = "scatter",
         save: bool = False,
         save_path: str = "plot.html",
         **kwargs):

    plot_type = plot_type.lower()

    mapping = {
        "scatter": plot_scatter,
        "line": plot_line,
        "bar": plot_bar,
        "hist": plot_histogram,
        "histogram": plot_histogram,
        "box": plot_box,
        "violin": plot_violin,
        "heatmap": plot_heatmap,
        "corr": plot_correlation_matrix,
        "correlation": plot_correlation_matrix,
        "density": plot_density_contour,
        "kde": plot_density_contour,
        "ecdf": plot_ecdf,
        "matrix": plot_scatter_matrix,
        "scatter_matrix": plot_scatter_matrix,
    }

    if plot_type not in mapping:
        raise ValueError(f"Unknown plot type: {plot_type}. Available: {list(mapping.keys())}")

    func = mapping[plot_type]

    # Call the appropriate plot function
    if plot_type in ["corr", "correlation", "matrix", "scatter_matrix"]:
        fig = func(df, **kwargs)
    elif plot_type == "heatmap":
        fig = func(df, xcol, ycol, kwargs.get("zcol"), **kwargs)
    elif plot_type in ["histogram", "hist"]:
        fig = func(df, xcol, **kwargs)
    elif plot_type == "box":
        fig = func(df, ycol=ycol, xcol=xcol, **kwargs)
    elif plot_type == "violin":
        fig = func(df, ycol=ycol, xcol=xcol, **kwargs)
    else:
        fig = func(df, xcol, ycol, **kwargs)

    # Save if needed
    if save:
        fig.write_html(save_path)

    return fig