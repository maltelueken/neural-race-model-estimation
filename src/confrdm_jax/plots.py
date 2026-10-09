"""Plotting helpers.

:func:`save_figure` is how every ``scripts/create_figure*.py`` script writes its figures, so
they all meet the minimum size in ``conf_jax/figures.yaml``. The rest is exploratory only —
the figures that end up in write-ups read the saved ``.nc`` recovery files rather than
calling anything here.
"""

import math

import matplotlib.pyplot as plt
from PIL import Image

from .runs import figure_settings


def save_figure(fig_or_grid, path, dpi=200, min_pixels=None):
    """Save a figure with a tight bounding box, at least ``min_pixels`` on each side.

    The figure keeps its layout: if it would come out smaller than ``min_pixels`` on its
    short side at ``dpi``, it is rendered at a higher dpi instead, so a wide strip of panels
    grows on both sides rather than being stretched.

    Args:
        fig_or_grid: A matplotlib ``Figure``, or anything with a ``.figure`` such as a
            seaborn ``FacetGrid``. Closed after saving.
        path: Output path. The size is checked by reading the file back, so it must be a
            raster format.
        dpi: Resolution used when it already meets ``min_pixels``.
        min_pixels: Minimum width and height in pixels. Defaults to ``min_pixels`` in
            ``conf_jax/figures.yaml``.

    Raises:
        RuntimeError: If the saved file is still smaller than ``min_pixels``.
    """
    fig = getattr(fig_or_grid, "figure", fig_or_grid)
    if min_pixels is None:
        min_pixels = figure_settings()["min_pixels"]

    # The tight bounding box is not quite the same at every dpi (text is laid out per
    # resolution), so the size is measured on the saved file and the dpi raised until it fits.
    for _ in range(3):
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        with Image.open(path) as image:
            width, height = image.size
        short_side = min(width, height)
        if short_side >= min_pixels:
            break
        # +1 so that rounding the pixel size down cannot land one short.
        dpi = math.ceil(dpi * (min_pixels + 1) / short_side)
    plt.close(fig)

    if short_side < min_pixels:
        raise RuntimeError(f"{path} is {width}x{height} px, below the minimum of {min_pixels}")


def plot_data_hist(data):
    """Overlay RT histograms for the two responses.

    Args:
        data: Array whose last axis holds ``[rt, resp]``; leading axes are
            flattened by the boolean indexing.

    Returns:
        ``(fig, ax)``.

    Note:
        Censored trials carry ``rt = -1.0`` and ``resp = -1``, so they fall in
        neither series and are silently dropped. Bin edges are chosen per
        series, so the two histograms are not directly comparable in height.
    """
    rt = data[..., 0]
    choice = data[..., 1]

    fig, ax = plt.subplots()

    ax.hist(rt[choice == 1], color="darkgreen", alpha=0.5)
    ax.hist(rt[choice == 0], color="indianred", alpha=0.5)

    return fig, ax
