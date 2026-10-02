"""Quick-look plotting helpers.

Exploratory only — the figures that end up in write-ups are built in
``notebooks/``, which reads the saved ``.nc`` recovery files rather than
calling anything here.
"""

import matplotlib.pyplot as plt

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
