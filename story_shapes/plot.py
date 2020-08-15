from typing import List

from matplotlib import pyplot as plt

def plot_story_shape(sentiments: List[float], title: str, shape_path: str) -> None:
    with plt.xkcd():
        ax = plt.gca()
        ax.plot(sentiments, color="black")
        ax.set_xlim(-0.5, len(sentiments))
        ax.set_ylim(-1, 1)
        ax.axis("off")
        ax.axhline(0, color="dimgray")
        ax.axvline(0, color="dimgray")
        ax.annotate(
            "BEGINNING",
            xy=(-0.25, 0.5),
            ha="left",
            va="center",
            fontsize=12,
            xycoords="axes fraction",
            color="dimgray",
        )
        ax.annotate(
            "END",
            xy=(1.1, 0.5),
            ha="right",
            va="center",
            fontsize=12,
            xycoords="axes fraction",
            color="dimgray",
        )
        ax.annotate(
            "GOOD FORTUNE",
            xy=(0, 1.05),
            ha="center",
            va="center",
            fontsize=12,
            xycoords="axes fraction",
            color="dimgray",
        )
        ax.annotate(
            "ILL FORTUNE",
            xy=(0, -0.05),
            ha="center",
            va="center",
            fontsize=12,
            xycoords="axes fraction",
            color="dimgray",
        )
        ax.annotate(
            title.upper(),
            xy=(0.5, -0.1),
            va="bottom",
            fontsize=16,
            xycoords="axes fraction",
        )
        plt.tight_layout()
        plt.savefig(shape_path)
