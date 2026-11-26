"""
Shared matplotlib style + fixed colour assignment so every figure in the
project (and the UI) uses the same colour for the same model / class.
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

from src.config import FIG_DIR  # noqa: E402

# Chart chrome
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

# Categorical slots (fixed order)
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
         "#e87ba4", "#008300", "#6250d6", "#e34948"]

# Colour follows the entity: each model always gets the same colour.
# The plain Logistic Regression is a reference baseline -> neutral grey.
MODEL_COLORS = {
    "Logistic Regression": MUTED,
    "PCA + Logistic Regression": SLOTS[0],
    "Decision Tree": SLOTS[1],
    "Random Forest": SLOTS[2],
    "AdaBoost": SLOTS[3],
    "Gradient Boosting": SLOTS[4],
    "XGBoost": SLOTS[5],
    "LightGBM": SLOTS[6],
    "CatBoost": SLOTS[7],
}
MODEL_STYLE = {"Logistic Regression": "--"}

CLASS_COLORS = {0: SLOTS[0], 1: SLOTS[1]}           # repaid = blue, default = orange
CLASS_LABELS = {0: "Fully paid", 1: "Defaulted"}

SEQ_BLUE = LinearSegmentedColormap.from_list(
    "seq_blue", ["#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#184f95", "#0d366b"])
DIVERGING = LinearSegmentedColormap.from_list(
    "div_blue_red", ["#184f95", "#6da7ec", "#f0efec", "#ec8a89", "#b02a2a"])


def set_style() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": AXIS,
        "axes.labelcolor": INK_2,
        "axes.titlecolor": INK,
        "axes.titleweight": "bold",
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "axes.axisbelow": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.frameon": False,
        "legend.fontsize": 9,
        "lines.linewidth": 1.6,
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "figure.dpi": 110,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
    })


def save(fig, name: str, subdir: str | None = None) -> None:
    out = FIG_DIR / subdir if subdir else FIG_DIR
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{name}.png")
    plt.close(fig)


set_style()
