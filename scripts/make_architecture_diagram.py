"""Render a simple box-and-arrow architecture diagram to reports/figures/architecture.png.

Pure matplotlib (no graphviz dependency) so it runs anywhere requirements.txt is installed.
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from config import FIGURES_DIR

STAGES = [
    ("UCI dataset\n(src/ingest.py)", "#CFE8FF"),
    ("Clean + binarise\n(src/clean.py)", "#CFE8FF"),
    ("Feature pipeline\n(src/features.py)", "#CFE8FF"),
    ("Train + tune\n3 models, MLflow\n(src/train.py)", "#FFE8B3"),
    ("Package model\njoblib + MLflow +\nregistry (package_model.py)", "#FFE8B3"),
    ("FastAPI service\n/predict /health /metrics\n(src/main.py)", "#D8F5D0"),
    ("Docker image\n(Dockerfile)", "#D8F5D0"),
    ("Kubernetes\nDeployment + Service + HPA\n(deployment/k8s)", "#E3D6FF"),
]

MONITORING = [
    ("Prometheus\n(scrapes /metrics)", "#FFD6D6"),
    ("Grafana\ndashboard", "#FFD6D6"),
]


def draw_box(ax, xy, text, color, width=2.0, height=1.1):
    x, y = xy
    box = FancyBboxPatch(
        (x - width / 2, y - height / 2), width, height,
        boxstyle="round,pad=0.08,rounding_size=0.08",
        linewidth=1.2, edgecolor="#333333", facecolor=color,
    )
    ax.add_patch(box)
    ax.text(x, y, text, ha="center", va="center", fontsize=9)


def draw_arrow(ax, start, end, style="-|>", color="#333333"):
    arrow = FancyArrowPatch(
        start, end, arrowstyle=style, mutation_scale=14,
        linewidth=1.2, color=color, shrinkA=8, shrinkB=8,
    )
    ax.add_patch(arrow)


def main() -> None:
    fig, ax = plt.subplots(figsize=(16, 5))
    n = len(STAGES)
    box_width, step = 1.9, 2.1
    first_x = box_width / 2 + 0.3
    x_positions = [first_x + i * step for i in range(n)]
    x_max = x_positions[-1] + box_width / 2 + 0.3

    ax.set_xlim(0, x_max)
    ax.set_ylim(0, 5)
    ax.axis("off")
    ax.set_title("Heart Disease Risk - MLOps Pipeline", fontsize=14, fontweight="bold")

    y_main = 3.2
    for x, (label, color) in zip(x_positions, STAGES, strict=True):
        draw_box(ax, (x, y_main), label, color, width=box_width)
    for x1, x2 in zip(x_positions, x_positions[1:], strict=False):
        draw_arrow(ax, (x1 + box_width / 2, y_main), (x2 - box_width / 2, y_main))


    # Monitoring sits below the Kubernetes box and the FastAPI box, fed by /metrics
    api_x = x_positions[5]
    k8s_x = x_positions[7]
    mon_y = 1.0
    draw_box(ax, (api_x, mon_y), MONITORING[0][0], MONITORING[0][1])
    draw_box(ax, (k8s_x, mon_y), MONITORING[1][0], MONITORING[1][1])
    draw_arrow(ax, (api_x, y_main - 0.55), (api_x, mon_y + 0.55))
    draw_arrow(ax, (api_x + 1.0, mon_y), (k8s_x - 1.0, mon_y))

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = FIGURES_DIR / "architecture.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
