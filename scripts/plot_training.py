"""Render a training run's CSV as the figure used in the README.

Kept separate from the trainer, and importing matplotlib only when run, so
training never depends on a plotting library being installed.

    python scripts/plot_training.py --run runs/main --out results/curves.png

Four panels, because four different questions get asked of a training run: how
good did it get, how many of them can drive, how far round do they get, and how
quickly. Each panel is one measure on one axis - never two scales sharing a
frame.

The two series that recur - the best individual and the population mean - are
identified once in a figure-level legend rather than four times in panel
legends that would sit on top of the data.
"""

import argparse
import csv
import os

import _bootstrap  # noqa: F401

# Surface and ink are fixed rather than following a theme: the PNG is embedded
# in a README that may be read on a light or a dark page, and a committed light
# surface reads correctly on both.
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#8a8880"
GRID = "#e6e5e1"

# Categorical slots 1 and 2 from the validated palette; at most two series share
# a panel, so the pair only ever has to separate from each other.
BLUE = "#2a78d6"
ORANGE = "#eb6834"
BAND = "#cddef5"


def read_run(path):
    with open(path, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        raise SystemExit("no generations logged in %s" % path)
    return rows


def column(rows, name, cast=float, default=None):
    values = []
    for row in rows:
        raw = row.get(name, "")
        values.append(default if raw in ("", None) else cast(raw))
    return values


def running_best(values, better=min):
    """The record so far, so the reader sees progress rather than variance."""
    out = []
    record = None
    for value in values:
        if value is not None:
            record = value if record is None else better(record, value)
        out.append(record)
    return out


def style_axes(ax, title, ylabel):
    ax.set_title(title, color=INK, fontsize=11.5, loc="left", pad=8)
    ax.set_xlabel("generation", color=INK_SECONDARY, fontsize=9)
    ax.set_ylabel(ylabel, color=INK_SECONDARY, fontsize=9)
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_MUTED, labelsize=8.5, length=0)


def label_last(ax, xs, ys, text, colour):
    """Direct-label the final point, so the eye need not travel to an axis."""
    points = [(x, y) for x, y in zip(xs, ys) if y is not None]
    if not points:
        return
    x, y = points[-1]
    ax.annotate(text, xy=(x, y), xytext=(5, 0), textcoords="offset points",
                color=colour, fontsize=9.5, va="center", fontweight="bold",
                annotation_clip=False)


def plot(rows, out_path, title=None):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    generations = column(rows, "generation", int)
    population = int(rows[0]["evaluations"]) // 2

    figure, axes = plt.subplots(2, 2, figsize=(11.5, 8.2), facecolor=SURFACE)
    # Generous top margin: the title block and the shared legend live above the
    # panels, so nothing has to compete with a panel title for space.
    figure.subplots_adjust(hspace=0.45, wspace=0.24,
                           left=0.075, right=0.955, top=0.795, bottom=0.085)

    # --- fitness -----------------------------------------------------------
    ax = axes[0][0]
    best = column(rows, "fitness_best")
    mean = column(rows, "fitness_mean")
    ax.fill_between(generations, column(rows, "fitness_p25"),
                    column(rows, "fitness_p75"),
                    color=BAND, alpha=0.6, linewidth=0, zorder=1)
    ax.plot(generations, mean, color=ORANGE, linewidth=2, zorder=2)
    ax.plot(generations, best, color=BLUE, linewidth=2, zorder=3)
    style_axes(ax, "Fitness", "score")

    # --- how many can complete a lap ---------------------------------------
    ax = axes[0][1]
    rate = [100 * value for value in column(rows, "lap_completion_rate")]
    ax.fill_between(generations, 0, rate, color=BAND, alpha=0.5, linewidth=0, zorder=1)
    ax.plot(generations, rate, color=BLUE, linewidth=2, zorder=3)
    ax.set_ylim(0, 105)
    style_axes(ax, "Population completing a full lap", "% of population")
    label_last(ax, generations, rate, "%.0f%%" % rate[-1] if rate else "", BLUE)

    first = next((g for g, value in zip(generations, rate) if value > 0), None)
    if first is not None:
        ax.axvline(first, color=INK_MUTED, linewidth=1, linestyle=":", zorder=2)
        # Placed high in the panel, where the curve has not reached yet.
        ax.annotate("first completed lap\ngeneration %d" % first,
                    xy=(first, 100), xytext=(6, -2), textcoords="offset points",
                    color=INK_SECONDARY, fontsize=8.5, va="top")

    # --- checkpoint progress ------------------------------------------------
    ax = axes[1][0]
    ax.plot(generations, column(rows, "gates_mean"), color=ORANGE, linewidth=2, zorder=2)
    ax.plot(generations, column(rows, "gates_best"), color=BLUE, linewidth=2, zorder=3)
    # A lap is the eight-gate outer circuit; the remaining two checkpoints sit
    # on an inner section the outer ring does not pass. See scripts/plot_track.py.
    ax.axhline(8, color=INK_MUTED, linewidth=1, linestyle=":", zorder=1)
    ax.annotate("a full circuit (8 of 10)", xy=(generations[0], 8), xytext=(10, 5),
                textcoords="offset points", color=INK_SECONDARY, fontsize=8.5)
    ax.set_ylim(0, 10.5)
    style_axes(ax, "Checkpoints cleared", "checkpoints")

    # --- time to complete a lap ---------------------------------------------
    # This is measured from the spawn point, so it includes the partial circuit
    # a car drives before it first reaches the finish line - it is "how long to
    # get round", not a flying lap time. Per generation it is noisy (a
    # generation with no finisher has no entry at all), so the record so far is
    # the headline and the per-generation figure sits behind it as context.
    ax = axes[1][1]
    seconds = column(rows, "best_lap_seconds", float, default=None)
    xs = [g for g, value in zip(generations, seconds) if value is not None]
    ys = [value for value in seconds if value is not None]
    if ys:
        ax.plot(xs, ys, color=ORANGE, linewidth=1.5, alpha=0.55, zorder=2)
        record = running_best(ys, better=min)
        ax.plot(xs, record, color=BLUE, linewidth=2, zorder=3)
        ax.set_ylim(0, max(ys) * 1.15)
        label_last(ax, xs, record, "%.1fs" % record[-1], BLUE)
        style_axes(ax, "Time to complete a lap", "seconds (simulated)")
    else:
        style_axes(ax, "Time to complete a lap", "seconds (simulated)")
        ax.text(0.5, 0.5, "no lap completed in this run", transform=ax.transAxes,
                ha="center", va="center", color=INK_MUTED, fontsize=10)

    # --- title block and one shared legend ---------------------------------
    figure.suptitle(title or "Training run", x=0.075, y=0.965, ha="left",
                    color=INK, fontsize=15, fontweight="bold")
    figure.text(0.075, 0.925,
                "%d generations, population %d, %d evaluations per generation"
                % (len(rows), population, population * 2),
                ha="left", color=INK_SECONDARY, fontsize=10)

    handles = [
        Line2D([], [], color=BLUE, linewidth=2, label="best individual (or record so far)"),
        Line2D([], [], color=ORANGE, linewidth=2, label="population mean (or per generation)"),
        Patch(facecolor=BAND, edgecolor="none", label="middle 50% of the population"),
    ]
    legend = figure.legend(handles=handles, loc="upper left",
                           bbox_to_anchor=(0.072, 0.895), ncol=3,
                           frameon=False, fontsize=9.5, handlelength=1.8,
                           columnspacing=1.8)
    for text in legend.get_texts():
        text.set_color(INK_SECONDARY)

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    figure.savefig(out_path, dpi=150, facecolor=SURFACE)
    plt.close(figure)
    print("wrote", out_path)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", default="runs/main")
    parser.add_argument("--out", default="results/curves.png")
    parser.add_argument("--title", default=None)
    args = parser.parse_args()

    csv_path = os.path.join(args.run, "generations.csv")
    if not os.path.exists(csv_path):
        raise SystemExit("no run at %s" % csv_path)

    plot(read_run(csv_path), args.out, args.title)


if __name__ == "__main__":
    main()
