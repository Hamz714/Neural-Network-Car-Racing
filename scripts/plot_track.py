"""Draw the circuit, its checkpoints, and a trained network's route.

    python scripts/plot_track.py --model models/hard.pkl --out results/route.png

The figure exists to make one thing obvious: the track is a double loop. Two of
the ten checkpoints sit on an inner section, and the other eight lie on the
outer ring - so driving the outer ring is a complete closed lap that simply is
not the route the checkpoint list describes. Seeing that on a map explains the
evolved driver's behaviour immediately, where a table of gate counts does not.
"""

import argparse
import os
import pickle

import _bootstrap  # noqa: F401

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
BLUE = "#2a78d6"
ORANGE = "#eb6834"
GREEN = "#1baf7a"


def load_network(path):
    with open(path, "rb") as fh:
        payload = pickle.load(fh)
    if isinstance(payload, dict):
        return (payload.get("networks") or [payload["network"]])[0]
    return payload


def trace(network, start_index, ticks):
    """The path a network drives, in grid coordinates, and the gates it clears."""
    from nncar import entities as v
    from nncar.sim import clock as sim_clock
    from nncar.sim.rollout import RolloutConfig, build_track, make_car

    cfg = RolloutConfig(laps=1, max_ticks=ticks, normalise_inputs=True)
    track = build_track(cfg)
    v.track = track
    sim_clock.set_clock(sim_clock.TickClock(cfg.fps))

    car = make_car(network, v.NPC_START_POS[start_index], cfg)
    for index, checkpoint in enumerate(car.checkpoints):
        checkpoint.index = index

    outstanding = {c.index for c in car.checkpoints}
    cleared = set()
    path = []

    for _ in range(ticks):
        car.update_sensors()
        car.move()
        car.reset_checkpoints()
        car.check_checkpoints()
        still = {c.index for c in car.checkpoints if hasattr(c, "index")}
        cleared |= outstanding - still
        outstanding = still
        path.append((car.world_x - track.x, car.world_y - track.y))

    return path, cleared, track


def plot(model_path, out_path, start_index=0, ticks=2400):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from nncar import entities as v
    from nncar.sim import occupancy as occ

    grid = occ.load_grid()
    walls = (grid[1:-1, 1:-1] & occ.RAY_BIT).astype(bool)

    path, cleared, track = ([], set(), None)
    if model_path and os.path.exists(model_path):
        path, cleared, track = trace(load_network(model_path), start_index, ticks)
    else:
        from nncar.sim.rollout import RolloutConfig, build_track

        track = build_track(RolloutConfig())

    figure, ax = plt.subplots(figsize=(10, 8.5), facecolor=SURFACE)
    ax.imshow(walls.T, cmap="Greys", origin="upper", interpolation="nearest", alpha=0.75)

    if path:
        ax.plot([p[0] for p in path], [p[1] for p in path],
                color=BLUE, linewidth=1.7, zorder=3)
        ax.plot(path[0][0], path[0][1], "o", color=GREEN, markersize=10, zorder=4)

    for index, (x1, y1, x2, y2) in enumerate(v.CHECKPOINT_GATES):
        gx1, gy1 = x1 - track.x, y1 - track.y
        gx2, gy2 = x2 - track.x, y2 - track.y
        reached = index in cleared if path else True
        colour = ORANGE if reached else "#e34948"
        ax.plot([gx1, gx2], [gy1, gy2], color=colour, linewidth=3, zorder=2)
        ax.annotate(str(index), ((gx1 + gx2) / 2, (gy1 + gy2) / 2),
                    color=colour, fontsize=13, fontweight="bold",
                    ha="center", va="center", zorder=5,
                    bbox=dict(boxstyle="circle,pad=0.22", fc=SURFACE, ec=colour, lw=1.6))

    missed = sorted(set(range(len(v.CHECKPOINT_GATES))) - cleared) if path else []
    ax.set_title("The circuit, its ten checkpoints, and the evolved route",
                 color=INK, fontsize=14, fontweight="bold", loc="left", pad=14)
    if path:
        note = ("blue: the trained network's path   -   "
                "red gates never reached: %s" % (missed or "none"))
        ax.text(0, -0.035, note, transform=ax.transAxes, color=INK_SECONDARY,
                fontsize=10, va="top")

    ax.axis("off")
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    figure.savefig(out_path, dpi=110, bbox_inches="tight", facecolor=SURFACE)
    plt.close(figure)
    print("wrote %s (missed gates: %s)" % (out_path, missed or "none"))


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="models/hard.pkl")
    parser.add_argument("--out", default="results/route.png")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--ticks", type=int, default=2400)
    args = parser.parse_args()
    plot(args.model, args.out, args.start, args.ticks)


if __name__ == "__main__":
    main()
