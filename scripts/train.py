"""Evolve a network that can drive the circuit.

    python scripts/train.py --generations 200 --population 100 --workers 8

Writes a per-generation CSV, the configuration needed to repeat the run, and a
checkpoint every time the champion improves. Everything is seeded: the same
--seed produces the same run, and the same run whether it uses one worker or
eight.

To ship the result as a difficulty tier:

    python scripts/train.py --export models/hard.pkl --from-run runs/main
"""

import argparse
import os
import random
import sys
import time

# Must happen before pygame is imported anywhere, including in spawned workers,
# which inherit this environment.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "src"))
from nncar.sim import headless  # noqa: E402

headless.enable()

import _bootstrap  # noqa: F401,E402

from nncar.ga import population as ga  # noqa: E402
from nncar.ga import runlog  # noqa: E402
from nncar.ga.evaluate import Evaluator, resolve_workers  # noqa: E402
from nncar.sim import fitness as fit  # noqa: E402
from nncar.sim.rollout import RolloutConfig  # noqa: E402


def build_parser():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    search = parser.add_argument_group("search")
    search.add_argument("--generations", type=int, default=200)
    search.add_argument("--population", type=int, default=100)
    search.add_argument("--elite", type=int, default=5,
                        help="carried through unchanged; guarantees the best never drops")
    search.add_argument("--parents", type=int, default=20,
                        help="size of the breeding pool (truncation selection)")
    search.add_argument("--sigma", type=float, default=0.15, help="initial mutation step")
    search.add_argument("--sigma-final", type=float, default=0.02)
    search.add_argument("--sigma-schedule", choices=["constant", "linear", "exp"],
                        default="exp")
    search.add_argument("--crossover-rate", type=float, default=0.0,
                        help="off by default; see nncar.ga.population for why")
    search.add_argument("--random-inject", type=float, default=0.05)
    search.add_argument("--tournament-k", type=int, default=3)

    env = parser.add_argument_group("environment")
    env.add_argument("--laps", type=int, default=1)
    env.add_argument("--fps", type=int, default=50)
    env.add_argument("--max-ticks", type=int, default=3000)
    env.add_argument("--stall-ticks", type=int, default=400)
    env.add_argument("--displacement-window", type=int, default=100)
    env.add_argument("--displacement-min", type=float, default=50.0)
    env.add_argument("--collision-limit", type=int, default=5)
    env.add_argument("--min-gates-per-lap", type=int, default=8,
                     help="a lap only counts once this many checkpoints are cleared")
    env.add_argument("--start-positions", default="0,1",
                     help="comma-separated spawn indices; scores are averaged")
    env.add_argument("--eval-noise", type=float, default=0.0,
                     help="output noise during evaluation; 0 keeps fitness deterministic")

    weights = parser.add_argument_group("fitness")
    weights.add_argument("--w-progress", type=float, default=100.0)
    weights.add_argument("--w-speed", type=float, default=25.0)
    weights.add_argument("--w-lap", type=float, default=250.0)
    weights.add_argument("--w-collision", type=float, default=5.0)
    weights.add_argument("--w-finish", type=float, default=500.0)

    run = parser.add_argument_group("execution")
    run.add_argument("--seed", type=int, default=1234)
    run.add_argument("--workers", type=int, default=0,
                     help="0 uses one fewer than the core count")
    run.add_argument("--out", default="runs")
    run.add_argument("--run-name", default="main")
    run.add_argument("--log-every", type=int, default=1)

    export = parser.add_argument_group("export")
    export.add_argument("--export", metavar="PATH",
                        help="write a champion as a playable model and exit")
    export.add_argument("--from-run", metavar="DIR",
                        help="run directory to export from")
    export.add_argument("--from-generation", type=int,
                        help="checkpoint to export; defaults to the best overall")
    return parser


def export_only(args):
    import pickle

    run_dir = args.from_run or os.path.join(args.out, args.run_name)
    if args.from_generation is None:
        source = os.path.join(run_dir, "champion.pkl")
    else:
        source = os.path.join(run_dir, "champions", "gen%04d.pkl" % args.from_generation)

    if not os.path.exists(source):
        raise SystemExit("no champion at %s" % source)

    with open(source, "rb") as fh:
        payload = pickle.load(fh)

    digest = runlog.export_network(args.export, payload["networks"], {
        "source_run": os.path.abspath(run_dir),
        "source_checkpoint": os.path.basename(source),
        "generation": payload.get("meta", {}).get("generation"),
        "fitness": payload.get("meta", {}).get("fitness"),
        "exported": time.strftime("%Y-%m-%dT%H:%M:%S"),
    })
    print("wrote %s (sha1 %s)" % (args.export, digest[:12]))


def main(argv=None):
    args = build_parser().parse_args(argv)

    if args.export:
        return export_only(args)

    weights = fit.FitnessWeights(args.w_progress, args.w_speed, args.w_lap,
                                 args.w_collision, args.w_finish)
    # Fail before an eight-hour run rather than after it.
    fit.check_weights(weights, args.collision_limit)

    start_indices = tuple(int(value) for value in args.start_positions.split(","))

    cfg = RolloutConfig(
        fps=args.fps, laps=args.laps, max_ticks=args.max_ticks,
        stall_ticks=args.stall_ticks, collision_limit=args.collision_limit,
        displacement_window=args.displacement_window,
        displacement_min=args.displacement_min,
        min_gates_per_lap=args.min_gates_per_lap,
        exploration_noise=args.eval_noise, normalise_inputs=True)

    config = ga.GAConfig(
        population=args.population, elite=args.elite, parents=args.parents,
        sigma0=args.sigma, sigma1=args.sigma_final,
        sigma_schedule=args.sigma_schedule, crossover_rate=args.crossover_rate,
        random_inject=args.random_inject, tournament_k=args.tournament_k)

    workers = resolve_workers(args.workers)

    # Build the occupancy cache here, in the parent, so the workers never do.
    from nncar.sim import occupancy

    occupancy.load_grid()

    directory = os.path.join(args.out, args.run_name)
    os.makedirs(directory, exist_ok=True)

    rng = random.Random(args.seed)
    networks = ga.initial_population(config.population, rng)

    print("population %d, %d generations, %d worker(s), seed %d"
          % (config.population, args.generations, workers, args.seed))
    print("start positions %s, %d evaluations per generation"
          % (list(start_indices), config.population * len(start_indices)))
    print("logging to %s" % os.path.abspath(directory))
    print()

    settings = dict(vars(args))
    settings.update({"resolved_workers": workers,
                     "rollout": cfg.as_dict(), "ga": config.as_dict(),
                     "fitness_weights": weights.as_dict()})

    best_overall = None
    cumulative = 0.0

    with runlog.RunLog(directory, settings) as log, \
            Evaluator(cfg, workers, start_indices) as evaluator:

        header = ("%5s %9s %9s %7s %6s %7s %9s %8s"
                  % ("gen", "best", "mean", "gates", "laps", "sigma", "evals/s", "elapsed"))
        print(header)
        print("-" * len(header))

        for generation in range(args.generations):
            sigma = ga.sigma_for(generation, args.generations, config)

            started = time.time()
            results = evaluator.evaluate(networks, generation, args.seed)
            elapsed = max(time.time() - started, 1e-9)
            cumulative += elapsed

            # A network is scored on its worst start position, not its best:
            # driving one corner well by luck should not read as competence.
            scores = [min(fit.fitness(r, weights, cfg.fps, cfg.max_ticks) for r in group)
                      for group in results]

            row = runlog.summarise(generation, sigma, scores, results,
                                   elapsed, cumulative, cfg.fps, cfg.laps)

            champion_index = scores.index(max(scores))
            row["champion_fitness"] = round(scores[champion_index], 4)

            if best_overall is None or scores[champion_index] > best_overall:
                best_overall = scores[champion_index]
                row["champion_sha1"] = log.save_champion(
                    networks[champion_index], generation,
                    {"generation": generation,
                     "fitness": round(scores[champion_index], 4),
                     "gates": row["gates_best"], "laps": row["laps_best"],
                     "seed": args.seed, "sigma": sigma})

            log.write(row)

            if generation % args.log_every == 0 or generation == args.generations - 1:
                print("%5d %9.1f %9.1f %7d %6d %7.4f %9.1f %7.1fs"
                      % (generation, row["fitness_best"], row["fitness_mean"],
                         row["gates_best"], row["laps_best"], sigma,
                         row["evals_per_sec"] or 0, cumulative))

            scored = [(scores[i], i, networks[i]) for i in range(len(networks))]
            networks = ga.next_generation(scored, config, sigma, rng)

    print()
    print("done in %.1fs; best fitness %.1f" % (cumulative, best_overall))
    print("champion: %s" % os.path.join(directory, "champion.pkl"))


if __name__ == "__main__":
    main()
