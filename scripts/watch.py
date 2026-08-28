"""Watch a trained network drive, on screen.

    python scripts/watch.py --model models/hard.pkl
    python scripts/watch.py --run runs/main --generation 6

This is the debugging tool that keeps the trainer honest: it replays through
the same rollout the trainer scores with, so if a network looks nothing like
its fitness suggests, the two have diverged and one of them is wrong.

It is also what the README's demo is recorded from. Press Esc or close the
window to stop.
"""

import argparse
import os
import pickle

import _bootstrap  # noqa: F401



def load_networks(path):
    with open(path, "rb") as fh:
        payload = pickle.load(fh)
    if isinstance(payload, dict):
        networks = payload.get("networks") or [payload["network"]]
        return networks, payload.get("normalise_inputs", True), payload.get("meta", {})
    return [payload], False, {}


def resolve(args):
    if args.model:
        return args.model
    run = args.run or "runs/main"
    if args.generation is not None:
        return os.path.join(run, "champions", "gen%04d.pkl" % args.generation)
    return os.path.join(run, "champion.pkl")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", help="path to a saved model")
    parser.add_argument("--run", help="run directory to take a champion from")
    parser.add_argument("--generation", type=int, help="checkpoint to watch")
    parser.add_argument("--start", type=int, default=0, help="spawn position, 0-4")
    parser.add_argument("--laps", type=int, default=1)
    parser.add_argument("--max-ticks", type=int, default=6000)
    parser.add_argument("--fps", type=int, default=50,
                        help="0 runs as fast as it can")
    parser.add_argument("--follow", action="store_true", default=True,
                        help="keep the camera on the car (default)")
    args = parser.parse_args()

    path = resolve(args)
    if not os.path.exists(path):
        raise SystemExit("no model at %s" % path)

    networks, normalise, meta = load_networks(path)
    print("watching %s" % path)
    if meta:
        print("  %s" % ", ".join("%s=%s" % kv for kv in sorted(meta.items())))

    import pygame

    from nncar import entities as v
    from nncar import window as w
    from nncar.sim import clock as sim_clock
    from nncar.sim.rollout import RolloutConfig, make_car

    pygame.init()
    pygame.display.set_caption("nncar - %s" % os.path.basename(path))

    cfg = RolloutConfig(laps=args.laps, max_ticks=args.max_ticks,
                        exploration_noise=0.0, normalise_inputs=normalise)

    clock = sim_clock.TickClock(cfg.fps)
    sim_clock.set_clock(clock)

    v.track = v.Track(cfg.laps, load_visuals=True)
    car = make_car(networks[0], v.NPC_START_POS[args.start % len(v.NPC_START_POS)], cfg)

    # The renderer draws everything relative to `player`'s offsets, so a stand-in
    # player is used purely as the camera.
    camera = v.PlayerCar()
    v.player = camera
    v.NPC_cars = [car]
    v.track.leaderboard = [car]

    font = pygame.font.Font(None, 40)
    ticker = pygame.time.Clock()

    running = True
    tick = 0
    while running and tick < cfg.max_ticks:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False

        car.update_sensors()
        car.move()
        car.reset_checkpoints()
        car.check_checkpoints()

        # Centre the camera on the car.
        camera.offset_x = w.window_width // 2 - car.world_x - car.width // 2
        camera.offset_y = w.window_height // 2 - car.world_y - car.height // 2

        w.window.fill((128, 128, 128))
        v.track.draw()
        car.draw()

        for index, line in enumerate([
            "lap %d/%d" % (car.laps, cfg.laps),
            "checkpoints %d/10" % car.checkpoints_passed,
            "speed %.1f" % car.velocity,
            "%.1fs" % (tick / float(cfg.fps)),
        ]):
            w.window.blit(font.render(line, True, (255, 255, 255)), (20, 20 + index * 34))

        pygame.display.flip()
        clock.advance()
        tick += 1
        if args.fps:
            ticker.tick(args.fps)

        if car.laps >= cfg.laps:
            print("completed %d lap(s) in %.2f simulated seconds"
                  % (car.laps, tick / float(cfg.fps)))
            break

    pygame.quit()


if __name__ == "__main__":
    main()
