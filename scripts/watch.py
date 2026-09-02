"""Watch a trained network drive - on screen, or straight to a GIF.

    python scripts/watch.py --model models/hard.pkl
    python scripts/watch.py --run runs/main --generation 6
    python scripts/watch.py --model models/hard.pkl --record results/demo.gif

This is the debugging tool that keeps the trainer honest: it replays through
the same rollout the trainer scores with, so if a network looks nothing like
its fitness suggests, the two have diverged and one of them is wrong.

With --record it also produces the README's demo. Recording renders the frames
rather than capturing the screen, which means no window borders or cursor in
the picture, no manual trimming, identical output on any machine, and anyone
who clones the repo can regenerate it with one command. It runs headless and as
fast as it can, so a fifteen-second clip takes a couple of seconds to make.

Press Esc or close the window to stop an on-screen run.
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


def capture(pygame, surface, size):
    """One frame as a PIL image, taken from the render surface.

    Reading the surface rather than the screen is what keeps window borders,
    the cursor and the desktop out of the picture.
    """
    from PIL import Image

    raw = pygame.image.tostring(surface, "RGB")
    image = Image.frombytes("RGB", surface.get_size(), raw)
    return image.resize(size, Image.LANCZOS)


def write_gif(frames, path, every, fps, colours, dither=False, hold=0.0):
    """Assemble frames into an optimised, looping GIF.

    Two choices keep the file small enough to sit in a README.

    All frames share one palette, built from a frame in the middle of the clip.
    Per-frame palettes look marginally better and roughly double the size.

    Dithering is off by default. It is the single biggest cost here: the art is
    flat-coloured, so dithering replaces large uniform areas with pixel noise
    that GIF's run-length compression cannot pack - on this clip it tripled the
    file for no visible benefit. --record-dither turns it back on.

    The camera follows the car, so every pixel changes every frame and
    inter-frame compression cannot help. That is why resolution and frame count
    matter more here than they would for a static shot.
    """
    if not frames:
        raise SystemExit("no frames captured")

    from PIL import Image

    palette = frames[len(frames) // 2].quantize(colors=colours, method=Image.MEDIANCUT)
    mode = Image.FLOYDSTEINBERG if dither else Image.NONE
    quantised = [frame.quantize(palette=palette, dither=mode) for frame in frames]

    duration = int(round(1000.0 * every / fps))
    # A per-frame duration list lets the closing frame linger, so the finished
    # lap registers before the loop restarts.
    durations = [duration] * len(quantised)
    if hold > 0:
        durations[-1] = max(duration, int(round(hold * 1000)))

    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    quantised[0].save(path, save_all=True, append_images=quantised[1:],
                      duration=durations, loop=0, optimize=True, disposal=1)

    size_mb = os.path.getsize(path) / 1e6
    print("wrote %s - %d frames at %dx%d, %d ms each (%.1f fps), %.1f s hold, %.2f MB"
          % (path, len(quantised), quantised[0].width, quantised[0].height,
             duration, 1000.0 / duration, hold, size_mb))
    if size_mb > 10:
        print("  warning: GitHub will not render a GIF this large inline; "
              "reduce --record-scale, --record-seconds or --record-colours")


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

    record = parser.add_argument_group("recording")
    record.add_argument("--record", metavar="PATH.gif",
                        help="render to a GIF instead of watching; implies headless")
    record.add_argument("--record-every", type=int, default=3,
                        help="keep every Nth simulated frame (3 -> 16.7 fps); the camera "
                             "tracks a car doing 12 px per tick, so a larger value moves "
                             "the view several percent of the frame width between frames "
                             "and reads as judder. 2 gives 25 fps but pushes the file "
                             "past the ~10 MB GitHub renders inline")
    record.add_argument("--record-scale", type=float, default=0.28,
                        help="scale factor applied to the 1400x750 frame")
    record.add_argument("--record-seconds", type=float, default=17.0,
                        help="stop after this many simulated seconds")
    record.add_argument("--record-colours", type=int, default=32,
                        help="GIF palette size; fewer colours means a smaller file")
    record.add_argument("--record-dither", action="store_true",
                        help="dither the palette; looks smoother, roughly triples the file")
    record.add_argument("--record-view", type=float, default=1.5,
                        help="how much more of the track to show than the game "
                             "window does; 1.0 matches the game exactly")
    record.add_argument("--record-hold", type=float, default=1.2,
                        help="seconds to hold the final frame before looping")
    record.add_argument("--record-from-lap", type=int, default=1,
                        help="start capturing after this many finish-line "
                             "crossings; 1 skips the partial circuit driven "
                             "from the starting grid, 0 records from the grid")
    args = parser.parse_args()

    if args.record:
        # Must precede the pygame import, and there is no window to show.
        from nncar.sim import headless

        headless.enable()
        args.fps = 0

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

    frames = []
    frame_size = None
    if args.record:
        # Draw onto a surface larger than the game window and shrink it down.
        # At 1:1 the camera is so tight that the clip reads as "a car on a
        # road" rather than "a car driving a circuit"; widening the view is
        # what makes the corners and the racing line legible.
        view_width = int(w.window_width * args.record_view)
        view_height = int(w.window_height * args.record_view)
        w.window = pygame.Surface((view_width, view_height))
        w.window_width, w.window_height = view_width, view_height

        frame_size = (int(view_width * args.record_scale / args.record_view),
                      int(view_height * args.record_scale / args.record_view))
        # Keep the HUD legible after the downscale.
        font = pygame.font.Font(None, int(40 * args.record_view))

        # Budget enough ticks to drive the skipped circuits and then the one
        # being captured.
        budget = args.record_seconds * (args.record_from_lap + 1) + 10
        cfg = RolloutConfig(laps=args.laps,
                            max_ticks=int(budget * cfg.fps),
                            exploration_noise=0.0, normalise_inputs=normalise)
        print("recording %d x %d at %.1fx view, every %d frames, up to %.0f s"
              % (frame_size[0], frame_size[1], args.record_view,
                 args.record_every, args.record_seconds))
        if args.record_from_lap:
            print("  skipping %d finish-line crossing(s) first, so the clip is "
                  "one whole circuit" % args.record_from_lap)

    running = True
    announced = False
    capture_start = None
    tick = 0
    while running and tick < cfg.max_ticks:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False

        car.update_sensors()
        car.move()

        # Crossing the line resets the checkpoint count for the new lap, so
        # bank the closing figure first: on that frame the interesting number
        # is what the circuit just completed cleared, not what the next one has
        # managed in its first tick.
        gates_completed = car.checkpoints_passed
        laps_before = car.laps
        car.reset_checkpoints()
        just_finished = car.laps > laps_before
        car.check_checkpoints()

        # Centre the camera on the car.
        camera.offset_x = w.window_width // 2 - car.world_x - car.width // 2
        camera.offset_y = w.window_height // 2 - car.world_y - car.height // 2

        w.window.fill((128, 128, 128))
        v.track.draw()
        car.draw()

        # While recording, the clip has its own clock and lap counter so that
        # skipped warm-up circuits do not appear in the numbers.
        closing = just_finished and (not args.record
                                     or car.laps > args.record_from_lap)
        if args.record:
            shown_lap = max(0, car.laps - args.record_from_lap)
            # capture_start is set later in this iteration, so it is None on the
            # frame that opens the clip - which is exactly 0.0 s.
            elapsed = 0.0 if capture_start is None else (tick - capture_start) / float(cfg.fps)
        else:
            shown_lap = car.laps
            elapsed = tick / float(cfg.fps)

        hud_scale = args.record_view if args.record else 1.0
        for index, line in enumerate([
            "lap %d/%d" % (shown_lap, cfg.laps),
            # Freeze the count only on the crossing that ENDS the clip. The
            # crossing that opens it is also a lap boundary, and there the
            # interesting number is the new circuit's, not the warm-up's.
            "checkpoints %d/10" % (gates_completed if closing else car.checkpoints_passed),
            "speed %.1f" % car.velocity,
            "%.1fs" % elapsed,
        ]):
            w.window.blit(font.render(line, True, (255, 255, 255)),
                          (int(20 * hud_scale), int((20 + index * 34) * hud_scale)))

        if args.record:
            if capture_start is None and car.laps >= args.record_from_lap:
                capture_start = tick
            if capture_start is not None and (tick - capture_start) % args.record_every == 0:
                frames.append(capture(pygame, w.window, frame_size))
        else:
            pygame.display.flip()

        clock.advance()
        tick += 1
        if args.fps:
            ticker.tick(args.fps)

        if args.record:
            # Stop once the circuit being captured is complete. The crossing
            # seldom falls on the capture cadence, so take the closing frame
            # regardless - otherwise the clip stops a few frames short and
            # never shows the completed lap.
            if capture_start is not None and car.laps > args.record_from_lap:
                if frames:
                    frames[-1] = capture(pygame, w.window, frame_size)
                print("captured one full circuit in %.2f simulated seconds"
                      % ((tick - capture_start) / float(cfg.fps)))
                break
            if (capture_start is not None
                    and tick - capture_start > args.record_seconds * cfg.fps):
                print("reached the %.0f s limit before the circuit closed"
                      % args.record_seconds)
                break
        elif car.laps >= cfg.laps and not announced:
            announced = True
            print("completed %d lap(s) in %.2f simulated seconds"
                  % (car.laps, tick / float(cfg.fps)))
            break

    pygame.quit()

    if args.record:
        write_gif(frames, args.record, args.record_every, cfg.fps,
                  args.record_colours, args.record_dither, args.record_hold)


if __name__ == "__main__":
    main()
