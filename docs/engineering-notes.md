# Engineering notes

Supporting material for the README: how the measurements were taken, what was
tried, and the things that turned out to be different from what they looked
like. The README states conclusions; this is the working.

---

## Benchmark methodology

`scripts/benchmark.py` produces `docs/baseline.json` and `docs/optimised.json`.

- `time.perf_counter`, garbage collector disabled for the timed region.
- One discarded warm-up pass, then seven independent repeats; the **minimum** is
  reported, with the median and standard deviation recorded alongside. Minimum
  is the right statistic for a microbenchmark: the distribution is bounded
  below by the real cost and everything above it is scheduler noise.
- Both implementations are timed **in the same process, moments apart**, so the
  comparison cannot be contaminated by a different machine state on a different
  day.
- **Equivalence is asserted before timing.** A speedup between two functions
  that compute different things is not a speedup, so the raycast benchmark
  refuses to report a number until the two implementations have been shown to
  agree on the sample it is about to time.

Machine: Intel i7-1165G7, 4 physical cores / 8 threads, Windows 11,
Python 3.12.6, numpy 2.5.2, pygame 2.6.1. No display (SDL dummy driver).

Ray cost depends entirely on how far a ray travels before it hits something, so
results are bucketed by regime rather than averaged into one figure that would
hide where the benefit is.

---

## Measurements

### Sensor raycasting, five rays per car

| Regime | 5-px march (pygame) | Vectorised (grid) | Factor |
|---|---|---|---|
| Corridors, 150–450 px | 789 µs | 38.0 µs | **20.8×** |
| Open space, > 450 px | 989 µs | 33.7 µs | **29.4×** |

The vectorised cost is essentially **flat** — 34–38 µs regardless of distance —
while the original scaled with it. That flatness matters more than the factor:
the expensive case for the old code was a ray that saw nothing, which is
exactly what an untrained network produces constantly, so the worst case was
also the most common one during early training.

### Forward pass

| | µs | Factor |
|---|---|---|
| Composed from the matrix helpers | 68.4 | — |
| Hand-fused, still pure Python | 35.5 | **1.93×** |

Three things account for it: the input is known to be a column vector, so the
generic matrix code's innermost loop over columns always ran over one element;
the multiply-accumulate, bias and activation happen in one pass rather than
three, avoiding two intermediate matrices per layer; and values move as plain
floats rather than as one-element lists.

The accumulation is written as an explicit `for ...: total += w * v`.
`sum(map(mul, ...))` measures faster still, but CPython 3.12 gave `sum()`
compensated floating-point summation, so it does not produce identical results.
Identical was the requirement.

### End to end

| | Ticks/sec | µs/tick |
|---|---|---|
| Before | 1,030 | 970 |
| After | 8,373 | 119 |

**8.1×**, measured directly rather than by multiplying the component factors
together.

### Parallel scaling

Averaged over generations 2–7 of an identical seeded run, excluding pool
startup:

| Workers | 1 | 2 | 4 | 6 | 8 |
|---|---|---|---|---|---|
| evals/sec | 23.5 | 41.7 | 59.8 | 68.0 | 73.4 |

3.1× at eight workers. The CPU has four physical cores and eight threads, so
this is roughly 78% of what the physical cores can give — the second thread on
each core adds much less than the first, as expected for work that is arithmetic
rather than latency-bound.

`chunksize=1` measured fastest (96 evals/sec) against 2, 4 and 8 (90, 87, 86) in
a direct comparison, despite the extra inter-process traffic. Rollout lengths
vary more than tenfold — a crasher stops in a few dozen ticks, a lap-completer
runs for thousands — so static chunking leaves workers idle at the end of every
generation.

---

## Correctness before speed

Each optimisation was gated on evidence that it changed nothing.

**The forward pass** is bit-identical to the matrix-helper version on 500
random networks and on saturating inputs. It also still reproduces
`tests/golden/network_golden.json`, which was recorded from the original
pre-refactor code — so the hot path is provably unchanged all the way back to
the starting point.

**Raycasting** agrees with the original `Sensor` class to within 1e-9 on 99.9%
of rays. The residual is inherent and understood: the old march accumulated its
position one 5-px step at a time while the vectorised form evaluates a
closed-form distance, so a sample landing within a rounding error of a pixel
boundary can fall either side of it. When that happens the two differ by
exactly one step. None differ by more, and the test asserts that.

**Grid collision** agrees with `pygame.mask.overlap` on 5,000 poses across the
real track, with no disagreements, and the test also asserts that a meaningful
fraction of those poses actually touched a wall — so it cannot pass by sampling
only empty space.

**The whole simulation** is pinned by a recorded 400-tick, five-car trajectory.
The physics is chaotic — each position depends on rays cast from the previous
one — so a one-ulp change compounds into a visible divergence within a few
hundred ticks. That makes a stored trajectory a far sharper instrument than
unit tests. The package restructure was verified against it as bit-identical;
the grid swap moved it by at most 1.2e-4 px over 400 ticks, with collision,
checkpoint and lap counts unchanged.

---

## Two thresholds, one grid

The occupancy grid carries two bitplanes because its two callers disagree about
what counts as solid:

- sensors treat **any non-zero alpha** as a wall (`get_pixel_alpha`);
- `pygame.mask.from_surface` defaults to **alpha > 127**.

On this track that is **19,478 pixels** of anti-aliased fringe. A single-plane
grid could only have reproduced one of them, and would have been quietly wrong
for the other — the kind of discrepancy that shows up as cars clipping walls
occasionally rather than as an error.

---

## Things that were not what they looked like

**The sensors were not drifting.** Each `Sensor` integrated its own position
from the velocity and angle it was handed, which are a frame stale, and nothing
ever resynchronised them against the car — which reads exactly like a bug. It
is not one. The deltas applied are identical to the car's own, and
`update_sensors` runs before `move`, so at the moment the rays are cast the two
are in the same place. Measured over 600 ticks including 66 collisions: **0.000
px of origin drift, 0.0000° of angle drift.** The rays are now derived from the
car anyway, because five stateful objects per car earn nothing when the car's
position already holds the answer, and because deriving them is what allows all
five to be cast in one batch — but it is a simplification, not a fix.

**The `+ velocity * cos` in the crossing tests is correct.** Position updates
subtract that term while the crossing predicates add it, which looks like a
sign error. For `pass_checkpoint` the predicate is a symmetric disjunction over
both directions and is therefore invariant under the flip. For
`reset_checkpoints` it makes the test fire one tick late, but still exactly
once per crossing. Left alone, with characterisation tests pinning the
behaviour.

**Box–Muller throws away half its output.** It produces two independent normals
per pair of uniforms and only the cosine term is kept. Reclaiming the sine
would halve the calls into the random number generator — and would shift every
seeded value in the project, invalidating the golden files that make the rest of
this document checkable. Deliberately left as it is.

**A bare `except` was hiding a real error.** `Track.get_pixel_alpha` caught
everything and answered "no wall". The catch exists for a good reason — rays
routinely sample past the edge of the map, and off the map *is* open — but as
written it also turned "you asked a grid-backed track for a pixel it does not
have" into open space. Narrowed to `IndexError`.

---

## The reward exploit

The most interesting thing in the project, and the reason fitness is measured
in checkpoints rather than laps.

`Car.reset_checkpoints` increments the lap counter whenever a car is just north
of the finish line and moving north. All five spawn points are north of that
line. So a network that reverses roughly 120 px and drives forward again banks
a lap in about **30 ticks**, having passed **zero** checkpoints — and can
repeat it indefinitely. The same crossing also resets the collision count,
which would have laundered the crash penalty too.

Any fitness of the form `laps * w + checkpoints` is therefore trivially farmed,
and a search will find it: it is a far shorter path to a high score than
learning to steer. Rather than change the game's lap detection, the rollout
banks each lap's checkpoint count as it happens, fitness counts checkpoints,
and a lap only counts once it has cleared eight of the ten.

The guard is visible in the training log. Champions frequently show 16
checkpoints for one completed lap: a first pass that clipped a corner and
cleared only six gates did not count, so the car had to go round again
properly.

`tests/test_fitness.py::test_the_free_lap_exploit_scores_nothing` pins it at
zero.

---

## Why the inputs are scaled

Raw sensor distances reach 700 while weights are initialised from N(0,1), so
first-layer pre-activations average around 10³. The `±20` saturation clamp then
pins **99.7%** of hidden units to exactly ±1, and the first layer degenerates
into `sign(z)` — a piecewise-constant function of its inputs.

That is close to unlearnable by mutation: almost every small perturbation
changes nothing at all, and the rare one that flips a sign changes the output
discontinuously. The search sees a flat landscape with occasional cliffs and no
gradient to follow.

Scaled to [0, 1], mean |z| falls from **1060 to 1.98** and saturation from
99.7% to 12.1%, putting tanh back in the part of its range where a small change
to a weight makes a small change to the output.

Measured over 200 random networks in
`tests/test_forward.py::test_raw_pixel_inputs_saturate_the_first_layer`.

---

## Rejected

- **Vectorising physics across the whole population.** Simulating all hundred
  cars in lockstep would give roughly another order of magnitude, but it means a
  second implementation of the physics that the game does not use — and two
  copies of a simulation drift apart. The trainer and the game share one
  `Car`/`NPC` for exactly this reason.
- **Per-angle rotated collision masks.** The original built its mask from the
  unrotated sprite, so rotation never affected collision. Adding it would be
  more correct and would change how the game feels, for no training benefit.
- **`math.tanh`.** About five times faster than the hand-written
  `(e^x - e^-x)/(e^x + e^-x)`, but not bit-identical, and it erodes the
  from-scratch claim for roughly 2 µs per tick.
- **Reclaiming the second Box–Muller normal.** See above.
