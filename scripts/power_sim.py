"""Power of the primary test (A15) for design options: paired task success, clustered by task. Free, offline.

Model: task t has a baseline logit u_t ~ N(mu, su) and an arm-by-task deviation v_t ~ N(0, sv); the
comparison arm's logit is u_t + beta + v_t. Seeds of a task are copies of seed 1 with probability rho
(rho = 0.8 models temperature 0, 0.3 temperature > 0), else fresh draws. The test is a two-sided
t-test on per-task mean differences (|t| > 2), the large-sample equivalent of the task bootstrap.
Defaults are calibrated to the pilot: about 30% discordant blocks and a 33% baseline.

    uv run python scripts/power_sim.py            # the grid in docs/plan/V2_OPTIONS.md
"""

from __future__ import annotations

import argparse
import math
import random
import statistics as st


def sim(n_tasks: int, seeds: int, rho: float, beta: float, sims: int = 1500, mu: float = -0.9, su: float = 2.2,
        sv: float = 1.6, seed: int = 1) -> tuple[float, float, float]:
    """(power, mean discordance, mean success difference)."""
    rng = random.Random(seed)
    hits = 0
    disc: list[float] = []
    eff: list[float] = []
    for _ in range(sims):
        diffs = []
        d = tot = ea = eb = 0
        for _t in range(n_tasks):
            u = rng.gauss(mu, su)
            v = rng.gauss(0, sv)
            pa = 1 / (1 + math.exp(-u))
            pb = 1 / (1 + math.exp(-(u + beta + v)))
            ra: list[bool] = []
            rb: list[bool] = []
            for s in range(seeds):
                if s > 0 and rng.random() < rho:
                    ra.append(ra[0])
                    rb.append(rb[0])
                    continue
                ra.append(rng.random() < pa)
                rb.append(rng.random() < pb)
            diffs.append((sum(rb) - sum(ra)) / seeds)
            d += sum(1 for x, y in zip(ra, rb) if x != y)
            tot += seeds
            ea += sum(ra)
            eb += sum(rb)
        m = st.mean(diffs)
        sd = st.pstdev(diffs) * math.sqrt(n_tasks / (n_tasks - 1))
        se = sd / math.sqrt(n_tasks) if sd > 0 else 1e-9
        hits += abs(m / se) > 2.0
        disc.append(d / tot)
        eff.append((eb - ea) / tot)
    return hits / sims, st.mean(disc), st.mean(eff)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sims", type=int, default=1500)
    args = parser.parse_args()
    for n in (40, 60):
        for seeds, rho, label in ((1, 0.0, "1 seed"), (2, 0.8, "2 seeds T=0"), (2, 0.3, "2 seeds T>0"), (3, 0.3, "3 seeds T>0")):
            cells = []
            for beta in (0.75, 1.1, 1.5):
                power, _disc, effect = sim(n, seeds, rho, beta, sims=args.sims)
                cells.append(f"{effect * 100:+.0f}pp:{power:.2f}")
            print(n, label, " ".join(cells))


if __name__ == "__main__":
    main()
