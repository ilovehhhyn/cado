# Cado

Open-source instruments for trustworthy reinforcement learning.

**[Website](https://ilovehhhyn.github.io/cado/)** · **[Python SDK](tplane/README.md)** · **[Benchmark methodology](tplane/docs/results/rft-faultbench.md)**

Cado's first instrument, `tplane`, records typed outcomes for RL work and separates infrastructure, grader, timeout, and agent failures. Your trainer applies the resulting decision.

## Start

Requires Python 3.11 or 3.12.

```sh
git clone https://github.com/ilovehhhyn/cado.git
cd cado/tplane
python -m pip install -e .
```

See the [SDK tour](tplane/README.md#ten-minute-tour) for integration and the [website](https://ilovehhhyn.github.io/cado/#how-to) for a local example.

## Website development

The website is plain HTML, CSS, and JavaScript with self-hosted fonts. No build step is needed.

```sh
python3 -m http.server 4173 --directory website
```

Open `http://localhost:4173`. Changes pushed to `main` under `website/` deploy automatically to GitHub Pages.

## Source snapshot

The initial SDK publication is the committed tplane snapshot `40993c9`. In-progress local research changes are not part of this publication. Benchmark results are early research; the linked methodology documents dataset imbalance, label choices, and selection bias.

The SDK is MIT licensed. Font licenses are included in `website/assets/`.
