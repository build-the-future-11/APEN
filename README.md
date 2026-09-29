# APEN

> **GitHub snapshot reproducibility guard (2026-09-29).** The live GitHub tree at base commit `42f8917efc10019a693e7bf3a4faa8440a8ed055` contains only `README.md`; the package metadata, source tree, training script, and frozen protocol/status files referenced below are absent. The install/import/training/benchmark commands below are therefore not runnable from that snapshot. A verified repair patch is attached to this sprint; it adds a fail-closed repository-shape preflight and regression tests without changing experiment definitions.


**APEN now uses the spatial-memory architecture previously named APENNext.**
The normal `APEN` import, `main.py`, `scripts/train.py`, and `python -m apen`
all use this version. It combines spatial residual forecasting, temporal history,
learnable episodic retrieval, and a validation-selected gated correction.

```sh
python -m apen train --output runs_local/my-apen-run
python -m apen verify runs_local/my-apen-run
```

See [APEN.md](APEN.md) for training, progress, checkpoint loading, and forecasting.
The current [research draft](paper/apen_v2/README.md) describes the spatial-memory
method and reports both its gains and its negative distribution-shift results.
[STATUS.md](STATUS.md) separates execution results from research readiness.
The source folder has no Git history. This is not a submission-ready project.

The previous architecture is available as `models.legacy.apen.APEN`, with an
unmodified source/evidence archive in `legacy/pre_apen_v2_2026-09-27/`.
Its [negative research report](paper/APEN_FALSIFICATION_MANUSCRIPT.md) and
[table](paper/current_burgers_table.md) remain historical evidence for that model;
they do not measure this new architecture. The bundled Synthica portfolio is
outside the APEN migration.

## Historical research references

The references and protocols below describe the archived architecture unless
explicitly marked APEN v2. See [REPRODUCE.md](REPRODUCE.md) for that distinction.

Deep-audit entry points: [repository map](audit/REPOSITORY_MAP.md),
[mathematical specification](research/MATHEMATICAL_SPEC.md),
[hypotheses](research/HYPOTHESES.md), [evidence ledger](EVIDENCE_LEDGER.md), and
[conference-readiness checklist](audit/CONFERENCE_READINESS_CHECKLIST.md).
The full implementation queue, including pseudocode/scaffolds, unsafe code,
abstractions, additions, and execution order, is in
[`audit/ULTIMATE_IMPLEMENTATION_CHECKLIST.md`](audit/ULTIMATE_IMPLEMENTATION_CHECKLIST.md).

**Validity boundary (2026-09-08):** training previously supplied future-target
salience to the prediction gate. The maintained path is now causal. Retained
comparative APEN results are historical pre-fix development evidence until a
new frozen comparison is executed; see `RESEARCH_TRUTH.md`.

> **Evidence status:** read [`RESEARCH_TRUTH.md`](RESEARCH_TRUTH.md) before quoting benchmark or paper results. The current repository contains implemented experimental/statistical machinery and structural hotfixes, but its own status files state that the full multi-seed benchmark, baseline-table, OOD-scaling, and paper-grade final-metric program is not complete. The committed legacy benchmark report currently records failed cells, not successful paper results.

Global invariants:

Tensor format: [B, T, C, *spatial]  
Dtype: torch.float32  
Device must be passed explicitly  
No module may call .cuda() internally  
All randomness must be controlled via set_seed  

Install:

```bash
uv sync --locked --all-extras
```

Pip fallback:

```bash
pip install -e '.[synthica-report,synthica-engineering,apen-plots,test]'
```

All scripts must run from project root.

Research checks:

```bash
python3 scripts/preflight.py
python3 scripts/verify_apen_artifact.py
python3 scripts/verify_apen_physics_smoke.py
ruff check analysis benchmarks data evaluation experiments.py models scripts synthica_foundry training utils tests
PYTHONPATH=. pytest -q
python3 -m synthica_foundry.cli build-research-packages
```

Paper-oriented smoke run:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python3 scripts/apen_physics_smoke.py --output runs_local/apen-physics-smoke.json

OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python3 scripts/paper_suite.py --smoke --run-baselines --seeds 61 --dataset burgers --device cpu

OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python3 scripts/apen_matched_benchmark.py

OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python3 scripts/apen_validation_selected_benchmark.py

OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python3 scripts/apen_matched_navier_stokes.py

OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python3 scripts/apen_matched_navier_stokes_v2.py

OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python3 scripts/apen_ablation_burgers.py

OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python3 scripts/apen_ood_burgers.py

python3 scripts/plot_apen_matched.py
python3 scripts/plot_apen_ablation.py
python3 scripts/plot_apen_ood.py
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python3 scripts/profile_navier_compute.py
python3 scripts/plot_navier_compute.py
```

The first command verifies real APEN training/evaluation against the Burgers,
Kuramoto–Sivashinsky, and 2-D Navier–Stokes generators. The second verifies a
small APEN/FNO/DeepONet workflow. Both are implementation smoke tests, not
paper-scale performance evidence.

The third command runs the reduced two-epoch Burgers v1 comparison. The fourth
runs its locally frozen validation-selected extension: the same five models and
2%-matched parameter capacities receive a common 20-epoch ceiling; best states
are selected only by validation rollout MSE and the test split is evaluated
once afterward. APEN remains last and wins zero of three fresh seeds. The next
two commands cover the reduced APEN/FNO2d/ResNet2d Navier–Stokes comparison,
followed by RNG-paired component ablations and generated Burgers viscosity
shifts. These grids and data budgets remain too small for a general superiority
or application claim.

The compute profiler launches each 2-D model in a fresh process and records
PyTorch-counted FLOPs, latency, and process RSS. The FLOP count is explicitly
bounded because FFT and custom operations may be omitted; latency and RSS are
host-specific and excluded from its deterministic signature.

The older Navier–Stokes, ablation, OOD, and fixed-epoch comparison numbers above
are historical pre-repair evidence. Their conclusions must not be attributed to
the repaired implementation. New diagnostic replays have separate dated paths
and are classified in `ASTRA_FINAL_REPORT.md`.

The portfolio research-package command materializes evidence-bounded packages
for all 64 canonical ideas under `research/projects/`. It deliberately reports
zero submission-ready standalone projects: most entries remain operational
proxies, and the APEN flagship has negative matched-baseline and mechanism
evidence. The APEN literature synthesis includes FNO, DeepONet, Adaptive
Computation Time, and the ICLR 2025 Memory Neural Operator paper.

The default APEN objective combines prediction error with spectral, gradient, Laplacian, fractional Sobolev, Haar-Besov multiscale, empirical Wasserstein, cumulant, gate, memory, and consolidation terms. The analysis pipeline implements paired ablation statistics with bootstrap intervals, sign-flip permutation tests, Wilcoxon tests, t-tests, Hedges g, Holm correction, and a readiness audit.

Those implemented analyses become scientific evidence only when they are run on a complete valid frozen experiment matrix with retained provenance. See `STATUS.md`, `HOTFIX_STATUS.json`, and `RESEARCH_TRUTH.md` for the current execution boundary.

Maintained training and evaluation paths fail closed on non-finite tensors and empty
loaders. They do not replace NaN/Inf predictions with finite numbers, because that
would make a failed run look measurable. Checkpoint loading accepts only regular,
non-symlink files and uses PyTorch's restricted weights-only loader.
# APEN
