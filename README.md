# PINO for mass-conserving cavitation in journal bearings

Solver, network and training code for the paper

Y. Dong, "Physics-informed neural operators for mass-conserving cavitation in textured journal bearings: label
efficiency and label-free extension to new operating ranges" (submitted).

The network maps the film thickness H, the length-to-diameter ratio L/D, the groove position and the supply pressure
of a finite journal bearing to its pressure P and void fraction r. It uses Fourier layers along the circumference and
sine layers along the axis, so the edge conditions hold exactly; the groove condition is imposed on the output. The
PINO adds the discrete JFO (Jakobsson-Floberg-Olsson) residual of the reference solver and a complementarity term to
the data loss. This physics loss can also be evaluated on bearings without solver labels.

## Files

- `pino.py`: network (FNO2d with Fourier or Fourier-sine layers)
- `pino_jfo.py`: training of the data-only FNO and the PINO
- `cut_residual.py`: residual rows next to the supply groove (cut-cell treatment)
- `solver.py`: finite-volume Reynolds/JFO solver (primal-dual active-set method), film thickness, dimples, groove,
  load and attitude angle
- `solver2.py`: JFO solver with the supply groove imposed with sub-cell accuracy (cut-cell), used for all labels
- `make_dataset.py`: random bearing geometry (eccentricity, L/D, misalignment, dimples)
- `make_dataset_jfo_cut.py`: labelled dataset, bearings solved with `solver2.py`
- `make_dataset_jfo_ood.py`: test sets outside the training ranges and the unlabelled bearings of the extrapolation study

## Requirements

Python 3.10, numpy, scipy and PyTorch (see `requirements.txt`). Tested with PyTorch 2.10 and CUDA 12.8.

## Labelled dataset

    python make_dataset_jfo_cut.py 64 10000                                  # -> dataset_jfo_cut.npz
    for s in eps ld depth ps; do SOLVER=cut python make_dataset_jfo_ood.py $s 1000; done   # extrapolation test sets

The random draws use fixed seeds, so the datasets of the paper are reproduced exactly.

## Training

    python pino_jfo.py <tag> <n_labelled> <physics_weight> <epochs> [n_unlabelled]

Options are set through environment variables (listed at the top of `pino_jfo.py`). The labelled dataset is read from
the file given by `DATA` (arrays `H`, `LD`, `G`, `Ps`, `P`, `Tf`, `params`, `theta`, `zeta`). Settings used in the
paper:

    # data-only FNO
    DATA=dataset_jfo_cut.npz CUT=1 GEO=1 python pino_jfo.py fno 8000 0 200
    # PINO with 5000 unlabelled bearings from the training ranges
    DATA=dataset_jfo_cut.npz CUT=2 CUTW=0.03 GEO=1 WARMUP=0.25 python pino_jfo.py pino_unl 8000 0.1 200 5000
    # PINO with unlabelled bearings from the training and extrapolation ranges
    DATA=dataset_jfo_cut.npz CUT=2 CUTW=0.03 GEO=1 WARMUP=0.25 UNL_RANGE=wide python pino_jfo.py pino_wide 8000 0.1 200 5000
    # few labels, e.g. 250 labelled bearings
    DATA=dataset_jfo_cut.npz CUT=2 CUTW=0.03 GEO=1 WARMUP=0.25 STEPS=250 python pino_jfo.py pino_n250 250 0.1 200 5000

Each run writes `<tag>.pt`, `<tag>_results.json` and `<tag>_test_pred.npy`. Use `SEED=1`, `SEED=2` for repeated runs.

## License

MIT, see `LICENSE`.
