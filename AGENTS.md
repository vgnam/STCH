# AGENTS.md — STCH Project

This file provides context and instructions for AI agents working on the **STCH** (Smooth Tchebycheff Scalarization for Multi-Objective Optimization) codebase.

---

## Project Overview

STCH is a research codebase accompanying the ICML 2024 paper:
> *Smooth Tchebycheff Scalarization for Multi-Objective Optimization*  
> Xi Lin, Xiaoyuan Zhang, Zhiyuan Yang, Fei Liu, Zhenkun Wang, Qingfu Zhang

The repository contains two independent components:

1. **STCH_PSL** — Pareto Set Learning on synthetic and real-world benchmark problems.
2. **STCH_MTL** — Multi-Task Learning integration via the `LibMTL` library.

For deep algorithmic details (probabilistic model, energy function, MCMC procedure), see `AGENT.md` in the repo root.

---

## Directory Structure

```
STCH/
├── AGENT.md                # Algorithm design doc (mathematical spec)
├── AGENTS.md               # This file — agent instructions
├── README.md               # Human-facing project overview
│
├── STCH_PSL/               # Pareto Set Learning component
│   ├── run_stch_psl.py     # Main training loop (~200 lines)
│   ├── model.py            # Simple FC ParetoSetModel
│   ├── problem.py          # Benchmark problem definitions
│   ├── data/RE/            # RE problem Pareto-front / ideal-nadir data
│   └── README.md
│
└── STCH_MTL/               # Multi-Task Learning component
    ├── LibMTL/             # Core library (forked from LibMTL)
    │   ├── weighting/
    │   │   ├── STCH.py     # <-- Our method implementation
    │   │   └── ...         # 15+ other weighting strategies
    │   ├── architecture/   # MTL architectures (MTAN, MMoE, PLE, ...)
    │   ├── model/          # Backbone networks (ResNet, SegNet)
    │   ├── trainer.py      # Training orchestrator
    │   ├── config.py       # CLI argument parser
    │   └── ...
    ├── examples/
    │   ├── nyu/            # NYUv2 scene-understanding example
    │   ├── office/         # Office-31 / Office-Home domain adaptation
    │   └── qm9/            # QM9 molecular property prediction
    ├── docs/               # Sphinx documentation source
    └── README.md
```

---

## Technology Stack

- **Language**: Python 3.8+
- **Deep Learning**: PyTorch 1.8+ (CUDA 11.x recommended)
- **Scientific**: NumPy, SciPy, matplotlib
- **MOO**: `pymoo` (for HV indicator)
- **Docs**: Sphinx + ReadTheDocs theme

---

## How to Run

### STCH_PSL (Pareto Set Learning)

No installation step required; pure Python with standard PyTorch/NumPy.

```bash
cd STCH_PSL
python run_stch_psl.py
```

Editable hyper-parameters are at the top of `run_stch_psl.py`:
- `ins_list` — which benchmarks to run (`f1`..`f6`, `re21`, `re24`, `re33`, `re36`, `re37`)
- `method` — scalarization: `'ls'` (linear), `'tch'` (hard Tchebycheff), `'stch'` (smooth Tchebycheff)
- `n_steps` — training iterations (default 2000)
- `n_pref_update` — preferences sampled per step (default 10)
- `n_run` — independent runs for statistics

Outputs:
- Prints hypervolume mean per run.
- Generates matplotlib scatter plots of the learned Pareto front (2-D or 3-D).

### STCH_MTL (Multi-Task Learning)

Requires installing `LibMTL` in editable mode:

```bash
cd STCH_MTL
pip install -r requirements.txt
pip install -e .
```

Example training command (NYUv2):

```bash
cd examples/nyu
python main.py \
  --weighting STCH \
  --arch HPS \
  --dataset_path /path/to/nyuv2 \
  --gpu_id 0 \
  --scheduler step \
  --mode train \
  --save_path ./results
```

Key CLI flags:
- `--weighting {EW,GradNorm,UW,MGDA,DWA,GLS,PCGrad,GradDrop,IMTL,GradVac,CAGrad,Nash_MTL,RLW,MoCo,Aligned_MTL,DB_MTL,STCH}`
- `--arch {HPS,Cross_stitch,MMoE,MTAN,CGC,PLE,LTB,DSelect_k}`

---

## Code Style & Conventions

- Follow **PEP 8** with a 100-character line limit.
- Use **4-space indentation** (no tabs).
- Import order: standard library → third-party → local modules.
- Doc-strings: Google-style or concise plain-text; not mandatory for one-liners.
- Type hints are welcome but not required (the original codebase does not use them).
- Keep scripts self-contained and runnable without external configuration files when possible.

---

## Where to Find Key Logic

| Concern | File(s) |
|---------|---------|
| Smooth Tchebycheff loss (PSL) | `STCH_PSL/run_stch_psl.py` lines 110–113 |
| Hard Tchebycheff loss (PSL) | `STCH_PSL/run_stch_psl.py` lines 106–108 |
| Pareto Set network (FC) | `STCH_PSL/model.py` |
| Benchmark problems | `STCH_PSL/problem.py` |
| STCH weighting strategy (MTL) | `STCH_MTL/LibMTL/weighting/STCH.py` |
| Abstract weighting base class | `STCH_MTL/LibMTL/weighting/abstract_weighting.py` |
| Trainer / training loop | `STCH_MTL/LibMTL/trainer.py` |

---

## Adding a New Benchmark Problem (PSL)

1. Define the problem class in `STCH_PSL/problem.py`.
   - Required attributes: `n_dim`, `n_obj`, `nadir_point`
   - Required method: `evaluate(x: torch.Tensor) -> torch.Tensor`
2. Add the problem name to `ins_list` in `run_stch_psl.py`.
3. Provide `ideal_point` and `nadir_point` in the main loop if it is not an RE problem.

---

## Adding a New Weighting Strategy (MTL)

1. Create a new file under `STCH_MTL/LibMTL/weighting/<MyMethod>.py`.
2. Inherit from `AbsWeighting` (see `abstract_weighting.py`).
3. Implement `init_param(self)` and `backward(self, losses, **kwargs)`.
4. Register the class in `STCH_MTL/LibMTL/weighting/__init__.py`.
5. Add CLI mapping in `STCH_MTL/LibMTL/config.py` if needed.

Reference implementation: `STCH_MTL/LibMTL/weighting/STCH.py`.

---

## Testing

There is **no automated test suite** in this research repository.  
Verification is manual:

1. Run `run_stch_psl.py` and confirm hypervolume prints without errors.
2. Run any MTL example with `--weighting STCH` and confirm loss decreases.
3. Check that generated Pareto-front plots look reasonable (no NaN / collapse).

When modifying existing code, preserve backward compatibility with the default CLI commands shown in the README files.

---

## Common Pitfalls

| Pitfall | Why it happens | Fix |
|---------|---------------|-----|
| `mu` too large in STCH_PSL | Gradient explosion in `logsumexp` | Keep `mu <= 0.1` during warmup; the MTL version caps it via `kwargs['STCH_mu']`. |
| `nadir_vector` is `None` in MTL | `warmup_epoch` not reached yet | The trainer automatically falls back to equal weighting until `epoch >= warmup_epoch`. |
| NaN losses | Log of zero/negative numbers | A small epsilon (`1e-20`) is already added in `STCH.py` and `run_stch_psl.py`; do not remove it. |
| Missing RE data files | `data/RE/` not on disk | The `.dat` files are tracked in Git; clone with `--recursive` or restore them. |
| LibMTL import errors | Package not installed in editable mode | Run `pip install -e .` from `STCH_MTL/`. |

---

## Citation

If you modify this codebase, retain the original citations in README files and doc-strings:

```bibtex
@inproceedings{lin2024smooth,
  title={Smooth Tchebycheff Scalarization for Multi-Objective Optimization},
  author={Lin, Xi and Zhang, Xiaoyuan and Yang, Zhiyuan and Liu, Fei and Wang, Zhenkun and Zhang, Qingfu},
  booktitle={International Conference on Machine Learning},
  year={2024}
}
```

For the MTL component, also cite LibMTL:

```bibtex
@article{lin2023libmtl,
  title={{LibMTL}: A {P}ython Library for Multi-Task Learning},
  author={Baijiong Lin and Yu Zhang},
  journal={Journal of Machine Learning Research},
  volume={24},
  number={209},
  pages={1--7},
  year={2023}
}
```

---

## Contact & Issues

- Paper / general questions: see `README.md` for author emails.
- LibMTL-specific issues: https://github.com/median-research-group/LibMTL/issues

---

*Last updated: 2024-05-11*
