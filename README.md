# TRIAD: Resolvability-Typed Adaptive Intent Hierarchies for Multimodal Recommendation

This repository contains the reference implementation of

> **TRIAD: Resolvability-Typed Adaptive Intent Hierarchies for Multimodal Recommendation.**

TRIAD is a multimodal recommender organized around a single object: an **exact
three-term variance decomposition** of a user's multimodal evidence. Each item
modality is given a closed-form distributional representation whose dispersion is
read in one forward pass and **certified offline** against a neighborhood-coupled
flow-matching gold standard. By the law of total variance, a user's total
dispersion splits exactly into

- **`B_u` — between-item breadth** (how far apart the consumed items lie; *resolvable*),
- **`G_u` — between-modality gap** (how much the visual and textual signals disagree),
- **`W_u` — within-item fuzziness** (how noisy each item's evidence is; *irreducible*).

This typed ambiguity drives a **resolvability-typed adaptive intent depth**: a
residual-quantized intent hierarchy whose per-user depth goes *deeper* on
resolvable breadth and *halts earlier* on irreducible fuzziness.

The implementation is built on the [MMRec](https://github.com/enoche/MMRec)
toolbox; we thank its authors for their open-source contribution.

---

## Paper ↔ Code map

The core model lives in [`src/models/triad.py`](src/models/triad.py). Classes and
methods are named to mirror the methodology section of the paper:

| Paper component (section) | Code object (`src/models/triad.py`) |
| --- | --- |
| Multimodal graph backbone (Sec. IV-D) | `CollaborativeGraphConv`, `ContentGraphConv`, `ModalityProjection`, `GraphConvLayer` |
| Within-item dispersion `w̃_i` / flow audit `L_FM` (Sec. IV-B, IV-E) | `FlowDispersionSensor` (+ `ConditionalVectorField`) |
| Three-term variance decomposition `B_u + G_u + W_u` (Sec. IV-C) | `TRIAD._user_dispersion`, `FlowDispersionSensor.served_dispersion` |
| Stable residual-quantized codebook level (Sec. IV-F) | `ResidualQuantizer` |
| Resolvability-typed adaptive intent depth (Sec. IV-F, IV-G) | `AdaptiveIntentHierarchy` |
| Depth-attention aggregation `u_u^{(d)} = h_u + Σ α_l c_u^{(l)}` (Sec. IV-G) | `DepthAttentionAggregator` |
| Full model, prediction & optimization (Sec. IV-H) | `TRIAD` |

### Loss terms (Sec. IV-H)

The training objective combines a ranking loss with the module-specific terms;
the configuration weights map to the paper symbols as:

| Config key (`configs/model/TRIAD.yaml`) | Paper symbol | Role |
| --- | --- | --- |
| `beta1` | `β₁` | dispersion (flow audit / NLL) loss weight |
| `beta2` | `β₂` | intent codebook commitment loss weight |
| `lambda_cross` | — | between-modality (cross-modal flow) synergy weight |
| `lambda1` | — | self-supervised contrastive loss weight |

> **Note.** A few non-core routines (the user–user graph construction in
> `get_knn_uu_mat` and the low-level message passing in `GraphConvLayer`) are
> stubbed in this release and will be published in full upon paper acceptance.

---

## Repository structure

```
.
├── README.md
├── preprocessing/
│   ├── README.md                     # how to build the user–user graph
│   └── dualgnn-gen-u-u-matrix.py
└── src/
    ├── main.py                       # entry point
    ├── models/
    │   └── triad.py                  # ← the TRIAD model
    ├── configs/
    │   ├── overall.yaml              # global training / evaluation settings
    │   ├── dataset/<name>.yaml       # per-dataset feature & field settings
    │   └── model/TRIAD.yaml          # TRIAD hyper-parameters
    ├── common/                       # abstract recommender, trainer, losses, init
    └── utils_package/                # data loading, evaluation, config, logging
```

---

## Environment

The code is developed and tested under the same environment as MMRec. Install
the dependencies from the MMRec `requirements.txt`:

```bash
pip install -r requirements.txt
```

Key dependencies: PyTorch, PyTorch Geometric (`torch_geometric`, `torch_scatter`),
NumPy, SciPy, PyYAML.

---

## Data

We use three public multimodal benchmarks with pre-extracted visual (V) and
textual (T) item features:

- **Amazon-Baby** and **Amazon-Sports** (Amazon review collection)
- **TikTok** (keyframe visual + textual features)

The Baby and Sports subsets follow the MMRec data format. Place each dataset
under `./data/<dataset_name>/` (the path is set by `data_path` in
`src/configs/overall.yaml`, default `../data/`), e.g.:

```
data/
└── baby/
    ├── baby.inter
    ├── image_feat.npy
    ├── text_feat.npy
    └── user_graph_dict.npy
```

### Preprocessing: user–user graph

Before training, build the user–user graph for a dataset:

```bash
cd preprocessing
python dualgnn-gen-u-u-matrix.py -d baby      # or: -d sports
```

This produces the `user_graph_dict.npy` consumed by the backbone.

---

## Running

From the `src` directory, run `main.py` and pick a dataset:

```bash
cd src
python main.py --model TRIAD --dataset baby
```

Arguments:

| Flag | Default | Meaning |
| --- | --- | --- |
| `--model`, `-m` | `TRIAD` | model name (resolves to `models/triad.py` and `configs/model/TRIAD.yaml`) |
| `--dataset`, `-d` | `baby` | dataset name (resolves to `configs/dataset/<name>.yaml`) |
| `--gpu_id` | `0` | CUDA device id |

Hyper-parameters listed under `hyper_parameters` in the YAML configs are
grid-searched automatically; results for every combination, and the best one,
are written to the run log.

---

## Citation

If you find this work useful, please cite:

```bibtex
@article{miao2024triad,
  title   = {TRIAD: Resolvability-Typed Adaptive Intent Hierarchies for Multimodal Recommendation},
  author  = {Miao, Yuchen and Wang, Zijun and Liu, Ke and Liu, Xulong},
  year    = {2024}
}
```

## Acknowledgements

This implementation is based on the
[MMRec: A modern MultiModal Recommendation toolbox](https://github.com/enoche/MMRec).
We are grateful to its authors.
