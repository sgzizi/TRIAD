# TRIAD: Resolvability-Typed Adaptive Intent Hierarchies for Multimodal Recommendation

Reference implementation of **TRIAD**, a multimodal recommender that decides
*how much* intent modeling each user needs from the structure of their own
multimodal evidence.

The idea is one exact object. For every user we take the total dispersion of the
multimodal signals across the items they consumed and split it, by the law of
total variance, into three additive parts:

- **`B_u` — between-item breadth:** how far apart the consumed items lie (broad,
  diverse taste). This is *resolvable* — finer intent codes can separate it.
- **`G_u` — between-modality gap:** how much an item's visual and textual signals
  disagree.
- **`W_u` — within-item fuzziness:** how noisy each item's evidence is on its own.
  This is *irreducible* — no amount of depth removes it.

TRIAD turns this typed ambiguity into a **per-user adaptive intent depth**: a
residual-quantized intent hierarchy that goes **deeper** when a user's breadth is
resolvable and **halts earlier** when their evidence is merely fuzzy. The
dispersion that drives this is read cheaply at serving time and certified
offline against a flow-matching gold standard, so it never costs a per-query
solver.

This code is built on the [MMRec](https://github.com/enoche/MMRec) toolbox; we
thank its authors for their open-source work.

> ⚠️ **Status.** This repository accompanies a paper currently under review. A
> few non-core routines (the user–user graph construction in `get_knn_uu_mat`
> and the low-level message passing in `GraphConvLayer`) are stubbed here and
> will be released in full once the paper is accepted.

---

## How the model is organized

The whole model lives in [`src/models/triad.py`](src/models/triad.py). A forward
pass flows through these components, each implemented as its own class:

1. **Multimodal graph backbone** — a LightGCN-style backbone with a
   *collaborative* branch over user/item ID embeddings and a *content* branch
   that smooths projected visual/textual features over a frozen item–item kNN
   graph.
   → `CollaborativeGraphConv`, `ContentGraphConv`, `ModalityProjection`,
   `GraphConvLayer`

2. **Within-item dispersion sensor** — a per-modality conditional flow matched to
   each item's collaborative neighborhood. It produces the per-item dispersion
   used downstream, and doubles as the offline audit that certifies the cheap
   served signal.
   → `FlowDispersionSensor` (with `ConditionalVectorField`)

3. **Per-user variance decomposition** — item-level dispersion is aggregated into
   each user's within-item fuzziness `W_u`, the typed signal the depth rule
   consumes.
   → `TRIAD._user_dispersion`, `FlowDispersionSensor.served_dispersion`

4. **Adaptive intent hierarchy** — the user representation is decomposed into a
   coarse-to-fine sequence of intent codes by residual quantization, with a
   per-user halting rule that stops going deeper once the user's fuzziness says
   extra codes would only chase noise.
   → `AdaptiveIntentHierarchy` (with `ResidualQuantizer`)

5. **Depth-attention aggregation** — the active intent codes are combined back
   into the final user representation by content-based attention.
   → `DepthAttentionAggregator`

The final score for a user–item pair is the dot product of the depth-selected
user representation and the item representation.

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
its dependencies:

```bash
pip install -r requirements.txt
```

Key packages: PyTorch, PyTorch Geometric (`torch_geometric`, `torch_scatter`),
NumPy, SciPy, PyYAML.

---

## Data

We use three public multimodal benchmarks with pre-extracted visual (V) and
textual (T) item features: **Amazon-Baby**, **Amazon-Sports**, and **TikTok**.
The Baby and Sports subsets follow the MMRec data format.

Place each dataset under `./data/<dataset_name>/` (the path is set by `data_path`
in `src/configs/overall.yaml`, default `../data/`):

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

| Flag | Default | Meaning |
| --- | --- | --- |
| `--model`, `-m` | `TRIAD` | model name (resolves to `models/triad.py` and `configs/model/TRIAD.yaml`) |
| `--dataset`, `-d` | `baby` | dataset name (resolves to `configs/dataset/<name>.yaml`) |
| `--gpu_id` | `0` | CUDA device id |

Hyper-parameters listed under `hyper_parameters` in the YAML configs are
grid-searched automatically; results for every combination, and the best one,
are written to the run log.

---

## Configuration

The main knobs live in `src/configs/model/TRIAD.yaml`:

| Key | Meaning |
| --- | --- |
| `embedding_size`, `n_layers`, `n_mm_layers`, `knn_k`, `mm_image_weight` | backbone size and graph propagation |
| `intent_depth` | maximum number of intent-codebook levels `L` |
| `intent_codebook_size` | entries per codebook level |
| `halt_rate` | rate of the fuzziness-typed halting threshold |
| `depth_gamma`, `lambda_div` | depth-attention penalty and codebook diversity weights |
| `lambda_cross` | cross-modal (between-modality) synergy weight in the dispersion sensor |
| `beta1` | weight of the dispersion (flow audit) loss |
| `beta2` | weight of the intent commitment loss |
| `cl_weight`, `lambda1`, `epsilon` | self-supervised contrastive view and its loss weight |

Global training and evaluation settings (epochs, batch size, metrics, top-K)
live in `src/configs/overall.yaml`.

---

## Acknowledgements

This implementation is based on the
[MMRec: A modern MultiModal Recommendation toolbox](https://github.com/enoche/MMRec).
We are grateful to its authors.
