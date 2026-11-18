# Behavior-Specialized Mixture of Experts for Continual Intrusion Detection on the CAN Bus

Official implementation of a two-branch Mixture-of-Experts (MoE) intrusion detection system for in-vehicle networks that learns new attack types over time (class-incremental continual learning) without forgetting previously learned ones.

> **Paper:** *Title, authors and venue to be added upon publication.*

---

## Overview

Each window of 64 consecutive CAN messages is represented as a graph whose nodes are the CAN IDs observed in the window. The model processes this graph with two branches:

| Component | Role |
|---|---|
| **Identity branch** | Node features and a CAN-ID embedding, followed by graph mixing. Captures the normal context of each CAN ID. IDs never observed in normal traffic share a single embedding, so the branch cannot memorise attack-only IDs. |
| **Behavior branch** | A mixture of experts that **never sees the CAN ID**. Each node is described by how unusual its behavior is relative to the normal traffic of its own ID (conformal surprise), together with payload-continuity and window-level features. A router sends each node to the top-2 of a **null expert** and the behavior experts; normal nodes are routed to the null expert and contribute nothing to the representation. |
| **Readout and classifier** | Mean and max pooling over nodes, followed by a cosine classifier over normal traffic and the attack classes. |

When a new attack type arrives, a **label-gated, validation-verified expansion rule** decides how the model is updated:

1. A phase with a new label is always learned; a phase without new labels is ignored if the current model already detects it.
2. The model first tries to **reuse** the existing behavior experts (experts and their routing frozen).
3. Reuse is kept only if, on validation data, old classes do not degrade, every new class is learned, and the existing experts actually serve every new class (limited routing to the null expert). Otherwise the model **adds one expert** for the new behavior.

Old experts and their router rows remain frozen after they are learned. A bounded replay memory (reservoir sampling) and the normal traffic of each new recording are replayed during updates.

## Highlights

- **Behavior-level experts:** experts specialise on attack behaviors (frequency anomalies, ID injection, signal spoofing) rather than on CAN identities.
- **Selective growth:** one small expert is added only when a genuinely new behavior appears; known behaviors reuse existing experts.
- **Vehicle-agnostic behavior features:** conformal surprise places every signal of every CAN ID on the same scale.
- **Strict continual-learning protocol:** the normal reference used to build features comes only from the recordings available at the first phase.

---

## Repository structure

```
.
├── preprocess_window.py         # CSV logs -> window graphs, behavior features, splits
├── train_continual_window.py    # continual training of the proposed model (and its ablations)
├── train_baselines.py           # continual-learning baselines under the same setting
├── diag_two_branch.py           # diagnostics: null expert, branch ablation, ID reliance, probes
├── summarize_v5.py              # result tables for the proposed model (seeds, phase orders)
├── summarize_baselines.py       # result tables for the baseline comparison
├── check_vehicles.py            # dataset statistics: CAN IDs and attack targets per vehicle
├── fast_data_window.py          # GPU-resident packing and batching of window graphs
├── utils.py, losses.py, preprocess.py   # shared helpers
├── networks/
│   ├── moe_ivn_ids_window.py    # proposed model: two-branch behavior MoE with window features
│   ├── moe_ivn_ids_behavior.py  # behavior MoE (null expert, routing losses, expansion)
│   ├── moe_ivn_ids_cosine.py    # one-branch MoE (ID-routed) and the cosine classifier
│   ├── moe_ivn_ids.py           # earlier MoE variant (kept for the --arch v2 option)
│   └── moe_layer.py             # MoE layer used by the one-branch model
├── run/
│   ├── run_v5.sh                # proposed model: three seeds and three phase orders
│   ├── run_v4.sh                # ablations: number of initial experts, attacked-node balance loss
│   └── run_baselines.sh         # all baselines
├── notebooks/                   # figures: routing heatmaps, comparisons, diagnostics
└── data/                        # raw CSV logs (not tracked), see "Dataset"
```

Scripts in `run/` use paths relative to the repository root and must be launched from it, for example `bash run/run_v5.sh`. Notebooks in `notebooks/` expect the repository root one level up (`ROOT = Path("..")`).

## Requirements

- Python 3.10 or later
- PyTorch 2.x with CUDA (experiments were run on a single NVIDIA RTX 4090)
- numpy, pandas, matplotlib, tqdm

```bash
pip install torch numpy pandas matplotlib tqdm
```

## Dataset

Experiments use the **can-ml** dataset (Lampe and Meng, *Computers & Security*, 2024): the 2011 Chevrolet Impala for the main experiments, and the 2011 Chevrolet Traverse and 2016 Chevrolet Silverado for cross-vehicle analysis. Each vehicle has nine attack recordings: `DoS`, `fuzzing`, `gear`, `interval`, `rpm`, `speed`, `standstill`, `systematic` and `combined`. Each CSV file provides `timestamp`, `arbitration_id` (decimal), `data_field` and `attack`.

Place the raw files as follows (preprocessed data is written next to them):

```
data/
├── 2011-chevrolet-impala/merged/{DoS,fuzzing,gear,interval,rpm,speed,standstill,systematic,combined}.csv
├── 2011-chevrolet-traverse/*.csv
└── 2016-chevrolet-silverado/*.csv
```

---

## Usage

### 1. Preprocessing

```bash
python preprocess_window.py \
    --input_dir  ./data/2011-chevrolet-impala/merged \
    --output_dir ./data/2011-chevrolet-impala/preprocessed_v3_variant \
    --window_size 64 --max_normal_per_file 10000 --seed 42 --profile_min_count 20 \
    --behavior_norm surprise --payload_features \
    --x_norm_source normal --profile_files dos,fuzzing,gear,interval,rpm
```

`--profile_files` lists the recordings available at the first phase; only their normal traffic is used to build the normal reference. Each phase order therefore has its own preprocessed copy:

| Phase order | `--output_dir` | `--profile_files` |
|---|---|---|
| `variant` | `preprocessed_v3_variant` | `dos,fuzzing,gear,interval,rpm` |
| `late_injection` | `preprocessed_v3_late_injection` | `dos,gear,interval,rpm` |
| `late_spoof` | `preprocessed_v3_late_spoof` | `dos,fuzzing,interval,systematic` |

### 2. Training the proposed model

```bash
python train_continual_window.py \
    --data_dir ./data/2011-chevrolet-impala/preprocessed_v3_variant \
    --save_dir ./save/impala/ours_variant_s42 --seed 42 \
    --arch window --num_experts 4 --phase_config variant \
    --mode expand_auto --decision label --reuse_null_rule max \
    --reuse_tol 0.01 --reuse_min_f1 0.95 --reuse_max_null 0.25 --refresh_min_acc 0.95 \
    --null_scale --mask_unknown_ids \
    --train_scope backbone --freeze_old_gate --no_balance_on_expand \
    --null_weight 0.1 --active_weight 0.5 --excl_weight 0.1 --identity_drop 0.3 \
    --memory_size 10000 --memory_policy reservoir --normal_protocol recording \
    --phase1_epochs 100 --phase_epochs 60 --phase_scheduler cosine --logit_adjust 0.6
```

All runs of the paper (three seeds, three phase orders) can be reproduced with:

```bash
bash run/run_v5.sh
python summarize_v5.py
```

### 3. Baselines

`train_baselines.py` shares the data pipeline, phase orders, replay memory, training budget, checkpoint selection and metrics with the proposed model; only the network and the continual-learning strategy differ. All baselines use the same dense graph encoder with the same inputs.

| `--method` | Method |
|---|---|
| `finetune` | Fine-tuning on new data only (lower bound) |
| `joint` | Retraining on all data seen so far (upper bound) |
| `er` | Experience Replay (Chaudhry et al., 2019) |
| `cbrs` | Class-balanced reservoir memory (Chrysakis and Moens, ICML 2020) |
| `ewc` | Elastic Weight Consolidation (Kirkpatrick et al., PNAS 2017) |
| `lwf` | Learning without Forgetting (Li and Hoiem, ECCV 2016) |
| `icarl` | iCaRL (Rebuffi et al., CVPR 2017) |
| `derpp` | DER++ (Buzzega et al., NeurIPS 2020) |
| `der` | Dynamically Expandable Representation (Yan et al., CVPR 2021) |

```bash
SMOKE=1 bash run/run_baselines.sh              # one epoch per phase, sanity check
SEEDS="42 43 44" bash run/run_baselines.sh     # full runs
python summarize_baselines.py
```

### 4. Diagnostics

```bash
python diag_two_branch.py --run ./save/impala/ours_variant_s42 --out ./save/diag/ours_variant_s42.json
```

Reports the contribution of normal nodes to the behavior output (null expert), the effect of switching off each branch, the reliance on CAN identities, and linear probes of each representation.

---

## Phase orders

| Order | Phase 1 | Phase 2 | Phase 3 | Phase 4 | Phase 5 |
|---|---|---|---|---|---|
| `variant` | normal, DoS (part 1), fuzzing, gear, interval, rpm | speed | DoS (part 2) | standstill | systematic, combined |
| `late_injection` | normal, DoS (part 1), gear, interval, rpm | fuzzing | DoS (part 2) | speed, standstill | systematic, combined |
| `late_spoof` | normal, DoS (part 1), fuzzing, interval, systematic | gear | DoS (part 2) | rpm, speed | standstill, combined |

`late_injection` and `late_spoof` introduce a behavior type (ID injection or signal spoofing) only after the first phase, to test whether the model expands when a new behavior appears.

## Outputs and metrics

Each run directory contains `results.json` (per-phase metrics, decisions, routing statistics), `cl_summary.json` (final metrics), `preds_phase*.npz` (test predictions and per-node routing), the best checkpoint of each phase and a log file.

| Metric | Definition |
|---|---|
| Macro F1 | One-vs-rest F1 averaged over all classes seen so far, on the test split |
| Normal FPR | Fraction of normal windows classified as an attack |
| Forgetting | Best earlier F1 of a class minus its F1 after the last phase (mean and max over classes) |
| New-class F1 | F1 of each new class right after the phase in which it is learned |

## Results

Impala, phase order `variant`, after the last phase, three seeds (mean ± std):

| Method | Macro F1 | Normal FPR | Parameters |
|---|---|---|---|
| **Ours** | 0.9885 ± 0.0009 | 0.0033 ± 0.0003 | 265K initially, +12K per added expert |
| Baselines | see `summarize_baselines.py` | | |

*Full comparison tables will be added with the paper.*

---

## Citation

```bibtex
@article{tbd,
  title   = {Behavior-Specialized Mixture of Experts for Continual Intrusion Detection on the CAN Bus},
  author  = {To be added},
  journal = {To be added},
  year    = {2026}
}
```

Please also cite the can-ml dataset if you use it.

## License

To be added.

## Contact

For questions about the code, please open an issue in this repository.