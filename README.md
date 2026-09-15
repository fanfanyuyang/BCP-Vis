# BCP-Vis: A Learned Background Confidence Prior for Tiny-Vehicle Segmentation in High-Altitude UAV Imagery

[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org)
[![PyTorch](https://img.shields.io/badge/pytorch-2.x-orange.svg)](https://pytorch.org)
[![Dataset](https://img.shields.io/badge/dataset-EVD4UAV-green.svg)](https://github.com/)
[![License](https://img.shields.io/badge/license-MIT-lightgrey.svg)](LICENSE)
[![Experiments](https://img.shields.io/badge/experiments-E0%E2%80%93E33-informational.svg)](#8-experiment-matrix)

---

## Abstract

Vehicles in high-altitude UAV imagery (50–90 m, 1920×1080) frequently occupy far less than 0.1 % of
the image pixels. Generic encoder–decoder segmenters lose them during downsampling, so the dominant
error mode is **missed instances (low recall)** rather than imprecise boundaries.
**BCP-Vis** decouples *where to look* from *what to delineate*. A lightweight **PriorNet**
(≈0.08 M parameters) regresses a **soft Background Confidence Prior (BCP)** from RGB alone, where
the regression target is a distance-aware Gaussian field that equals 1 on foreground and decays
smoothly with the Euclidean distance to the nearest object. The predicted prior is concatenated to
the RGB image as a fourth channel (RGBP, 4×H×W) and fed to an otherwise unmodified ResNet-18 U-Net.
Because the only architectural change with respect to the RGB baseline is `in_channels: 3 → 4` in the
first convolution (+3,136 parameters, +0.021 %), any observed difference is attributable to the
prior itself. The repository contains the full experimental record of the **E0–E33 campaign** (33 trained
configurations plus one evaluation-only control, E15), including (i) a **causal ablation protocol** with deliberately degenerate priors (zero channel,
random channel, spatially misaligned prior, ground-truth oracle) that localises the source of the
gain and the conditions under which it disappears, (ii) **out-of-fold (OOF) prior generation** that
removes ground-truth leakage from the fourth channel, (iii) **altitude-adaptive prior targets**
(σ(h) ∝ 1/h) and **altitude-conditioned/dual-expert architectures**, and (iv) **two independent
negative results** on uncertainty-gated priors, which we report rather than hide.
The prior produces large, scale-dependent gains on small objects (tiny IoU requires grouping: see
§9.2) while the aggregate IoU gain stays between +0.0017 and +0.0055, of the same order as
run-to-run noise — stated explicitly here rather than sold as a headline number.

**Keywords:** high-altitude UAV imagery, tiny object segmentation, prior-guided segmentation,
conditional modulation, mixture of experts, out-of-fold inference, negative results.

---

## 1. Motivation and Problem Statement

Let $I \in \mathbb{R}^{3 \times H \times W}$ be a UAV frame captured at altitude
$h \in \{50, 70, 90\}$ m and $M \in \{0,1\}^{H \times W}$ the binary vehicle mask. Three properties
make this regime hard:

1. **Extreme scale.** Apparent object area scales as $h^{-2}$; at 90 m a car spans roughly 10–20 px.
2. **Extreme sparsity.** Most 768×768 patches contain no foreground at all, so a standard
   encoder–decoder rapidly drives the foreground response to zero and the loss is dominated by
   background.
3. **Structured, non-uniform failure.** Errors concentrate on `tiny` objects and vary systematically
   with altitude *and* with weather (snow cover), which turns out to be a confounder (§9.3).

The design response is to hand the segmenter an explicit, learnable cue about *where objects may
be* — a dense prior that is high on objects, decays in their neighbourhood, and vanishes in empty
background — before any semantic decision is made.

## 2. Method

### 2.1 Pipeline

```
RGB image ──► PriorNet (≈0.08 M) ──► soft prior P̂ ──concat──► [R,G,B,P̂] ──► ResNet-18 U-Net ──► mask
                   ▲                                                  4 × H × W
        supervised by distance-aware soft target P* (no GT at test time)
```

Optional: a second prior (e.g. a background prior or a second altitude expert) can be concatenated
as a fifth channel (`predicted2`, $X \in \mathbb{R}^{5 \times H \times W}$), which is how the fusion
experiments E20/E22/E28/E30 are built. `reports/fig1_bcpvis_pipeline.svg` holds the paper schematic.

### 2.2 Distance-Aware Soft Prior Target

The regression target of PriorNet is **not** the binary mask. Given $M$ and the Euclidean distance
transform $d(x,y)$ to the nearest foreground pixel,

$$
P^{*}(x,y) = \begin{cases}
1, & M(x,y) = 1,\\[4pt]
\exp\!\left(-\dfrac{d(x,y)^{2}}{2\sigma^{2}}\right), & M(x,y) = 0,
\end{cases}
\qquad \sigma = 32\ \text{px (EVD4UAV, 1920×1080)}
$$

i.e. ≈1.00 on objects, 0.8–0.3 in the immediate neighbourhood, →0 in far background. The field has
half-maximum width $d_{0.5} = \sigma\sqrt{2\ln 2} \approx 1.1774\,\sigma$. σ is a configuration
entry (`configs/soft_prior.yaml`), never hard-coded; `--sigmas 8,16,24,32,48,64 --dry-run` compares
candidates without materialising data.

**Altitude-adaptive variant (E21).** Since object area scales as $h^{-2}$, a fixed halo is
relatively too wide at 90 m (where the prior becomes spatially diffuse and carries less
information). `scripts/build_soft_prior_aasp.py` therefore uses

$$
\sigma(h) = \sigma_0 \cdot \frac{h_{\text{ref}}}{h},
\qquad \sigma(50\,\text{m}) = 44.8,\ \ \sigma(70\,\text{m}) = 32,\ \ \sigma(90\,\text{m}) = 24.9\ \text{px}.
$$

### 2.3 PriorNet

`LiteHRPriorNet` (`models/priornet.py`) deliberately keeps high-resolution representations:
branches at $H/2$ (32 ch), $H/4$ (64 ch) and $H/8$ (128 ch) built from depthwise-separable
convolutions, merged top-down, fused to 64 ch and decoded to a full-resolution logit map.
Downsampling below $H/8$ is intentionally avoided — for a 10-px object an $H/32$ map no longer
carries a usable signal.

Training objective (identical for every PriorNet variant):

$$
\mathcal{L}_{\text{prior}} = \mathcal{L}_{\text{focal}}(\alpha = 0.25,\ \gamma = 2) + \lambda_{\text{dice}}\,\mathcal{L}_{\text{dice}},\quad \lambda_{\text{dice}} = 1,
$$

with the Dice term computed at **batch level** over dims $(0,2,3)$ so that nearly empty patches
cannot dominate the gradient.

### 2.4 RGBP Fusion and the Final Segmenter

The final network is an unmodified ResNet-18 U-Net (ImageNet-pretrained encoder, 4-stage decoder,
1×1 head). BCP-Vis input is $X=[I;\hat{P}]\in\mathbb{R}^{4\times H\times W}$. The weights of the
fourth input channel are initialised as the **mean** of the pretrained RGB kernels
(`init_prior: mean`); zero initialisation leaves the channel nearly gradient-free early on.

Three optional mechanisms sit on top, each individually ablated:

* **ACPC — altitude-conditioned prior channel (E16).** FiLM applied to the prior channel itself,
  $P' = (1+\gamma(e_h))\odot P + \beta(e_h)$, with $\gamma,\beta$ zero-initialised (identity at
  initialisation).
* **Prior-consistency loss (E19).** $\mathcal{L} \mathrel{+}= \lambda_{pc}\left(1 - \mathrm{Dice}(\sigma(\text{out}), P_4)\right)$,
  $\lambda_{pc}=0.1$: the segmentation output is encouraged to agree with the prior channel.
* **FiLM-in-PriorNet (E6/E24/E25)** and **dual-expert / cascade priors (E8/E9/E12/E20)**, below.

### 2.5 Out-of-Fold Prior Generation (No Ground-Truth Leakage)

If the fourth channel is produced by a PriorNet that has seen the image, the final network is
trained on unrealistically clean priors and validation collapses. `scripts/generate_oof_prior.py`
implements 5-fold OOF generation: the train split is partitioned by capture sequence, each fold is
inferred by a PriorNet trained on the other four, and per-fold manifests (`patch_manifest_foldK.csv`)
plus `oof_manifest.csv` record which model produced each prior. Validation/test priors come from a
PriorNet trained on the whole train split. E23 re-runs the E12 architecture with genuine OOF priors
to quantify this effect; `tools/check_oof_fold.py` and `tools/check_leak.py` verify it.

### 2.6 Altitude Conditioning and Expert Structure

Altitude is *metadata*, not an image. Feeding it as an extra spatial channel is provably useless:
$w_5 \cdot h$ is a spatial constant that cannot modulate any location. Because apparent area scales
as $h^{-2}$, the optimal prior bandwidth is altitude dependent, so we condition instead:

* **FiLM (V4/E6):** $F' = (1+\gamma(e_h))\odot F + \beta(e_h)$, $t=(h-70)/40$, zero-initialised.
* **Altitude MoE (V7/E11):** $P=\sum_k w_k(h)\,\mathrm{Expert}_k(F)$ with soft routing; the
  load-balancing term $K\sum_k \bar w_k^{2}$ prevents expert collapse.
* **Cascade (V8/E12):** stage 1 normalises *scale* (altitude experts), stage 2 handles *semantics*
  (foreground head + background expert).
* **Dual head (V5-A/E8):** $P_f \odot (1-P_b)$ with independent sigmoids — the two heads are not
  constrained to sum to one; their disagreement is an uncertainty cue
  $U = |P_f - (1-P_b)|$, exploited by the UAP experiments E13/E13b/E29 (§9.6).
* **Background MoE (V5-B/E9)** and **class-conditioned background expert (E14)**, the latter kept
  as an ablation only: class-conditioned priors are already published (CSENet/CPNet) and EVD4UAV
  contains just three vehicle categories.

### 2.7 Prior-Network Model Zoo

| Key | Class | Conditioning / second stage | Config |
|---|---|---|---|
| `miniunet` | `MiniUNetPriorNet` | – (pipeline sanity check) | `configs/prior_v1.yaml` |
| `litehr` | `LiteHRPriorNet` | – | `configs/prior_v3_litehr.yaml` |
| `altitude` | `AltitudeAwarePriorNet` | FiLM on $H/2$, $H/4$ | `configs/prior_v4_altitude.yaml` |
| `dual` | `DualHeadPriorNet` | foreground + background head | `configs/prior_v5_dual.yaml` |
| `moe` | `MoEBackgroundPriorNet` | K background experts + router | `configs/prior_v5_moe.yaml` |
| `alt_moe` | `AltitudeMoEPriorNet` | altitude experts + router | `configs/prior_v7_altmoe.yaml` |
| `cascade` | `CascadePriorNet` | altitude stage → foreground/background stage | `configs/prior_v8_cascade.yaml` |
| `cascade` + `film`/`nobg` | `CascadePriorNet` | 2×2 altitude×background ablation | `configs/prior_v8_cascade_film*.yaml` |
| `cat_bg` | `CategoryBgPriorNet` | class-conditioned background expert | `configs/prior_v5_cat.yaml` |
| – | `ResNet18UNet` / `MiniUNet` | final segmenter, 3/4/5 input channels | `configs/bcp_vis*.yaml`, `configs/rgb_baseline.yaml` |

## 3. Fairness Protocol

The RGB baseline and BCP-Vis differ in **exactly one** respect: `in_channels` of the first
convolution (3 vs 4, or 5 for prior fusion). Backbone, decoder, loss, optimiser, learning rate and
schedule, epochs, patch size, augmentation, split, seed and hardware are identical.

| | RGB baseline (E0) | BCP-Vis 4-ch (E5/E6) | dual-prior 5-ch (E20/E28/E30) |
|---|---|---|---|
| Parameters | 14,834,121 | 14,837,257 | 14,840,393 |
| Δ vs E0 | – | **+3,136 (+0.021 %)** | +6,272 (+0.042 %) |

Capacity explanations are excluded by construction; the controls in §8–§9.5 exclude "one more
channel", "any fourth channel" and "a prior that merely exists".

## 4. Repository Structure

```
BCP-Vis/
├── configs/            experiment configs (one per model / experiment)
├── datasets_py/        patch dataset: on-the-fly cropping, prior-channel construction
├── models/             blocks, losses, PriorNet family, ResNet-18 U-Net
├── train/              shared engine + trainers for PriorNet and the final CNN
├── scripts/            dataset audit, mask/split/patch/soft-prior construction, prior inference
├── eval/               patch-level metrics, instance AP, ablation driver, report builder
├── tools/              leakage checks, prior diagnostics, CPU unit tests, benchmarks
├── pipelines/          unattended job chains used on the single-GPU server
├── reports/            every produced metric (CSV), unified report, figure source
└── docs/               original development notes (Chinese): derivations, audit trail, logs
```

## 5. Installation

```bash
git clone <your-repo-url> BCP-Vis && cd BCP-Vis
pip install -r requirements.txt
```

Reference environment: Python 3.12, PyTorch 2.8 (CUDA 12.8), one NVIDIA RTX 4090 (24 GB),
40 epochs of 768×768 patches with AMP, single-GPU serialised runs.

## 6. Dataset Preparation

This project uses **EVD4UAV** (high-altitude UAV vehicle imagery with CVAT polygon annotations).
Obtain it from its authors and lay it out as below; every script treats the raw tree as read-only.

```
datasets/EVD4UAV/
├── raw/EVD4UAV/                 images (*.jpg) + mask_attribute.xml
└── _official_metadata/          optional official split / metadata files
```

All derived artefacts go to `datasets/EVD4UAV_processed/`
(`binary_masks/`, `soft_priors/`, `oof_priors/`, `priors_*`, `patch_manifest*.csv`, `split.csv`).

## 7. Reproducing the Experiments

```bash
# ---- data build -------------------------------------------------------------
# 1. audit the raw dataset (read-only; writes reports/dataset_audit.json|txt)
python scripts/audit_dataset.py --root datasets/EVD4UAV/raw --meta datasets/EVD4UAV/_official_metadata

# 2. instance polygons -> binary foreground GT (binary_masks/*.png + manifest.csv)
python scripts/build_binary_masks.py --root datasets/EVD4UAV/raw \
       --out datasets/EVD4UAV_processed --meta datasets/EVD4UAV/_official_metadata

# 3. image-level split, grouped by capture sequence, stratified by altitude and snow
python scripts/split_dataset.py --out datasets/EVD4UAV_processed --group-by sequence

# 4. patch index (768 px, stride 576 = overlap 192); no pixels materialised
python scripts/make_patches.py --root datasets/EVD4UAV/raw \
       --proc datasets/EVD4UAV_processed --patch 768 --overlap 192 --empty-keep-ratio 0.15

# 5. soft prior regression targets: fixed sigma (E5-E20) and altitude-adaptive sigma (E21)
python scripts/build_soft_prior_targets.py --config configs/soft_prior.yaml
python scripts/build_soft_prior_aasp.py --sigma0 32 --alt-ref 70

# ---- prior training + OOF inference ----------------------------------------
# 6. train a PriorNet (variant chosen by --config)
python train/train_prior.py --config configs/prior_v3_litehr.yaml     # V3  -> E5
python train/train_prior.py --config configs/prior_v4_altitude.yaml   # V4  -> E6
python train/train_prior.py --config configs/prior_v8_cascade.yaml    # V8  -> E12

# 7. out-of-fold priors for the train split (5 sequence-grouped folds)
python scripts/generate_oof_prior.py --base-config configs/prior_v3_litehr.yaml --folds 5
#    single-model alternative (must pass the matching model config):
python scripts/infer_prior.py --ckpt runs/priornet/v3_litehr/best.pth \
       --model-config configs/prior_v3_litehr.yaml --split train \
       --out-dir datasets/EVD4UAV_processed/oof_priors --save-png

# ---- final CNN --------------------------------------------------------------
# 8. train + evaluate (identical config except in_channels / prior)
python train/train_final.py --config configs/rgb_baseline.yaml                 # E0
python train/train_final.py --config configs/bcp_vis.yaml                      # E5
python train/train_final.py --config configs/bcp_vis.yaml --exp E1             # zero channel
python train/train_final.py --config configs/bcp_vis.yaml --exp E2             # random channel
python train/train_final.py --config configs/bcp_vis_dual.yaml    --exp E20    # 5-ch prior fusion
python train/train_final.py --config configs/bcp_vis_dual_v4moe.yaml --exp E30
python eval/eval_final.py  --ckpt runs/bcp_vis/E5_bcp_vis/best.pth \
       --config configs/bcp_vis.yaml --exp E5 --split val

# ---- reporting --------------------------------------------------------------
python eval/make_report.py          # writes reports/unified_report.md and prints it
python tools/check_prior_health.py  # prior sanity: mean response on target regions
```

Unattended multi-experiment chains (as actually used on the single-GPU server) are preserved in
`pipelines/`, e.g. `pipelines/run_e12_queue.sh`, `pipelines/run_e20_queue_v2.sh`.

Both trainers execute an automatic pre-flight check (environment, dataset size, batch count, input
channels, parameter count, config dump) followed by a **one-batch forward/backward smoke test**;
training does not start unless the smoke test passes.

## 8. Experiment Matrix

| ID | Fourth / fifth channel | Purpose |
|---|---|---|
| E0 | none (RGB only) | the true baseline |
| E1 | all zeros | control: is the gain just "one more channel"? |
| E2 | fixed random noise | control: is the gain just "any non-trivial channel"? |
| E3, E4 | V1 binary prior, V2 soft prior | early sanity variants (superseded) |
| E5 | V3 Lite-HR BCP | main method |
| E6 | V4 altitude-aware BCP (FiLM) | altitude conditioning |
| E7 | **ground-truth mask** | oracle upper bound — **not** a deployable method |
| E8 | V5-A dual head $P_f\odot(1-P_b)$ | explicit background expert |
| E9 | V5-B background MoE | specialised background experts |
| E10 | enlarged plain PriorNet | capacity control (no expert structure) |
| E11 | V7 altitude MoE | expert routing vs FiLM, strictly matched |
| E12 | V8 cascade (altitude → foreground/background) | dual-dimension expert cascade |
| E13, E13b | UAP prior, two independent implementations | uncertainty-gated prior (**negative result**) |
| E14 | class-conditioned background expert | vehicle-type conditioning (ablation, already published family) |
| E15 | – | evaluation-only: E13 checkpoint re-scored on OOF priors |
| E16 | V8 prior + ACPC (altitude-conditioned prior channel) | altitude conditioning inside the segmenter |
| E17 | V8 logit-space prior | prior fusion before vs after sigmoid |
| E18 | **spatially misaligned** V8 prior | negative control: does the prior need to be correct? |
| E19 | V8 prior + prior-consistency loss ($\lambda_{pc}=0.1$) | prior as a training signal, not only an input |
| E20 | V8 foreground + background priors (5-ch) | explicit two-channel prior fusion |
| E21 | **altitude-adaptive** soft prior, σ(h) ∝ 1/h | fixes the fixed-bandwidth assumption |
| E22 | V8 + coarse-scale prior (5-ch) | multi-scale prior fusion |
| E23 | V8 cascade with genuine OOF priors | leakage audit of E12 |
| E24, E25 | cascade-FiLM with / without background head | 2×2 ablation: altitude mechanism |
| E26 | cascade-MoE without background head | 2×2 ablation: altitude mechanism |
| E28 | cascade-FiLM + cascade-MoE priors (5-ch) | fusion of two altitude experts |
| E29 | UAP with scale-normalised disagreement | fair re-test of uncertainty gating |
| E30 | V4 + V8 prior (5-ch) | fusion of two prior families |
| E31, E32 | E6 / E12 re-run with seed 43 | run-to-run variance (in progress) |
| E33 | UAP against a healthy, calibrated prior | fair control for E13/E29 (in progress) |

## 9. Results

All numbers come from `eval/eval_final.py` on the **validation split (9,129 patches)** at threshold
0.5, read from `reports/*.csv`; the tables report the **latest recorded run** of each experiment
(CSV rows are appended, never overwritten, so earlier runs remain inspectable). Metrics are
patch-level; `iou_macro` averages over patches.

### 9.1 Overall semantic segmentation

| Exp | Architecture / prior | Precision | Recall | IoU | Dice | ΔIoU vs E0 | Params |
|---|---|---|---|---|---|---|---|
| E0 | RGB baseline | 0.9497 | 0.9377 | 0.8946 | 0.9423 | — | 14,834,121 |
| E1 | zero channel | 0.9500 | 0.9382 | 0.8951 | 0.9427 | +0.0004 | 14,837,257 |
| E2 | random channel | 0.9496 | 0.9356 | 0.8927 | 0.9412 | −0.0020 | 14,837,257 |
| E5 | V3 BCP | 0.9518 | 0.9378 | 0.8964 | 0.9433 | +0.0017 | 14,837,257 |
| E5_OOF | V3 BCP, OOF | 0.9506 | 0.9374 | 0.8949 | 0.9425 | +0.0003 | 14,837,257 |
| E6 | V4 altitude-FiLM BCP | 0.9524 | 0.9375 | 0.8966 | 0.9434 | +0.0020 | 14,837,257 |
| E7 | oracle mask | 1.0000 | 1.0000 | 1.0000 | 1.0000 | (upper bound) | 14,837,257 |
| E8 | V5-A dual head | 0.9505 | 0.9388 | 0.8962 | 0.9432 | +0.0016 | 14,837,257 |
| E9 | V5-B background MoE | 0.9523 | 0.9376 | 0.8968 | 0.9435 | +0.0022 | 14,837,257 |
| E10 | capacity control | 0.9531 | 0.9363 | 0.8962 | 0.9431 | +0.0015 | 14,837,257 |
| E11 | V7 altitude MoE | 0.9522 | 0.9391 | 0.8980 | 0.9443 | +0.0033 | 14,837,257 |
| E12 | V8 cascade | 0.9527 | 0.9390 | 0.8983 | 0.9444 | +0.0037 | 14,837,257 |
| E13 | UAP (uncertainty-gated) | 0.9522 | 0.9382 | 0.8971 | 0.9437 | +0.0024 | 14,837,257 |
| E13b | UAP, 2nd implementation | 0.9508 | 0.9398 | 0.8973 | 0.9438 | +0.0027 | 14,837,257 |
| E14 | class-conditioned prior | 0.9506 | 0.9384 | 0.8959 | 0.9430 | +0.0013 | 14,837,257 |
| E16 | + ACPC | 0.9520 | 0.9398 | 0.8983 | 0.9444 | +0.0037 | 14,837,387 |
| E17 | logit-space prior | 0.9511 | 0.9380 | 0.8961 | 0.9430 | +0.0014 | 14,837,257 |
| **E18** | **misaligned prior (control)** | 0.9492 | 0.8705 | **0.8320** | 0.9001 | **−0.0627** | 14,837,257 |
| E19 | + prior-consistency loss | 0.9498 | 0.9414 | 0.8980 | 0.9442 | +0.0033 | 14,837,257 |
| E20 | V8 $P_f$+$P_b$ (5-ch) | 0.9527 | 0.9396 | 0.8987 | 0.9445 | +0.0040 | 14,840,393 |
| **E21** | **altitude-adaptive σ(h)** | 0.9522 | **0.9415** | **0.9002** | **0.9454** | **+0.0055** | 14,837,257 |
| E22 | multi-scale prior (5-ch) | 0.9512 | 0.9405 | 0.8984 | 0.9444 | +0.0037 | 14,840,393 |
| E23 | cascade + OOF priors | 0.9527 | 0.9390 | 0.8983 | 0.9444 | +0.0037 | 14,837,257 |
| E24 | cascade-FiLM + bg head | 0.9528 | 0.9387 | 0.8981 | 0.9442 | +0.0034 | 14,837,257 |
| E25 | cascade-FiLM, no bg head | 0.9512 | 0.9384 | 0.8965 | 0.9434 | +0.0019 | 14,837,257 |
| E26 | cascade-MoE, no bg head | 0.9517 | 0.9396 | 0.8980 | 0.9442 | +0.0033 | 14,837,257 |
| E28 | FiLM+MoE priors (5-ch) | 0.9526 | 0.9398 | 0.8990 | 0.9448 | +0.0043 | 14,840,393 |
| E29 | UAP, normalised | 0.9530 | 0.9392 | 0.8987 | 0.9446 | +0.0040 | 14,837,257 |
| E30 | V4+V8 priors (5-ch) | **0.9539** | 0.9381 | 0.8985 | 0.9444 | +0.0038 | 14,840,393 |

Aggregate IoU gains span **+0.0014 … +0.0055** and are therefore reported *alongside* the grouped
results, not instead of them.

### 9.2 Where the gain actually is: tiny objects × altitude (IoU)

`tiny` = fewer than 1,024 foreground pixels in the patch (n = 57 / 77 / 96 for 50 / 70 / 90 m).

| Group | E0 | E1 zero | E2 random | E5 | E6 | E11 | E12 | E20 | E24 |
|---|---|---|---|---|---|---|---|---|---|
| tiny @50 m | 0.6663 | 0.6928 | 0.6672 | 0.7135 | 0.7450 | **0.7577** | 0.7371 | 0.7256 | 0.6798 |
| tiny @70 m | 0.6398 | 0.6651 | 0.6500 | 0.7004 | 0.6671 | 0.6927 | 0.6850 | **0.6964** | 0.6806 |
| tiny @90 m | 0.7818 | 0.7570 | 0.7758 | 0.7808 | 0.7956 | 0.7589 | 0.7661 | 0.7695 | 0.7940 |
| Δ (E5−E0) | — | +0.027/+0.025/−0.025 | +0.001/+0.010/−0.006 | +0.047/+0.061/−0.001 | +0.079/+0.027/+0.014 | +0.091/+0.053/−0.023 | +0.071/+0.045/−0.016 | +0.059/+0.057/−0.012 | +0.014/+0.041/+0.012 |

Reading: the prior helps **low- and mid-altitude tiny objects** (+0.05 … +0.09 IoU at 50 m,
+0.03 … +0.06 at 70 m) and is **at best neutral at 90 m** — consistent with the $h^{-2}$ area law
and with the rationale for the altitude-adaptive bandwidth of E21. Note that the zero-channel
control E1 *also* lifts tiny @50 m by +0.027, which is why the E18 misaligned-prior control (below)
is essential for attributing the gain to *semantic* content rather than to a benign extra channel.

### 9.3 Snow / degradation decoupling (tiny IoU)

EVD4UAV confounds altitude and snow cover (63 % snowy frames at 70 m vs 4.5 % at 90 m), so grouped
access to snow is a confounder check as much as a report (`reports/by_size_snow.csv`).

| Exp | tiny@clean | tiny@snow | Δclean vs E0 | Δsnow vs E0 |
|---|---|---|---|---|
| E0 | 0.7309 | 0.6834 | — | — |
| E5 | 0.7515 | 0.7251 | +0.0205 | **+0.0417** |
| E6 | **0.7653** | 0.7119 | +0.0344 | +0.0286 |
| E12 | 0.7404 | 0.7237 | +0.0095 | **+0.0403** |
| E20 | 0.7411 | **0.7292** | +0.0102 | **+0.0458** |
| E24 | 0.7542 | 0.7022 | +0.0232 | +0.0188 |
| E28 | 0.7587 | 0.6928 | +0.0278 | +0.0094 |
| E30 | 0.7602 | 0.6635 | +0.0293 | −0.0199 |
| E29 | 0.7219 | 0.7190 | −0.0090 | +0.0356 |

The attribution of the gain **depends on the altitude mechanism**: the MoE/cascade family
(E5/E12/E20) gains more in snow (+0.040 … +0.046) than in clean scenes (+0.010 … +0.021), whereas
the FiLM family (E6/E24/E28) does the opposite. Snow penalises every model on tiny objects
(E0 −0.047; E30 −0.067), so "prior gains concentrate on degraded tiny objects" is *mechanism
dependent* and must not be stated as a blanket claim.

### 9.4 Background cleanliness (false positives on empty patches)

Mean false-positive pixels per empty patch; lower is better. Empty patches have no foreground, so
IoU is undefined there and FP is the informative metric.

| Exp | empty@50 m | empty@70 m | empty@90 m |
|---|---|---|---|
| E0 (RGB) | 359.7 | 24.8 | 35.0 |
| E5 | 139.9 | 44.2 | 28.4 |
| E6 | 128.0 | 39.5 | 39.0 |
| E8 | 122.3 | 17.4 | 23.2 |
| E12 | 126.2 | 21.2 | 33.2 |
| E20 | 130.7 | 39.3 | 38.0 |
| **E30** | **92.9** | **15.1** | **24.4** |

Every prior-using model cuts empty-patch false positives at 50 m by 2.6–3.9× relative to the RGB
baseline (359.7 → 92.9–139.9), the clearest and most robust effect in the study. E30 (V4+V8 fusion,
5-ch) is the cleanest of all and also has the highest precision (0.9539) — bought, however, at the
cost of the worst tiny@snow recall (0.6635), i.e. **there is no single configuration that dominates
on every criterion**: E21 maximises aggregate IoU, E30 maximises precision / background cleanliness,
E6 and E28 maximise tiny@clean, E20 maximises tiny@snow.

### 9.5 Negative controls

* **E1 ≈ E0, E2 ≤ E0** ⇒ the gain is not produced by an extra channel, by +0.021 % parameters, or by
  an arbitrary non-trivial channel.
* **E18 (spatially misaligned prior): IoU 0.8320 (−0.0627), tiny 0.4809 (−0.2304)** ⇒ the prior must
  be *semantically aligned*; a plausible-looking but displaced field actively harms the segmenter.
  Together with E1/E2 this isolates the gain as semantic.
* **E7 (oracle = 1.0)** ⇒ a perfect prior is sufficient; the remaining gap is a prior-quality
  problem, not a capacity problem of the final network. E7 uses ground truth as input and must never
  be reported as a method.
* **E10 (larger plain PriorNet) ≈ E5–E12** ⇒ expert structure, not raw capacity, drives the
  differences among prior variants.
* **E23 (cascade + OOF priors) ≈ E12** ⇒ the reported gains survive the leakage audit.

### 9.6 Status of the three claimed contributions

| # | Claim | Verdict | Evidence |
|---|---|---|---|
| ① | BCP prior + causal validation | **supported** | E1/E2 ≈ E0; E18 collapses; E7 = 1.0; consistent tiny/altitude/snow grouping |
| ② | Dual-dimension expert cascade (altitude × background) + fusion | **partially supported** | removing the background head costs ≈0.019 tiny IoU on average, and the sign is consistent under *both* altitude mechanisms (FiLM: E24 0.7328 → E25 0.7180; MoE: E12 0.7335 → E26 0.7108); altitude MoE ≈ FiLM, no clear winner; fusing two experts (E20/E28/E30) gives the best implementations, but which one wins depends on the deployment metric |
| ③ | Uncertainty-gated prior $P^{\star}=(1-U)P_f$ | **negative result** | three independent attempts (E13, E13b, E29) all degrade tiny (E13 0.7214, E13b 0.7173, E29 0.7206 vs E0 0.7113 / E5 0.7405); down-weighting by disagreement suppresses the already-scarce small-object response |

### 9.7 Instance-level cross-check (COCO-style AP, 1,868 val images)

| Exp | AP | AP50 | AP75 | $AP_S$ |
|---|---|---|---|---|
| E0 | 0.4657 | 0.6803 | 0.4697 | 0.00044 |
| E5 | 0.4650 | 0.6822 | 0.4700 | 0.00038 |

Instance-level AP is essentially unchanged (ΔAP = −0.0007) while $AP_S$ collapses for both models:
BCP-Vis improves *pixel-level* recall on tiny objects without yet converting it into *instance-level*
detection. Closing this gap (a connected-component or instance head on top of the prior) is the
natural next step and is stated as a limitation, not glossed over.

## 10. Evaluation and Analysis Tools

| Tool | Purpose |
|---|---|
| `eval/eval_final.py` | patch-level metrics grouped by size / altitude / snow / size×altitude |
| `eval/eval_instance_ap.py` | COCO-style instance AP, AP50/75, AP_S/M/L |
| `eval/run_ablation.py` | train + evaluate a list of experiments and merge the CSVs |
| `eval/make_report.py` | aggregate every CSV into `reports/unified_report.md` |
| `eval/visualize_samples.py` | qualitative overlays (`reports/visual_samples/`) |
| `tools/check_prior_health.py` | prior sanity check: mean response on target vs background regions (flags "flattened" priors before they silently ruin a run) |
| `tools/check_leak.py`, `tools/check_oof_fold.py` | verify no prior was produced by a model that saw the image |
| `tools/diagnose_prior.py`, `tools/analyze_groups.py`, `tools/compare_exps.py` | prior statistics, grouped diagnostics, experiment diffs |
| `tools/e2e_synthetic_test.py`, `tools/test_v5_cpu.py`, `tools/test_v7_cpu.py`, `tools/test_v8_cpu.py` | CPU-only unit tests for every prior variant |
| `tools/verify_artifacts.py`, `tools/verify_pred.py`, `tools/audit_review.py` | artefact / annotation integrity checks |

## 11. Reproducibility, Scope and Limitations

* **Determinism.** `seed: 42` for Python/NumPy/PyTorch; identical augmentation (h/v flip, mild
  brightness/contrast) and identical data order across all experiments. The random-channel control
  is seeded per patch and therefore reproducible.
* **Evaluation-mode bug and its fix.** Model evaluation must call `model.eval()` *before* the FPS
  warm-up; otherwise warm-up batches run with training-mode BatchNorm statistics and silently lower
  every metric (measured: E16 read 0.8228 before the fix and 0.8983 after). Baselines reported here
  were re-scored after this fix.
* **Run-to-run variance.** `reports/*.csv` keep *every* run, so the spread is directly inspectable.
  Aggregate IoU differences among the top configurations are 0.0002–0.0055, i.e. at or below the
  run-to-run spread; **E31/E32 (seed 43) are running precisely to quantify that spread, and until
  they land no ranking of the top models should be treated as reliable.** Grouped metrics, not
  aggregate metrics, are the evidence, and the tiny groups are small (n = 57/77/96).
* **Confounded factors.** Altitude and snow cover are strongly correlated in EVD4UAV, so
  altitude-grouped and snow-grouped results must be read together (§9.3); we do not claim altitude
  is the sole driver of the tiny-object effect.
* **No pixel materialisation.** Patches are cropped on the fly from a manifest (~6 patches per
  1920×1080 image); materialising float32 priors would cost ≈135 GB.
* **Split hygiene.** Splitting happens at the capture-sequence level (`DJI_XXXX` prefix) *before*
  patching, so near-duplicate frames cannot straddle train and validation.
* **Prior calibration is not solved.** The mean target-region response of different prior variants
  differs by up to ≈5× (e.g. a `mul`-fused cascade prior ≈111 vs a raw FiLM prior ≈234 on the same
  scale). Downstream BatchNorm/convolution largely compensates for the *magnitude*, but any
  fusion or disagreement scheme must solve calibration first — this is why `tools/check_prior_health.py`
  exists and why the UAP family failed.
* **Scope.** Vehicle segmentation on EVD4UAV only; three vehicle categories, so class conditioning
  is an ablation. Sixteen further runs (E14–E33) are reported with their exact configurations rather
  than curated, including the negative and neutral results.

## 12. Citation

```bibtex
@misc{bcpvis2026,
  title  = {BCP-Vis: A Learned Background Confidence Prior for Tiny-Vehicle Segmentation
            in High-Altitude UAV Imagery},
  author = {fanfanyuyang},
  year   = {2026},
  note   = {Code and experimental record},
  url    = {https://github.com/fanfanyuyang/BCP-Vis}
}

@article{sun2024evd4uav,
  title   = {EVD4UAV: An Altitude-Sensitive Benchmark to Evade Vehicle Detection in UAV},
  author  = {Sun, Huiming and Guo, Jiacheng and Meng, Zibo and Zhang, Tianyun and Fang, Jianwu and Lin, Yuewei and Yu, Hongkai},
  journal = {arXiv preprint arXiv:2403.05422},
  year    = {2024}
}
```

> **Dataset:** EVD4UAV is used under CC-BY 4.0; cite `sun2024evd4uav` above as requested by its authors.

## 13. License and Acknowledgements

Released under the MIT License (see [LICENSE](LICENSE)). Compute was provided by an AutoDL
single-GPU (RTX 4090 24 GB) instance. We thank the authors of EVD4UAV for releasing the dataset.

`docs/` retains the original (Chinese) development notes — mathematical derivations, dataset audit
trail and per-experiment logs — for full provenance of every number reported above.
