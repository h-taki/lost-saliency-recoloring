# Lost-saliency-guided targeted recoloring (code release)

Code and result tables for the paper:

> H. Takimoto, "Lost-saliency-guided targeted recoloring with a resolution-calibrated
> few-shot saliency evaluation," *Journal of Electronic Imaging* (under review).

Targeted recoloring for color vision deficiency (CVD): a learned *lost-saliency map*
`M = relu(S_normal − S_cvd)` marks the regions that are salient in normal vision but
lose attention under dichromacy, and recoloring is restricted to those regions
(lightness fixed, chromatic channels optimized in CIELAB). The dichromat saliency is
predicted few-shot by adapting a frozen DINOv2 backbone with LoRA.

## Repository layout
```
src/sm_cvd/      saliency model + training framework (PyTorch Lightning; MSI-Net baseline, DINOv2/LoRA)
scripts/         training, prediction, recoloring, evaluation, and analysis scripts
configs/         YAML run configurations
results/         the per-image / aggregate result tables (CSV) behind the paper
tests/           unit tests
environment.yml  conda environment
```
The repository is curated to the code and result tables behind the paper. The
"Reproducing the paper" map below links each reported result to its script(s) and CSV(s).

## Setup
```bash
conda env create -f environment.yml
conda activate sm-cvd
pip install -e .
```

## Data and model weights
Datasets and pretrained backbones are **not** redistributed here (licensing); see
`DATA_AND_WEIGHTS.md` for how to obtain EToCVD, the Toronto saliency benchmark, SALICON,
and the DINOv2 / MSI-Net weights. After downloading, set the dataset root: scripts and
configs use the placeholder `/PATH/TO/EToCVD` (with `/D`, `/P`, `/normal` subfolders),
which you must replace with your local path (or pass `--dataset-root`).

## Reproducing the paper
| Paper item | Code | Result table(s) in `results/` |
|---|---|---|
| Table 1, Fig. 1 (targeted recoloring; Pareto) | `scripts/recolor_optimize.py`, `recolor_eval.py`, `recolor_kuhn.py`, `e_recoloring_pipeline.py` | `recolor_method_{d,p}.csv`, `recolor_external_{baseline,kuhn}_{d,p}.csv`, `recolor_pareto_coords.csv` |
| Independent-region & restoration (Sec. 4.2) | same as above | `distinguish_gain_independent_region.csv`, `recolor_attention_restoration.csv` |
| Predictor vs. baseline; data-bound (Sec. 4.3) | `scripts/dinov2_driver.py` (adapted), `src` MSI-Net (baseline), `predict_perimage_nativep.py`, `analyze_imagelevel_saliency.py` | per-image predictor CSVs |
| Predictor-agnostic recoloring & M-support IoU | `recolor_optimize.py` (baseline-M), IoU script | `ablation_predictor_M_*.csv`, `m_support_iou_{d,p}.csv` |
| Fully-predicted deployment | `generate_normal_sm_msinet.py`, `recolor_optimize.py` | `ablation_fullpred_M_ciede2000.csv` |
| L*-free ablation | `recolor_optimize.py --free-L`, `recolor_eval.py` | `ablation_Lfree_ciede2000.csv`, `ciede_abs2_{d,p}.csv` |
| Calibration confound (Toronto/ETTO) | `eval_cross_dataset.py` | `sens_calibration.csv` |
| Temporal window | `build_temporal_splits.py`, `predict_perimage_temporal.py` | `early_M_recoloring.csv` |
| Scaling / sensitivity | `build_scaling_splits.py` | `scaling_predictor_{d,p}.csv`, `sens_{threshold,severity}_{d,p}.csv` |
| Second color-difference formula (CIE76) and severity re-evaluation (Secs. 4.2, 5) | recomputed from the recolored images with the Machado matrices of `recolor_optimize.py` | `recheck_de76_table1.csv`, `recheck_severity_eval.csv` |
| Independent region: top 5% of S_normal with R_M excluded, CIEDE2000 (Sec. 4.2) | same recomputation; explicit region definition | `recheck_snorm_top5_excl_RM.csv` (summary), `recheck_snorm_top5_excl_RM_perimage.csv` |

Evaluation uses predictor-independent CIEDE2000 metrics; ground-truth saliency maps are
built by smoothing the public fixations to one degree of visual angle.

## Acknowledgment
The saliency framework in `src/` re-implements (in PyTorch) an in-house MSI-Net-style
TensorFlow implementation used in the laboratory.

## License
Code is released under the MIT License (`LICENSE`). Datasets and third-party model
weights are governed by their own licenses (see `DATA_AND_WEIGHTS.md`).
