# Obtaining data and model weights

None of the datasets or pretrained backbones are redistributed in this repository.
Download them from the original sources, then point the configs/scripts at your local
copy (replace the `/PATH/TO/EToCVD` placeholder, or pass `--dataset-root`).

## Datasets
- **EToCVD** (primary): A. Bruno et al., "Image content enhancement through salient
  regions segmentation for people with color vision deficiencies," *i-Perception* 10(3),
  2019, doi:10.1177/2041669519841073. Provides per-group fixations (8 normal, 5 protan,
  3 deutan) on ~100 images, time-windowed fixations, and a second session on the
  authors' recolored images. Use under the dataset's terms; do not redistribute its
  images. Expected layout: `/PATH/TO/EToCVD/{D,P,normal}`.
- **Toronto** saliency benchmark: Bruce & Tsotsos, NIPS 2006 (used for the calibration
  confound demonstration).
- **SALICON**: used for saliency pretraining.

## Ground truth
EToCVD provides fixation locations, not saliency maps. Build the saliency-map targets by
Gaussian smoothing calibrated to one degree of visual angle (~3% of image width); see the
evaluation protocol in the paper and `scripts/regen_gt_sigma.py` / `generate_*` scripts.

## Model weights (third-party; obtain from the official sources)
- **DINOv2** (ViT-B/14, with registers) — Meta AI; subject to the DINOv2 license.
- **MSI-Net** — Kröner et al., *Neural Networks* 129, 2020 (generic-saliency baseline).
- (Exploratory, not in the paper) DINOv3 / SAM2 backbones — Meta AI licenses.

Trained LoRA adapters are small; if you wish to share them, confirm the backbone license
permits redistribution of derived adapters before publishing.
