# CellMorphoTEM

A deep learning pipeline to quantify **Epithelial-to-Mesenchymal Transition (EMT)** in cells using fluorescence microscopy images and transfer learning.

---

## What is this project?

Epithelial-to-Mesenchymal Transition (EMT) is a biological process where epithelial cells progressively acquire mesenchymal properties, a key mechanism in cancer metastasis and tissue remodeling. Traditionally, measuring where a cell sits on this spectrum requires complex molecular assays.

The goal of this project is to **score EMT automatically from cell morphology alone**, using only fluorescence microscopy images of the actin cytoskeleton and nucleus. The output is a continuous **TEM score** between 0 and 1:

- `0` → epithelial (control cells, CTRL condition)
- `1` → mesenchymal (fully transitioned cells, TNFTGF condition)
- `0–1` → intermediate transitional states (TGF, TNF conditions)

---

## Dataset

**Celltool** is a dataset of 523 fluorescence microscopy images of cells in 4 biological conditions.

This project was assigned by a professor at my university to a student in the Biotechnology Engineering track. The biotech student owned the biological problem and the experimental data; I stepped in as a collaborator to handle the AI/ML part, designing and implementing the pipeline that extracts morphological features and quantifies the transition automatically.

| Condition | Description | Cells |
|-----------|-------------|-------|
| CTRL | Control (epithelial baseline) | 166   |
| TGF | Treated with TGF (transitional) | ,     |
| TNF | Treated with TNF (transitional) | ,     |
| TNFTGF | Treated with both (fully mesenchymal) | 204   |

Each cell comes with **two paired grayscale images**:
- **Actin channel**: highlights F-actin filaments in the cytoskeleton
- **Nucleus channel**: highlights DAPI-stained nuclei

---

## Approach

### Why transfer learning?

Training a model from scratch on 523 images is not feasible. Instead, I used **ResNet18 pre-trained on ImageNet** as a feature extractor, its learned visual representations generalize well to microscopy images, capturing texture, shape, and structure without any fine-tuning.

### Architecture

Two independent ResNet18 models process each modality:

- **ModelA**, processes actin images → 512-dimensional embedding
- **ModelN**, processes nucleus images → 512-dimensional embedding

Since microscopy images are single-channel (grayscale) while ResNet18 expects 3-channel RGB input, I adapted the first convolutional layer by **averaging the RGB weights** into a single channel, preserving the pretrained features while accepting grayscale input.

The classification head is removed; only the `avgpool` layer output (the 512-d feature vector) is used.

### TEM Score Computation

Rather than training a classifier, the TEM score is computed geometrically:

1. Compute the mean embedding of CTRL cells (epithelial center)
2. Compute the mean embedding of TNFTGF cells (mesenchymal center)
3. Define the **TEM axis** as the vector connecting these two centers
4. Project each cell's embedding onto this axis
5. Scale robustly to [0, 1] using 1st–99th percentile clipping

This yields three scores per cell:
- `tem_actin`, from actin modality alone
- `tem_nucleus`, from nucleus modality alone
- `tem_fused`, combined from both modalities

### Validation

- **ROC-AUC** (CTRL vs TNFTGF binary classification), measures separability
- **Student's t-test**, tests statistical significance of score differences
- **Visual inspection**, colorized composite images matched to TEM scores
- **Biological consistency**, intermediate conditions (TGF, TNF) should score between extremes

---

## Pipeline Summary

```
1. Load paired actin + nucleus images (PairedActinNucleusDataset)
2. Preprocess: resize to 224×224, normalize to [0, 1]
3. Extract 512-d embeddings via dual ResNet18 (no_grad, batch_size=64)
4. Compute TEM axis from CTRL / TNFTGF mean embeddings
5. Project & scale each cell → TEM score ∈ [0, 1]
6. Export results to tem_scores.csv
7. Compute AUC, t-test, and visualize distributions
```

---

## Tech Stack

- **PyTorch** + **torchvision**, model and data pipeline
- **ResNet18** (ImageNet pretrained), feature extractor
- **scikit-learn**, AUC computation, statistical metrics
- **pandas**, results tabulation and CSV export
- **matplotlib**, visualization
- **Google Colab**, cloud GPU execution (T4)

---

## Results

The pipeline produces a `tem_scores.csv` file with one row per cell containing:

| Column | Description |
|--------|-------------|
| `sample_id` | Cell identifier |
| `condition` | CTRL / TGF / TNF / TNFTGF |
| `tem_actin` | TEM score from actin channel |
| `tem_nucleus` | TEM score from nucleus channel |
| `tem_fused` | Combined TEM score |

Statistical validation confirms that CTRL and TNFTGF cells are well-separated on the TEM axis, with intermediate conditions landing between the two extremes, consistent with biological expectations.

---

## How to Run

Open `main.ipynb` in Google Colab (or locally with a GPU) and run all cells in order. Make sure `Celltool.zip` is available in the runtime directory.

```
Runtime → Run all
```

Results will be saved to `/content/outputs_tem/`.