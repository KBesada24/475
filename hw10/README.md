# HW10: Object Recognition With Local and Global Features

This submission is a self-contained Python image-processing project. It generates all images procedurally, extracts the required features, and performs object recognition with a nearest-centroid classifier.

## Requirements Covered

- Haralick texture features from LBP images for color texture classification
- Box-counting feature
- Chain-code feature
- Area feature
- Local Binary Pattern feature
- Generated image dataset
- Classification report and feature table

## Run

```bash
python3 image_recognition_hw10.py
```

The script creates these folders:

- `generated_images/` - generated RGB source images for three classes: `coin`, `leaf`, and `car`
- `output/` - masks, LBP/Haralick visualizations, classification montage, CSV feature table, and text report

## Output Files

- `output/features.csv` - one row per image with class label, prediction, and extracted feature values
- `output/classification_report.txt` - nearest-centroid accuracy, confusion matrix, and per-image predictions
- `output/01_dataset_contact_sheet.png` - generated dataset overview
- `output/02_lbp_contact_sheet.png` - LBP examples used before Haralick extraction
- `output/03_masks_contact_sheet.png` - segmentation masks used for area, box counting, and chain code
- `output/04_classification_montage.png` - generated images annotated with predicted labels
- `output/05_feature_summary.png` - class-level feature averages for global shape and Haralick/LBP texture descriptors

The project is deterministic: running the script again recreates the same dataset and results.
