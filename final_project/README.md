# Final Project: Development of an Image Classification System

This project implements a complete image classification system for the final assignment. It builds a deterministic image database, restores noisy images, segments foreground objects, extracts binary and texture features, concatenates those features, trains an SVM classifier, and reports classification accuracy.

## Requirements

```bash
python3 -m pip install -r requirements.txt
```

Dependencies:

- NumPy
- Pillow
- scikit-learn

## Run

```bash
python3 image_classification_system.py
```

The script creates:

- `generated_images/` for the image database
- `output/` for feature files, reports, and visual evidence

## Assignment Requirement Mapping

1. **Image Database Assembly**: The script generates 48 images across four classes: `coin`, `leaf`, `car`, and `traffic_sign`. Each class has six clean/high-quality images and six noisy/degraded images.
2. **Image Restoration and Enhancement**: Noisy images are processed with median denoising, Gaussian smoothing, unsharp masking, contrast stretching, and histogram equalization. A deterministic contrast/edge/noise score selects the best enhanced image.
3. **Image Segmentation**: Foreground objects are isolated using color-distance thresholding, grayscale thresholding, morphological opening/closing, and largest connected component selection.
4. **Binary Object Feature Extraction**: The best enhanced image is multiplied by the binary mask to form the segmented object image. Horizontal and vertical projections are computed from the binary mask.
5. **Projection Features**: The binary object mask is summarized with 16 horizontal and 16 vertical projection bins.
6. **Feature Vector Creation**: The feature vector includes Haralick texture, box-counting dimension, chain-code histograms, LBP histograms, RGB histograms, intensity histograms, shape area, aspect ratio, and binary projections.
7. **Feature Concatenation**: All features are concatenated into one vector per image and saved to `output/features.csv`.
8. **Classifier Training**: The classifier is `sklearn.svm.SVC` with an RBF kernel inside a `StandardScaler` pipeline.
9. **Classification**: The trained SVM predicts the class of held-out test images.
10. **Accuracy Evaluation**: The report calculates accuracy as correct predictions divided by total predictions, multiplied by 100.
11. **Documentation and Reporting**: This README and `output/classification_report.txt` document the full processing path, method rationale, parameters, and results.
12. **Performance Report**: The report discusses observed performance and practical improvement paths for real image datasets.

## Output Files

- `output/features.csv`: one row per image with class label, train/test split, prediction, and concatenated feature values
- `output/classification_report.txt`: SVM accuracy, confusion matrix, precision/recall/F1, per-image predictions, and performance discussion
- `output/01_dataset_contact_sheet.png`: clean and noisy database examples
- `output/02_restoration_enhancement.png`: noisy image restoration and enhancement comparison
- `output/03_segmentation_pipeline.png`: enhanced image, raw mask, cleaned mask, and segmented object
- `output/04_binary_projection_features.png`: binary mask with horizontal and vertical projections
- `output/05_feature_concatenation.png`: visual explanation of feature concatenation
- `output/06_classification_results.png`: test images annotated with SVM predictions
- `output/07_feature_summary.png`: class-level feature averages

## Method Rationale and Parameters

The database is generated procedurally so the project is reproducible and does not depend on external files. The classes differ by shape and texture, which makes them suitable for testing both binary object features and texture features.

Noisy images include Gaussian noise, salt-and-pepper noise, blur, contrast reduction, and small occlusions to simulate real-world image degradation. Median filtering targets impulse noise, Gaussian smoothing reduces high-frequency noise, unsharp masking restores edge definition, contrast stretching expands useful intensity range, and histogram equalization improves global contrast.

Segmentation uses the image border as a background estimate, then combines color distance, saturation, and dark-object cues. Morphological opening removes isolated false positives, closing fills gaps, and largest connected component selection keeps the main object of interest.

The feature vector intentionally combines several independent cues:

- Projection profiles describe binary object shape.
- Haralick features describe texture co-occurrence patterns.
- Box counting approximates shape complexity.
- Chain code describes boundary direction.
- LBP histograms describe local texture.
- RGB and intensity histograms describe color and brightness distribution.

The SVM uses an RBF kernel because the concatenated handcrafted features are not guaranteed to be linearly separable. `StandardScaler` is used before SVM training so features with different numeric ranges contribute fairly.

## Performance and Improvements

The generated classes are designed to be separable, so accuracy should be high. For real-world applications, this system could be improved by using labeled image folders, larger datasets, cross-validation for SVM `C` and `gamma`, stronger segmentation for cluttered scenes, and additional augmentation to handle rotation, scale, lighting, and viewpoint changes.
