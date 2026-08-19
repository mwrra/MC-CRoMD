# MC-CRoMD

**MC-CRoMD: A Multi-Channel Statistical Feature Extraction and Selection Framework for Lung and Colon Histopathological Image Classification Using Machine and Deep Learning**

This repository provides the implementation code, configuration files, and reproducibility resources for the revised MC-CRoMD study.

## Overview

MC-CRoMD is a statistical feature-extraction framework for lung and colon histopathological image classification. Each image is divided into non-overlapping patches, and each patch is represented using:

- Cronbach's Alpha as a heuristic inter-channel descriptor across the RGB channels.
- Eight channel-wise statistical dispersion measures:
  - Range
  - Variance
  - Standard Deviation
  - Q1
  - Q3
  - Interquartile Range
  - Mean Absolute Deviation
  - Coefficient of Variation

Each patch therefore produces 25 features.

The framework evaluates four patch sizes:

- 32×32
- 64×64
- 128×128
- 256×256

Three classification tasks are considered:

- Binary colon classification
- Three-class lung classification
- Five-class lung and colon classification

## Dataset

The experiments use the LC25000 lung and colon histopathological image dataset.

The original LC25000 images are not redistributed in this repository. Users should obtain the dataset from its official or authorized source and prepare it according to the instructions that will be provided in `DATASET_SETUP.md`.

To reduce augmentation-related leakage, images originating from the same source group are kept within the same data partition.

## Data Partitioning Policy

A fixed source-group-aware partition is used throughout the experiments:

- Training: approximately 72%
- Validation: approximately 8%
- Locked independent test: approximately 20%

All images sharing the same `SourceGroupID` are assigned to the same partition.

The same split assignments are used across patch sizes, feature-selection methods, machine-learning models, and DNN experiments.

## Leakage-Controlled Experimental Policy

The revised workflow follows these rules:

1. Preprocessing parameters are fitted using the training partition only.
2. Feature ranking and feature-selection parameters are fitted using the training partition only.
3. The validation partition is used for model/configuration selection and hyperparameter tuning.
4. The independent test partition remains locked during model development.
5. Final DNN and machine-learning configurations are frozen before the test set is opened.
6. The locked test set is used only for final evaluation.

## Feature Selection

Six feature-selection methods are evaluated together with the AllMsr baseline:

- Chi2
- Fclass
- Mutclass
- Variance
- Random Forest
- Logistic Regression
- AllMsr baseline

## Classification Models

Nine conventional machine-learning models are evaluated:

- SVM with RBF kernel
- SVM with polynomial kernel
- K-Nearest Neighbors
- Random Forest
- Decision Tree
- Logistic Regression
- SGD Classifier
- Gaussian Naive Bayes
- Voting Classifier

A fully connected Deep Neural Network is also evaluated using the MC-CRoMD feature vectors.

## Reproducibility

The repository will include:

- Source-group-aware split information
- Feature-extraction code
- Data-quality and preprocessing scripts
- Feature-selection scripts and selected-feature information
- Machine-learning screening and tuning scripts
- DNN training and validation scripts
- Frozen final-test evaluation scripts
- Random seeds
- Software requirements
- Final per-sample predictions
- Confusion-matrix counts
- Class-wise metrics
- Unrounded evaluation metrics
- Bootstrap confidence intervals
- Integrity metadata and checksums

## Final Evaluation

The final test set is not used for model selection.

The final DNN and conventional machine-learning configurations are selected using training and validation results only and are subsequently evaluated once on the locked independent test partition.

Detailed final results and reproducibility artifacts will be available in the `results` directory.

## Repository Structure

```text
MC-CRoMD/
├── README.md
├── requirements.txt
├── DATASET_SETUP.md
├── LICENSE
├── code/
├── split_information/
├── selected_features/
├── results/
└── reproducibility/
## Authors

Mohammed Thajeel Abdullah  
Raed Mohammed Hussein  
Ashraf Sabri Waheed Alameri

## License

The implementation code in this repository is released under the MIT License.

## Citation

Citation information will be added after publication of the final manuscript.
