# CT-EAIC

## An Interpretable CT-based Artificial Intelligence Tool for Classification of Solid Focal Liver Lesions

This repository contains a publication-oriented version of the source code associated with the final AI/ML framework reported in:

**“An Interpretable CT-based Artificial Intelligence Tool for Classification of Solid Focal Liver Lesions (sFLLs) in High-Risk Populations.”**

### Final study configuration

- Input: four-phase contrast-enhanced CT (precontrast, AP, PVP, and DP)
- Automated lesion detection/segmentation: **nnU-Net v2**
- Classification backbone: **UniFormer-Small**
- Imaging-feature branch: **24 predefined binary radiological imaging features**
- Final diagnostic classes: **benign lesion / non-HCC malignancy / HCC**
- Clinical variables in the **clinical-integrated model**: **sex, age, AFP, PLT, ALB, ALT, AST, ALP, TBIL, and PT**
- Clinical preprocessing: **age and AFP were Z-score standardized; sex and PLT/ALB/ALT/AST/ALP/TBIL/PT were encoded as binary 0/1 variables**
- Interpretability: **SHAP**
- Representation visualization: **t-SNE**

The CT-EAIC framework includes an image-only model and a clinical-integrated model. The clinical-integrated model combines global deep imaging features, imaging-feature probabilities, and clinical variables before three-class classification.

### Repository layout

```text
CT-EAIC/
├── README.md
├── FEATURES_24.md
├── requirements.txt
├── classification_models/
│   ├── main/                  # UniFormer-Small classification code
│   └── preprocess/            # Classification preprocessing
├── nnunetv2/                  # Custom nnU-Net v2 trainer/loss code
├── scripts/
│   ├── preprocessing/         # CT preprocessing
│   └── prediction/            # Segmentation prediction entry points
└── interpretability/
    ├── shap_image_only.py
    ├── shap_clinical_integrated.py
    ├── feature_recognition_5fold_cv.py
    └── tsne_image_only_vs_clinical_integrated.py
```

### Imaging-feature recognition

The imaging-feature head predicts the 24 features listed in `FEATURES_24.md`.  
The final manuscript reports that all retained imaging features achieved a mean recognition accuracy >0.80 in five-fold cross-validation of the training cohort.

### Classification models

**CT-EAIC is the study-level framework name encompassing both classification models described below.**


**Image-only model:** multiphase CECT global representation + imaging-feature probabilities.

**Clinical-integrated model:** the same imaging representation and imaging-feature probabilities, additionally integrated with 10 clinical variables using a learnable clinical scaling parameter.

### Code availability

The public repository contains model architecture, preprocessing, training, inference, segmentation customization, SHAP, and t-SNE code.

**Model checkpoints and patient-level imaging/clinical data are not included.**

### Data availability

Patient-level imaging and clinical data are not publicly distributed because of patient privacy, institutional, and ethical restrictions.

### Intended use

Research and academic use only. The model is not a standalone clinical diagnostic device.
