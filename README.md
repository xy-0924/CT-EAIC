# CT-EAIC

## An Interpretable CT-based Artificial Intelligence Tool for Classification of Solid Focal Liver Lesions

This repository contains the source code associated with the CT-based Explainable AI Classification (CT-EAIC) system and the image-only comparison model reported in:

**“An Interpretable CT-based Artificial Intelligence Tool for Classification of Solid Focal Liver Lesions (sFLLs) in High-Risk Populations.”**

### Final study configuration

- Input: four-phase contrast-enhanced CT (precontrast, AP, PVP, and DP)
- Automated lesion detection/segmentation: **nnU-Net v2**
- Classification backbone: **UniFormer-Small**
- Imaging-feature branch: **24 predefined binary radiological imaging features**
- Final diagnostic classes: **benign lesion / non-HCC malignancy / HCC**
- Clinical variables in the **CT-EAIC classification model**: **sex, age, AFP, PLT, ALB, ALT, AST, ALP, TBIL, and PT**
- Clinical preprocessing: **age and AFP were Z-score standardized; sex and PLT/ALB/ALT/AST/ALP/TBIL/PT were encoded as binary 0/1 variables**
- Interpretability: **SHAP**
- Representation visualization: **t-SNE**

CT-EAIC is the overall end-to-end AI system, comprising an automated lesion detection/segmentation module and a clinically interpretable three-class classification model. The segmentation module is based on nnU-Net v2. The final **CT-EAIC classification model** uses a UniFormer-Small backbone and integrates global deep imaging features, 24 imaging-feature probabilities, and clinical variables for classification of benign lesions, non-HCC malignancies, and HCC. An **image-only model**, using the same imaging backbone and imaging-feature branch but excluding clinical variables, was used as the comparison model. In classification-performance comparisons, the term **CT-EAIC** refers to the final CT-EAIC classification model.

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
    ├── shap_ct_eaic.py
    ├── feature_recognition_5fold_cv.py
    └── tsne_image_only_vs_ct_eaic.py
```

### Imaging-feature recognition

The imaging-feature head predicts the 24 features listed in `FEATURES_24.md`.  
The final manuscript reports that all retained imaging features achieved a mean recognition accuracy >0.80 in five-fold cross-validation of the training cohort.

### Classification models

**CT-EAIC:** the final classification model, integrating multiphase CECT global representations, 24 imaging-feature probabilities, and 10 clinical variables using a learnable clinical scaling parameter for three-class classification.

**Image-only model:** the comparison model, using the same imaging backbone and imaging-feature branch but excluding the clinical-variable component.


### Code availability

The public repository contains model architecture, preprocessing, training, inference, segmentation customization, SHAP, and t-SNE code.

**Model checkpoints are not included in this public repository. Patient-level imaging and clinical data are not publicly available because of patient privacy, institutional, and ethical restrictions.**

### Intended use

Research and academic use only. CT-EAIC is an investigational AI system and is not intended for standalone clinical diagnosis.
