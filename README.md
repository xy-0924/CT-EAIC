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
- Clinical variables in CT-EAIC: **sex, age, AFP, PLT, ALB, ALT, AST, ALP, TBIL, and PT**
- Clinical preprocessing: **age and AFP were Z-score standardized; sex and PLT/ALB/ALT/AST/ALP/TBIL/PT were encoded as binary 0/1 variables**
- Interpretability: **SHAP**
- Representation visualization: **t-SNE**

The final CT-EAIC model integrates global deep imaging features, imaging-feature probabilities, and clinical variables before three-class classification.

### Repository layout

```text
CT-EAIC_AJR_Public_Code/
├── README.md
├── FEATURES_24.md
├── CODE_AUDIT.md
├── requirements.txt
├── LIFT/
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

**Image-only model:** multiphase CECT global representation + imaging-feature probabilities.

**CT-EAIC:** the same imaging representation and imaging-feature probabilities, additionally integrated with 10 clinical variables using a learnable clinical scaling parameter.

### Code availability

The public repository contains model architecture, preprocessing, training, inference, segmentation customization, SHAP, and t-SNE code.

**Model checkpoints and patient-level imaging/clinical data are not included.**

### Data availability

Patient-level imaging and clinical data are not publicly distributed because of patient privacy, institutional, and ethical restrictions.

### Intended use

Research and academic use only. The model is not a standalone clinical diagnostic device.

### Contact

**Dajing Guo, PhD**  
Department of Radiology  
The Second Affiliated Hospital of Chongqing Medical University  
Email: guodaj@hospital.cqmu.edu.cn
