# Classification Models

This directory contains the classification component of the study framework.

It includes:

- **UniFormer-Small backbone**
- **24 predefined binary imaging-feature recognition branch**
- **Image-only classification model**
- **CT-EAIC classification model**
- Training and inference code
- Classification-specific preprocessing utilities

The final **CT-EAIC classification model** integrates imaging representations, imaging-feature probabilities, and clinical variables. The **image-only model** is the comparison model and excludes the clinical-variable component.
