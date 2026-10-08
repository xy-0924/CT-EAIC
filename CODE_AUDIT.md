# Code Audit Notes

This public package was curated from the uploaded development repository to remove development-only material and harmonize terminology with the final manuscript.

## Safe changes made

1. Development-stage model labels were removed from the public-facing interpretability scripts and replaced with **Image-only model** and **CT-EAIC**.
2. The final imaging-feature dimension is set/documented as **24**.
3. Legacy hard-coded 9-feature lists were removed from the public dataset code. Generic optional feature-subset support remains only for ablation/research use.
4. The default `num_feature_classes` for the final public classification configuration was harmonized to **24**.
5. Old result figures, flowchart images, temporary result-analysis scripts, and development comparison scripts were not included.
6. Machine-specific absolute paths were removed where straightforward.
7. A comment/count typo in the SHAP CT-EAIC script was corrected: the clinical feature list contains **10**, not 9, variables.

## Clinical-variable preprocessing confirmed

The final study uses 10 clinical variables. **Age and AFP are continuous variables and are standardized using training-set Z-score statistics. Sex and PLT, ALB, ALT, AST, ALP, TBIL, and PT are binary 0/1 variables and are not Z-score standardized.** The validation/test datasets use the normalization statistics derived from the training set.

## Checkpoints

The uploaded repository does not contain the final trained model checkpoint files. The public package therefore does not claim that trained weights are available.
