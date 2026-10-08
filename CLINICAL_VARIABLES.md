# Clinical Variables Used by CT-EAIC

The final CT-EAIC model integrates 10 clinical variables:

| Variable | Encoding / preprocessing |
|---|---|
| Sex | Binary 0/1 |
| Age | Continuous; Z-score standardized using training-set mean and SD |
| AFP | Continuous; Z-score standardized using training-set mean and SD |
| PLT | Binary 0/1 |
| ALB | Binary 0/1 |
| ALT | Binary 0/1 |
| AST | Binary 0/1 |
| ALP | Binary 0/1 |
| TBIL | Binary 0/1 |
| PT | Binary 0/1 |

For validation and external evaluation, the Z-score parameters for Age and AFP are loaded from the training cohort. Binary variables are kept unchanged.

The threshold definitions used to convert laboratory variables into 0/1 indicators should be interpreted according to the study data dictionary / manuscript definitions.
