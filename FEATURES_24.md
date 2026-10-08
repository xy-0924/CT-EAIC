# Final 24 Imaging Features

The final public code uses the following 24 predefined binary imaging features, in this order:

1. Nonrim arterial phase hyperenhancement
2. Rim APHE
3. Nonperipheral washout
4. Peripheral "washout"
5. Corona enhancement
6. Enhancing capsule
7. Nonenhancing capsule
8. Peripheral discontinuous nodular enhancement
9. Progressive enhancement
10. Centripetal enhancement
11. Parallels blood pool enhancement
12. Uniform AP enhancement
13. Uniform PVP enhancement
14. Uniform DP enhancement
15. Necrosis or severe ischemia
16. Blood products in mass
17. Nodule-in-nodule architecture
18. Mosaic architecture
19. Delayed central enhancement
20. Infiltrative appearance
21. Portal venous phase peritumoral hypoenhancement
22. Fat in mass, more than liver
23. Fat sparing in solid mass
24. Intratumoral artery


## Correspondence with the manuscript

This list is intended to match the 24 imaging features reported in the final Methods and Supplementary Tables.  
The standalone overall **arterial phase hyperenhancement (APHE)** label is not included as a separate feature; **nonrim APHE** and **rim APHE** are modeled separately.

The final study configuration uses all 24 features. Optional `--selected-features` support is retained in the generic training/inference code only for ablation or research experiments and is not the configuration used in the final study models.
