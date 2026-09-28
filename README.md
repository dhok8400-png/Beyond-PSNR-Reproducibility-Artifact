# Beyond PSNR — Reproducibility Artifact

Reproducibility materials for the manuscript:

**Beyond PSNR: A Reproducible Ablation and Detection Study of PSO–Huffman–Bit Match Image Steganography**

**Authors:** Ahmed Fouad Abdullah; Habib Izadkhah; Jaber Karimpour  
**Corresponding author:** Habib Izadkhah — izadkhah@tabrizu.ac.ir

This repository contains the Python reference implementation, frozen experiment metadata,
derived numerical results, matched-selector steganalysis records, and supplementary identity
information used in the final evidence-reconciled manuscript.

## Repository scope

The repository supports reproducibility of the technical experiments reported in the manuscript:

- HF01 Huffman source serialization and decoding
- disjoint 8 × 8 image-group selection
- PSO, greedy, random, and sequential selector comparisons
- groupwise Bit Match inversion
- SK04 external side-information accounting
- factorial ablation and PSO sensitivity experiments
- robustness results for the specified image-processing operations
- matched-stream SRM/FLD steganalysis
- nonclinical coordinate-based regional fidelity analysis
- reference-by-reference citation audit records

The repository does **not** establish clinical preservation, patient independence, original
DICOM provenance, universal robustness, or general steganographic security.

## Source images

The 36 source JPEGs are **not redistributed** in this repository.

They were selected from the publicly accessible Kaggle archive **“DICOM data”** by
Ashraf Alsinglawi. The experimental files are 512 × 512, 8-bit grayscale JPEG derivatives.
A dataset-level redistribution license was not independently verified from the accessible
metadata, so researchers should obtain the source files from the original provider under its
current terms.

Use:

`./supplementary/Supplementary_Table_S1_Image_Identity_Manifest.csv`

to match the study image IDs (`MED-001`–`MED-036`) using SHA-256 hashes.

## Reproducing the study

See `REPRODUCIBILITY.md`.

## Citation and authorship

The GitHub account hosting this repository does **not** determine scientific authorship.
The manuscript author list and order are recorded in `AUTHORS.md` and `CITATION.cff`.

GitHub's **Cite this repository** feature can use the `CITATION.cff` metadata. A Zenodo DOI
will be added after the first archived release.

## Repository status

This artifact is prepared for public archival and manuscript reproducibility. Source images
are intentionally excluded because redistribution rights were not independently verified.
