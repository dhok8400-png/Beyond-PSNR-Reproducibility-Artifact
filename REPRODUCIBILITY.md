# Reproducibility Guide

## 1. Obtain the source images

Obtain the source archive from the original Kaggle provider. The source images are not
redistributed in this artifact.

The study used 36 512 × 512, 8-bit grayscale JPEG derivatives identified as MED-001–MED-036.
Match files using SHA-256 against:

`./supplementary/Supplementary_Table_S1_Image_Identity_Manifest.csv`

You can scan a downloaded directory with:

```bash
python tools/match_images_by_sha256.py /path/to/dataset
```

## 2. Create an environment

A Python 3.12 environment is recommended because the audited run used Python 3.12.14.

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/macOS
source .venv/bin/activate

pip install -r requirements.txt
```

The original recorded environment is stored in `experiment/environment.json`.

## 3. Review the frozen protocol

The study protocol is stored in:

`experiment/protocol.json`

Important boundaries:

- the study images are JPEG derivatives, not verified original acquisition DICOM objects;
- archive grouping is not verified patient identity;
- technical seeds are repeated computational runs, not independent patient samples;
- the measured payload range is a tested range, not a demonstrated capacity limit;
- the Python implementation is not claimed to be bit-for-bit MATLAB parity.

## 4. Core implementation

Main reference files:

- `code/reference_codec.py`
- `code/pso_reference.py`

The matched-selector detector runner is retained as an archival experiment script:

- `code/run_matched_selector.py`

That runner reflects the frozen experiment directory layout used for the audited run.
Researchers adapting it to another machine should update its local root path while preserving
the documented selector, payload, seed, feature, classifier, and split settings.

## 5. Derived results

The `results/` directory contains the numerical outputs used to support the manuscript,
including ablation, sensitivity, robustness-related summaries, detector outputs, and the
nonclinical regional analysis.

The matched-selector steganalysis outputs are under:

`results/matched_selector/`

## 6. Exact-source recovery boundary

Exact recovery refers to the source message bytes recovered from the persisted stego PNG plus
the matching external SK04 record. It does not mean recovery of the original cover image.

## 7. Clinical boundary

No clinical ROI, blinded reader study, diagnostic-equivalence assessment, or original-acquisition
DICOM validation was performed. The fixed central rectangle used in the regional analysis is a
coordinate mask, not a clinician-defined anatomical ROI.

## 8. Releasing a citable snapshot

Recommended release sequence:

1. finalize author names/order in `AUTHORS.md`;
2. create `CITATION.cff` from the provided template;
3. choose a repository/code license;
4. create a GitHub release;
5. connect the repository to Zenodo;
6. archive the release and obtain the Zenodo DOI;
7. update the manuscript Data Availability Statement with the GitHub URL and Zenodo DOI.
