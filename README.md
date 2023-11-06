# CT-free Multi-Organ Segmentation for Total-Body PET Scans

[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## Setup

1. Set up Python 3.9 (e.g. with Miniconda)

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-py39_4.11.0-Linux-x86_64.sh
bash Miniconda3-py39_4.11.0-Linux-x86_64.sh
```

2. Install dependencies with [Poetry](https://python-poetry.org)

Install Poetry by following the install instructions for your OS on their [website](https://python-poetry.org/docs/#installation).
Then run the following commands to install the dependecies:

```bash
poetry install
```

3. Install [pre-commit](https://pre-commit.com) hooks from `.pre-commit-config.yaml`

```bash
git init
poetry run pre-commit install
```

## Data Preparation

### 0. Required dir structure
- `DATA_ROOT_DIR` defined in `settings.py` contains a `train` and `test` subdir, each containing subdirs named after the different scanners, e.g. Bern_Quadra, SH_uExplorer, Bern_Vision600, ... .
- `SCANNER_TO_STAGE` in `settings.py` defines which scanners are used for which stage.
- Each scanner dir needs a `raw` subdir containing the original data that Song collected.
  The structure for each scanner is a bit different, which is defined in `RAW_PATIENTS_DIRS` in `settings.py`.

```bash
train
├── Bern_Quadra
│   └── raw
│       ├── NAC_only
│       └── NASC_NSC
└── SH_uExplorer
    └── raw
        ├── Explorer_20230509_dicom_233
        └── SH_Explorer_first_84_334_dicom
```

### 1. Organize raw data
Creates a `raw-organized` dir in each scanner dir containing the patient dirs.
Each patient dir comprises the CT and PET data.

```bash
poetry run python pet_seg/01_organize_raw_data.py
```

### 2. Prepare MOOSE-to_run data
Creates a `MOOSE-to_run` dir in each scanner dir containing only the CT data for each patient, which is required by MOOSE.

```bash
poetry run python pet_seg/02_prepare_moose_to_run_data.py --scanner=Bern_Quadra
```

## Run MOOSE

### 0. Install MOOSE 0.1.4
Follow instructions from https://github.com/QIMP-Team/MOOSE/tree/d9c0a9bfb8d25920cf2624cd346828bb3071112f for your prefered way,
either as a local install or with Docker.

### 1. Run MOOSE

```bash
CUDA_VISIBLE_DEVICES=0 moose -f /home/user/Data/total-body-pet-segmentation/Bern_Quadra/MOOSE-to_run
```

## nn-UNet Training

### 1. Create data CSV file
Create a CSV file with the paths to the data for a given scanner.

```bash
poetry run python pet_seg/03_create_data_csv.py --scanner=Bern_Quadra
```

### 2. Convert data for nnU-Net
Convert data for nnU-Net (see https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/dataset_format.md).

```bash
source nnunetv2.env
poetry run python pet_seg/04_convert_data_for_nnunet.py --data_csv_path=...
```
