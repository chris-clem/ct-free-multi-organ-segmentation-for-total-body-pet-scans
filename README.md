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
- `DATA_ROOT_DIR` defined in `settings.py` contains a `train` and `test` subdir, each containing subdirs named after the different scanners, e.g. Bern_Quadra.
- Each scanner dir contains a subdir for each patient, named after the patient ID, e.g. 01122021_1_20211201_164050.
- Each patient dir contains various NIFTI files, e.g. CT and PET.

```bash
train
├── Bern_Quadra
│   ├── 01122021_1_20211201_164050
│   ├── 01122021_2_20211201_164139
│   ├── 01122021_3_20211201_164209
│   └── ...
└── SH_uExplorer
    ├── Anonymous_ANO_20220812_1618356_084329
    ├── Anonymous_ANO_20220812_1619483_104008
    ├── Anonymous_ANO_20220812_1621040_105304
    └── ...
```

### 1. Create CT-based segmentation masks with TotalSegmentator

```bash
poetry run python pet_seg/01_run_total_segmentator.py --scanner=Bern_Quadra
```

### 2. Resize TS segmentation to PET space

```bash
poetry run python pet_seg/02_resize_ts_seg_to_pet.py --scanner=Bern_Quadra
```

## nn-UNet Training and Inference

### 1. Create data CSV file
Creates a CSV file with the paths to the data for a given scanner.

```bash
poetry run python pet_seg/03_create_data_csv.py --scanner=Bern_Quadra
```

### 2. Convert data for nnU-Net
Converts data for nnU-Net (see https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/dataset_format.md).

```bash
source nnunetv2.env
poetry run python pet_seg/04_convert_data_for_nnunet.py --data_csv_path=...
```

### 3. Run nnU-Net planning and preprocessing
Checks the dataset integrity, and overwrites existing fingerprints for dataset 1

```bash
poetry run nnUNetv2_plan_and_preprocess -d 1 --verify_dataset_integrity --clean
```

### 4. Run nnU-Net Training
Use GPU 0, 3D fullres, dataset 1, and fold 0.

```bash
CUDA_VISIBLE_DEVICES=0 poetry run nnUNetv2_train 1 3d_fullres 0
```

### 5. Run nnU-Net Prediction
Runs nnU-Net prediction with model trained on dataset 3 on dataset with id 4.
Creates predicted segmentation masks in `NNUNET_RESULTS_DIR` and computes metrics saved in "summary.json" file.

```bash
CUDA_VISIBLE_DEVICES=0 poetry run python pet_seg/05_predict_nnunet.py --model_dataset_ids=3 --input_dataset_ids=4
```

## Results Analysis

### 1. Extract nnU-Net Results
Creates csv files containing the dice scores for each patient and organ.
Files are stored in `NNUNET_VAL_TEST_RESULTS_DIR`.

```bash

```bash
poetry run python pet_seg/06_extract_nnunet_results.py --model_dataset_ids=3 --test_datasets=test_internal
```

### 2. Analyze nnU-Net Results
Computes per organ dice scores for all extracted nnU-Net results stored in NNUNET_VAL_TEST_RESULTS_DIR.

```bash
poetry run python pet_seg/07_analyze_nnunet_results.py
```

### 3. Plot nnU-Net Results

```bash
poetry run streamlit run pet_seg/08_plot_nnunet_results.py
```
