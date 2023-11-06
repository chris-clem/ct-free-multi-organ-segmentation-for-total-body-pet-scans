# ct-free-multi-organ-segmentation-for-total-body-pet-scans

TODO: addd GitHub Actions Badge
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## Setup

1. Set up Python 3.9 (e.g. with Miniconda)

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-py39_4.11.0-Linux-x86_64.sh
bash Miniconda3-py39_4.11.0-Linux-x86_64.sh
```

2. Install dependencies with [Poetry](https://python-poetry.org)

Install Poetry by following the install instructions for your OS on their
[website](https://python-poetry.org/docs/#installation). Then run the following
commands to install the dependecies:

```bash
poetry install
```

3. Install [pre-commit](https://pre-commit.com) hooks from `.pre-commit-config.yaml`

```bash
git init
poetry run pre-commit install
```

4. Run the tests

```bash
poetry run pytest
```

## Run train

```bash
poetry run python pet_seg/train.py
```
