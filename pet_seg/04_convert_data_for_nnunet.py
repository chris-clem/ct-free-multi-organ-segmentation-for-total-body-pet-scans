import os
from pathlib import Path
from typing import List

import fire
import pandas as pd
from loguru import logger
from nnunetv2.dataset_conversion.generate_dataset_json import generate_dataset_json

from pet_seg.settings import ORGANS_TO_LABELS

NNUNET_RAW_DIR = Path(os.environ["nnUNet_raw"])


def main():
    fire.Fire(convert_data_for_nnunet)


def convert_data_for_nnunet(
    data_csv_path: str,
    pet_type: str = "nac",
):
    """Converts data for nnUNet.

    Follows instructions from https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/dataset_format.md

    Args:
        data_csv_path (str): Path to the CSV file containing the data.
        pet_type (str): PET type to use. Can be "ac" or "nac". Defaults to "nac".
    """
    data_csv_path = Path(data_csv_path)

    # Load data
    df = pd.read_csv(data_csv_path)
    df_train = df[df["stage"] == "train"]
    df_test = df[df["stage"] == "test"]

    # Create dataset name consisting of ID (a three digit integer) + freely chosen name
    existing_dataset_ids = [
        int(path.name.split("_")[0].replace("Dataset", "")) for path in sorted(NNUNET_RAW_DIR.iterdir())
    ]
    if len(existing_dataset_ids) > 0:
        dataset_id = existing_dataset_ids[-1] + 1
    else:
        dataset_id = 1

    dataset_name = f"Dataset{dataset_id:03d}_{data_csv_path.stem}_{pet_type.upper()}"

    logger.debug(f"{dataset_name=}")

    # Create dirs
    dataset_raw_dir = NNUNET_RAW_DIR / dataset_name
    dataset_raw_dir.mkdir(exist_ok=True)

    if (num_training_cases := len(df_train)) > 0:
        # Train images
        images_tr_dir = dataset_raw_dir / "imagesTr"
        images_tr_dir.mkdir(exist_ok=True)

        create_symlinks(image_paths=df_train[f"pet_{pet_type}"].values, nnunet_dir=images_tr_dir)
        logger.debug(f"Created {len(list(images_tr_dir.iterdir()))} symlinks in {images_tr_dir}")

        # Train labels
        labels_tr_dir = dataset_raw_dir / "labelsTr"
        labels_tr_dir.mkdir(exist_ok=True)

        create_symlinks(df_train["organ_seg"].values, nnunet_dir=labels_tr_dir)
        logger.debug(f"Created {len(list(labels_tr_dir.iterdir()))} symlinks in {labels_tr_dir}")

    # Test images
    images_ts_dir = dataset_raw_dir / "imagesTs"
    images_ts_dir.mkdir(exist_ok=True)

    create_symlinks(image_paths=df_test[f"pet_{pet_type}"].values, nnunet_dir=images_ts_dir)
    logger.debug(f"Created {len(list(images_ts_dir.iterdir()))} symlinks in {images_ts_dir}")

    # Test labels
    labels_ts_dir = dataset_raw_dir / "labelsTs"
    labels_ts_dir.mkdir(exist_ok=True)

    create_symlinks(df_test["organ_seg"].values, nnunet_dir=labels_ts_dir)
    logger.debug(f"Created {len(list(labels_ts_dir.iterdir()))} symlinks in {labels_ts_dir}")

    generate_dataset_json(
        output_folder=str(dataset_raw_dir),
        channel_names={0: pet_type.upper()},
        labels=ORGANS_TO_LABELS,
        num_training_cases=num_training_cases,
        file_ending=".nii.gz",
        dataset_name=dataset_name,
    )

    logger.info(f"Created {dataset_raw_dir / 'dataset.json'}")


def create_symlinks(
    image_paths: List[str],
    nnunet_dir: Path,
):
    """Creates symlinks from image paths (PET or Seg) to nnUNet image/ label dir.

    Renames images to case_identifier_XXXX.nii.gz format, where XXXX is the modality identifier (0000 for PET).
    Label files are saved as case_identifier.nii.gz

    Args:
        image_paths (List[str]): List of image paths.
        nnunet_dir (Path): nnUNet target directory (imagesTr or labelsTr).
    """
    for image_path in image_paths:
        is_dynamic = "dynamic" in image_path
        is_image = "NASC" in image_path

        image_path = Path(image_path)

        patient_id = image_path.parent.parent.name if is_image else image_path.parent.name
        image_name = f"{patient_id}_{image_path.name.split('.')[0]}" if is_dynamic else patient_id

        nnunet_image_name = f"{image_name}.nii.gz" if "label" in nnunet_dir.name else f"{image_name}_0000.nii.gz"
        nnunet_image_path = nnunet_dir / nnunet_image_name

        try:
            nnunet_image_path.symlink_to(image_path)
        except FileExistsError:
            continue


if __name__ == "__main__":
    main()
