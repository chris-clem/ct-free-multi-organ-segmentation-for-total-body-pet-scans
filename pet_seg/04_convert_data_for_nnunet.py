import os
from pathlib import Path
from typing import List
from typing import Optional

import fire
import pandas as pd
from loguru import logger
from nnunetv2.dataset_conversion.generate_dataset_json import generate_dataset_json

from pet_seg.settings import ANATOMICAL_STRUCTURES_TO_INDEX
from pet_seg.settings import MERGED_ANATOMICAL_STRUCTURES_TO_INDEX
from pet_seg.settings import OPTIMIZED_LABELS_TO_INDEX

NNUNET_RAW_DIR = Path(os.environ["nnUNet_raw"])


def main():
    fire.Fire(convert_data_for_nnunet)


def convert_data_for_nnunet(
    data_csv_path: str,
    pet_type: str = "nac",
    seg_type: str = "ts",
    dataset_id: Optional[int] = None,
):
    """Converts data for nnUNet.

    Follows instructions from https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/dataset_format.md

    Args:
        data_csv_path (str): Path to the CSV file containing the data.
        pet_type (str): PET type to use. Can be "ac" or "nac". Defaults to "nac".
        seg_type (str): Segmentation type to use. Can be "moose", "moose_optimized", "ts", or "ts_merged".
        dataset_id (Optional[int]): ID of the dataset. If not provided, the next available ID is used.
    """
    use_optimized_seg = seg_type == "moose_optimized"
    use_merged_seg = seg_type == "ts_merged"

    # Load data
    data_csv_path = Path(data_csv_path)
    df = pd.read_csv(data_csv_path)

    if use_optimized_seg:
        # Filter out patients without optimized segmentation
        df = df[df["seg_moose_optimized"].notnull()]
        # Set stage of rest of patients to train (optimized seg normally used for testing)
        df["stage"] = "train"

    df_train = df[df["stage"] == "train"]
    df_test = df[df["stage"] == "test"]

    # Create dataset name consisting of ID (a three digit integer) + freely chosen name
    if dataset_id is None:
        existing_dataset_ids = [
            int(path.name.split("_")[0].replace("Dataset", "")) for path in sorted(NNUNET_RAW_DIR.iterdir())
        ]
        if len(existing_dataset_ids) > 0:
            dataset_id = existing_dataset_ids[-1] + 1
        else:
            dataset_id = 1

    dataset_name = (
        f"Dataset{dataset_id:03d}_"
        f"{data_csv_path.name.split('-num_train')[0]}-num_train={len(df_train)}-num_test={len(df_test)}_"
        f"{pet_type.upper()}-{seg_type}"
    )

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

        create_symlinks(df_train[f"seg_{seg_type}"].values, nnunet_dir=labels_tr_dir)
        logger.debug(f"Created {len(list(labels_tr_dir.iterdir()))} symlinks in {labels_tr_dir}")

    # Test images
    images_ts_dir = dataset_raw_dir / "imagesTs"
    images_ts_dir.mkdir(exist_ok=True)

    create_symlinks(image_paths=df_test[f"pet_{pet_type}"].values, nnunet_dir=images_ts_dir)
    logger.debug(f"Created {len(list(images_ts_dir.iterdir()))} symlinks in {images_ts_dir}")

    # Test labels
    labels_ts_dir = dataset_raw_dir / "labelsTs"
    labels_ts_dir.mkdir(exist_ok=True)

    create_symlinks(df_test[f"pet_{pet_type}"].values, nnunet_dir=labels_ts_dir)
    logger.debug(f"Created {len(list(labels_ts_dir.iterdir()))} symlinks in {labels_ts_dir}")

    if use_merged_seg:
        labels = MERGED_ANATOMICAL_STRUCTURES_TO_INDEX
    elif use_optimized_seg:
        labels = OPTIMIZED_LABELS_TO_INDEX
    else:
        labels = ANATOMICAL_STRUCTURES_TO_INDEX

    generate_dataset_json(
        output_folder=str(dataset_raw_dir),
        channel_names={0: pet_type.upper()},
        labels=labels,
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
        is_dynamic = "dynamic" in image_path and "static" not in image_path

        image_path = Path(image_path)

        patient_id = image_path.parent.parent.name if is_dynamic else image_path.parent.name
        image_name = f"{patient_id}_{image_path.name.split('.')[0]}" if is_dynamic else patient_id

        nnunet_image_name = f"{image_name}.nii.gz" if "label" in nnunet_dir.name else f"{image_name}_0000.nii.gz"
        nnunet_image_path = nnunet_dir / nnunet_image_name

        try:
            nnunet_image_path.symlink_to(image_path)
        except FileExistsError:
            continue


if __name__ == "__main__":
    main()
