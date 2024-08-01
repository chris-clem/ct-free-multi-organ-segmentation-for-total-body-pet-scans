import os
from pathlib import Path

import fire
import pandas as pd
from loguru import logger

from pet_seg.settings import DICOM_HEADERS_DIR
from pet_seg.settings import MODEL_DATASET_IDS_TO_NAMES
from pet_seg.settings import RESULTS_DIR
from pet_seg.settings import TEST_DATASET_IDS_TO_NAMES
from pet_seg.settings import TEST_DATASETS_TO_IDS
from pet_seg.utils import create_patient_metrics_df

NNUNET_RESULTS_DIR = Path(os.environ["nnUNet_results"])


def main():
    fire.Fire(extract_nnunet_results)


def extract_nnunet_results(
    model_dataset_id: int = 501,
    trainer: str = "nnUNetTrainer",
    plans: str = "nnUNetResEncUNetMPlans",
    config: str = "3d_fullres",
    folds: str = "all",
    test_datasets: str = "internal_merged_new_ts",
    images_dir_name: str = "imagesTs",
    use_optimized_labels: bool = False,
    metric: str = "Dice",
):
    """Extract nnUNet results from the predictions of the given model and test datasets.

    Creates a CSV file containing the dice scores for each patient.

    Args:
        model_dataset_id (int): Model trained on the given dataset to use.
        trainer (str): Trainer to use.
        plans (str): nnUNet plans to use.
        config (str): nnUNet config to use. Can be "2d" or "3d_cascade_fullres".
        folds (str): Folds to use, separated by spaces.
        test_datasets (str): Test datasets to predict on. Can be "internal", "cross_scanner" or "cross_tracer".
        images_dir_name (str): Name of the images directory to use.
        use_optimized_labels (bool): Whether to use optimized labels.
        metric (str): Metric to use. Can be "Dice" or "IoU".
    """

    # Get dirs
    model_dataset_name = MODEL_DATASET_IDS_TO_NAMES[model_dataset_id]
    model_results_dir = NNUNET_RESULTS_DIR / model_dataset_name
    config_dir = model_results_dir / f"{trainer}__{plans}__{config}"
    fold_str = f"fold_{folds.replace(' ', '_')}"
    predictions_dir = config_dir / fold_str / "predictions"

    # Get paths to nnunet summary files
    summary_paths = [
        predictions_dir
        / f"{images_dir_name}_{TEST_DATASET_IDS_TO_NAMES[test_dataset_id]}"
        / ("summary_optimized.json" if use_optimized_labels else "summary.json")
        for test_dataset_id in TEST_DATASETS_TO_IDS[test_datasets]
    ]

    # Extract patient dice scores from nnunet summary file and merge with dicom header
    patient_dice_scores_dfs = []
    for summary_path in summary_paths:
        # Create patient dice scores df
        patient_metrics = create_patient_metrics_df(
            summary_path,
            use_merged_seg="merged" in test_datasets,
            use_optimized_seg="optimized" in test_datasets,
            metric=metric,
        )

        # Merge with dicom header
        if "internal" not in test_datasets:
            # Load corresponding dicom header
            scanner = summary_path.parent.name.split("-")[0].split("_", maxsplit=2)[-1]
            dicom_header_csv_path = DICOM_HEADERS_DIR / f"{scanner}.csv"
            dicom_header_df = pd.read_csv(dicom_header_csv_path)

            patient_metrics = patient_metrics.merge(dicom_header_df, left_on="patient_id", right_on="PID")

        patient_dice_scores_dfs.append(patient_metrics)

    # Save patient dice scores df
    patient_metrics_df = pd.concat(patient_dice_scores_dfs)
    output_file_path = (
        RESULTS_DIR
        / "patient_metrics"
        / f"{model_dataset_name}__{trainer}__{plans}__{config}__{fold_str}__{test_datasets}{'__optimized' if use_optimized_labels else ''}__{metric}.csv"  # noqa: E501
    )

    patient_metrics_df.to_csv(output_file_path, index=False)
    logger.info(f"Created {output_file_path}")


if __name__ == "__main__":
    main()
