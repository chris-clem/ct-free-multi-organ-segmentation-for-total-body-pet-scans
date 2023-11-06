import json
import os
from collections import defaultdict
from pathlib import Path

import fire
import pandas as pd

from pet_seg.settings import DATASET_IDS_TO_NAMES
from pet_seg.settings import DICOM_HEADERS_DIR
from pet_seg.settings import NNUNET_VAL_TEST_RESULTS_DIR
from pet_seg.settings import ORGANS
from pet_seg.settings import ORGANS_TO_LABELS
from pet_seg.settings import TEST_DATASETS_TO_IDS

NNUNET_RESULTS_DIR = Path(os.environ["nnUNet_results"])


def main():
    fire.Fire(extract_nnunet_results)


def extract_nnunet_results(
    model_dataset_id: str = "3",
    config: str = "3d_cascade_fullres",
    fold="folds_0_1_2_3_4",
    test_datasets: str = "test_internal_optimized",
):
    """Extract nnUNet results from the given models and datasets.

    Creates a CSV file containing the dice scores for each patient.

    Args:
        model_dataset_id (str): Model trained on the given dataset to use.
        config (str): nnUNet config to use. Can be "2d" or "3d_cascade_fullres".
        fold (str): Fold to use.
        test_datasets (str): Datasets to use. Can be "test_internal", "test_cross_scanner" or "test_cross_tracer".
    """

    # Get dirs
    model_dataset_name = DATASET_IDS_TO_NAMES[int(model_dataset_id)]
    model_results_dir = NNUNET_RESULTS_DIR / model_dataset_name
    config_dir = model_results_dir / f"nnUNetTrainer__nnUNetPlans__{config}"
    fold_dir = config_dir / fold

    # Get paths to nnunet summary files
    summary_paths = [
        fold_dir / "predictions" / f"imagesTs_{DATASET_IDS_TO_NAMES[int(test_dataset_id)]}" / "summary.json"
        for test_dataset_id in TEST_DATASETS_TO_IDS[test_datasets]
    ]

    # Extract patient dice scores from nnunet summary file and merge with dicom header
    patient_dice_scores_dfs = []
    for summary_path in summary_paths:
        # Load corresponding dicom header
        scanner = summary_path.parent.name.split("-")[0].split("_", maxsplit=2)[-1]
        dicom_header_csv_path = DICOM_HEADERS_DIR / f"{scanner}.csv"
        dicom_header_df = pd.read_csv(dicom_header_csv_path)

        # Create patient dice scores df
        patient_dice_scores = create_patient_dice_scores_df(summary_path)

        # Merge with dicom header
        patient_dice_scores = patient_dice_scores.merge(dicom_header_df, left_on="patient_id", right_on="PID")

        patient_dice_scores_dfs.append(patient_dice_scores)

    # Save patient dice scores df
    patient_dice_scores_df = pd.concat(patient_dice_scores_dfs)
    patient_dice_scores_df.to_csv(
        NNUNET_VAL_TEST_RESULTS_DIR
        / "patient_dice_scores"
        / f"{model_dataset_name}_{config}_{fold}_{test_datasets}.csv",
        index=False,
    )


def create_patient_dice_scores_df(nnunet_summary_path):
    """Create a DataFrame containing the organ dice scores for each patient.

    Args:
        nnunet_summary_path (Path): Path to nnUNet summary JSON file.
    """
    with open(nnunet_summary_path, "r") as f:
        nnunet_summary = json.load(f)

    patient_dice_scores = defaultdict(list)
    for patient_result in nnunet_summary["metric_per_case"]:
        patient_id = Path(patient_result["reference_file"]).stem.split(".")[0]

        patient_dice_scores["dataset"].append(nnunet_summary_path.parent.name)
        patient_dice_scores["patient_id"].append(patient_id)

        for organ in ORGANS:
            organ_key = ORGANS_TO_LABELS[organ]
            dice_score = patient_result["metrics"][str(organ_key)]["Dice"]
            patient_dice_scores[organ].append(dice_score)

    patient_dice_scores = pd.DataFrame(patient_dice_scores)

    patient_dice_scores["mean"] = patient_dice_scores[ORGANS].mean(axis=1)

    return patient_dice_scores


if __name__ == "__main__":
    main()
