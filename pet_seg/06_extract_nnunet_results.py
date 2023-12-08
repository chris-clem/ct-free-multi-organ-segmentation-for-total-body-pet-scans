import json
import os
from collections import defaultdict
from pathlib import Path

import fire
import pandas as pd

from pet_seg.settings import ANATOMICAL_REGIONS
from pet_seg.settings import DICOM_HEADERS_DIR
from pet_seg.settings import INDEX_TO_ANATOMICAL_STRUCTURES
from pet_seg.settings import MERGED_ANATOMICAL_STRUCTURES
from pet_seg.settings import MODEL_DATASET_IDS_TO_NAMES
from pet_seg.settings import RESULTS_DIR
from pet_seg.settings import TEST_DATASET_IDS_TO_NAMES
from pet_seg.settings import TEST_DATASETS_TO_IDS

NNUNET_RESULTS_DIR = Path(os.environ["nnUNet_results"])


def main():
    fire.Fire(extract_nnunet_results)


def extract_nnunet_results(
    model_dataset_id: int = 1,
    config: str = "3d_fullres",
    folds: str = "0 1 2 3 4",
    test_datasets: str = "internal",
):
    """Extract nnUNet results from the predictions of the given model and test datasets.

    Creates a CSV file containing the dice scores for each patient.

    Args:
        model_dataset_id (int): Model trained on the given dataset to use.
        config (str): nnUNet config to use. Can be "2d" or "3d_cascade_fullres".
        folds (str): Folds to use, separated by spaces.
        test_datasets (str): Test datasets to predict on. Can be "internal", "cross_scanner" or "cross_tracer".
    """

    # Get dirs
    model_dataset_name = MODEL_DATASET_IDS_TO_NAMES[model_dataset_id]
    model_results_dir = NNUNET_RESULTS_DIR / model_dataset_name
    config_dir = model_results_dir / f"nnUNetTrainerNoMirroring__nnUNetPlans__{config}"
    fold_str = f"fold_{folds.replace(' ', '_')}"
    predictions_dir = config_dir / fold_str / "predictions"

    # Get paths to nnunet summary files
    summary_paths = [
        predictions_dir / f"imagesTs_{TEST_DATASET_IDS_TO_NAMES[test_dataset_id]}" / "summary.json"
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
        RESULTS_DIR / "patient_dice_scores" / f"{model_dataset_name}_{config}_{fold_str}_{test_datasets}.csv",
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

        for index, anatomical_structure in INDEX_TO_ANATOMICAL_STRUCTURES.items():
            if index == 0:
                continue

            dice_score = patient_result["metrics"][str(index)]["Dice"]
            patient_dice_scores[anatomical_structure].append(dice_score)

    patient_dice_scores = pd.DataFrame(patient_dice_scores)

    # Add merged anatomical structures means
    for anatomical_structure, merged_anatomical_structures in MERGED_ANATOMICAL_STRUCTURES.items():
        patient_dice_scores[anatomical_structure] = patient_dice_scores[merged_anatomical_structures].mean(axis=1)

    # Add anatomical regions means
    for anatomical_region, anatomical_structures in ANATOMICAL_REGIONS.items():
        all_anatomical_structures = []
        for anatomical_structure in anatomical_structures:
            if anatomical_structure in patient_dice_scores.columns:
                all_anatomical_structures.append(anatomical_structure)
            else:
                all_anatomical_structures.extend(MERGED_ANATOMICAL_STRUCTURES[anatomical_structure])

        patient_dice_scores[anatomical_region] = patient_dice_scores[all_anatomical_structures].mean(axis=1)

    return patient_dice_scores


if __name__ == "__main__":
    main()
