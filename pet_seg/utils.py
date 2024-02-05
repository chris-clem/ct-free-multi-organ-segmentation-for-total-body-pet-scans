import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import scipy

from pet_seg.settings import ANATOMICAL_REGIONS
from pet_seg.settings import DATA_ROOT_DIR
from pet_seg.settings import INDEX_TO_ANATOMICAL_STRUCTURES
from pet_seg.settings import MERGED_ANATOMICAL_STRUCTURES
from pet_seg.settings import SCANNER_TO_STAGE


def get_scanner_dir(scanner: str) -> Path:
    return DATA_ROOT_DIR / SCANNER_TO_STAGE[scanner] / scanner


def get_sorted_patient_dirs(scanner: str) -> list[Path]:
    return sorted(get_scanner_dir(scanner).iterdir())


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


def compute_ci_intervals(confidence, dice_scores_df, col):
    """Compute confidence intervals for the given column of the given DataFrame.

    Args:
        confidence (float): Confidence level to use for the confidence intervals.
        dice_scores_df (pd.DataFrame): DataFrame containing the dice scores for each patient.
        col (str): Column to compute the confidence intervals for.

    Returns:
        tuple: Confidence intervals.
    """
    ci = scipy.stats.t.interval(
        confidence=confidence,
        df=len(dice_scores_df[col]) - 1,
        loc=np.nanmean(dice_scores_df[col]),
        scale=scipy.stats.sem(
            dice_scores_df[col],
            nan_policy="omit",
        ),
    )

    return ci


def create_ci_intervals_str(dice_scores_df, col, ci_intervals):
    """Create a string containing the confidence intervals for the given column of the given DataFrame.

    Args:
        dice_scores_df (pd.DataFrame): DataFrame containing the dice scores for each patient.
        col (str): Column to compute the confidence intervals for.
        ci_intervals (tuple): Confidence intervals.

    Returns:
        str: Confidence intervals string.
    """
    ci_str = f"{dice_scores_df[col].mean():.3f} (95% CI: {ci_intervals[0]:.3f}, {ci_intervals[1]:.3f}), n={len(dice_scores_df)}"  # noqa: E501
    return ci_str
