from collections import defaultdict

import fire
import numpy as np
import pandas as pd
import scipy

from pet_seg.settings import NNUNET_VAL_TEST_RESULTS_DIR
from pet_seg.settings import ORGANS

CONFIDENCE = 0.95


def main():
    fire.Fire(anaylze_nnunet_test_results)


def anaylze_nnunet_test_results():
    """Compute per organ dice scores for all extracted nnUNet results stored in NNUNET_VAL_TEST_RESULTS_DIR."""

    for test_results_path in (NNUNET_VAL_TEST_RESULTS_DIR / "patient_dice_scores").glob("*.csv"):
        all_patients_dice_scores_df = pd.read_csv(test_results_path)

        # Create dataset-radionuclide column as unique identifier (needed for cross tracer datasets)
        try:
            all_patients_dice_scores_df["dataset-radionuclide"] = (
                all_patients_dice_scores_df["dataset"] + "-" + all_patients_dice_scores_df["radionuclide_name"]
            )
        except KeyError:
            all_patients_dice_scores_df["dataset-radionuclide"] = all_patients_dice_scores_df["dataset"]

        organ_dice_scores = defaultdict(list)
        for test_dataset in all_patients_dice_scores_df["dataset-radionuclide"].unique():
            organ_dice_scores["dataset-radionuclide"].append(test_dataset)

            patients_dice_scores_df = all_patients_dice_scores_df[
                all_patients_dice_scores_df["dataset-radionuclide"] == test_dataset
            ]

            add_per_organ_dice_scores(organ_dice_scores, patients_dice_scores_df, CONFIDENCE)
            add_patient_characteristics(organ_dice_scores, patients_dice_scores_df)

        # Compute per organ dice scores for all patients combined
        organ_dice_scores["dataset-radionuclide"].append("all")
        add_per_organ_dice_scores(organ_dice_scores, all_patients_dice_scores_df, CONFIDENCE)
        add_patient_characteristics(organ_dice_scores, all_patients_dice_scores_df)

        # Save per organ dice scores
        organ_dice_scores_df = pd.DataFrame(organ_dice_scores)
        organ_dice_scores_df.to_csv(str(test_results_path).replace("patient", "organ"), index=False)


def add_per_organ_dice_scores(organ_dice_scores, patients_dice_scores_df, confidence):
    """Compute per organ dice scores for the given patients.

    Args:
        organ_dice_scores (defaultdict(list)): Dictionary to store the results in.
        patients_dice_scores_df (pd.DataFrame): DataFrame containing the dice scores for each patient.
        confidence (float): Confidence level to use for the confidence intervals.
    """
    for organ in ORGANS + ["mean"]:
        organ_series = patients_dice_scores_df[organ]

        ci_intervals = scipy.stats.t.interval(
            confidence=confidence,
            df=len(organ_series) - 1,
            loc=np.mean(organ_series),
            scale=scipy.stats.sem(organ_series),
        )
        ci_str = (
            f"{organ_series.mean():.3f} (95% CI: {ci_intervals[0]:.3f}, {ci_intervals[1]:.3f}), n={len(organ_series)}"
        )

        organ_dice_scores[organ].append(ci_str)


def add_patient_characteristics(organ_dice_scores, patients_dice_scores_df):
    """Compute patient characteristics (gender, Dose, post_inj_time, age, weight) for the given patients.

    Args:
        organ_dice_scores (defaultdict(list)): Dictionary to store the results in.
        patients_dice_scores_df (pd.DataFrame): DataFrame containing the dice scores for each patient.
    """
    gender_dict = patients_dice_scores_df["gender"].value_counts().to_dict()
    try:
        organ_dice_scores["gender"].append(f"{gender_dict['F']}/{gender_dict['M']}")
    except KeyError:
        organ_dice_scores["gender"].append("0/0")

    for col in ["Dose", "post_inj_time", "age", "weight"]:
        if col == "Dose":
            organ_dice_scores[col].append(
                f"{patients_dice_scores_df[col].mean() / 1e6:.1f}±{patients_dice_scores_df[col].std() / 1e6:.1f}"
            )
        else:
            organ_dice_scores[col].append(
                f"{patients_dice_scores_df[col].mean():.1f}±{patients_dice_scores_df[col].std():.1f}"
            )


if __name__ == "__main__":
    fire.Fire(main)
