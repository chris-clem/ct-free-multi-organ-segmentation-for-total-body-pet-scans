from pathlib import Path

import fire
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st

from pet_seg.settings import ORGANS

sns.set_theme()

TEST_SET_TYPES_TO_NAMES = {
    "internal": "Internal Test Sets with MOOSE-Generated Labels",
    "cross_scanner": "Cross Scanner External Test Sets",
    "cross_tracer": "Cross Tracer External Test Sets",
}


TEST_DATASET_IDS_TO_NAMES = {
    "001": "Bern Quadra (n=25)",
    "002": "Shanghai uExplorer (n=25)",
    "004": "Bern Quadra UHS (n=21)",
    "005": "Bern Vision 600 (n=51)",
    "006-PSMA": "Bern Vision 600 PSMA (n=12)",
    "006-Edotreotide (DOTATOC)": "Bern Vision 600 DOTATOC (n=6)",
    "007": "Shanghai GE Discovery (n=104)",
    "008": "Shanghai UI780 (n=97)",
    "009": "Shanghai Vision 450 (n=51)",
    "010-Edotreotide (DOTATOC)": "Shanghai Vision 450 DOTATOC (n=26)",
    "010-FAPI": "Shanghai Vision 450 FAPI (n=14)",
}


def main():
    fire.Fire(plot_nnunet_test_results)


def plot_nnunet_test_results(
    model_dataset_id: str = "3",
):
    """Plot nnUNet test results."""

    # For each test set type
    for test_set_type in [
        "internal",
        "cross_scanner",
        "cross_tracer",
    ]:
        # Load all patient dice scores for selected test set type
        patient_dice_scores_path = sorted(
            (Path("nnunet_results") / "patient_dice_scores").glob(f"Dataset00{model_dataset_id}*{test_set_type}.csv")
        )[0]
        dice_scores_df = pd.read_csv(patient_dice_scores_path)

        # Extract dataset id from dataset col
        dice_scores_df["test-dataset-id"] = dice_scores_df["dataset"].str.split("_").str[1].str.replace("Dataset", "")

        if test_set_type == "cross_tracer":
            dice_scores_df["test-dataset-id"] = (
                dice_scores_df["test-dataset-id"] + "-" + dice_scores_df["radionuclide_name"]
            )

        # Add test dataset name
        dice_scores_df["test-dataset-name"] = dice_scores_df["test-dataset-id"].map(TEST_DATASET_IDS_TO_NAMES)

        st.write(f"# {test_set_type}")

        # Create figure with mean dice scores for each test dataset and model
        fig, ax = plt.subplots(figsize=(len(dice_scores_df["test-dataset-name"].unique()) * 4.0, 8))

        # Add boxplot to ax
        sns.boxplot(data=dice_scores_df, x="test-dataset-name", y="mean", ax=ax)

        ax.set_title(TEST_SET_TYPES_TO_NAMES[test_set_type], fontsize=20)
        ax.set_xlabel(None)
        ax.set_ylabel("Dice score")

        st.write(fig)

        # Save figure
        fig.savefig(
            f"nnunet_results/figures/{test_set_type}_mean.png",
            bbox_inches="tight",
            dpi=300,
        )

        # Create figure with dice scores for each organ for each test dataset
        for test_dataset_name in dice_scores_df["test-dataset-name"].unique():
            test_dataset_dice_scores_df = dice_scores_df[dice_scores_df["test-dataset-name"] == test_dataset_name]

            test_dataset_dice_scores_df_melted = test_dataset_dice_scores_df[ORGANS].melt(
                var_name="organ",
                value_name="dice",
            )

            fig, ax = plt.subplots(figsize=(len(ORGANS) * 3.0, 8))
            sns.boxplot(data=test_dataset_dice_scores_df_melted, x="organ", y="dice", ax=ax)

            ax.set_title(test_dataset_name, fontsize=20)
            ax.set_xlabel(None)
            ax.set_ylabel("Dice score")

            st.write(fig)

            # Save figure
            fig.savefig(
                f"nnunet_results/figures/{test_set_type}_{test_dataset_name}_organs.png",
                bbox_inches="tight",
            )


if __name__ == "__main__":
    fire.Fire(main)
