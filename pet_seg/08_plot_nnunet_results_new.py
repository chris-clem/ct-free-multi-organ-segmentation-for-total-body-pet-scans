import fire
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st

from pet_seg.settings import ANATOMICAL_REGIONS
from pet_seg.settings import RESULTS_DIR
from pet_seg.utils import compute_ci_intervals
from pet_seg.utils import create_ci_intervals_str

sns.set_theme()


def main():
    fire.Fire(plot_nnunet_test_results)


def plot_nnunet_test_results():
    for patient_dice_scores_path in (RESULTS_DIR / "patient_dice_scores").glob("*_tracer.csv"):
        st.write(f"# {patient_dice_scores_path.stem}")
        dice_scores_df = pd.read_csv(patient_dice_scores_path)

        if "cross_tracer" in patient_dice_scores_path.stem:
            dice_scores_df["dataset"] = dice_scores_df["dataset"] + "-" + dice_scores_df["radionuclide_name"]
        # else:
        #     continue

        dataset_dice_scores_dfs = []
        for unique_dataset in dice_scores_df["dataset"].unique():
            if "Fluorodeoxyglucose" in unique_dataset:
                continue

            st.write(f"## {unique_dataset}")

            dataset_dice_scores_df = dice_scores_df[dice_scores_df["dataset"] == unique_dataset]
            dataset_dice_scores_dfs.append(dataset_dice_scores_df)
            st.write(dataset_dice_scores_df)
            st.write(dataset_dice_scores_df.describe())

            all_anatomical_structures = []
            for anatomical_region in ANATOMICAL_REGIONS:
                anatomical_structures_in_region = ANATOMICAL_REGIONS[anatomical_region]
                all_anatomical_structures.extend(anatomical_structures_in_region)

                regions_df = dataset_dice_scores_df[anatomical_structures_in_region]

                fig, ax = plt.subplots(figsize=(5, 0.5 * len(anatomical_structures_in_region)))
                sns.boxplot(
                    data=regions_df,
                    orient="h",
                    ax=ax,
                    palette="Set3",
                )
                ax.set_xlim(0, 1)
                ax.set_title(anatomical_region)
                st.write(fig)

                regions_df["mean"] = regions_df.mean(axis=1)
                st.write(
                    anatomical_region,
                    create_ci_intervals_str(regions_df, "mean", compute_ci_intervals(0.95, regions_df, "mean")),
                )

            all_anatomical_structures_df = dataset_dice_scores_df[all_anatomical_structures]
            all_anatomical_structures_df["mean"] = all_anatomical_structures_df.mean(axis=1)
            st.write(
                unique_dataset,
                create_ci_intervals_str(
                    all_anatomical_structures_df,
                    "mean",
                    compute_ci_intervals(0.95, all_anatomical_structures_df, "mean"),
                ),
            )

        dice_scores_df = pd.concat(dataset_dice_scores_dfs)
        dice_scores_df = dice_scores_df[all_anatomical_structures]
        dice_scores_df["mean"] = dice_scores_df.mean(axis=1)
        st.write(
            create_ci_intervals_str(
                dice_scores_df,
                "mean",
                compute_ci_intervals(0.95, dice_scores_df, "mean"),
            )
        )


if __name__ == "__main__":
    fire.Fire(main)
