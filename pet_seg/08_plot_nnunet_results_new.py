import fire
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st

from pet_seg.settings import ANATOMICAL_REGIONS
from pet_seg.settings import RESULTS_DIR

sns.set_theme()


def main():
    fire.Fire(plot_nnunet_test_results)


def plot_nnunet_test_results():
    for patient_dice_scores_path in (RESULTS_DIR / "patient_dice_scores").glob("*.csv"):
        st.write(f"# {patient_dice_scores_path.stem}")
        dice_scores_df = pd.read_csv(patient_dice_scores_path)

        if "cross_tracer" in patient_dice_scores_path.stem:
            dice_scores_df["dataset"] = dice_scores_df["dataset"] + "-" + dice_scores_df["radionuclide_name"]
        else:
            continue

        for unique_dataset in dice_scores_df["dataset"].unique():
            st.write(f"## {unique_dataset}")

            dataset_dice_scores_df = dice_scores_df[dice_scores_df["dataset"] == unique_dataset]
            st.write(dataset_dice_scores_df)
            st.write(dataset_dice_scores_df.describe())

            for anatomical_region in ANATOMICAL_REGIONS:
                anatomical_structures = ANATOMICAL_REGIONS[anatomical_region]
                regions_df = dataset_dice_scores_df[anatomical_structures]

                fig, ax = plt.subplots(figsize=(5, 0.5 * len(anatomical_structures)))
                sns.boxplot(
                    data=regions_df,
                    orient="h",
                    ax=ax,
                    palette="Set3",
                )
                ax.set_xlim(0, 1)
                ax.set_title(f"{anatomical_region} (mean dice: {regions_df.mean().mean():.2f})")
                st.write(fig)


if __name__ == "__main__":
    fire.Fire(main)
