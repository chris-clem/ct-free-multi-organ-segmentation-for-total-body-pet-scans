from collections import defaultdict

import fire
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st

from pet_seg.settings import ANATOMICAL_REGIONS
from pet_seg.settings import CONFIDENCE
from pet_seg.settings import RESULTS_DIR
from pet_seg.utils import create_ci_intervals_str

sns.set_theme()

NICE_DATASET_NAMES = {
    "imagesTs_Dataset001_Bern_Quadra-SH_uExplorer-num_train=956-num_test=50_NAC-Quadra": "Bern Quadra (n = 25)",
    "imagesTs_Dataset001_Bern_Quadra-SH_uExplorer-num_train=956-num_test=50_NAC-uExplorer": "Shared uExplorer (n = 25)",
    "imagesTs_Dataset101_Bern_Quadra_UHS-num_train=0-num_test=21_NAC-ts_merged": "Bern Quadra UHS (n = 21)",
    "imagesTs_Dataset102_Bern_Vision600-num_train=0-num_test=51_NAC-ts_merged": "Bern Vision 600 (n = 51)",
    "imagesTs_Dataset103_SH_GE_Discovery-num_train=0-num_test=104_NAC-ts_merged": "Shanghai MI (n = 104)",
    "imagesTs_Dataset104_SH_UI780-num_train=0-num_test=99_NAC-ts_merged": "Shanghai uMI780 (n = 99)",
    "imagesTs_Dataset105_SH_Vision450-num_train=0-num_test=51_NAC-ts_merged": "Shanghai Vision 450 (n = 51)",
    "imagesTs_Dataset106_Bern_Vision600_cross_tracer-num_train=0-num_test=30_NAC-ts_merged-PSMA": "Bern Vision 600 18F-PSMA (n = 12)",  # noqa: E501
    "imagesTs_Dataset106_Bern_Vision600_cross_tracer-num_train=0-num_test=30_NAC-ts_merged-Edotreotide (DOTATOC)": "Bern Vision 600 68Ga-DOTA-TOC (n = 6)",  # noqa: E501
    "imagesTs_Dataset107_SH_Vision_cross_tracer-num_train=0-num_test=41_NAC-ts_merged-Edotreotide (DOTATOC)": "Shanghai Vision 450 68Ga-FAPI (n = 13)",  # noqa: E501
    "imagesTs_Dataset107_SH_Vision_cross_tracer-num_train=0-num_test=41_NAC-ts_merged-FAPI": "Shanghai Vision 450 68Ga-DOTA-TATE (n = 26)",  # noqa: E501
    "imagesTs_Dataset501_Bern_Quadra-SH_uExplorer-num_train=938-num_test=68_NAC-ts_merged-Quadra": "Bern Quadra (n = 34)",  # noqa: E501
    "imagesTs_Dataset501_Bern_Quadra-SH_uExplorer-num_train=938-num_test=68_NAC-ts_merged-uExplorer": "Shared uExplorer (n = 34)",  # noqa: E501
}


def main():
    fire.Fire(plot_nnunet_test_results)


def plot_nnunet_test_results():
    for metric in ("Dice", "IoU"):
        st.write(f"# {metric}")
        for patient_dice_scores_path in (RESULTS_DIR / "patient_metrics").glob(f"Dataset501_*__{metric}.csv"):
            (
                dataset_name,
                trainer_name,
                plans_name,
                config,
                fold,
                test_datatets,
                metric,
            ) = patient_dice_scores_path.stem.split("__")
            st.write(f"# {dataset_name} - {test_datatets} - {metric}")

            dice_scores_df = pd.read_csv(patient_dice_scores_path)

            if "internal" in patient_dice_scores_path.stem:
                # Add scanner column: if Anonymous in patient_id, then "uExplorer" else "Quadra"
                dice_scores_df["scanner"] = dice_scores_df["patient_id"].apply(
                    lambda x: "uExplorer" if "Anonymous" in x else "Quadra"
                )

                # Add scanner to dataset column
                dice_scores_df["dataset"] = dice_scores_df["dataset"] + "-" + dice_scores_df["scanner"]

            if "cross_tracer" in patient_dice_scores_path.stem:
                dice_scores_df["dataset"] = dice_scores_df["dataset"] + "-" + dice_scores_df["radionuclide_name"]

            dataset_dice_scores_dfs = []
            per_organ_dice_scores = defaultdict(list)
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

                    # Add per region dice scores
                    if anatomical_region not in per_organ_dice_scores["Anatomical Structure"]:
                        per_organ_dice_scores["Anatomical Structure"].append(anatomical_region)
                    regions_df["mean"] = regions_df.mean(axis=1)
                    ci_str = create_ci_intervals_str(regions_df, "mean", CONFIDENCE)
                    per_organ_dice_scores[NICE_DATASET_NAMES[unique_dataset]].append(ci_str)

                    # Add per organ dice scores
                    for anatomical_structure in anatomical_structures_in_region:
                        if anatomical_structure not in per_organ_dice_scores["Anatomical Structure"]:
                            per_organ_dice_scores["Anatomical Structure"].append(anatomical_structure)
                        ci_str = create_ci_intervals_str(regions_df, anatomical_structure, CONFIDENCE)
                        per_organ_dice_scores[NICE_DATASET_NAMES[unique_dataset]].append(ci_str)

                all_anatomical_structures_df = dataset_dice_scores_df[all_anatomical_structures]
                all_anatomical_structures_df["mean"] = all_anatomical_structures_df.mean(axis=1)
                ci_str = create_ci_intervals_str(all_anatomical_structures_df, "mean", CONFIDENCE)
                st.write(unique_dataset, ci_str)

            per_organ_dice_scores_df = pd.DataFrame(per_organ_dice_scores)
            per_organ_dice_scores_df.set_index("Anatomical Structure", inplace=True)
            st.write(per_organ_dice_scores_df)

            dice_scores_df = pd.concat(dataset_dice_scores_dfs)
            dice_scores_df = dice_scores_df[all_anatomical_structures]
            dice_scores_df["mean"] = dice_scores_df.mean(axis=1)
            ci_str = create_ci_intervals_str(dice_scores_df, "mean", CONFIDENCE)
            st.write(test_datatets, ci_str)


if __name__ == "__main__":
    fire.Fire(main)
