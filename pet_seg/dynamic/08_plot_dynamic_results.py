from collections import defaultdict
from pathlib import Path

import fire
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from matplotlib import pyplot as plt

from pet_seg.settings import CUM_FRAME_TIMES_S

sns.set_theme()


def main():
    fire.Fire(plot_dynamic_results)


def plot_dynamic_results():
    # TODO: use NASC to create AUC table
    # TODO: create figures with ASC-TS_organ seg, AC_gen-dynamic and AC_gen-optimized_dynamic

    num_frames = st.sidebar.selectbox("Num frames", [12, 92])
    pet_type = st.sidebar.selectbox("PET type", ["ASC", "NASC"])
    results_path = Path(".") / f"dynamic_results_F{num_frames}_{pet_type}_1.csv"

    show_volume = st.sidebar.checkbox("Show volume", value=False)
    use_mean = st.sidebar.checkbox("Use mean", value=True)

    max_frame_num = st.sidebar.slider("Max frame num", min_value=1, max_value=92, value=92)

    results_df = pd.read_csv(results_path)

    # Translate column names
    results_df = results_df.rename(
        columns={
            "TS_pred_organ_size": "CT-based size",
            "TS_pred_organ_activity": "CT-based activity",
            "dynamic_pred_organ_activity": "Dynamic PET-based activity",
            "dynamic_pred_organ_size": "Dynamic PET-based size",
            "static_pred_organ_activity": "Static PET-based activity",
            "static_pred_organ_size": "Static PET-based size",
        }
    )

    st.write(results_df.head())

    # Compute dynamic_pred_organ_dice per organ
    # st.write(results_df.groupby(["anatomical_structure"])[["dynamic_pred_organ_dice", "TS_pred_organ_dice"]].mean())

    st.write(f"# Dynamic Results {results_path.stem.replace('dynamic_results_', '')}")

    for patient_id in list(results_df["patient_id"].unique())[3:4]:
        patient_df = results_df[results_df["patient_id"] == patient_id]

        st.write(f"## {patient_id}")

        # Sort by frame_num
        patient_df = patient_df.sort_values("frame_num")

        patient_df = patient_df[patient_df["frame_num"] <= max_frame_num]

        # Use static pet-based values for dynamic pet-based if frame_num < 25
        patient_df["Dynamic PET-based activity"] = np.where(
            patient_df["frame_num"] < 25,
            patient_df["Static PET-based activity"],
            patient_df["Dynamic PET-based activity"],
        )
        patient_df["Dynamic PET-based size"] = np.where(
            patient_df["frame_num"] < 25,
            patient_df["Static PET-based size"],
            patient_df["Dynamic PET-based size"],
        )

        st.write(patient_df)

        frame_types = [
            "CT-based",
            "Dynamic PET-based",
            # "Static PET-based",
            # "optimized_dynamic",
        ]

        aucs = defaultdict(list)
        for anatomical_structure in sorted(patient_df["anatomical_structure"].unique()):
            organ_df = patient_df[patient_df["anatomical_structure"] == anatomical_structure]

            organ_df["cum_frame_time_s"] = CUM_FRAME_TIMES_S[f"F{num_frames}"][:max_frame_num]

            # Smooth dynamic PET-based after frame 25
            organ_df["Dynamic PET-based activity"] = np.where(
                organ_df["frame_num"] >= 25,
                organ_df["Dynamic PET-based activity"].rolling(window=3).mean(),
                organ_df["Dynamic PET-based activity"],
            )
            organ_df["Dynamic PET-based size"] = np.where(
                organ_df["frame_num"] >= 25,
                organ_df["Dynamic PET-based size"].rolling(window=3).mean(),
                organ_df["Dynamic PET-based size"],
            )

            if use_mean:
                for type in frame_types:
                    organ_df[f"{type} activity"] = organ_df[f"{type} activity"] / organ_df[f"{type} size"]

            # Backfill/ interpolate NaNs
            # organ_df["optimized_dynamic_pred_organ_size"] = organ_df["optimized_dynamic_pred_organ_size"].fillna(
            #     method="bfill"
            # )

            # anatomical_structure_df["dynamic_pred_organ_dice"] = anatomical_structure_df[
            #     "dynamic_pred_organ_dice"
            # ].fillna(method="bfill")

            # organ_df["optimized_dynamic_pred_organ_activity"] = organ_df[
            #     "optimized_dynamic_pred_organ_activity"
            # ].interpolate()

            anatomical_structure = organ_df["anatomical_structure"].iloc[0]

            st.write(f"### {anatomical_structure}")
            # st.write(organ_df)
            # st.write(organ_df.describe())

            aucs["anatomical_structure"].append(anatomical_structure)
            for type in frame_types:
                auc = np.trapz(organ_df[f"{type} activity"], organ_df["cum_frame_time_s"])
                aucs[type].append(auc)

            if show_volume:
                st.line_chart(
                    organ_df,
                    x="cum_frame_time_s",
                    y=[f"{type} size" for type in frame_types],
                )

            # st.line_chart(
            #     organ_df,
            #     x="cum_frame_time_s",
            #     y=[f"{type} activity" for type in frame_types],
            # )

            fig, ax = plt.subplots(figsize=(7, 4))
            for type in frame_types:
                sns.lineplot(
                    data=organ_df,
                    x="cum_frame_time_s",
                    y=f"{type} activity",
                    ax=ax,
                    label=type,
                )
            ax.set_title(anatomical_structure.capitalize())
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Activity")
            st.write(fig)

            # st.line_chart(
            #     anatomical_structure_df,
            #     x="cum_frame_time_s",
            #     y="dynamic_pred_organ_dice",
            # )

            # break

        aucs_df = pd.DataFrame(aucs)
        st.write("## AUCs")
        st.write(aucs_df)

        # Divide TS, static, dynamic AUCs by optimized_dynamic
        # aucs_df["TS"] = ((aucs_df["TS"] - aucs_df["optimized_dynamic"]).abs() / aucs_df["optimized_dynamic"]) * 100
        # aucs_df["static"] = (
        #     (aucs_df["static"] - aucs_df["optimized_dynamic"]).abs() / aucs_df["optimized_dynamic"]
        # ) * 100
        # aucs_df["dynamic"] = (
        #     (aucs_df["dynamic"] - aucs_df["optimized_dynamic"]).abs() / aucs_df["optimized_dynamic"]
        # ) * 100

        # st.write("## AUCs normalized by optimized_dynamic")
        # st.write(aucs_df)
        # st.write(aucs_df.describe())


if __name__ == "__main__":
    main()
