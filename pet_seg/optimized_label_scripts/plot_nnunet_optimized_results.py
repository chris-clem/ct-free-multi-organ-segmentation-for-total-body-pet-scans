import os
from collections import OrderedDict
from pathlib import Path

import fire
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
import seaborn as sns
import skimage
import streamlit as st
from matplotlib.colors import ListedColormap
from monai.visualize.utils import blend_images
from skimage.morphology import dilation

from pet_seg.settings import CONFIDENCE
from pet_seg.settings import MERGED_ANATOMICAL_STRUCTURES_TO_INDEX
from pet_seg.settings import RESULTS_DIR
from pet_seg.utils import create_ci_intervals_str

NNUNET_RAW_DIR = Path(os.environ["nnUNet_raw"])
NNUNET_RESULTS_DIR = Path(os.environ["nnUNet_results"])

HEX_COLORS = {
    "Brain": "#8dd3c7",
    "Heart": "#ffffb3",
    "Kidneys": "#bebada",
    "Liver": "#fb8072",
    "Lungs": "#80b1d3",
    "Pancreas": "#fdb462",
    "Spleen": "#b3de69",
    "Thyroid Gland": "#fccde5",
    "Urinary Bladder": "#d9d9d9",
}

sns.set_theme()


def main():
    fire.Fire(plot_nnunet_test_results)


def plot_nnunet_test_results():
    optimized_label_names = [
        "Brain",
        "Heart",
        "Kidneys",
        "Liver",
        "Lungs",
        "Pancreas",
        "Spleen",
        "Thyroid Gland",
        "Urinary Bladder",
    ]

    for metric in ("Dice", "IoU"):
        st.write(f"# {metric}")

        for patient_dice_scores_path in (RESULTS_DIR / "patient_metrics").glob(f"ts_vs_optimized_labels__{metric}.csv"):
            is_ts_vs_optimized = "ts_vs_optimized" in patient_dice_scores_path.stem

            if is_ts_vs_optimized:
                st.write(f"# {patient_dice_scores_path.stem}")
            else:
                path_stem = patient_dice_scores_path.stem
                st.write(path_stem)
                (
                    model_dataset_name,
                    trainer,
                    plans,
                    config,
                    fold_str,
                    test_datasets,
                    optimized,
                    metric,
                ) = path_stem.split("__")

                st.write(f"# {model_dataset_name} - {trainer} - {plans} - {config} - {fold_str} - {test_datasets}")

            # Load the dice scores
            dice_scores_df = pd.read_csv(patient_dice_scores_path)
            dice_scores_df = dice_scores_df.set_index("patient_id")
            dice_scores_df = dice_scores_df.dropna(axis=1)
            dice_scores_df = dice_scores_df.loc[:, (dice_scores_df != 0).any(axis=0)]

            for dataset in dice_scores_df["dataset"].unique():
                # Get dirs containing images, labels, and preds
                if not is_ts_vs_optimized:
                    raw_dataset_dir = NNUNET_RAW_DIR / dataset.replace("imagesTs_", "")
                    predictions_dir = (
                        NNUNET_RESULTS_DIR
                        / model_dataset_name
                        / f"{trainer}__{plans}__{config}"
                        / fold_str
                        / "predictions"
                        / dataset
                        # / "merged_labels"
                    )

                st.write(f"## {dataset}")

                # Get dice scores for the dataset
                dataset_dsc_df = dice_scores_df[dice_scores_df["dataset"] == dataset]

                # Get dice scores for uExplorer and Quadra scanners
                uexplorer_dsc_df = dataset_dsc_df[dataset_dsc_df.index.str.contains("Anonymous")]
                quadra_dsc_df = dataset_dsc_df[~dataset_dsc_df.index.str.contains("Anonymous")]

                for dataset_name, scanner_dsc_df in zip(
                    ("Bern Quadra (n = 34)", "Shanghai uExplorer (n = 34)"), (quadra_dsc_df, uexplorer_dsc_df)
                ):
                    st.write(f"### {dataset_name}")
                    st.write(scanner_dsc_df)

                    # Get dice scores for optimized labels
                    regions_dsc_df = scanner_dsc_df[optimized_label_names]

                    st.write(regions_dsc_df)
                    st.write(regions_dsc_df.describe())

                    if not is_ts_vs_optimized:
                        patient_ids_to_plot = extract_patient_ids_to_plot(optimized_label_names, regions_dsc_df)
                        plot_patient_ids(patient_ids_to_plot, raw_dataset_dir, predictions_dir)

                    plot_dsc_boxplots(optimized_label_names, dataset_name, regions_dsc_df)

                    regions_dsc_df["mean"] = regions_dsc_df.mean(axis=1)
                    st.write(create_ci_intervals_str(regions_dsc_df, "mean", CONFIDENCE))


def extract_patient_ids_to_plot(optimized_label_names, regions_dsc_df):
    patient_ids_to_plot = OrderedDict()
    for label_name in optimized_label_names:
        # Get some quantiles
        label_dice_scores = regions_dsc_df[label_name]
        label_dice_scores_quantiles = label_dice_scores.quantile([0.25, 0.5, 0.75])

        # Find corresponding patient ids
        patient_ids_to_dsc = OrderedDict()
        for dice_score in label_dice_scores_quantiles:
            # Calculate the absolute difference between the quantile value and all scores
            # Then find the index of the minimum difference
            index = (label_dice_scores - dice_score).abs().idxmin()
            patient_ids_to_dsc[index] = dice_score

        patient_ids_to_plot[label_name] = patient_ids_to_dsc

    return patient_ids_to_plot


def plot_patient_ids(patient_ids_to_plot, raw_dataset_dir, predictions_dir):
    nrows = len(patient_ids_to_plot)
    ncols = len(patient_ids_to_plot["Brain"])
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(3.96 * ncols, 3.08 * nrows))

    for i, (label_name, patient_ids_to_dsc) in enumerate(patient_ids_to_plot.items()):
        hex_color = HEX_COLORS[label_name]

        for j, (patient_id, dice_score) in enumerate(patient_ids_to_dsc.items()):
            blended_slice = create_blended_slice(raw_dataset_dir, predictions_dir, label_name, patient_id, hex_color)

            axes[i, j].imshow(blended_slice)
            axes[i, j].text(x=4, y=20, s=f"Dice Score: {dice_score:.2f}", color="white", fontsize=16)
            # axes[i, j].set_title(f"Dice Score: {dice_score:.2f}")
            axes[i, j].axis("off")

    fig.tight_layout()
    st.write(fig)


def create_blended_slice(raw_dataset_dir, predictions_dir, label_name, patient_id, hex_color):
    image_path = raw_dataset_dir / "imagesTs" / f"{patient_id}_0000.nii.gz"
    label_path = raw_dataset_dir / "labelsTs_optimized_changed_order" / f"{patient_id}.nii.gz"
    pred_path = predictions_dir / f"{patient_id}.nii.gz"

    image_npy = nib.load(image_path).get_fdata()
    label_npy = nib.load(label_path).get_fdata()
    pred_npy = nib.load(pred_path).get_fdata()

    # Keep only the optimized label
    label_npy = (label_npy == MERGED_ANATOMICAL_STRUCTURES_TO_INDEX[label_name]).astype(int)
    pred_npy = (pred_npy == MERGED_ANATOMICAL_STRUCTURES_TO_INDEX[label_name]).astype(int)

    # Find the slice with the largest area
    slice_areas = pred_npy.sum(axis=(0, 1))
    slice_idx = slice_areas.argmax()

    image_slice = image_npy[..., slice_idx]
    label_slice = label_npy[..., slice_idx]
    pred_slice = pred_npy[..., slice_idx]

    # Rotate by 90 degrees
    image_slice = np.rot90(image_slice, k=3)
    label_slice = np.rot90(label_slice, k=3)
    pred_slice = np.rot90(pred_slice, k=3)

    # Mirror the slices
    image_slice = np.fliplr(image_slice)
    label_slice = np.fliplr(label_slice)
    pred_slice = np.fliplr(pred_slice)

    # Remove 15% at top and bottom
    H, W = pred_slice.shape
    slice_start_H, slice_start_W = int(H * 0.15), int(W * 0.05)
    slice_end_H, slice_end_W = int(H * 0.85), int(W * 0.95)
    image_slice = image_slice[slice_start_H:slice_end_H, slice_start_W:slice_end_W]
    label_slice = label_slice[slice_start_H:slice_end_H, slice_start_W:slice_end_W]
    pred_slice = pred_slice[slice_start_H:slice_end_H, slice_start_W:slice_end_W]

    # Extract contour from slices
    # label_slice_contour = extract_contour_from_slice(label_slice)
    pred_slice_contour = extract_contour_from_slice(pred_slice)

    # Scale image to percentiles
    percentile = 0.4
    percentile_lower, percentile_upper = np.percentile(image_slice, (percentile, 100 - percentile))
    image_slice = np.clip(image_slice, percentile_lower, percentile_upper)

    # Blend the image and the label
    custom_cmap = ListedColormap("#2ca02c")
    blended_slice = blend_images(image_slice[None, ...], label_slice[None, ...], alpha=0.5, cmap=custom_cmap)

    # Blend the image and the prediction
    custom_cmap = ListedColormap(hex_color)
    blended_slice = blend_images(blended_slice, pred_slice_contour[None, ...], alpha=1, cmap=custom_cmap)

    # Convert from CHW to HWC
    blended_slice = np.moveaxis(blended_slice, 0, -1)

    return blended_slice


def extract_contour_from_slice(slice):
    contours = skimage.measure.find_contours(slice, 0.5)
    slice_contour = np.zeros_like(slice)
    for contour in contours:
        slice_contour[contour[:, 0].astype(int), contour[:, 1].astype(int)] = 1

    # Make it thicker
    slice_contour = dilation(dilation(slice_contour))

    return slice_contour


def plot_dsc_boxplots(optimized_label_names, dataset_name, regions_dsc_df):
    fig, ax = plt.subplots(figsize=(5, 0.5 * len(optimized_label_names)))
    sns.boxplot(
        data=regions_dsc_df,
        orient="h",
        ax=ax,
        palette="Set3",
    )
    ax.set_xlim(0, 1)
    # ax.set_title(dataset_name)
    st.write(fig)


if __name__ == "__main__":
    fire.Fire(main)
