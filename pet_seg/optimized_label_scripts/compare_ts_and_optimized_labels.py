import os
from pathlib import Path

import fire
from loguru import logger
from nnunetv2.evaluation.evaluate_predictions import compute_metrics_on_folder2

from pet_seg.settings import MODEL_DATASET_IDS_TO_NAMES

NNUNET_RAW_DIR = Path(os.environ["nnUNet_raw"])
NNUNET_RESULTS_DIR = Path(os.environ["nnUNet_results"])


def main():
    fire.Fire(compare_ts_and_optimized_labels)


def compare_ts_and_optimized_labels(
    dataset_id: int = 1,
):
    dataset_name = MODEL_DATASET_IDS_TO_NAMES[dataset_id]
    raw_dir = NNUNET_RAW_DIR / dataset_name
    model_results_dir = NNUNET_RESULTS_DIR / dataset_name / "nnUNetTrainerNoMirroring__nnUNetPlans__3d_fullres"

    labels_ts_optimized_dir = raw_dir / "labelsTs_optimized"
    labels_ts_dir = raw_dir / "labelsTs_merged"

    compute_metrics_on_folder2(
        folder_ref=labels_ts_optimized_dir,
        folder_pred=labels_ts_dir,
        dataset_json_file=raw_dir / "dataset.json",
        plans_file=model_results_dir / "plans.json",
        output_file=str(output_file := labels_ts_optimized_dir / "summary_labelsTs.json"),
        chill=True,
    )

    logger.info(f"Created {output_file}")


if __name__ == "__main__":
    main()
