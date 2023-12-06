import os
from pathlib import Path

import fire
from loguru import logger
from nnunetv2.evaluation.evaluate_predictions import compute_metrics_on_folder2
from tqdm import tqdm

from pet_seg.settings import DATASET_IDS_TO_NAMES, TEST_DATASETS_TO_IDS

NNUNET_RAW_DIR = Path(os.environ["nnUNet_raw"])
NNUNET_RESULTS_DIR = Path(os.environ["nnUNet_results"])


def main():
    fire.Fire(predict_nnunet)


def predict_nnunet(
    model_dataset_id: int = 1,
    config: str = "3d_fullres",
    folds: str = "0 1 2 3 4",
    checkpoint_name: str = "checkpoint_best",
    test_datasets: str = "internal",
    test_images_dir_name: str = "imagesTs",
    compute_metrics_only: bool = False,
):
    """Run nnUNet prediction on the given model and input datasets.

    Args:
        model_dataset_id (int): Model trained on the given dataset to use.
        config (str): nnUNet config to use. Can be "2d" or "3d_cascade_fullres".
        folds (str): Folds to use, separated by spaces.
        checkpoint_name (str): Name of the checkpoint to use. Can be "checkpoint_best" or "checkpoint_latest".
        test_datasets (str): Test datasets to predict on. Can be "test_internal", "test_cross_scanner" or "test_cross_tracer".
        images_dir_name (str): Name of the images directory to use.
        compute_metrics_only (bool): Whether to only compute metrics on existing predictions.
    """
    model_dataset_name = DATASET_IDS_TO_NAMES[model_dataset_id]
    model_results_dir = NNUNET_RESULTS_DIR / model_dataset_name / f"nnUNetTrainerNoMirroring__nnUNetPlans__{config}"

    for test_dataset_id in tqdm(TEST_DATASETS_TO_IDS[test_datasets]):
        test_dataset_name = DATASET_IDS_TO_NAMES[test_dataset_id]
        raw_dir = NNUNET_RAW_DIR / test_dataset_name
        output_dir_name = f"{test_images_dir_name}_{test_dataset_name}"
        output_dir = model_results_dir / f"fold_{folds.replace(' ', '_')}" / "predictions" / output_dir_name

        # Run prediction
        if not compute_metrics_only:
            cmd = (
                "nnUNetv2_predict "
                f"-i {raw_dir / test_images_dir_name} "
                f"-o {output_dir} "
                f"-d {model_dataset_name} "
                "-tr nnUNetTrainerNoMirroring "
                f"-c {config} "
                f"-f {folds} "
                # "--verbose "
                f"-chk {checkpoint_name}.pth "
            )

            os.system(cmd)

        # Compute metrics
        folder_ref_name = test_images_dir_name.replace("images", "labels")

        compute_metrics_on_folder2(
            folder_ref=raw_dir / folder_ref_name,
            folder_pred=output_dir,
            dataset_json_file=raw_dir / "dataset.json",
            plans_file=model_results_dir / "plans.json",
            output_file=str(output_file := output_dir / "summary.json"),
            chill=True,
        )

        logger.info(f"Created {output_file}")


if __name__ == "__main__":
    main()
