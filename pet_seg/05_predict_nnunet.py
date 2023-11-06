import os
from pathlib import Path

import fire
from loguru import logger
from nnunetv2.evaluation.evaluate_predictions import compute_metrics_on_folder2
from tqdm import tqdm

from pet_seg.settings import DATASET_IDS_TO_NAMES

NNUNET_RAW_DIR = Path(os.environ["nnUNet_raw"])
NNUNET_RESULTS_DIR = Path(os.environ["nnUNet_results"])


def main():
    fire.Fire(predict_nnunet)


def predict_nnunet(
    model_dataset_id: str = "3",
    input_dataset_ids: str = "4_5_6_7_8_9_10",
    config: str = "3d_cascade_fullres",
    folds: str = "0 1 2 3 4",
    checkpoint_name: str = "checkpoint_best",
    compute_metrics_only: bool = False,
):
    """Run nnUNet prediction on the given model and input datasets.

    Args:
        model_dataset_id (str): Models trained on the given datasets to use.
        input_dataset_ids (str): Datasets to use as input to the models, separated by underscores.
        config (str): nnUNet config to use. Can be "2d" or "3d_cascade_fullres".
        folds (str): Folds to use, separated by spaces.
        checkpoint_name (str): Name of the checkpoint to use. Can be "checkpoint_best" or "checkpoint_latest".
        compute_metrics_only (bool): Whether to only compute metrics on existing predictions.
    """
    for input_dataset_id in tqdm(input_dataset_ids.split("_")):
        # Get names from ids
        model_dataset_name = DATASET_IDS_TO_NAMES[int(model_dataset_id)]
        input_dataset_name = DATASET_IDS_TO_NAMES[int(input_dataset_id)]

        # Get dirs
        raw_dir = NNUNET_RAW_DIR / input_dataset_name
        results_dir = NNUNET_RESULTS_DIR / model_dataset_name / f"nnUNetTrainer__nnUNetPlans__{config}"

        # Set up output dir
        images_dir_name = "imagesTs"
        dir_name = f"{images_dir_name}_{input_dataset_name}"

        if len(folds_splitted := str(folds).split(" ")) == 1:
            output_dir = results_dir / f"fold_{folds}" / "predictions" / dir_name
        else:
            output_dir = results_dir / f"folds_{'_'.join(folds_splitted)}" / "predictions" / dir_name

        # Run prediction
        if not compute_metrics_only:
            cmd = (
                "nnUNetv2_predict "
                f"-i {raw_dir / images_dir_name} "
                f"-o {output_dir} "
                f"-d {model_dataset_name} "
                f"-c {config} "
                f"-f {folds} "
                f"-chk {checkpoint_name}.pth"
            )

            os.system(cmd)

        # Compute metrics
        folder_ref_name = "labelsTs"
        output_file_name = "summary.json"

        compute_metrics_on_folder2(
            folder_ref=raw_dir / folder_ref_name,
            folder_pred=output_dir,
            dataset_json_file=raw_dir / "dataset.json",
            plans_file=results_dir / "plans.json",
            output_file=str(output_dir / output_file_name),
            chill=False,
        )

        logger.info(f"Created {output_dir / output_file_name}")


if __name__ == "__main__":
    main()
