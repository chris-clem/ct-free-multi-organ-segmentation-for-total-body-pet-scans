# flake8: noqa
import os
import tempfile
from pathlib import Path

import fire
import nibabel as nib
import numpy as np
from loguru import logger
from nnunetv2.evaluation.evaluate_predictions import compute_metrics_on_folder2
from tqdm import tqdm

from pet_seg.settings import MODEL_DATASET_IDS_TO_NAMES
from pet_seg.settings import TEST_DATASET_IDS_TO_NAMES
from pet_seg.settings import TEST_DATASETS_TO_IDS

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
    split_images: bool = False,
    compute_metrics_only: bool = False,
):
    """Run nnUNet prediction on the given model and input datasets.

    Args:
        model_dataset_id (int): Model trained on the given dataset to use.
        config (str): nnUNet config to use. Can be "2d" or "3d_cascade_fullres".
        folds (str): Folds to use, separated by spaces.
        checkpoint_name (str): Name of the checkpoint to use. Can be "checkpoint_best" or "checkpoint_latest".
        test_datasets (str): Test datasets to predict on. Can be "internal", "cross_scanner" or "cross_tracer".
        images_dir_name (str): Name of the images directory to use.
        compute_metrics_only (bool): Whether to only compute metrics on existing predictions.
    """
    model_dataset_name = MODEL_DATASET_IDS_TO_NAMES[model_dataset_id]
    model_results_dir = NNUNET_RESULTS_DIR / model_dataset_name / f"nnUNetTrainerNoMirroring__nnUNetPlans__{config}"

    for test_dataset_id in tqdm(TEST_DATASETS_TO_IDS[test_datasets]):
        test_dataset_name = TEST_DATASET_IDS_TO_NAMES[test_dataset_id]
        raw_dir = NNUNET_RAW_DIR / test_dataset_name
        images_dir = raw_dir / test_images_dir_name

        output_dir_name = f"{test_images_dir_name}_{test_dataset_name}"
        output_dir = model_results_dir / f"fold_{folds.replace(' ', '_')}" / "predictions" / output_dir_name
        output_dir.mkdir(parents=True, exist_ok=True)

        if not compute_metrics_only:
            for image_path in tqdm(sorted(images_dir.glob("*.nii.gz"))):
                image_name = image_path.name

                pred_path = output_dir / image_name.replace("_0000", "")

                if pred_path.exists():
                    logger.debug(f"Prediction already exists for {image_name}")
                    continue

                with tempfile.TemporaryDirectory() as tmp_dir:
                    tmp_dir = Path(tmp_dir)

                    if split_images:
                        logger.debug(f"Splitting {image_name}")

                        image_nifti = nib.load(image_path)

                        # From https://github.com/wasserth/TotalSegmentator/blob/master/totalsegmentator/nnunet.py#L333C1-L342C49
                        third = image_nifti.shape[2] // 3
                        margin = 20
                        image_data = image_nifti.get_fdata()
                        nib.save(
                            nib.Nifti1Image(image_data[:, :, : third + margin], image_nifti.affine),
                            tmp_dir / "s01_0000.nii.gz",
                        )
                        nib.save(
                            nib.Nifti1Image(
                                image_data[:, :, third + 1 - margin : third * 2 + margin], image_nifti.affine
                            ),
                            tmp_dir / "s02_0000.nii.gz",
                        )
                        nib.save(
                            nib.Nifti1Image(image_data[:, :, third * 2 + 1 - margin :], image_nifti.affine),
                            tmp_dir / "s03_0000.nii.gz",
                        )
                    else:
                        # Create symlink to image
                        image_tmp_path = tmp_dir / image_name
                        image_tmp_path.symlink_to(image_path)

                    # Run prediction
                    logger.debug(f"Predicting {image_name}")
                    cmd = (
                        "nnUNetv2_predict "
                        f"-i {tmp_dir} "
                        f"-o {tmp_dir} "
                        f"-d {model_dataset_name} "
                        "-tr nnUNetTrainerNoMirroring "
                        f"-c {config} "
                        f"-f {folds} "
                        f"-chk {checkpoint_name}.pth "
                        f"-npp 1 "
                        f"-nps 1 "
                    )

                    os.system(cmd)

                    pred_tmp_path = tmp_dir / pred_path.name

                    if split_images:
                        logger.debug(f"Combining {image_name}")
                        combined_img = np.zeros(image_nifti.shape, dtype=np.uint8)
                        combined_img[:, :, :third] = nib.load(tmp_dir / "s01.nii.gz").get_fdata()[:, :, :-margin]
                        combined_img[:, :, third : third * 2] = nib.load(tmp_dir / "s02.nii.gz").get_fdata()[
                            :, :, margin - 1 : -margin
                        ]
                        combined_img[:, :, third * 2 :] = nib.load(tmp_dir / "s03.nii.gz").get_fdata()[
                            :, :, margin - 1 :
                        ]
                        nib.save(nib.Nifti1Image(combined_img, image_nifti.affine), pred_tmp_path)

                    pred_tmp_path.rename(pred_path)

                    logger.debug(f"Created {pred_path}")

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
