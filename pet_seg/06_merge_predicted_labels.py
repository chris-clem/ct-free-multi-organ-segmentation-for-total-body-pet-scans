import os
from pathlib import Path

import fire
import nibabel as nib
from loguru import logger
from tqdm import tqdm

from pet_seg.settings import ANATOMICAL_STRUCTURES_TO_INDEX
from pet_seg.settings import MERGED_ANATOMICAL_STRUCTURES
from pet_seg.settings import MODEL_DATASET_IDS_TO_NAMES

NNUNET_RESULTS_DIR = Path(os.environ["nnUNet_results"])


def main():
    fire.Fire(merge_predicted_labels)


def merge_predicted_labels(
    model_dataset_id: int = 1,
    config: str = "3d_fullres",
    folds: str = "0 1 2 3 4",
    test_dataset: str = "imagesTs_Dataset001_Bern_Quadra-SH_uExplorer-num_train=956-num_test=50_NAC",
):
    # Get dirs
    model_dataset_name = MODEL_DATASET_IDS_TO_NAMES[model_dataset_id]
    model_results_dir = NNUNET_RESULTS_DIR / model_dataset_name
    config_dir = model_results_dir / f"nnUNetTrainerNoMirroring__nnUNetPlans__{config}"
    fold_str = f"fold_{folds.replace(' ', '_')}"
    preds_dir = config_dir / fold_str / "predictions" / test_dataset

    merged_labels_dir = preds_dir / "merged_labels"
    merged_labels_dir.mkdir(parents=True, exist_ok=True)

    for preds_path in tqdm(sorted(preds_dir.glob("*.nii.gz"))):
        # Load predictions
        preds_nii = nib.load(preds_path)
        preds_npy = preds_nii.get_fdata()

        for merged_anatomical_structures in tqdm(sorted(MERGED_ANATOMICAL_STRUCTURES.values()), leave=False):
            labels_to_merge = [
                ANATOMICAL_STRUCTURES_TO_INDEX[anatomical_structure]
                for anatomical_structure in merged_anatomical_structures
            ]

            label_to_use = labels_to_merge[0]

            for label_to_merge in labels_to_merge[1:]:
                preds_npy[preds_npy == label_to_merge] = label_to_use

        # Save merged labels
        merged_labels_path = merged_labels_dir / preds_path.name
        merged_labels_nii = nib.Nifti1Image(preds_npy, preds_nii.affine, preds_nii.header, extra=preds_nii.extra)
        nib.save(merged_labels_nii, merged_labels_path)

    logger.info(f"Saved merged labels to {merged_labels_dir}")


if __name__ == "__main__":
    main()
