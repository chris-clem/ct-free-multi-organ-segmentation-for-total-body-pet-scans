import os
from pathlib import Path

import fire
import nibabel as nib
import numpy as np
from joblib import delayed
from joblib import Parallel
from loguru import logger
from tqdm import tqdm

from pet_seg.settings import ANATOMICAL_STRUCTURES_TO_INDEX
from pet_seg.settings import MERGED_ANATOMICAL_STRUCTURES

NNUNET_RESULTS_DIR = Path(os.environ["nnUNet_results"])


def main():
    fire.Fire(merge_labels)


def merge_labels(
    labels_to_merge_dir: str,
    n_jobs: int = 1,
):
    labels_to_merge_dir = Path(labels_to_merge_dir)

    labels_merged_dir = labels_to_merge_dir.parent / f"{labels_to_merge_dir.name}_merged"
    labels_merged_dir.mkdir(parents=True, exist_ok=True)

    Parallel(n_jobs=n_jobs)(
        delayed(merge_one_label)(label_path, labels_merged_dir)
        for label_path in tqdm(sorted(labels_to_merge_dir.glob("*.nii.gz")))
    )

    logger.info(f"Saved merged labels to {labels_merged_dir}")


def merge_one_label(label_path, labels_merged_dir):
    # Load predictions
    preds_nii = nib.load(label_path)
    preds_npy = preds_nii.get_fdata()

    # Merge
    for merged_anatomical_structures in tqdm(sorted(MERGED_ANATOMICAL_STRUCTURES.values()), leave=False):
        labels_to_merge = [
            ANATOMICAL_STRUCTURES_TO_INDEX[anatomical_structure]
            for anatomical_structure in merged_anatomical_structures
        ]

        label_to_use = labels_to_merge[0]

        mask = np.isin(preds_npy, labels_to_merge[1:])
        preds_npy[mask] = label_to_use

    preds_npy = preds_npy.astype(int)

    # Save merged labels
    merged_labels_path = labels_merged_dir / label_path.name
    merged_labels_nii = nib.Nifti1Image(preds_npy, preds_nii.affine, preds_nii.header, extra=preds_nii.extra)
    nib.save(merged_labels_nii, merged_labels_path)


if __name__ == "__main__":
    main()
