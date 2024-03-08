import fire
import nibabel as nib
import numpy as np
from joblib import delayed
from joblib import Parallel
from tqdm import tqdm

from pet_seg.settings import ANATOMICAL_STRUCTURES_TO_INDEX
from pet_seg.settings import INDEX_TO_ANATOMICAL_STRUCTURES
from pet_seg.settings import MERGED_ANATOMICAL_STRUCTURES
from pet_seg.settings import MERGED_ANATOMICAL_STRUCTURES_TO_INDEX
from pet_seg.utils import get_sorted_patient_dirs


def main():
    fire.Fire(merge_ts_seg_labels)


def merge_ts_seg_labels(
    scanner: str = "Bern_Quadra",
    n_jobs: int = 1,
):
    Parallel(n_jobs=n_jobs)(
        delayed(merge_one_ts_seg)(patient_dir / "organ_TS_seg.nii.gz")
        for patient_dir in tqdm(get_sorted_patient_dirs(scanner))
    )


def merge_one_ts_seg(ts_seg_path):
    # Load
    try:
        ts_seg_nii = nib.load(ts_seg_path)
    except FileNotFoundError:
        print(f"File not found: {ts_seg_path}")
        return

    ts_seg_npy = ts_seg_nii.get_fdata()

    ts_seg_merged_npy = np.zeros_like(ts_seg_npy, dtype=np.uint8)

    missing_indices = set(sorted(INDEX_TO_ANATOMICAL_STRUCTURES.keys()))

    # Merge
    for merged_structure, structures_to_merge in sorted(MERGED_ANATOMICAL_STRUCTURES.items()):
        indices_to_merge = [ANATOMICAL_STRUCTURES_TO_INDEX[structure] for structure in structures_to_merge]

        # Remove labels_to_merge from all_label_indices
        missing_indices -= set(indices_to_merge)

        idx_to_use = MERGED_ANATOMICAL_STRUCTURES_TO_INDEX[merged_structure]

        mask = np.isin(ts_seg_npy, indices_to_merge[1:])
        ts_seg_merged_npy[mask] = idx_to_use

    # Add the remaining structures
    for idx in sorted(missing_indices):
        missing_anatomical_structure = INDEX_TO_ANATOMICAL_STRUCTURES[idx]
        idx_to_use = MERGED_ANATOMICAL_STRUCTURES_TO_INDEX[missing_anatomical_structure]
        ts_seg_merged_npy[ts_seg_npy == idx] = idx_to_use

    # Save
    ts_seg_merged_path = ts_seg_path.parent / f"{ts_seg_path.name.split('.')[0]}_merged.nii.gz"
    ts_seg_merged_nii = nib.Nifti1Image(ts_seg_merged_npy, ts_seg_nii.affine, ts_seg_nii.header, extra=ts_seg_nii.extra)
    nib.save(ts_seg_merged_nii, ts_seg_merged_path)


if __name__ == "__main__":
    main()
