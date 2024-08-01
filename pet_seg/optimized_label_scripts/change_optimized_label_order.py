from pathlib import Path

import fire
import nibabel as nib
import numpy as np
from joblib import delayed
from joblib import Parallel
from tqdm import tqdm

OPTIMIZED_LABELS_TO_ORGANS = {
    0: "background",
    # 1: "Adrenal-glands",
    2: "Aorta",
    3: "Bladder",
    4: "Brain",
    5: "Heart",
    6: "Kidneys",
    7: "Liver",
    8: "Pancreas",
    9: "Spleen",
    10: "Thyroid",
    # 11: "Inferior-vena-cava",
    12: "Lung",
}

OPTIMIZED_TO_TS_LABEL_MAPPING = {
    0: 0,
    # 1: 8,
    2: 52,
    3: 21,
    4: 90,
    5: 51,
    6: 2,
    7: 5,
    8: 7,
    9: 1,
    10: 17,
    # 11: 63,
    12: 10,
}

OPTIMIZED_TO_TS_MERGED_LABEL_MAPPING = {
    0: 0,
    # 1: 7,
    2: 21,
    3: 15,
    4: 41,
    5: 20,
    6: 2,
    7: 4,
    8: 6,
    9: 1,
    10: 11,
    # 11: 28,
    12: 8,
}

TS_TO_OPTIMIZED_LABEL_MAPPING = {v: k for k, v in OPTIMIZED_TO_TS_MERGED_LABEL_MAPPING.items()}


def main():
    fire.Fire(change_optimized_label_order)


def change_optimized_label_order(
    optimized_label_dir: str,
    n_jobs: int = 1,
):
    optimized_label_dir = Path(optimized_label_dir)

    optimized_label_dir_changed_order = optimized_label_dir.parent / f"{optimized_label_dir.name}_changed_order"
    optimized_label_dir_changed_order.mkdir(parents=True, exist_ok=True)

    Parallel(n_jobs=n_jobs)(
        delayed(process_one)(optimized_label_path, optimized_label_dir_changed_order)
        for optimized_label_path in tqdm(sorted(optimized_label_dir.glob("*.nii.gz")))
    )


def process_one(optimized_label_path, optimized_label_dir_changed_order):
    optimized_label_nii = nib.load(optimized_label_path)
    optimized_label_npy = optimized_label_nii.get_fdata()

    # print(f"unique labels: {np.unique(optimized_label_npy)}")

    optimized_label_ts_npy = np.zeros_like(optimized_label_npy)
    for optimized_label, ts_label in OPTIMIZED_TO_TS_MERGED_LABEL_MAPPING.items():
        optimized_label_ts_npy[optimized_label_npy == optimized_label] = ts_label

    # print(f"unique labels: {np.unique(optimized_label_ts_npy)}")

    optimized_label_ts_path = optimized_label_dir_changed_order / optimized_label_path.name
    optimized_label_ts_nii = nib.Nifti1Image(optimized_label_ts_npy, optimized_label_nii.affine)
    nib.save(optimized_label_ts_nii, optimized_label_ts_path)


if __name__ == "__main__":
    main()
