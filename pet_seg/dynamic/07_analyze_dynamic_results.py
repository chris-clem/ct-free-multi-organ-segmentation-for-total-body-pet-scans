from collections import defaultdict
from pathlib import Path

import fire
import nibabel as nib
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm

from pet_seg.utils import get_sorted_patient_dirs

MERGED_ANATOMICAL_STRUCTURES = {
    "kidneys": [2, 3],
    "lungs": [10, 11, 12, 13, 14],
}

SELECTED_ANATOMICAL_STRUCTURES = {
    1: "spleen",
    2: "kidneys",
    5: "liver",
    7: "pancreas",
    10: "lungs",
    51: "heart",
    52: "aorta",
    90: "brain",
}


FRAME_TIMES_S = [5] * 24 + [30] * 20 + [60] * 48
CUM_FRAME_TIMES_S = list(np.cumsum(FRAME_TIMES_S))
OPTIMIZED_SEG_FRAME_NUMS = [15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 92]


def main():
    fire.Fire(analyze_dynamic_results)


def analyze_dynamic_results(
    scanner: str = "SH_uExplorer_Renji_dynamic",
    image_name: str = "F92_NASC",
    dynamic_pred_dir: str = "/home/user/Data/ct-free-multi-organ-segmentation-for-total-body-pet-scans/nnunetv2/results/Dataset002_SH_uExplorer_Renji-num_train=900-num_test=99_NAC/nnUNetTrainerNoMirroring__nnUNetPlans__3d_fullres_resenc/fold_all/predictions/imagesTs_Dataset092_SH_uExplorer_Renji_dynamic-num_dynamic_frames=92-num_train=0-num_test=1840_NAC",  # noqa
    first_dynamic_frame_num: int = 1,
    n_jobs: int = 1,
):
    dynamic_pred_dir = Path(dynamic_pred_dir)

    all_data = []
    for patient_dir in tqdm(get_sorted_patient_dirs(scanner), desc="Patients"):
        patient_id = patient_dir.name

        if patient_id in ["XUE_MEI_FANG_PET082008_094155"]:
            continue

        image_paths = sorted((patient_dir / image_name).glob("*.nii.gz"))
        patient_data = Parallel(n_jobs=n_jobs)(
            delayed(process_frame)(image_path, image_name, dynamic_pred_dir, patient_id, first_dynamic_frame_num)
            for image_path in tqdm(image_paths, leave=False, desc="Frames")
        )

        all_data.extend(patient_data)

    dfs = [pd.DataFrame(frame_data) for frame_data in all_data]
    data_df = pd.concat(dfs, ignore_index=True)

    print(data_df.describe())

    data_csv_name = f"dynamic_results_{image_name}_{first_dynamic_frame_num}"

    data_df.to_csv(f"{data_csv_name}.csv", index=False)


def process_frame(image_path, image_name, dynamic_pred_dir, patient_id, first_dynamic_frame_num):
    frame_num = int(image_path.name.split(".")[0].replace(f"{image_name}_", ""))
    pred_frame_num = frame_num if frame_num > first_dynamic_frame_num else first_dynamic_frame_num

    # Take closest frame_num from OPTIMIZED_SEG_FRAME_NUMS
    # optimized_seg_frame_num = min(OPTIMIZED_SEG_FRAME_NUMS, key=lambda x: abs(x - frame_num))

    patient_dir = image_path.parent.parent
    TS_pred_path = patient_dir / "ts_seg.nii.gz"
    static_pred_path = dynamic_pred_dir / f"{patient_id}.nii.gz"
    dynamic_pred_path = next(iter(dynamic_pred_dir.glob(f"{patient_id}*{pred_frame_num:04d}.nii.gz")))
    # dynamic_pred_path = next(iter(dynamic_pred_dir.glob(f"{patient_id}*.nii.gz")))

    # Get optimized seg path
    # optimized_segs_dir = patient_dir.parent.parent / "Renji_uExplorer_dynamic_optimized_seg"
    # optimized_dynamic_patient_segs_dir = optimized_segs_dir / patient_id / "dynamic_seg_optimized"
    # optimized_dynamic_seg_path = next(
    #     iter(optimized_dynamic_patient_segs_dir.glob(f"{optimized_seg_frame_num:02d}*.nii.gz"))
    # )

    _, pet_npy = load_nifti(image_path)
    _, TS_pred_npy = load_nifti(TS_pred_path)
    _, static_pred_npy = load_nifti(static_pred_path)
    _, dynamic_pred_npy = load_nifti(dynamic_pred_path)
    # _, optimized_dynamic_seg_npy = load_nifti(optimized_dynamic_seg_path)

    # Merge anatomical structures
    for indices_to_merge in MERGED_ANATOMICAL_STRUCTURES.values():
        merged_idx = indices_to_merge[0]

        for idx in indices_to_merge:
            TS_pred_npy[TS_pred_npy == idx] = merged_idx
            static_pred_npy[static_pred_npy == idx] = merged_idx
            dynamic_pred_npy[dynamic_pred_npy == idx] = merged_idx
            # optimized_dynamic_seg_npy[optimized_dynamic_seg_npy == idx] = merged_idx

    frame_data = defaultdict(list)
    for idx, anatomical_structure in SELECTED_ANATOMICAL_STRUCTURES.items():
        if idx == 0:
            continue

        TS_pred_organ = TS_pred_npy == idx
        static_pred_organ = static_pred_npy == idx
        dynamic_pred_organ = dynamic_pred_npy == idx
        # optimized_dynamic_pred_organ = optimized_dynamic_seg_npy == idx

        TS_pred_organ_size = TS_pred_organ.sum()
        static_pred_organ_size = static_pred_organ.sum()
        dynamic_pred_organ_size = dynamic_pred_organ.sum()
        # optimized_dynamic_pred_organ_size = optimized_dynamic_pred_organ.sum()

        TS_pred_organ_activity = (TS_pred_organ * pet_npy).sum()
        static_pred_organ_activity = (static_pred_organ * pet_npy).sum()
        dynamic_pred_organ_activity = (dynamic_pred_organ * pet_npy).sum()
        # optimized_dynamic_pred_organ_activity = (optimized_dynamic_pred_organ * pet_npy).sum()

        # Compute foreground dice
        # dynamic_pred_organ_dice = (
        #     (dynamic_pred_organ * optimized_dynamic_pred_organ).sum()
        #     * 2
        #     / (dynamic_pred_organ_size + optimized_dynamic_pred_organ_size)
        # )

        # TS_pred_organ_dice = (
        #     (TS_pred_organ * optimized_dynamic_pred_organ).sum()
        #     * 2
        #     / (TS_pred_organ_size + optimized_dynamic_pred_organ_size)
        # )

        frame_data["patient_id"].append(patient_id)
        frame_data["frame_num"].append(frame_num)
        frame_data["anatomical_structure"].append(anatomical_structure)
        frame_data["TS_pred_organ_size"].append(TS_pred_organ_size)
        frame_data["TS_pred_organ_activity"].append(TS_pred_organ_activity)
        frame_data["static_pred_organ_size"].append(static_pred_organ_size)
        frame_data["static_pred_organ_activity"].append(static_pred_organ_activity)
        frame_data["dynamic_pred_organ_size"].append(dynamic_pred_organ_size)
        frame_data["dynamic_pred_organ_activity"].append(dynamic_pred_organ_activity)
        # frame_data["optimized_dynamic_pred_organ_size"].append(optimized_dynamic_pred_organ_size)
        # frame_data["optimized_dynamic_pred_organ_activity"].append(optimized_dynamic_pred_organ_activity)
        # frame_data["dynamic_pred_organ_dice"].append(dynamic_pred_organ_dice)
        # frame_data["TS_pred_organ_dice"].append(TS_pred_organ_dice)

    return frame_data


def load_nifti(image_path):
    image_nii = nib.load(image_path)
    image_npy = image_nii.get_fdata()
    return image_nii, image_npy


if __name__ == "__main__":
    main()
