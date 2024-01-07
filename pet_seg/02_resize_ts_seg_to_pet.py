import fire
import nibabel as nib
import numpy as np
import skimage.transform as skitran
from tqdm import tqdm

from pet_seg.settings import DATA_ROOT_DIR
from pet_seg.utils import get_sorted_patient_dirs


def main():
    fire.Fire(resize_ts_seg_to_pet)


def resize_ts_seg_to_pet(
    scanner: str,
):
    ts_output_dir = DATA_ROOT_DIR / "totalsegmentator" / scanner

    for patient_dir in tqdm(get_sorted_patient_dirs(scanner)):
        patient_id = patient_dir.name

        nac_path = patient_dir / "NAC.nii.gz"
        ts_seg_path = ts_output_dir / f"{patient_id}.nii.gz"

        try:
            nac_nifti = nib.load(nac_path)
            seg_nifti = nib.load(ts_seg_path)
        except Exception as e:
            print(e)
            continue

        seg_npy = np.array(seg_nifti.dataobj)

        target_shape = (
            seg_npy.shape[0] // (nac_nifti.header.get_zooms()[0] / seg_nifti.header.get_zooms()[0]),
            seg_npy.shape[1] // (nac_nifti.header.get_zooms()[1] / seg_nifti.header.get_zooms()[1]),
            nac_nifti.shape[2],
        )

        seg_npy = skitran.resize(
            seg_npy,
            target_shape,
            mode="edge",
            anti_aliasing=False,
            anti_aliasing_sigma=None,
            preserve_range=True,
            order=0,
        )

        seg_npy = np.around(seg_npy)
        seg_npy = seg_npy.astype("uint8")

        seg_npy = cut_or_pad(nac_nifti, seg_npy)

        seg_nifti = nib.Nifti1Image(seg_npy, nac_nifti.affine)
        nib.save(seg_nifti, patient_dir / "organ_TS_seg.nii.gz")


def cut_or_pad(nac_nifti, seg_npy):
    if seg_npy.shape[0] > nac_nifti.shape[0]:
        cut_num = int((seg_npy.shape[0] - nac_nifti.shape[0]) / 2)
        seg_npy = seg_npy[
            cut_num : -(seg_npy.shape[0] - nac_nifti.shape[0] - cut_num),  # noqa E203
            cut_num : -(seg_npy.shape[0] - nac_nifti.shape[0] - cut_num),  # noqa E203
            :,
        ]
    elif seg_npy.shape[0] < nac_nifti.shape[0]:
        pad_num = int((nac_nifti.shape[0] - seg_npy.shape[0]) / 2)
        seg_npy = np.pad(
            seg_npy,
            (
                (pad_num, (nac_nifti.shape[0] - seg_npy.shape[0]) - pad_num),
                (pad_num, (nac_nifti.shape[0] - seg_npy.shape[0]) - pad_num),
                (0, 0),
            ),
            "constant",
            constant_values=(0, 0),
        )

    return seg_npy


if __name__ == "__main__":
    main()
