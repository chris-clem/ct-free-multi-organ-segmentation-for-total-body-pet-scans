from collections import defaultdict

import fire
import pandas as pd
from loguru import logger
from tqdm import tqdm

from pet_seg.settings import DATA_CSVS_DIR
from pet_seg.utils import get_sorted_patient_dirs


def main():
    fire.Fire(create_data_csv)


def create_data_csv(
    scanners: str = "Renji_uExplorer_dynamic",
):
    data = defaultdict(list)

    for scanner in scanners.split("-"):
        for patient_dir in tqdm(get_sorted_patient_dirs(scanner)):
            patient_id = patient_dir.name

            ct_path = patient_dir / "CT.nii.gz"
            static_pet_path = patient_dir / "static_PET.nii.gz"
            ts_seg_path = patient_dir / "organ_TS_seg.nii.gz"

            asc_paths = sorted((patient_dir / "ASC").glob("*.nii.gz"))
            nasc_paths = sorted((patient_dir / "NASC").glob("*.nii.gz"))

            assert len(asc_paths) == len(nasc_paths), f"{patient_id} has different number of AC and NAC images"

            if not all([ct_path.exists(), static_pet_path.exists(), ts_seg_path.exists()]):
                logger.warning(f"Skipping {patient_id} because not all files exist")
                continue

            for ac_path, nac_path in zip(asc_paths, nasc_paths):
                data["patient_id"].append(patient_id)
                data["ct"].append(ct_path)
                data["pet_ac"].append(ac_path)
                data["pet_nac"].append(nac_path)
                data["organ_seg"].append(ts_seg_path)
                data["stage"].append("test")

    # Save dataframe
    df = pd.DataFrame(data)

    num_train = (df["stage"] == "train").sum()
    num_test = (df["stage"] == "test").sum()
    file_path = DATA_CSVS_DIR / f"{scanners}-{num_train=}-{num_test=}.csv"
    df.to_csv(file_path, index=False)
    logger.info(f"Created {file_path}.")


if __name__ == "__main__":
    main()
