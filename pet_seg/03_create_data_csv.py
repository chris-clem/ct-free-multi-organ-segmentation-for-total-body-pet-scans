from collections import defaultdict

import fire
import pandas as pd
from loguru import logger
from tqdm import tqdm

from pet_seg.settings import DATA_CSVS_DIR
from pet_seg.settings import TEST_PATIENT_IDS
from pet_seg.utils import get_sorted_patient_dirs


def main():
    fire.Fire(create_data_csv)


def create_data_csv(
    scanners: str = "Bern_Quadra-SH_uExplorer",
    all_test: bool = False,
):
    """
    Creates a CSV file with the paths to the data for a given scanner.

    Each row in the CSV file corresponds to a patient and contains the paths
    to the CT, PET (AC + NAC), and organ segmentation.

    Args:
        scanners (str): The scanner to create the CSV for. Can be multiple scanners separated by a dash.
        all_test (bool): Whether to use all patients as test patients.
    """
    data = defaultdict(list)

    for scanner in scanners.split("-"):
        for patient_dir in tqdm(get_sorted_patient_dirs(scanner)):
            patient_id = patient_dir.name

            ct_path = patient_dir / "CT.nii.gz"
            ac_path = patient_dir / "AC.nii.gz"
            nac_path = patient_dir / "NAC.nii.gz"
            ts_seg_path = patient_dir / "organ_TS_seg.nii.gz"

            if not all([ct_path.exists(), ac_path.exists(), nac_path.exists(), ts_seg_path.exists()]):
                logger.warning(f"Skipping {patient_id} because not all files exist")
                continue

            data["patient_id"].append(patient_id)
            data["ct"].append(ct_path)
            data["pet_ac"].append(ac_path)
            data["pet_nac"].append(nac_path)
            data["organ_seg"].append(ts_seg_path)

            # Add stage
            if all_test or patient_id.split("_")[-1] in TEST_PATIENT_IDS[scanner]:
                stage = "test"
            else:
                stage = "train"

            data["stage"].append(stage)

    # Save dataframe
    df = pd.DataFrame(data)

    num_train = (df["stage"] == "train").sum()
    num_test = (df["stage"] == "test").sum()
    file_path = DATA_CSVS_DIR / f"{scanners}-{num_train=}-{num_test=}.csv"
    df.to_csv(file_path, index=False)
    logger.info(f"Created {file_path}.")


if __name__ == "__main__":
    main()
