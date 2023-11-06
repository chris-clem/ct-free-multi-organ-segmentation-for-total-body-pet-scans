from collections import defaultdict

import fire
import pandas as pd
from loguru import logger

from pet_seg.settings import DATA_CSVS_DIR
from pet_seg.settings import DATA_ROOT_DIR
from pet_seg.settings import SCANNER_TO_STAGE
from pet_seg.settings import TEST_PATIENT_IDS


def main():
    fire.Fire(create_data_csv)


def create_data_csv(
    scanners: str = "Bern_Quadra,SH_uExplorer",
    all_test: bool = False,
):
    """
    Creates a CSV file with the paths to the data for a given scanner.

    Each row in the CSV file corresponds to a patient and contains the paths
    to the CT, PET (AC + NAC), and organ segmentation.

    Args:
        scanners (str): The scanner to create the CSV for. Can be multiple scanners separated by commas.
        all_test (bool): Whether to use all patients as test patients.
    """
    data = defaultdict(list)

    for scanner in scanners.split(","):
        # Prepare dirs
        scanner_dir = DATA_ROOT_DIR / SCANNER_TO_STAGE[scanner] / scanner

        raw_organized_data_dir = scanner_dir / "raw-organized"

        patient_dirs = sorted(raw_organized_data_dir.iterdir())
        logger.info(f"Found {len(patient_dirs)} patients in {raw_organized_data_dir}")

        # Iterate over all patients and get CT, PET, and organ segmentation paths
        for patient_dir in patient_dirs:
            patient_id = patient_dir.name

            ct = patient_dir / "CT.nii.gz"
            pet_ac = patient_dir / "AC.nii.gz"
            pet_nac = patient_dir / "NAC.nii.gz"
            organ_seg = patient_dir / "organ_seg.nii.gz"

            data["patient_id"].append(patient_id)
            data["ct"].append(ct)
            data["pet_ac"].append(pet_ac)
            data["pet_nac"].append(pet_nac)
            data["organ_seg"].append(organ_seg)

            # Add stage
            if all_test or patient_id.split("_")[-1] in TEST_PATIENT_IDS[scanner]:
                stage = "test"
            else:
                stage = "train"

            data["stage"].append(stage)

    # Create dataframe
    df = pd.DataFrame(data)

    # Save dataframe
    num_train = (df["stage"] == "train").sum()
    num_test = (df["stage"] == "test").sum()
    file_path = DATA_CSVS_DIR / f"{scanners}-{num_train=}-{num_test=}.csv"
    df.to_csv(file_path, index=False)
    logger.info(f"Created {file_path}.")


if __name__ == "__main__":
    main()
