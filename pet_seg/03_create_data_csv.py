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

            pet_ac_path = patient_dir / "AC.nii.gz"  # e.g. array[440, 440, 644]  x∈[0., 1.267e+05]
            pet_nac_path = patient_dir / "NAC.nii.gz"  # array[440, 440, 644] x∈[0., 9.649e+03]
            seg_moose_path = patient_dir / "organ_seg.nii.gz"  # array[440, 440, 644] x∈[0., 12.000]
            seg_moose_optimized_path = patient_dir / "optimized_seg.nii.gz"  # array[440, 440, 644] x∈[0., 12.000]
            seg_ts_path = patient_dir / "organ_TS_seg.nii.gz"  # array[440, 440, 644] x∈[0., 117.000]
            seg_ts_merged_path = patient_dir / "organ_TS_seg_merged.nii.gz"  # array[440, 440, 644] x∈[0., 45.000]

            if not all(
                [
                    pet_ac_path.exists(),
                    pet_nac_path.exists(),
                    seg_moose_path.exists(),
                    seg_ts_path.exists(),
                    seg_ts_merged_path.exists(),
                ]
            ):
                logger.warning(f"Skipping {patient_id} because not all files exist")
                continue

            data["patient_id"].append(patient_id)
            data["pet_ac"].append(pet_ac_path)
            data["pet_nac"].append(pet_nac_path)
            data["seg_moose"].append(seg_moose_path)
            data["seg_ts"].append(seg_ts_path)
            data["seg_ts_merged"].append(seg_ts_merged_path)

            # Add stage
            if all_test or patient_id.split("_")[-1] in TEST_PATIENT_IDS[scanner]:
                data["stage"].append("test")
                if seg_moose_optimized_path.exists():
                    data["seg_moose_optimized"].append(seg_moose_optimized_path)
                else:
                    logger.warning(f"Optimized segmentation not found for {patient_id}")
                    data["seg_moose_optimized"].append(None)
            else:
                data["stage"].append("train")
                data["seg_moose_optimized"].append(None)

    # Save dataframe
    df = pd.DataFrame(data)

    num_train = (df["stage"] == "train").sum()
    num_test = (df["stage"] == "test").sum()
    file_path = DATA_CSVS_DIR / f"{scanners}-{num_train=}-{num_test=}.csv"
    df.to_csv(file_path, index=False)
    logger.info(f"Created {file_path}.")


if __name__ == "__main__":
    main()
