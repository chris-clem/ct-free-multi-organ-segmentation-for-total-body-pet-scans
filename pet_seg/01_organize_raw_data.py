import fire
from loguru import logger

from pet_seg.settings import DATA_ROOT_DIR
from pet_seg.settings import RAW_PATIENTS_DIRS


def main():
    fire.Fire(organize_raw_data)


def organize_raw_data():
    """Organize original scanner data into a single directory with symlinks to the original data.

    See README.md for more information about the required directory structure.
    """

    for scanner in RAW_PATIENTS_DIRS:
        logger.info(f"Organizing raw data for {scanner}")

        raw_organized_data_dir = DATA_ROOT_DIR / "train" / scanner / "raw-organized"

        for patients_dir in RAW_PATIENTS_DIRS[scanner]:
            for patient_dir in patients_dir.iterdir():
                patient_id = patient_dir.name
                patient_symlink = raw_organized_data_dir / patient_id

                try:
                    patient_symlink.symlink_to(patient_dir)
                except FileExistsError:
                    logger.warning(f"Symlink already exists for {patient_symlink}")

        logger.info(f"Created {len(list(raw_organized_data_dir.iterdir()))} symlinks in {raw_organized_data_dir}")


if __name__ == "__main__":
    main()
