import shutil

import fire
from loguru import logger
from tqdm import tqdm

from pet_seg.settings import DATA_ROOT_DIR
from pet_seg.settings import SCANNER_TO_STAGE


def main():
    fire.Fire(prepare_moose_to_run_data)


def prepare_moose_to_run_data(
    scanner: str = "Bern_Quadra",
):
    """Creates a `MOOSE-to_run` dir in each scanner dir containing only the CT data for each patient.

    Args:
        scanner (str): Scanner to prepare data for, e.g. Bern_Quadra.
    """

    # Prepare dirs
    scanner_dir = DATA_ROOT_DIR / SCANNER_TO_STAGE[scanner] / scanner

    raw_organized_data_dir = scanner_dir / "raw-organized"

    moose_to_run_dir = scanner_dir / "MOOSE-to_run"
    moose_to_run_dir.mkdir(parents=True, exist_ok=True)

    # Copy CT data
    for ct_dir in tqdm(sorted(raw_organized_data_dir.glob("*/*CT*"))):
        patient_id = ct_dir.parent.name

        moose_to_run_patient_dir = moose_to_run_dir / patient_id
        moose_to_run_patient_dir.mkdir(parents=True, exist_ok=True)

        moose_to_run_patient_ct_dir = moose_to_run_patient_dir / "CT"
        shutil.copytree(ct_dir, moose_to_run_patient_ct_dir, symlinks=True)

    logger.info(f"Created {len(list(moose_to_run_dir.iterdir()))} patients in {moose_to_run_dir}")


if __name__ == "__main__":
    main()
