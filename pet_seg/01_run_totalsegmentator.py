import fire
from loguru import logger
from totalsegmentator.python_api import totalsegmentator
from tqdm import tqdm

from pet_seg.settings import DATA_ROOT_DIR
from pet_seg.settings import SCANNER_TO_STAGE


def main():
    fire.Fire(run_totalsegmentator)


def run_totalsegmentator(
    scanner: str,
):
    """Creates CT-based segmentation masks with TotalSegmentator for each patient for a given scanner."""
    scanner_dir = DATA_ROOT_DIR / SCANNER_TO_STAGE[scanner] / scanner

    for patient_dir in tqdm(sorted(scanner_dir.iterdir())):
        patient_id = patient_dir.name

        if (ts_seg_file_path := patient_dir / "CT_TS_seg.nii.gz").exists():
            logger.info(f"Skipping {patient_id} because {ts_seg_file_path.name} already exists")
            continue

        logger.info(f"Running TotalSegmentator for {patient_id}")

        try:
            totalsegmentator(
                input=patient_dir / "CT.nii.gz",
                output=ts_seg_file_path,
                ml=True,
                nr_thr_resamp=4,
                nr_thr_saving=4,
            )
        except Exception as e:
            logger.error(f"TotalSegmentator failed for {patient_id}: {e}")

        break


if __name__ == "__main__":
    main()
