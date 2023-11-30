from pathlib import Path

from pet_seg.settings import DATA_ROOT_DIR
from pet_seg.settings import SCANNER_TO_STAGE


def get_scanner_dir(scanner: str) -> Path:
    return DATA_ROOT_DIR / SCANNER_TO_STAGE[scanner] / scanner


def get_sorted_patient_dirs(scanner: str) -> list[Path]:
    return sorted(get_scanner_dir(scanner).iterdir())
