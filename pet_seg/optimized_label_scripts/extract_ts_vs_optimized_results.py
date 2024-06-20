from pathlib import Path

import fire

from pet_seg.settings import RESULTS_DIR
from pet_seg.utils import create_patient_dice_scores_df


def main():
    fire.Fire(extract_ts_vs_optimized_results)


def extract_ts_vs_optimized_results(
    ts_vs_optimized_summary_path: str = "/home/user/Data/ct-free-multi-organ-segmentation-for-total-body-pet-scans/nnunetv2/raw/Dataset001_Bern_Quadra-SH_uExplorer-num_train=956-num_test=50_NAC/labelsTs_optimized/summary_labelsTs.json",  # noqa: E501
):
    patient_dice_scores_df = create_patient_dice_scores_df(Path(ts_vs_optimized_summary_path))
    patient_dice_scores_df.to_csv(RESULTS_DIR / "patient_dice_scores" / "ts_vs_optimized_labels.csv", index=False)


if __name__ == "__main__":
    main()
