from pathlib import Path

import fire

from pet_seg.settings import RESULTS_DIR
from pet_seg.utils import create_patient_metrics_df


def main():
    fire.Fire(extract_ts_vs_optimized_results)


def extract_ts_vs_optimized_results(
    ts_vs_optimized_summary_path: str,
):
    for metric in ["Dice", "IoU"]:
        patient_dice_scores_df = create_patient_metrics_df(
            Path(ts_vs_optimized_summary_path),
            use_merged_seg="merged" in ts_vs_optimized_summary_path,
            use_optimized_seg=True,
            metric=metric,
        )
        patient_dice_scores_df.to_csv(
            RESULTS_DIR / "patient_metrics" / f"ts_vs_optimized_labels__{metric}.csv", index=False
        )


if __name__ == "__main__":
    main()
