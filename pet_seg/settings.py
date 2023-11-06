from pathlib import Path

DATA_ROOT_DIR = Path.home() / "Data" / "total-body-pet-segmentation"

BERN_QUADRA_RAW_DIR = DATA_ROOT_DIR / "train" / "Bern_Quadra" / "raw"
SH_UEXPLORER_RAW_DIR = DATA_ROOT_DIR / "train" / "SH_uExplorer" / "raw"

RAW_PATIENTS_DIRS = {
    "Bern_Quadra": [
        BERN_QUADRA_RAW_DIR / "NAC_only" / "Dose_red_0805_NAC_38_dicom",
        BERN_QUADRA_RAW_DIR / "NAC_only" / "Quadra_0718_NAC_92_dicom",
        BERN_QUADRA_RAW_DIR / "NAC_only" / "Quadra_NAC_114_dicom",
        BERN_QUADRA_RAW_DIR / "NASC_NSC" / "Bern_Quadra_115_dicom",
    ],
    "SH_uExplorer": [
        SH_UEXPLORER_RAW_DIR / "Explorer_20230509_dicom_233",
        SH_UEXPLORER_RAW_DIR / "SH_Explorer_first_84_334_dicom" / "PART1",
        SH_UEXPLORER_RAW_DIR / "SH_Explorer_first_84_334_dicom" / "PART2",
    ],
}

SCANNER_TO_STAGE = {
    "Bern_Quadra": "train",
    "SH_uExplorer": "train",
    "Bern_Quadra_UHS": "test",
    "Bern_Vision600_cross_tracer": "test",
    "Bern_Vision600": "test",
    "SH_GE_Discovery": "test",
    "SH_UI780": "test",
    "SH_Vision450": "test",
    "SH_Vision450_cross_tracer": "test",
}
