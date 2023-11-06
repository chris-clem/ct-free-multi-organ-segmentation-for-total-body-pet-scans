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

BERN_QUADRA_TEST_PATIENT_IDS = [
    "170411",
    "170820",
    "170911",
    "170943",
    "165653",
    "165730",
    "165937",
    "125353",
    "125423",
    "173158",
    "173219",
    "170505",
    "170657",
    "165659",
    "165216",
    "165152",
    "153649",
    "153822",
    "164152",
    "164220",
    "164253",
    "170053",
    "171616",
    "171657",
    "173413",
]

SH_UEXPLORER_TEST_PATIENT_IDS = [
    "102159",
    "110206",
    "092713",
    "095024",
    "100125",
    "092548",
    "102338",
    "103422",
    "091447",
    "095155",
    "090704",
    "091833",
    "102336",
    "105823",
    "090554",
    "091537",
    "094924",
    "112634",
    "090643",
    "093107",
    "095439",
    "104114",
    "105153",
    "110757",
    "113047",
]

TEST_PATIENT_IDS = {
    "Bern_Quadra": BERN_QUADRA_TEST_PATIENT_IDS,
    "SH_uExplorer": SH_UEXPLORER_TEST_PATIENT_IDS,
}

DATA_CSVS_DIR = DATA_ROOT_DIR / "data_csvs"

LABELS_TO_ORGANS = {
    0: "background",
    1: "Adrenal-glands",
    2: "Aorta",
    3: "Bladder",
    4: "Brain",
    5: "Heart",
    6: "Kidneys",
    7: "Liver",
    8: "Pancreas",
    9: "Spleen",
    10: "Thyroid",
    11: "Inferior-vena-cava",
    12: "Lung",
}

ORGANS_TO_LABELS = {v: k for k, v in LABELS_TO_ORGANS.items()}
