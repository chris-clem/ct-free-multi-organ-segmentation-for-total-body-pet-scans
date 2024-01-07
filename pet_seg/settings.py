from pathlib import Path

# Paths
DATA_ROOT_DIR = Path.home() / "Data" / "ct-free-multi-organ-segmentation-for-total-body-pet-scans"
DATA_CSVS_DIR = DATA_ROOT_DIR / "csvs"
RESULTS_DIR = DATA_ROOT_DIR / "results"
DICOM_HEADERS_DIR = DATA_ROOT_DIR / "dicom_headers"

# Constants
SCANNER_TO_STAGE = {
    "Bern_Quadra": "train",
    "SH_uExplorer": "train",
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

INDEX_TO_ANATOMICAL_STRUCTURES = {
    0: "background",
    1: "spleen",
    2: "kidney_right",
    3: "kidney_left",
    4: "gallbladder",
    5: "liver",
    6: "stomach",
    7: "pancreas",
    8: "adrenal_gland_right",
    9: "adrenal_gland_left",
    10: "lung_upper_lobe_left",
    11: "lung_lower_lobe_left",
    12: "lung_upper_lobe_right",
    13: "lung_middle_lobe_right",
    14: "lung_lower_lobe_right",
    15: "esophagus",
    16: "trachea",
    17: "thyroid_gland",
    18: "small_bowel",
    19: "duodenum",
    20: "colon",
    21: "urinary_bladder",
    22: "prostate",
    23: "kidney_cyst_left",
    24: "kidney_cyst_right",
    25: "sacrum",
    26: "vertebrae_S1",
    27: "vertebrae_L5",
    28: "vertebrae_L4",
    29: "vertebrae_L3",
    30: "vertebrae_L2",
    31: "vertebrae_L1",
    32: "vertebrae_T12",
    33: "vertebrae_T11",
    34: "vertebrae_T10",
    35: "vertebrae_T9",
    36: "vertebrae_T8",
    37: "vertebrae_T7",
    38: "vertebrae_T6",
    39: "vertebrae_T5",
    40: "vertebrae_T4",
    41: "vertebrae_T3",
    42: "vertebrae_T2",
    43: "vertebrae_T1",
    44: "vertebrae_C7",
    45: "vertebrae_C6",
    46: "vertebrae_C5",
    47: "vertebrae_C4",
    48: "vertebrae_C3",
    49: "vertebrae_C2",
    50: "vertebrae_C1",
    51: "heart",
    52: "aorta",
    53: "pulmonary_vein",
    54: "brachiocephalic_trunk",
    55: "subclavian_artery_right",
    56: "subclavian_artery_left",
    57: "common_carotid_artery_right",
    58: "common_carotid_artery_left",
    59: "brachiocephalic_vein_left",
    60: "brachiocephalic_vein_right",
    61: "atrial_appendage_left",
    62: "superior_vena_cava",
    63: "inferior_vena_cava",
    64: "portal_vein_and_splenic_vein",
    65: "iliac_artery_left",
    66: "iliac_artery_right",
    67: "iliac_vena_left",
    68: "iliac_vena_right",
    69: "humerus_left",
    70: "humerus_right",
    71: "scapula_left",
    72: "scapula_right",
    73: "clavicula_left",
    74: "clavicula_right",
    75: "femur_left",
    76: "femur_right",
    77: "hip_left",
    78: "hip_right",
    79: "spinal_cord",
    80: "gluteus_maximus_left",
    81: "gluteus_maximus_right",
    82: "gluteus_medius_left",
    83: "gluteus_medius_right",
    84: "gluteus_minimus_left",
    85: "gluteus_minimus_right",
    86: "autochthon_left",
    87: "autochthon_right",
    88: "iliopsoas_left",
    89: "iliopsoas_right",
    90: "brain",
    91: "skull",
    92: "rib_right_4",
    93: "rib_right_3",
    94: "rib_left_1",
    95: "rib_left_2",
    96: "rib_left_3",
    97: "rib_left_4",
    98: "rib_left_5",
    99: "rib_left_6",
    100: "rib_left_7",
    101: "rib_left_8",
    102: "rib_left_9",
    103: "rib_left_10",
    104: "rib_left_11",
    105: "rib_left_12",
    106: "rib_right_1",
    107: "rib_right_2",
    108: "rib_right_5",
    109: "rib_right_6",
    110: "rib_right_7",
    111: "rib_right_8",
    112: "rib_right_9",
    113: "rib_right_10",
    114: "rib_right_11",
    115: "rib_right_12",
    116: "sternum",
    117: "costal_cartilages",
}

MERGED_ANATOMICAL_STRUCTURES = {
    "kidneys": [
        "kidney_right",
        "kidney_left",
    ],
    "adrenal_glands": [
        "adrenal_gland_right",
        "adrenal_gland_left",
    ],
    "lungs": [
        "lung_upper_lobe_left",
        "lung_lower_lobe_left",
        "lung_upper_lobe_right",
        "lung_middle_lobe_right",
        "lung_lower_lobe_right",
    ],
    "kidney_cysts": [
        "kidney_cyst_left",
        "kidney_cyst_right",
    ],
    "vertebraes": [
        "vertebrae_S1",
        "vertebrae_L5",
        "vertebrae_L4",
        "vertebrae_L3",
        "vertebrae_L2",
        "vertebrae_L1",
        "vertebrae_T12",
        "vertebrae_T11",
        "vertebrae_T10",
        "vertebrae_T9",
        "vertebrae_T8",
        "vertebrae_T7",
        "vertebrae_T6",
        "vertebrae_T5",
        "vertebrae_T4",
        "vertebrae_T3",
        "vertebrae_T2",
        "vertebrae_T1",
        "vertebrae_C7",
        "vertebrae_C6",
        "vertebrae_C5",
        "vertebrae_C4",
        "vertebrae_C3",
        "vertebrae_C2",
        "vertebrae_C1",
    ],
    "subclavian_arteries": [
        "subclavian_artery_right",
        "subclavian_artery_left",
    ],
    "common_carotid_arteries": [
        "common_carotid_artery_right",
        "common_carotid_artery_left",
    ],
    "brachiocephalic_veins": [
        "brachiocephalic_vein_left",
        "brachiocephalic_vein_right",
    ],
    "vena_cavas": [
        "superior_vena_cava",
        "inferior_vena_cava",
    ],
    "iliac_arteries": [
        "iliac_artery_left",
        "iliac_artery_right",
    ],
    "iliac_venas": [
        "iliac_vena_left",
        "iliac_vena_right",
    ],
    "humeruses": [
        "humerus_left",
        "humerus_right",
    ],
    "scapulas": [
        "scapula_left",
        "scapula_right",
    ],
    "claviculas": [
        "clavicula_left",
        "clavicula_right",
    ],
    "femurs": [
        "femur_left",
        "femur_right",
    ],
    "hips": [
        "hip_left",
        "hip_right",
    ],
    "gluteus": [
        "gluteus_maximus_left",
        "gluteus_maximus_right",
        "gluteus_medius_left",
        "gluteus_medius_right",
        "gluteus_minimus_left",
        "gluteus_minimus_right",
    ],
    "autochthons": [
        "autochthon_left",
        "autochthon_right",
    ],
    "iliopsoas": [
        "iliopsoas_left",
        "iliopsoas_right",
    ],
    "ribs": [
        "rib_left_1",
        "rib_left_2",
        "rib_left_3",
        "rib_left_4",
        "rib_left_5",
        "rib_left_6",
        "rib_left_7",
        "rib_left_8",
        "rib_left_9",
        "rib_left_10",
        "rib_left_11",
        "rib_left_12",
        "rib_right_1",
        "rib_right_2",
        "rib_right_3",
        "rib_right_4",
        "rib_right_5",
        "rib_right_6",
        "rib_right_7",
        "rib_right_8",
        "rib_right_9",
        "rib_right_10",
        "rib_right_11",
        "rib_right_12",
    ],
}

ANATOMICAL_REGIONS = {
    "skeleton": [  # 63
        "skull",  # 1
        "claviculas",  # 2
        "scapulas",  # 2
        "humeruses",  # 2
        "vertebraes",  # 24
        "sternum",  # 1
        "ribs",  # 24
        "costal_cartilages",  # 2
        "hips",  # 2
        "sacrum",  # 1
        "femurs",  # 2
    ],
    "cardiovascular_system": [  # 17
        "common_carotid_arteries",  # 2
        "brachiocephalic_veins",  # 2
        "subclavian_arteries",  # 2
        "brachiocephalic_trunk",  # 1
        "pulmonary_vein",  # 1
        "vena_cavas",  # 2
        "atrial_appendage_left",  # 1
        "aorta",  # 1
        "portal_vein_and_splenic_vein",  # 1
        "iliac_arteries",  # 2
        "iliac_venas",  # 2
    ],
    "other_organs": [  # 21
        "brain",
        "spinal_cord",
        "thyroid_gland",
        "trachea",
        "lungs",  # 5
        "heart",
        "adrenal_glands",  # 2
        "spleen",
        "liver",
        # "gallbladder",
        "kidneys",  # 2
        # "kidney_cysts",  # 2
        "pancreas",
        # "prostate",
    ],
    "gastrointestinal_tract": [  # 6
        "esophagus",
        "stomach",
        "duodenum",
        "small_bowel",
        "colon",
        "urinary_bladder",
    ],
    "muscles": [  # 10
        "autochthons",  # 2
        "iliopsoas",  # 2
        "gluteus",  # 6
    ],
}


ANATOMICAL_STRUCTURES_TO_INDEX = {v: k for k, v in INDEX_TO_ANATOMICAL_STRUCTURES.items()}

MODEL_DATASET_IDS_TO_NAMES = {
    1: "Dataset001_Bern_Quadra-SH_uExplorer-num_train=956-num_test=50_NAC",
}

TEST_DATASET_IDS_TO_NAMES = {
    0: "Dataset000_Bern_Quadra-num_train=0-num_test=25_NAC",
    1: "Dataset001_SH_uExplorer-num_train=0-num_test=25_NAC",
    2: "Dataset002_Bern_Quadra_UHS-num_train=0-num_test=21_NAC",
    3: "Dataset003_Bern_Vision600-num_train=0-num_test=52_NAC",
    4: "Dataset004_SH_GE_Discovery-num_train=0-num_test=104_NAC",
    5: "Dataset005_SH_UI780-num_train=0-num_test=100_NAC",
    6: "Dataset006_SH_Vision450-num_train=0-num_test=51_NAC",
    7: "Dataset007_Bern_Vision600_cross_tracer-num_train=0-num_test=30_NAC",
    8: "Dataset008_SH_Vision450_cross_tracer-num_train=0-num_test=41_NAC",
}

TEST_DATASETS_TO_IDS = {
    "internal": [0, 1],
    "cross_scanner": [
        2,
        3,
        4,
        5,
        6,
    ],
    "cross_tracer": [7, 8],
}
