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
    "Bern_Quadra_UHS": "test",
    "Bern_Vision600": "test",
    "Bern_Vision600_cross_tracer": "test",
    "SH_GE_Discovery": "test",
    "SH_UI780": "test",
    "SH_Vision450": "test",
    "SH_Vision_cross_tracer": "test",
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
    1: "Spleen",
    2: "Kidney Right",
    3: "Kidney Left",
    4: "Gallbladder",
    5: "Liver",
    6: "Stomach",
    7: "Pancreas",
    8: "Adrenal Gland Right",
    9: "Adrenal Gland Left",
    10: "Lung Upper Lobe Left",
    11: "Lung Lower Lobe Left",
    12: "Lung Upper Lobe Right",
    13: "Lung Middle Lobe Right",
    14: "Lung Lower Lobe Right",
    15: "Esophagus",
    16: "Trachea",
    17: "Thyroid Gland",
    18: "Small Bowel",
    19: "Duodenum",
    20: "Colon",
    21: "Urinary Bladder",
    22: "Prostate",
    23: "Kidney Cyst Left",
    24: "Kidney Cyst Right",
    25: "Sacrum",
    26: "Vertebrae S1",
    27: "Vertebrae L5",
    28: "Vertebrae L4",
    29: "Vertebrae L3",
    30: "Vertebrae L2",
    31: "Vertebrae L1",
    32: "Vertebrae T12",
    33: "Vertebrae T11",
    34: "Vertebrae T10",
    35: "Vertebrae T9",
    36: "Vertebrae T8",
    37: "Vertebrae T7",
    38: "Vertebrae T6",
    39: "Vertebrae T5",
    40: "Vertebrae T4",
    41: "Vertebrae T3",
    42: "Vertebrae T2",
    43: "Vertebrae T1",
    44: "Vertebrae C7",
    45: "Vertebrae C6",
    46: "Vertebrae C5",
    47: "Vertebrae C4",
    48: "Vertebrae C3",
    49: "Vertebrae C2",
    50: "Vertebrae C1",
    51: "Heart",
    52: "Aorta",
    53: "Pulmonary Vein",
    54: "Brachiocephalic Trunk",
    55: "Subclavian Artery Right",
    56: "Subclavian Artery Left",
    57: "Common Carotid Artery Right",
    58: "Common Carotid Artery Left",
    59: "Brachiocephalic Vein Left",
    60: "Brachiocephalic Vein Right",
    61: "Atrial Appendage Left",
    62: "Superior Vena Cava",
    63: "Inferior Vena Cava",
    64: "Portal Vein and Splenic Vein",
    65: "Iliac Artery Left",
    66: "Iliac Artery Right",
    67: "Iliac Vena Left",
    68: "Iliac Vena Right",
    69: "Humerus Left",
    70: "Humerus Right",
    71: "Scapula Left",
    72: "Scapula Right",
    73: "Clavicula Left",
    74: "Clavicula Right",
    75: "Femur Left",
    76: "Femur Right",
    77: "Hip Left",
    78: "Hip Right",
    79: "Spinal Cord",
    80: "Gluteus Maximus Left",
    81: "Gluteus Maximus Right",
    82: "Gluteus Medius Left",
    83: "Gluteus Medius Right",
    84: "Gluteus Minimus Left",
    85: "Gluteus Minimus Right",
    86: "Autochthon Left",
    87: "Autochthon Right",
    88: "Iliopsoas Left",
    89: "Iliopsoas Right",
    90: "Brain",
    91: "Skull",
    92: "Rib Right 4",
    93: "Rib Right 3",
    94: "Rib Left 1",
    95: "Rib Left 2",
    96: "Rib Left 3",
    97: "Rib Left 4",
    98: "Rib Left 5",
    99: "Rib Left 6",
    100: "Rib Left 7",
    101: "Rib Left 8",
    102: "Rib Left 9",
    103: "Rib Left 10",
    104: "Rib Left 11",
    105: "Rib Left 12",
    106: "Rib Right 1",
    107: "Rib Right 2",
    108: "Rib Right 5",
    109: "Rib Right 6",
    110: "Rib Right 7",
    111: "Rib Right 8",
    112: "Rib Right 9",
    113: "Rib Right 10",
    114: "Rib Right 11",
    115: "Rib Right 12",
    116: "Sternum",
    117: "Costal Cartilages",
}

MERGED_ANATOMICAL_STRUCTURES = {
    "Kidneys": [
        "Kidney Right",
        "Kidney Left",
    ],
    "Adrenal Glands": [
        "Adrenal Gland Right",
        "Adrenal Gland Left",
    ],
    "Lungs": [
        "Lung Upper Lobe Left",
        "Lung Lower Lobe Left",
        "Lung Upper Lobe Right",
        "Lung Middle Lobe Right",
        "Lung Lower Lobe Right",
    ],
    "Kidney Cysts": [
        "Kidney Cyst Left",
        "Kidney Cyst Right",
    ],
    "Vertebraes": [
        "Vertebrae S1",
        "Vertebrae L5",
        "Vertebrae L4",
        "Vertebrae L3",
        "Vertebrae L2",
        "Vertebrae L1",
        "Vertebrae T12",
        "Vertebrae T11",
        "Vertebrae T10",
        "Vertebrae T9",
        "Vertebrae T8",
        "Vertebrae T7",
        "Vertebrae T6",
        "Vertebrae T5",
        "Vertebrae T4",
        "Vertebrae T3",
        "Vertebrae T2",
        "Vertebrae T1",
        "Vertebrae C7",
        "Vertebrae C6",
        "Vertebrae C5",
        "Vertebrae C4",
        "Vertebrae C3",
        "Vertebrae C2",
        "Vertebrae C1",
    ],
    "Subclavian Arteries": [
        "Subclavian Artery Right",
        "Subclavian Artery Left",
    ],
    "Common Carotid Arteries": [
        "Common Carotid Artery Right",
        "Common Carotid Artery Left",
    ],
    "Brachiocephalic Veins": [
        "Brachiocephalic Vein Left",
        "Brachiocephalic Vein Right",
    ],
    "Vena Cavas": [
        "Superior Vena Cava",
        "Inferior Vena Cava",
    ],
    "Iliac Arteries": [
        "Iliac Artery Left",
        "Iliac Artery Right",
    ],
    "Iliac Venas": [
        "Iliac Vena Left",
        "Iliac Vena Right",
    ],
    "Humeruses": [
        "Humerus Left",
        "Humerus Right",
    ],
    "Scapulas": [
        "Scapula Left",
        "Scapula Right",
    ],
    "Claviculas": [
        "Clavicula Left",
        "Clavicula Right",
    ],
    "Femurs": [
        "Femur Left",
        "Femur Right",
    ],
    "Hips": [
        "Hip Left",
        "Hip Right",
    ],
    "Gluteus": [
        "Gluteus Maximus Left",
        "Gluteus Maximus Right",
        "Gluteus Medius Left",
        "Gluteus Medius Right",
        "Gluteus Minimus Left",
        "Gluteus Minimus Right",
    ],
    "Autochthons": [
        "Autochthon Left",
        "Autochthon Right",
    ],
    "Iliopsoas": [
        "Iliopsoas Left",
        "Iliopsoas Right",
    ],
    "Ribs": [
        "Rib Right 4",
        "Rib Right 3",
        "Rib Left 1",
        "Rib Left 2",
        "Rib Left 3",
        "Rib Left 4",
        "Rib Left 5",
        "Rib Left 6",
        "Rib Left 7",
        "Rib Left 8",
        "Rib Left 9",
        "Rib Left 10",
        "Rib Left 11",
        "Rib Left 12",
        "Rib Right 1",
        "Rib Right 2",
        "Rib Right 5",
        "Rib Right 6",
        "Rib Right 7",
        "Rib Right 8",
        "Rib Right 9",
        "Rib Right 10",
        "Rib Right 11",
        "Rib Right 12",
    ],
}

ANATOMICAL_REGIONS = {
    "Skeleton": [  # 63
        "Skull",  # 1
        "Claviculas",  # 2
        "Scapulas",  # 2
        "Humeruses",  # 2
        # "Vertebraes",  # 24
        "Sternum",  # 1
        # "Ribs",  # 24
        # "Costal Cartilages",  # 2
        "Hips",  # 2
        "Sacrum",  # 1
        "Femurs",  # 2
    ],
    "Cardiovascular System": [  # 17
        "Common Carotid Arteries",  # 2
        "Brachiocephalic Veins",  # 2
        "Subclavian Arteries",  # 2
        # "Brachiocephalic Trunk",  # 1
        "Pulmonary Vein",  # 1
        "Vena Cavas",  # 2
        # "Atrial Appendage Left",  # 1
        "Aorta",  # 1
        # "Portal Vein and Splenic Vein",  # 1
        "Iliac Arteries",  # 2
        "Iliac Venas",  # 2
    ],
    "Other Organs": [  # 21
        # "Brain",
        "Spinal Cord",
        "Thyroid Gland",
        # "Trachea",
        "Lungs",  # 5
        "Heart",
        # "Adrenal Glands",  # 2
        "Spleen",
        "Liver",
        # "Gallbladder",
        "Kidneys",  # 2
        # "Kidney Cysts",  # 2
        "Pancreas",
        # "Prostate",
    ],
    "Gastrointestinal Tract": [  # 6
        "Esophagus",
        "Stomach",
        "Duodenum",
        "Small Bowel",
        "Colon",
        "Urinary Bladder",
    ],
    "Muscles": [  # 10
        "Autochthons",  # 2
        "Iliopsoas",  # 2
        "Gluteus",  # 6
    ],
}


ANATOMICAL_STRUCTURES_TO_INDEX = {v: k for k, v in INDEX_TO_ANATOMICAL_STRUCTURES.items()}

INDEX_TO_MERGED_ANATOMICAL_STRUCTURES = {
    0: "background",
    1: "Spleen",
    2: "Kidneys",
    3: "Gallbladder",
    4: "Liver",
    5: "Stomach",
    6: "Pancreas",
    7: "Adrenal Glands",
    8: "Lungs",
    9: "Esophagus",
    10: "Trachea",
    11: "Thyroid Gland",
    12: "Small Bowel",
    13: "Duodenum",
    14: "Colon",
    15: "Urinary Bladder",
    16: "Prostate",
    17: "Kidney Cysts",
    18: "Sacrum",
    19: "Vertebraes",
    20: "Heart",
    21: "Aorta",
    22: "Pulmonary Vein",
    23: "Brachiocephalic Trunk",
    24: "Subclavian Arteries",
    25: "Common Carotid Arteries",
    26: "Brachiocephalic Veins",
    27: "Atrial Appendage Left",
    28: "Vena Cavas",
    29: "Portal Vein and Splenic Vein",
    30: "Iliac Arteries",
    31: "Iliac Venas",
    32: "Humeruses",
    33: "Scapulas",
    34: "Claviculas",
    35: "Femurs",
    36: "Hips",
    37: "Spinal Cord",
    38: "Gluteus",
    39: "Autochthons",
    40: "Iliopsoas",
    41: "Brain",
    42: "Skull",
    43: "Ribs",
    44: "Sternum",
    45: "Costal Cartilages",
}

MERGED_ANATOMICAL_STRUCTURES_TO_INDEX = {v: k for k, v in INDEX_TO_MERGED_ANATOMICAL_STRUCTURES.items()}

MODEL_DATASET_IDS_TO_NAMES = {
    1: "Dataset001_Bern_Quadra-SH_uExplorer-num_train=956-num_test=50_NAC",
    9: "Dataset009_Renji_uExplorer-num_train=378-num_test=50_NAC",
    100: "Dataset100_Bern_Quadra-SH_uExplorer-num_train=956-num_test=50_NAC",
}

TEST_DATASET_IDS_TO_NAMES = {
    0: "Dataset000_Bern_Quadra-num_train=0-num_test=25_NAC",
    # 1: "Dataset001_SH_uExplorer-num_train=0-num_test=25_NAC",
    1: "Dataset001_Bern_Quadra-SH_uExplorer-num_train=956-num_test=50_NAC",
    2: "Dataset002_Bern_Quadra_UHS-num_train=0-num_test=21_NAC",
    3: "Dataset003_Bern_Vision600-num_train=0-num_test=52_NAC",
    4: "Dataset004_SH_GE_Discovery-num_train=0-num_test=104_NAC",
    5: "Dataset005_SH_UI780-num_train=0-num_test=100_NAC",
    6: "Dataset006_SH_Vision450-num_train=0-num_test=51_NAC",
    7: "Dataset007_Bern_Vision600_cross_tracer-num_train=0-num_test=30_NAC",
    8: "Dataset008_SH_Vision450_cross_tracer-num_train=0-num_test=41_NAC",
    10: "Dataset010_Renji_uExplorer_dynamic-num_train=0-num_test=276_NAC",
    11: "Dataset011_Renji_uExplorer_dynamic-num_train=0-num_test=276_STATIC",
    100: "Dataset100_Bern_Quadra-SH_uExplorer-num_train=956-num_test=50_NAC",
}

TEST_DATASETS_TO_IDS = {
    "internal": [
        # 0,
        1,
    ],
    "cross_scanner": [
        2,
        3,
        4,
        5,
        6,
    ],
    "cross_tracer": [7, 8],
    "dynamic": [10],
    "dynamic_static": [11],
    "internal_combined": [100],
}
