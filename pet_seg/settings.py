from pathlib import Path

# Paths
DATA_ROOT_DIR = Path.home() / "Data" / "ct-free-multi-organ-segmentation-for-total-body-pet-scans"

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
