from pathlib import Path


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"

RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
SEQUENCES_DIR = DATA_DIR / "sequences"

SRC_DIR = PROJECT_ROOT / "src"
SDN_DIR = PROJECT_ROOT / "sdn"

RESULTS_DIR = PROJECT_ROOT / "results"
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
DASHBOARD_DIR = PROJECT_ROOT / "dashboard"


# ============================================================
# PROJECT INFORMATION
# ============================================================

PROJECT_TITLE = (
    "AN EXPLAINABLE AI FRAMEWORK FOR "
    "DDoS NETWORK TRAFFIC CLASSIFICATION"
)


# ============================================================
# CLASS DEFINITIONS
# ============================================================

CLASS_NAMES = [
    "BENIGN",
    "SYN_FLOOD",
    "UDP_FLOOD",
    "ICMP_FLOOD",
    "SLOWLORIS",
]

CLASS_TO_ID = {
    name: idx
    for idx, name in enumerate(CLASS_NAMES)
}

ID_TO_CLASS = {
    idx: name
    for name, idx in CLASS_TO_ID.items()
}


# ============================================================
# BASIC TEST
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("DDoS XAI SDN PROJECT CONFIGURATION")
    print("=" * 60)

    print(f"Project root : {PROJECT_ROOT}")
    print(f"Project      : {PROJECT_TITLE}")

    print("\nClasses:")
    for class_id, class_name in ID_TO_CLASS.items():
        print(f"  {class_id}: {class_name}")

    print("\nConfiguration loaded successfully.")