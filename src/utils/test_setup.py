from src.utils.project_config import (
    PROJECT_ROOT,
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    SEQUENCES_DIR,
    CLASS_NAMES,
    PROJECT_TITLE,
)


def main():

    print("=" * 60)
    print("PROJECT SETUP TEST")
    print("=" * 60)

    print("\nProject:")
    print(f"  {PROJECT_TITLE}")

    print("\nPaths:")
    print(f"  Project root : {PROJECT_ROOT}")
    print(f"  Raw data     : {RAW_DATA_DIR}")
    print(f"  Processed    : {PROCESSED_DATA_DIR}")
    print(f"  Sequences    : {SEQUENCES_DIR}")

    print("\nTraffic classes:")

    for class_id, class_name in enumerate(CLASS_NAMES):
        print(f"  {class_id} -> {class_name}")

    print("\nDirectory checks:")

    directories = [
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR,
        SEQUENCES_DIR,
    ]

    for directory in directories:

        status = "OK" if directory.exists() else "MISSING"

        print(f"  [{status}] {directory}")

    print("\nProject setup successful.")


if __name__ == "__main__":
    main()