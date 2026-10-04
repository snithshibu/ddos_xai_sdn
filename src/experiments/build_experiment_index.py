from __future__ import annotations

from pathlib import Path
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

EXPERIMENT_DIR = (
    PROJECT_ROOT / "data" / "experiments"
)

OUTPUT_PATH = (
    EXPERIMENT_DIR / "experiment_index.csv"
)


def describe_csv(path: Path) -> dict:

    try:
        df = pd.read_csv(path)

        return {
            "experiment": path.stem,
            "file": str(
                path.relative_to(PROJECT_ROOT)
            ),
            "rows": len(df),
            "columns": len(df.columns),
            "status": "AVAILABLE",
        }

    except Exception as exc:

        return {
            "experiment": path.stem,
            "file": str(
                path.relative_to(PROJECT_ROOT)
            ),
            "rows": 0,
            "columns": 0,
            "status": f"ERROR: {exc}",
        }


def main():

    print("=" * 70)
    print("EXPERIMENT RESULTS INDEX")
    print("=" * 70)

    files = sorted(
        EXPERIMENT_DIR.rglob("*.csv")
    )

    records = [
        describe_csv(path)
        for path in files
    ]

    index = pd.DataFrame(records)

    index.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print(
        f"\nExperiments indexed: "
        f"{len(index)}"
    )

    print("\n" + "-" * 70)

    for _, row in index.iterrows():

        print(
            f"{row['experiment']:<35} "
            f"{row['rows']:>7} rows  "
            f"{row['status']}"
        )

    print("\n" + "-" * 70)

    print(
        f"Saved: {OUTPUT_PATH}"
    )

    print(
        "\n[PASS] Experiment index created."
    )


if __name__ == "__main__":
    main()