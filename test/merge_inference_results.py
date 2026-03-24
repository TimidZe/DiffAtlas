import csv
import sys
from pathlib import Path


def collect_rows(results_root: Path, filename: str):
    paths = sorted(results_root.glob(f"gpu*/{filename}"))
    if not paths:
        return [], []

    fieldnames = None
    rows = []
    for path in paths:
        with path.open("r", newline="", encoding="utf-8") as csv_file:
            reader = csv.DictReader(csv_file)
            if fieldnames is None:
                fieldnames = reader.fieldnames
            for row in reader:
                rows.append(row)
    return fieldnames or [], rows


def write_rows(path: Path, fieldnames, rows):
    if not fieldnames:
        return
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    if len(sys.argv) != 2:
        raise SystemExit("Usage: merge_inference_results.py <results_root>")

    results_root = Path(sys.argv[1]).resolve()
    if not results_root.exists():
        raise SystemExit(f"Results root does not exist: {results_root}")

    summary_fields, summary_rows = collect_rows(results_root, "run_summary.csv")
    case_fields, case_rows = collect_rows(results_root, "case_metrics.csv")

    write_rows(results_root / "run_summary.csv", summary_fields, summary_rows)
    write_rows(results_root / "case_metrics.csv", case_fields, case_rows)

    print(f"Merged {len(summary_rows)} run rows into {results_root / 'run_summary.csv'}")
    print(f"Merged {len(case_rows)} case rows into {results_root / 'case_metrics.csv'}")


if __name__ == "__main__":
    main()
