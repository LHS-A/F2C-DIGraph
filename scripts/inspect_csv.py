"""Inspect RetinaDUAL CSV schema and inferred columns."""

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from data.csv_parser import write_inspection_report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    write_inspection_report(args.csv, args.output)

