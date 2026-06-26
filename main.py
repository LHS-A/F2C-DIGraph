"""Convenience dispatcher for F2C-DIGraph reproduction commands."""

import argparse

from data.csv_parser import write_inspection_report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inspect_csv")
    parser.add_argument("--inspection_output", default="outputs/logs/csv_inspection.txt")
    args = parser.parse_args()
    if args.inspect_csv:
        write_inspection_report(args.inspect_csv, args.inspection_output)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
