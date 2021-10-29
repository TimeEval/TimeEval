import argparse
import struct
from pathlib import Path

import numpy as np


def convert2bin(input: Path, output: Path):
    data = np.loadtxt(input)
    numpy2bin(data, output)


def numpy2bin(data: np.ndarray, output: Path):
    total_lines: int = len(data)

    log_progress_step: int = total_lines // 100000
    next_progress_log: int = log_progress_step

    with output.open("wb+") as out_file:
        lines_written: int = 0
        for line in data:
            value = float(line)
            out_file.write(bytearray(struct.pack("!d", value)))
            lines_written += 1

            if lines_written >= next_progress_log:
                next_progress_log += log_progress_step


def _number_of_lines(file: Path) -> int:
    number_of_lines: int = 0
    with file.open("r", encoding="utf-8", errors="ignore") as in_file:
        while True:
            block = in_file.read(65536)
            if not block:
                break

            number_of_lines += block.count("\n")

    return number_of_lines


def _create_arg_parser():
    argument_parser = argparse.ArgumentParser()

    argument_parser.add_argument(
        "--input",
        type=Path,
        required=True
    )

    argument_parser.add_argument(
        "--output",
        type=Path,
        required=True
    )

    return argument_parser


if __name__ == "__main__":
    parser = _create_arg_parser()
    args = parser.parse_args()
    convert2bin(args.input, args.output)
