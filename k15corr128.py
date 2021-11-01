# Python 3.6
"""
Readme

"""

import sys
import os
import math

OUT_PREFIX = "zcr_"


def get_fnames():
    """
    Checks input args and returns input filename and not used output filename.

    Usage: input_filename, output_filename = get_fnames()

    :return:  input filename and output filename
    :rtype: tuple
    """
    # right args number
    assert len(sys.argv) == 2, "Usage:\n{} <inputfile> ".format(sys.argv[0])
    in_file = os.path.abspath(sys.argv[1])
    assert os.path.isfile(in_file), "No such file ({})".format(in_file)
    out_file = os.path.join(os.path.dirname(in_file), OUT_PREFIX + os.path.basename(in_file))
    assert not os.path.isfile(out_file), "File ({}) already exists".format(out_file)
    return in_file, out_file


def get_stat_and_corr(file_lines):
    """Fix 128-overflow bug
    and return the sum of all counts.

    :param file_lines: list of file lines
    :return: list of corrected file lines and sum
    :rtype: tuple
    """
    total = 0
    for line in file_lines:
        line = line.split()
        for idx in range(2, len(line)):
            val = int(line[idx])
            if val > 127:
                val %= 128
                line[idx] = str(val)
            total += val
        # print("{date} {time} {val}".format(date=line[0], time=line[1], val=total))
    return file_lines, total


def print_stats(file_lines, sum_):
    """Prints file stats (number of lines, sum, average, precision)

    :param file_lines: list of file lines
    :param sum_: the sum of all counts
    :return: None
    """
    n = len(file_lines)
    print("lines\tsum\tavg.\tprecision:\n{}\t{}\t{:.2f}\t+- {:.2f}"
          "".format(n, sum_, sum_ / n if n > 0 else 0.,
                    math.sqrt(sum_) / n if n > 0 else 0.))


def main():
    in_fname, out_fname = get_fnames()
    data_lines = None
    with open(in_fname, "r") as fid:
        data_lines = fid.readlines()
    data_lines, total = get_stat_and_corr(data_lines)
    print_stats(data_lines, total)
    with open(out_fname, "w") as fid:
        fid.writelines(data_lines)
    print("reloaded: {} ==> {}".format(os.path.basename(in_fname),
                                       os.path.basename(out_fname)))


if __name__ == "__main__":
    main()
