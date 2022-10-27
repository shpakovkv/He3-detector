import numpy as np
from tools import filter_128
from k15reader import get_raw_lines
from k15reader import get_k15_data


def stats_file(filename, verbose=False):
    raw_lines = get_raw_lines(filename)
    data = get_k15_data(raw_lines)
    stats_dict = stats_data(data, verbose=True)
    return stats_dict


def stats_data(data, verbose=False):
    keys = ["count",          "min",          "max",          "mean",          "std",
            "count_filtered", "min_filtered", "max_filtered", "mean_filtered", "std_filtered"]
    func_list = [len, np.nanmin, np.nanmax, np.nanmean, np.nanstd]

    # data[1] is time column
    stat_dict = get_stats(data[1:], keys, func_list)
    print_stat_dict(stat_dict, keys)
    return stat_dict


def get_stats(data, keys, func_list):
    assert len(keys) == 2 * len(func_list), "Error len(keys) % len(func_list) != 0"
    stat_dict = dict((label, list()) for label in keys)
    for col in data:
        for idx, func in enumerate(func_list):
            stat_dict[keys[idx]].append(func(col))

    filter_128(data, verbose=0)
    for col in data:
        for idx, func in enumerate(func_list):
            stat_dict[keys[idx + len(func_list)]].append(func(col))

    return stat_dict


def print_stat_dict(stat_dict, keys):
    print("Statistics non-filtered(filtered)")
    # non-filtered length is half of length
    half = len(keys) // 2
    head_len = 8
    for idx, key in enumerate(keys[:half]):
        head = f"{key.upper()}:"
        if len(head) < head_len:
            head += " " * (head_len - len(head))
        print(head, end="")
        if isinstance(stat_dict[key][0], int):
            print(";  ".join(f"{val:>6d}({fval:>6d})"
                             for val, fval in zip(stat_dict[key], stat_dict[keys[idx + half]])))
        else:
            print(";  ".join(f"{val:>6.2F}({fval:>6.2F})"
                             for val, fval in zip(stat_dict[key], stat_dict[keys[idx + half]])))


def main():
    filename = "D:\\GELIS\\Experiments\\2022-10-04\\04-10-2022\\12_Ti+D-30keV-80mkA-t-0gr-d-45gr-418A.dat"
    stats_file(filename, verbose=True)


if __name__ == "__main__":
    main()
