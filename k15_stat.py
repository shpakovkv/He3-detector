import numpy as np
from tools import filter_128
from k15reader import get_raw_lines
from k15reader import get_k15_data
from k15reader import get_file_list
from he3analysis import ll_filter_128
from math import nan


def stats_file(filename, verbose=False):
    raw_lines = get_raw_lines(filename)
    data = get_k15_data(raw_lines)
    stats_dict = stats_data(data, verbose=True)
    return stats_dict


def stats_data(data, verbose=False):
    keys = ["count",          "min",          "max",          "mean",          "std",
            "count_128_only", "min_128_only", "max_128_only", "mean_128_only", "std_128_only",
            "count_filtered", "min_filtered", "max_filtered", "mean_filtered", "std_filtered"]
    func_list = [len, np.nanmin, np.nanmax, np.nanmean, np.nanstd]

    # data[1] is time column
    stat_dict = get_stats(data[1:], keys, func_list)
    print_stat_dict(stat_dict, keys)
    return stat_dict


def get_stats(data, keys, func_list):
    assert len(keys) == 3 * len(func_list), "Error len(keys) != 3 * len(func_list)"
    stat_dict = dict((label, list()) for label in keys)
    below128_idx_list = list()
    for ch, col in enumerate(data):
        below128_idx_list.append(col < 128)
        for idx, func in enumerate(func_list):
            if len(col[below128_idx_list[ch]]) == 0:
                stat_dict[keys[idx]].append(0)
            else:
                stat_dict[keys[idx]].append(func(col[below128_idx_list[ch]]))

    data = ll_filter_128(data)

    for ch, col in enumerate(data):
        for idx, func in enumerate(func_list):
            if len(col[np.invert(below128_idx_list[ch])]) == 0:
                stat_dict[keys[idx + len(func_list)]].append(0)
            else:
                stat_dict[keys[idx + len(func_list)]].append(func(col[np.invert(below128_idx_list[ch])]))

    for col in data:
        for idx, func in enumerate(func_list):
            if len(col) == 0:
                stat_dict[keys[idx + 2 * len(func_list)]].append(0)
            else:
                stat_dict[keys[idx + 2 * len(func_list)]].append(func(col))

    return stat_dict


def print_stat_dict(stat_dict, keys):
    print("Statistics non-filtered(filtered)")
    # non-filtered length is half of length
    part = len(keys) // 3
    head_len = 8
    for idx, key in enumerate(keys[:part]):
        head = f"{key.upper()}:"
        if len(head) < head_len:
            head += " " * (head_len - len(head))
        print(head, end="")
        if isinstance(stat_dict[key][0], int):
            print(";  ".join(f"{val:>6d}({val128:>6d})({fval:>6d})"
                             for val, val128, fval in zip(stat_dict[key], stat_dict[keys[idx + part]], stat_dict[keys[idx + 2 * part]])))
        else:
            print(";  ".join(f"{val:>6.2F}({val128:>6.2F})({fval:>6.2F})"
                             for val, val128, fval in zip(stat_dict[key], stat_dict[keys[idx + part]], stat_dict[keys[idx + 2 * part]])))


def main():
    k15_dir = "D:\\GELIS\\Experiments\\Test\\data"
    k15_files = get_file_list(k15_dir, sort_by_name=True)
    for filename in k15_files:
        print("==================================================================================")
        print(f"Processing file '{filename}'")
        stats_file(filename, verbose=True)



if __name__ == "__main__":
    main()
