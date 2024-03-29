#!/usr/bin/env python
# coding: utf-8

"""He3 readout system data analysis and statistics.

Author: Konstantin Shpakov, march 2021.
"""
import matplotlib.pyplot as plt
import numpy as np
from numba import njit, vectorize, float64
from file_handler import save_signals_csv
import os
import datetime
from matplotlib import dates as md
from k15reader import get_time, get_date
from datetime import datetime
from datetime import timezone
from math import isnan, nan
import pytz
from dataclasses import dataclass, fields

DEFAULT_SEC_PER_RECORD = 1.025
ERR_COEF = 1.1
TIMEZONE = pytz.timezone("Etc/GMT+3")
FILTER128_TOLERANCE = 0.25

# minimum time step in the records
MIN_TIME_STEP = 1


@dataclass
class Stats:
    # field index 0(mean), 1(min), 2(max) is used in validate_filter128() !!
    # do not change field order
    mean: float = nan
    min: float = nan
    max: float = nan
    length: int = 0


def get_time_str(timestamp):
    """Returns time string in "%H:%M:%S" format.

    :param timestamp: time in timestamp format
    :type timestamp: float
    :return: time in "%H:%M:%S" format
    :rtype: str
    """
    val_str = datetime.fromtimestamp(timestamp).strftime("%H:%M:%S")
    return val_str


def get_datetime_str(timestamp):
    """Returns date and time string in "%Y.%m.%d %H:%M:%S" format.

    :param timestamp: time in timestamp format
    :type timestamp: float
    :return: time in "%Y.%m.%d %H:%M:%S" format
    :rtype: str
    """
    val_str = datetime.fromtimestamp(timestamp).strftime("%Y.%m.%d %H:%M:%S")
    return val_str


def get_vector_stats(array_1d):
    """Calculates array Stats (length, mean, max, min).

    :param array_1d: 1D array with data
    :type array_1d: np.ndarray

    :return: array Stats
    :rtype: Stats
    """

    if array_1d.size == 0:
        return Stats(array_1d.size)

    the_sats = Stats(length=array_1d.size,
                     mean=np.nanmean(array_1d),
                     min=np.nanmin(array_1d),
                     max=np.nanmax(array_1d))

    return the_sats


def get_below128_and_above128_stats(data):
    """Calculates statistics (length, mean, min, max) of below128 values and above127 values
    for all columns of the input data array.

    :param data: 2D array with numeric data
    :type data: np.ndarray

    :return: below128 Stats list and above127 Stats list
    :rtype: tuple of list
    """
    below128_stats_list = list()
    above127_stats_list = list()
    for column in data:
        below = get_vector_stats(column[column < 128])
        below128_stats_list.append(below)

        above127 = column[column > 127].astype(np.float64)
        above127 = ll_filter_128(above127)
        above127_stats_list.append(get_vector_stats(above127))

    return below128_stats_list, above127_stats_list


def get_filter128_relative_diff(below128_stats, above127_stats):
    """ Calculates relative difference of below128 and above127
     mean value, minimum value and maximum value.
     Rel_diff = abs(below_val - above_val) / below_val

    :param below128_stats: below 128 values Stats
    :type below128_stats: Stats
    :param above127_stats: above 127 values stats
    :type above127_stats: Stats
    :return: relative difference of mean value, minimum value and maximum value
    :rtype: tuple of float
    """
    min_meaningful_value = 7

    if below128_stats.length == 0:
        return 1.0, 1.0, 1.0
    if above127_stats.length == 0:
        return 0.0, 0.0, 0.0

    mean_rel_diff = nan
    min_rel_diff = nan
    max_rel_diff = nan

    if below128_stats.mean > 0.0:
        mean_rel_diff = abs(below128_stats.mean - above127_stats.mean) / float(below128_stats.mean)
        if abs(below128_stats.mean - above127_stats.mean) < min_meaningful_value:
            mean_rel_diff = 0
    if below128_stats.min > 0:
        min_rel_diff = abs(below128_stats.min - above127_stats.min) / float(below128_stats.min)
        if abs(below128_stats.min - above127_stats.min) < min_meaningful_value:
            min_rel_diff = 0
    if below128_stats.max > 0:
        max_rel_diff = abs(below128_stats.max - above127_stats.max) / float(below128_stats.max)
        if abs(below128_stats.max - above127_stats.max) < min_meaningful_value:
            max_rel_diff = 0

    return mean_rel_diff, min_rel_diff, max_rel_diff


def get_filter128_relative_diff_list(below128_stats_list, above127_stats_list):
    """ Calculates relative difference of below128 and above127
     mean value, minimum value and maximum value for all data columns.

    :param below128_stats_list: list of below 128 values Stats
    :type below128_stats_list: list of Stats
    :param above127_stats_list: list of above 127 values stats
    :type above127_stats_list: list of Stats
    :return: relative difference of mean value, minimum value and maximum value
    :rtype: list of tuple
    """
    rel_diff_list = list()
    for below_stats, above_stats in zip(below128_stats_list, above127_stats_list):
        rel_diff_list.append(get_filter128_relative_diff(below_stats, above_stats))
    return rel_diff_list


def get_validation_of_filter128(data):
    """Validate relative error for 128-filtering process
    for all channels of the input data array.

    Returns below128 Stats list, above127 Stats list, relative diff list

    :param data: 2D array with detectors data, data[1] - time col, data[1:] - data channels
    :type data: np.ndarray

    :return: tuple of: below128 Stats list, above127 Stats list, relative diffs
    :rtype: tuple
    """
    below128_stat_list, above127_stat_list = get_below128_and_above128_stats(data[1:])

    # diff list:
    # 0: mean
    # 1: min
    # 2: max
    diff_list = get_filter128_relative_diff_list(below128_stat_list, above127_stat_list)

    return below128_stat_list, above127_stat_list, diff_list


def get_extreme_deviation_intervals(time, value):
    """
    Checks curve data for extreme deviation from mean (signal drop or spurts).
    The input data should be 1-dimensional array (time column and value column).

    Extreme deviation is: TODO: deviation metric

    Returns all deviation intervals as pair (list) of time values [start, stop].
    All time pairs are combined into list.

    If there are no extreme deviation, returns None.

    Return example: [[0.0, 1.0], [3.0, 10.0]]

    :param time: 1-dimensional array with time data
    :param value: 1-dimensional array with value data
    :type time: np.ndarray
    :type value: np.ndarray
    :return: all deviation intervals (list of lists) or None
    :rtype: list
    """
    res = None
    first_diff = np.diff(value, n=1)
    data_mean = value.mean()
    fig, axes = plt.subplots(2, 1, sharex='all', squeeze=False)
    axes[0, 0].plot(time, value)
    axes[0, 0].plot([time[0], time[-1]], [data_mean, data_mean], 'r-')
    axes[1, 0].plot(time[:-1], first_diff)

    from scipy.signal import savgol_filter
    smooth = savgol_filter(value, 501, 2)
    print("Value shape = {},  Smooth shape = {}".format(value.shape, smooth.shape))
    axes[0, 0].plot(time, smooth, 'm-')

    print("Diff mean = {}".format(first_diff.mean()))
    print("Diff std err = {}".format(first_diff.std()))
    plt.show()
    return res


def cut_out_all_intervals(data, list_of_intervals, with_gaps=False, verbose=0):
    """
        Cuts out data from input array.

        Intervals is the list of some start-stop time pairs.

        If with_gaps flag is True, all cut data sections will be replaced by one NaN value.

        Returns modified data array.

        :param data: 2-dimensional array with data
        :param list_of_intervals: list or array with two time points
        :param verbose: if > 0, than print date-time of first and last points of deleted intervals
        :type data: np.ndarray
        :type list_of_intervals: list or tuple or np.ndarray
        :type verbose: int

        :return: modified data array
        :rtype: np.ndarray
        """
    supported_arr_types = "np.ndarray"
    supported_interval_types = "list or tuple or np.ndarray"
    assert isinstance(data, np.ndarray), \
        "Arr value is of an unsupported type. " \
        "Expected {}, got {} instead.".format(supported_arr_types, type(data))
    assert data.ndim == 2, \
        "Data must be 2-dimensional array. Got {} ndims instead.".format(data.ndim)
    assert isinstance(list_of_intervals, list) or \
           isinstance(list_of_intervals, tuple) or \
           isinstance(list_of_intervals, np.ndarray), \
           "Interval value is of an unsupported type. " \
           "Expected {}, got {} instead." \
           "".format(supported_interval_types, type(list_of_intervals))
    assert len(list_of_intervals) > 0, \
        "Unsupported interval length. " \
        "Expected >= 1, got {} instead.".format(len(list_of_intervals))

    # if nothing to cut, return
    if not list_of_intervals:
        print(" NO CUT")
        return data

    # TODO: verbose cutting (output of real boundaries of cut out intervals)

    list_of_intervals = convert_intervals_to_timestamp(list_of_intervals, data)

    at_least_one_cut = 0
    for time_pair in list_of_intervals:
        data, start_str, stop_str = cut_out_interval(data, time_pair, with_gaps=with_gaps)

        if at_least_one_cut < 1 and start_str is not None:
            at_least_one_cut = 1
        if verbose > 0:
            if at_least_one_cut == 1:
                print("Удалены интервалы: ", end="")
                at_least_one_cut = 2
            if start_str is not None:
                print("[{} - {}]".format(start_str, stop_str), end=" ")
    if verbose > 0 and at_least_one_cut > 0:
        print()
    return data


def convert_intervals_to_timestamp(list_of_intervals, data):
    list_of_intervals_ts = []
    for idx, time_pair in enumerate(list_of_intervals):
        if isinstance(time_pair[0], float) and isinstance(time_pair[1], float):
            continue

        assert isinstance(time_pair[0], str) and isinstance(time_pair[1], str), \
            "Wrong time value type ({}). " \
            "Expected [str, str], got [{}, {}] instead" \
            "".format(time_pair, type(time_pair[0]), type(time_pair[0]))

        interval_ts = []
        for j, time in enumerate(time_pair):
            day, month, year = None, None, None
            try:
                day, month, year = get_date(time)
            except AssertionError:
                base = datetime.fromtimestamp(data[0, 0], tz=TIMEZONE)
                day, month, year = base.day, base.month, base.year
            hour, mins, sec = get_time(time)
            date_and_time = datetime(year, month, day, hour, mins, sec, 0, TIMEZONE)
            interval_ts.append(date_and_time.timestamp())
        assert interval_ts[1] > interval_ts[0], \
            "Left interval border ({}) is greater than the right ({}).".format(list_of_intervals[idx][0], list_of_intervals[idx][1])
        list_of_intervals[idx][0] = interval_ts[0]
        list_of_intervals[idx][1] = interval_ts[1]
    return list_of_intervals


def cut_out_interval(data, interval, with_gaps=False):
    """
    Cuts out data from input array.
    Interval is the start-stop time pair.
    If with_gaps flag is True, then one NaN value will be added
    between the remaining two pieces of data.

    Returns modified data array.

    :param data: 2-dimensional array with data
    :param interval: list or array with two time points
    :type data: np.ndarray
    :type interval: list or tuple or np.ndarray

    :return: modified data array, start and stop point of deleted interval
    :rtype: tuple
    """
    supported_arr_types = "np.ndarray"
    supported_interval_types = "list or tuple or np.ndarray"
    assert isinstance(data, np.ndarray), \
        "Arr value is of an unsupported type. " \
        "Expected {}, got {} instead.".format(supported_arr_types, type(data))
    assert data.ndim == 2, \
        "Data must be 2-dimensional array. Got {} ndims instead.".format(data.ndim)
    assert isinstance(interval, list) or \
           isinstance(interval, tuple) or \
           isinstance(interval, np.ndarray), \
           "Interval value is of an unsupported type. " \
           "Expected {}, got {} instead." \
           "".format(supported_interval_types, type(interval))
    assert len(interval) == 2, \
        "Unsupported interval length. " \
        "Expected 2, got {} instead.".format(len(interval))
    assert interval[0] <= interval[1], \
        "Left interval border ({}) is greater than the right ({})." \
        "".format(interval[0], interval[1])

    idx_start, idx_stop = _get_interval_idx(data, interval)

    if idx_start is None or idx_stop is None:
        return data, None, None

    # 1-dimensional mask
    mask = np.ones(shape=data.shape[1], dtype=bool)

    # right border value is included
    mask[idx_start:idx_stop + 1] = False

    start_str = datetime.fromtimestamp(data[0, idx_start], tz=TIMEZONE).strftime("%Y.%m.%d %H:%M:%S")
    stop_str = datetime.fromtimestamp(data[0, idx_stop], tz=TIMEZONE).strftime("%Y.%m.%d %H:%M:%S")

    # add nan if cutting inner interval
    if with_gaps and idx_start > 0 and idx_stop < data.shape[1] - 1:
        # leave one element and replace it with nan
        mask[idx_stop] = True
        data[:, idx_stop] = np.nan
        # masking (cutting out) all columns
        data = data[:, mask]
    else:
        # masking (cutting out) all columns
        data = data[:, mask]

    return data, start_str, stop_str


def _get_interval_idx(data, interval):
    """
        Convert time interval to index interval.
        Np input data checks!

        Returns start idx and stop idx.

        Where start idx is the idx of time column (data[0]) element >= start time,
        and stop idx is the idx of time column (data[0]) element <= stop time.

        :param data: 2-dimensional array with data
        :param interval: list or array with start-stop time pair
        :type data: np.ndarray
        :type interval: list or tuple or np.ndarray

        :return: None or a tuple with interval start idx and stop idx (both included)
        :rtype: tuple or None
        """
    if interval[0] > data[0, -1] or interval[1] < data[0, 0]:
        return None, None

    # search interval[0] <= time[start]
    start = np.searchsorted(data[0], interval[0], side='left')

    # search time[stop - 1] <= interval[1]
    stop = np.searchsorted(data[0], interval[1], side='right')
    stop -= 1

    # interval between two points == nothing to cut
    if stop < start:
        return None, None

    return start, stop


# def convert_time(time, unixtime):
#     if unixtime:
#         return md.epoch2num(time)
#     return time


def convert_time(time, unixtime):
    if unixtime:
        return [datetime.fromtimestamp(ts, tz=TIMEZONE) for ts in time]
    return time


def print_k15_rates(data, rates, err_rates, gaps, group_by_4, verbose):
    rates_str = ",  ".join("{:.4f}".format(val) for val in rates)
    err_rates_str = ", ".join("±{:.4f}".format(err) for err in err_rates)
    if group_by_4:
        print("Средний счет (сумма по 4) [1/с] = [{}]".format(rates_str))
        print("Ср. кв. отклонение              = [{}]".format(err_rates_str))
    else:
        print("Средний счет по каналам [1/с] = [{}]".format(rates_str))
        print("Ср. кв. отклонение            = [{}]".format(err_rates_str))
    # print(", Погрешность [1/с] = {}".format(err_rates))

    if verbose > 1:
        print("Длительность регистрации: {} сек. Количество строк: {}."
              "".format(data[0, -1] - data[0, 0] + (data[0, 1] - data[0, 0]), data.shape[1]))
    if gaps:
        print("Присутствуют пропуски ({} шт) длительностью: {} сек"
              "".format(len(gaps), gaps))

    real_time_per_record, err_real_time_per_records = get_real_time_per_record(data)

    if real_time_per_record > DEFAULT_SEC_PER_RECORD * ERR_COEF:
        print("WARNING! Calculated time-per-record {:.4f} significantly exceeds the default value {:.4f}."
              "".format(real_time_per_record, DEFAULT_SEC_PER_RECORD))
    if verbose > 1:
        print("Длительность одной записи: {:.4f} сек ±{:.6f} сек"
              "".format(real_time_per_record, err_real_time_per_records))


def print_sc_average(data, rates, err_rates, gaps, verbose=2):

    rates_str = ",  ".join("{:.4f}".format(val) for val in rates)
    err_rates_str = ", ".join("±{:.4f}".format(err) for err in err_rates)
    print("Ср. напряжение[В], ток[мкА] = [{}]".format(rates_str))
    print("Среднее кв. отклонение      = [{}]".format(err_rates_str))
    # print(", Погрешность [1/с] = {}".format(err_rates))

    if verbose > 1:
        print("Длительность регистрации: {} сек. Количество строк: {}."
              "".format(data[0, -1] - data[0, 0] + (data[0, 1] - data[0, 0]), data.shape[1]))
    if gaps:
        print("Присутствуют пропуски ({} шт) длительностью: {} сек"
              "".format(len(gaps), gaps))

    # time spent from 1st record to last
    # (!!) registration time of the 1st event is not included
    time_spent = data[0, -1] - data[0, 0]

    # number of records made during time_spent
    records_num = data.shape[1] - 1

    real_time_per_record = time_spent / records_num

    err_real_time_per_records = (time_spent + 2) / records_num - real_time_per_record
    if real_time_per_record > DEFAULT_SEC_PER_RECORD * ERR_COEF:
        print("WARNING! Calculated time-per-record {:.4f} significantly exceeds the default value {:.4f}."
              "".format(real_time_per_record, DEFAULT_SEC_PER_RECORD))
    if verbose > 1:
        print("Длительность одной записи: {:.4f} сек ±{:.6f} сек"
              "".format(real_time_per_record, err_real_time_per_records))


def get_base_output_fname(source_filename, group_by_4, group_by_sec):
    base_out_name = os.path.dirname(source_filename)
    fname = os.path.basename(source_filename)
    if group_by_sec > 0:
        base_out_name = os.path.join(base_out_name, fname + "_sum_by_{}_sec".format(group_by_sec))
    elif group_by_4:
        base_out_name = os.path.join(base_out_name, fname + "_sum_by_ch")
    else:
        base_out_name = os.path.join(base_out_name, fname)
    return base_out_name


def write_data(data, base_out_name, group_by_sec, verbose):
    save_as = base_out_name + ".csv"
    if group_by_sec > 0:
        # averaging gives floating point values
        save_signals_csv(save_as, data, integer=False)
    else:
        save_signals_csv(save_as, data, integer=True)
    if verbose > 1:
        print("Данные сохранены: {}".format(os.path.basename(save_as)))


@njit()
def fill_with_sum_by_ch(dest, source, group_by):
    old_rows = source.shape[0]
    for point in range(dest.shape[1]):
        dest[0, point] = source[0, point]
        for old_row in range(1, old_rows, group_by):
            res = 0
            for val in source[old_row: old_row + group_by, point]:
                res += val
            new_row = (old_row - 1) // group_by + 1
            dest[new_row, point] = res
    return dest


def get_sum_by_number_of_channels(data, group_by):
    new_shape = ((data.shape[0] - 1) // group_by + 1, data.shape[1])
    new_data = np.ndarray(shape=new_shape, dtype=np.int64)
    new_data = fill_with_sum_by_ch(new_data, data, group_by)
    return new_data


def get_average_by_time_interval(data, group_by_sec, sec_per_record=DEFAULT_SEC_PER_RECORD, include_tail=False, verbose=0):
    @njit()
    def fill_value_rows(dest, source, group_by_sec):
        old_len = source.shape[1]
        left_border = 0
        right_border = 1
        for pt in range(dest.shape[1]):
            start_time = source[0, left_border]
            stop_time = start_time
            while stop_time - start_time <= group_by_sec:
                right_border += 1
                if right_border == source.shape[1]:
                    break
                stop_time = source[0, right_border]
            for ch in range(1, dest.shape[0]):
                dest[ch, pt] = np.sum(source[ch, left_border: right_border]) / (right_border - left_border)
            left_border = right_border
        return dest

    @njit()
    def fill_time_row(dest, source, group_by_sec):
        start_sec = np.round(np.mean(source[0, 0: group_by_sec]))
        for pt in range(dest.shape[1]):
            dest[0, pt] = start_sec + pt * group_by_sec
        return dest

    def find_tail_start(source, group_by):
        for idx in range(source.shape[1] - 1, 0, -1):
            if (source[0, idx] - source[0, 0]) % group_by == 0:
                return idx + 1
        return 0

    number_of_intervals = (data[0, -1] - data[0, 0]) // group_by_sec

    tail_points = 1
    tail_start = None
    there_is_tail = ((data[0, -1] - data[0, 0]) % group_by_sec) != 0
    # print("DEBUG: there is tail == {}".format(there_is_tail))
    # print("number_of_intervals == {}".format(number_of_intervals))
    # print("Full duration == {} seconds".format(data[0, -1] - data[0, 0]))
    # print("Number of records == {}".format(data.shape[1]))
    if there_is_tail:
        tail_start = find_tail_start(data, group_by_sec)
        tail_points = data.shape[1] - tail_start
    # print("tail_start == {}".format(tail_start))
    # print("tail_points == {}".format(tail_points))

    new_shape = (data.shape[0], number_of_intervals)
    new_data = np.ndarray(shape=new_shape, dtype=np.float64)
    if number_of_intervals > 0:
        new_data = fill_time_row(new_data, data[:, :-tail_points], group_by_sec)
        new_data = fill_value_rows(new_data, data[:, :-tail_points], group_by_sec)
    if verbose > 0:
        print("{} групп по {} сек.".format(new_data.shape[1], group_by_sec), end="")
    if include_tail and there_is_tail:
        tail = np.zeros(shape=(data.shape[0], 1), dtype=data.dtype)
        # simple solution "(data[0, tail_start] + data[0, -1]) // 2" may overflow in timestamp (for int32 format)
        tail[0, 0] = data[0, tail_start] // 2 + data[0, -1] // 2 + (data[0, tail_start] % 2 + data[0, -1] % 2) // 2
        duration = data[0, -1] - data[0, tail_start]
        duration += sec_per_record
        for ch in range(1, data.shape[0]):
            tail[ch, 0] = np.sum(data[ch, tail_start:]) / tail_points
        new_data = np.concatenate((new_data, tail), axis=1)
        if verbose > 0:
            print(", а также 'хвост' длительностью {} сек.".format(duration), end="")

    if verbose > 0 and not there_is_tail:
        print(", без 'хвоста'", end="")

    if verbose:
        print()
    return new_data


def get_counting_rate(data, sec_per_record=DEFAULT_SEC_PER_RECORD):
    # time spent from 1st record to last
    # (!!) registration time of the 1st event is not included
    duration = data[0, -1] - data[0, 0]

    # TODO: check rate calculation

    # number of records made during time_spent
    records_num = data.shape[1] - 1

    # duration of one record
    real_sec_per_record = duration / (records_num - 1)

    res = list()
    std_dev = list()
    # adding the 1st record time
    duration += sec_per_record
    # adding the 1st record
    records_num += 1

    there_are_gaps = list()

    intervals = list()
    start = 0
    for idx in range(1, records_num):
        if data[0, idx] - data[0, idx - 1] > sec_per_record + MIN_TIME_STEP:
            intervals.append([start, idx - 1])
            there_are_gaps.append(data[0, idx] - data[0, idx - 1])
            start = idx

    real_time_per_record, _ = get_real_time_per_record(data)

    for row in range(1, data.shape[0]):
        # calc rate per record
        rate = sum(data[row, :]) / float(records_num)
        # recalc rate per second
        rate *= real_time_per_record
        res.append(rate)
        std_dev.append(np.std(data[row, :] / float(records_num)))
    return res, std_dev, there_are_gaps


def precompile():
    __temp1 = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], dtype=np.int64)
    __temp1 = __temp1.reshape(3, 4)
    __temp2 = np.ndarray(shape=(3, 2), dtype=np.int64)
    fill_with_sum_by_ch(__temp2, __temp1, 3)
    __temp3 = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], dtype=np.int64)
    __temp3 = __temp1.reshape(2, 6)
    get_sc_ibounds(__temp1, __temp3)


@vectorize([float64(float64)])
def ll_filter_128(x):
    return x % 128


@vectorize([float64(float64)])
def ll_get_128_only(x):
    return (x // 128) * 128


def leave_128_only(data):
    data[1:] = ll_get_128_only(data[1:])


def validate_filter128(data, verbose):
    """ Validate relative results of 128-filtering process
    for all channels of the input data array.

    Prints warning messages during process (uses 3 levels of verbosity).

    Returns valid list (True if the filtered data makes sense,
    False if filtered data is out of statistics of unfiltered data) for all columns.

    :param data: He3 data loaded from file
    :type data: np.ndarray
    :param verbose: level of verbosity
    :type verbose: int
    :return: valid list
    :rtype: list
    """

    # diff list:
    # 0: mean
    # 1: min
    # 2: max
    below128_stat_list, above127_stat_list, diff_list = get_validation_of_filter128(data)
    labels = ("mean", "minimum", "maximum")

    # number of values below 128
    below_list = [val.length for val in below128_stat_list]
    # number of values above 127
    above_list = [val.length for val in above127_stat_list]

    if verbose > 0:
        print(f"Соотношение значений превышающих 127 к общему количеству записей (в %): "
              f"[{', '.join((f'{(above / (below + above) * 100):.1f}' for below, above in zip(below_list, above_list)))}]")

    valid_list = [True] * len(diff_list)
    for ch, diff_tuple in enumerate(diff_list):
        for idx, err in enumerate(diff_tuple):
            if not isnan(err) and err >= 0.25:
                valid_list[ch] = False

                warn_level = 1
                if err > 0.5:
                    warn_level += 1
                if err > 0.75:
                    warn_level += 1
                warn_symbol = '!' * warn_level
                warn_lv_list = ["1st", "2nd", "3rd"]
                attr = fields(Stats)[idx]
                if verbose > 1:
                    print(
                        f"({warn_symbol}) Warning! CH[{ch + 1}] {labels[idx].upper()} value filter128 error ({err * 100:.2f}%) "
                        f"({getattr(above127_stat_list[ch], attr.name)} vs {getattr(below128_stat_list[ch], attr.name)}) "
                        f"has exceeded the {warn_lv_list[warn_level - 1]} allowable level ({25 * warn_level}%)!")

        if below128_stat_list[ch].length == 0:
            valid_list[ch] = False
            if verbose > 0:
                print(f"(!!!) ERROR! CH[{ch + 1}] has NO VALUES below 128 and "
                      f"{above127_stat_list[ch].length} values above 127!")

        if below128_stat_list[ch].max > 128.0 - FILTER128_TOLERANCE * 128.0 and above127_stat_list[ch].length > 0:
            if verbose > 0:
                print(f"(!) WARNING! CH[{ch + 1}] maximum value below 128 ({below128_stat_list[ch].max}) is bigger than "
                      f"the allowable value (128 - {FILTER128_TOLERANCE * 100}%)!")
    if verbose > 0:
        if not all(valid_list):
            print(f"WARNING! filter128 operation error has exceeded the allowable value "
                  f"{FILTER128_TOLERANCE * 100}% for {len(valid_list) - sum(valid_list)} channel(s)")

    if verbose > 2:
        zipped_diff = list(zip(*diff_list))
        # ERR VALUES
        print(f"Filter128 mean value difference by channel: "
              f"[{', '.join(f'{diff * 100:.2f}% ({below_stat.mean:.2f}/{above_stat.mean:.2f})' for diff, below_stat, above_stat in zip(zipped_diff[0], below128_stat_list, above127_stat_list))}]")
        print(f"Filter128 minimum value difference by channel: "
              f"[{', '.join(f'{diff * 100:.2f}% ({below_stat.min}/{above_stat.min})' for diff, below_stat, above_stat in zip(zipped_diff[1], below128_stat_list, above127_stat_list))}]")
        print(f"Filter128 maximum value difference by channel: "
              f"[{', '.join(f'{diff * 100:.2f}% ({below_stat.max}/{above_stat.max})' for diff, below_stat, above_stat in zip(zipped_diff[2], below128_stat_list, above127_stat_list))}]")

    return valid_list


def filter_128(data, verbose):
    data[1:] = ll_filter_128(data[1:])


def print_overflow_128(data):
    data_filtered = np.copy(data)
    filter_128(data_filtered)
    fmt = "%d.%m.%Y %H:%M:%S"
    number_of_records = data.shape[1]
    for idx in range(number_of_records):
        if any(val > 127 for val in data[1:, idx]):
            print()
            msg = unix_datetime_to_str(data[0, idx], tz=TIMEZONE)
            msg += "         "
            msg += ""
            msg += " ".join(space_padded_num(val, 3) for val in data[1:, idx])
            print(msg)

            msg = "Остаток от деления на 128:  "
            msg += " ".join(space_padded_num(val, 3) for val in data_filtered[1:, idx])
            print(msg)


def unix_datetime_to_str(utime, fmt=None, tz=TIMEZONE):
    if fmt is None:
        fmt = "%Y_%m_%d %H:%M:%S"
    return datetime.fromtimestamp(utime, tz=TIMEZONE).strftime(fmt)
    # return datetime.utcfromtimestamp(utime, tz=TIMEZONE).strftime(fmt)


def space_padded_num(num, digits):
    msg = str(num)
    if len(msg) < digits:
        msg = " " * (digits - len(msg)) + msg
    return msg


def get_sc_ibounds(k15_data, sc_data):
    start_k15_time = k15_data[0, 0]
    stop_k15_time = k15_data[0, -1]
    
    # print("K15 Started at ({}) and finished at ({})"
    #       "".format(datetime.fromtimestamp(start_k15_time, tz=TIMEZONE).strftime("%d.%m.%Y %H:%M:%S"),
    #                 datetime.fromtimestamp(stop_k15_time, tz=TIMEZONE).strftime("%d.%m.%Y %H:%M:%S")))
    # print("SlowControl Started at ({}) and finished at ({})"
    #       "".format(datetime.fromtimestamp(sc_data[0, 0], tz=TIMEZONE).strftime("%d.%m.%Y %H:%M:%S"),
    #                 datetime.fromtimestamp(sc_data[0, -1], tz=TIMEZONE).strftime("%d.%m.%Y %H:%M:%S")))
    assert start_k15_time < sc_data[0, -1], \
        "Error! SlowControl data registration finished earlier ({}) " \
        "than K15 data registration started ({})." \
        "".format(datetime.fromtimestamp(sc_data[0, -1], tz=TIMEZONE).strftime("%d.%m.%Y %H:%M:%S"),
                  datetime.fromtimestamp(start_k15_time, tz=TIMEZONE).strftime("%d.%m.%Y %H:%M:%S"))
    assert stop_k15_time > sc_data[0, 0], \
        "Error! K15 data registration finished earlier ({}) " \
        "than SlowControl data registration started ({})." \
        "".format(datetime.fromtimestamp(stop_k15_time, tz=TIMEZONE).strftime("%d.%m.%Y %H:%M:%S"),
                  datetime.fromtimestamp(sc_data[0, 0], tz=TIMEZONE).strftime("%d.%m.%Y %H:%M:%S"))
    start_sc_idx = None
    stop_sc_idx = None
    for idx in range(sc_data.shape[1]):
        if start_sc_idx is None and sc_data[0, idx] >= start_k15_time:
            start_sc_idx = idx
        if stop_sc_idx is None and sc_data[0, idx] > stop_k15_time:
            stop_sc_idx = idx
        if start_sc_idx is not None and stop_sc_idx is not None:
            break
    # last bound is not included
    return start_sc_idx, stop_sc_idx


def get_real_time_per_record(k15_data):
    """ Calculates time-per-record value in seconds
    for k15 data.

    WARNING! Time gaps in data are not handled!

    # TODO: handle time gaps in records

    :param k15_data: k15 data
    :type k15_data: np.ndarray
    :return: tuple with time-per-record value and error of its calculation in seconds
    :rtype: tuple of float
    """
    time_spent = k15_data[0, -1] - k15_data[0, 0]

    # number of records made during time_spent
    records_num = k15_data.shape[1] - 1
    real_time_per_record = time_spent / float(records_num)

    err_real_time_per_records = (time_spent + 2) / records_num - real_time_per_record

    return real_time_per_record, err_real_time_per_records


def check_bounds(start_k15_time, stop_k15_time, start_sc_time, stop_sc_time):
    start_deviation = start_sc_time - start_k15_time
    stop_deviation = stop_sc_time - stop_k15_time
    assert start_deviation >= 0, "Algorithm error"
    assert stop_deviation >= 0, "Algorithm error"
    if start_deviation > 0:
        print("Warning! SlowControl data starts {} seconds later than k15 data!"
              "".format(start_deviation))
    if stop_deviation > 0:
        print("Warning! SlowControl data ends {} seconds earlier than k15 data!"
              "".format(stop_deviation))


precompile()
