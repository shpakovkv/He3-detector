#!/usr/bin/env python
# coding: utf-8

"""He3 readout system data analysis and statistics.

Author: Konstantin Shpakov, march 2021.
"""
import matplotlib.pyplot as plt
import numpy as np
from numba import njit, vectorize, float64
from k15reader import get_raw_lines, get_k15_data, get_slow_control_data
from file_handler import save_signals_csv
import os
import datetime
from matplotlib import dates as md
from statistics import stdev

DEFAULT_SEC_PER_RECORD = 1.025
ERR_COEF = 1.1

# minimum time step in the records
MIN_TIME_STEP = 1


def convert_time(time, unixtime):
    if unixtime:
        return md.epoch2num(time)
    return time


def print_k15_rates(data, group_by_4, verbose):
    rates, err_rates, gaps = get_counting_rate(data)
    rates_str = ",  ".join("{:.4f}".format(val) for val in rates)
    err_rates_str = ", ".join("±{:.4f}".format(err) for err in err_rates)
    if group_by_4:
        print("Средний счет (сумма по 4) [1/с] = [{}]".format(rates_str))
        print("Погрешность вычисления [1/с]    = [{}]".format(err_rates_str))
    else:
        print("Средний счет по каналам [1/с] = [{}]".format(rates_str))
        print("Погрешность вычисления [1/с]  = [{}]".format(err_rates_str))
    # print(", Погрешность [1/с] = {}".format(err_rates))

    print("Длительность регистрации: {} сек. Количество записей: {}."
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


def print_sc_average(data, verbose):
    rates, err_rates, gaps = get_counting_rate(data)

    rates_str = ",  ".join("{:.4f}".format(val) for val in rates)
    err_rates_str = ", ".join("±{:.4f}".format(err) for err in err_rates)
    print("Ср. напряжение[В], ток[мкА] = [{}]".format(rates_str))
    print("Среднее кв. отклонение      = [{}]".format(err_rates_str))
    # print(", Погрешность [1/с] = {}".format(err_rates))

    print("Длительность регистрации: {} сек. Количество записей: {}."
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
        base_out_name = os.path.join(base_out_name, fname + "_sum")
    else:
        base_out_name = os.path.join(base_out_name, "graph_1-4_and_9-12", fname)
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

    real_sec_per_record = duration / records_num

    res = list()
    std_dev = list()
    # adding the 1st record time
    duration += sec_per_record
    # adding the 1st record
    records_num += 1

    there_are_gaps = list()
    if real_sec_per_record <= sec_per_record * 1.1:
        for row in range(1, data.shape[0]):
            rate = sum(data[row, :]) / records_num
            res.append(rate)
            std_dev.append(stdev(data[row, :]))
    else:
        # there are gaps in the records
        intervals = list()
        start = 0
        for idx in range(1, records_num):
            if data[0, idx] - data[0, idx - 1] > sec_per_record + MIN_TIME_STEP:
                intervals.append([start, idx - 1])
                there_are_gaps.append(data[0, idx] - data[0, idx - 1])
                start = idx
        for row in range(1, data.shape[0]):
            rate = sum(data[row, :]) / records_num
            res.append(rate)
            std_dev.append(stdev(data[row, :]))
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


def filter_128(data):
    data[1:] = ll_filter_128(data[1:])


def print_overflow_128(data):
    data_filtered = np.copy(data)
    filter_128(data_filtered)
    fmt = "%d.%m.%Y %H:%M:%S"
    number_of_records = data.shape[1]
    for idx in range(number_of_records):
        if any(val > 127 for val in data[1:, idx]):
            print()
            msg = unix_datetime_to_str(data[0, idx])
            msg += "         "
            msg += ""
            msg += " ".join(space_padded_num(val, 3) for val in data[1:, idx])
            print(msg)

            msg = "Остаток от деления на 128:  "
            msg += " ".join(space_padded_num(val, 3) for val in data_filtered[1:, idx])
            print(msg)


def unix_datetime_to_str(utime, fmt=None):
    if fmt is None:
        fmt = "%Y_%m_%d %H:%M:%S"
    return datetime.datetime.fromtimestamp(utime).strftime(fmt)
    # return datetime.datetime.utcfromtimestamp(utime).strftime(fmt)


def space_padded_num(num, digits):
    msg = str(num)
    if len(msg) < digits:
        msg = " " * (digits - len(msg)) + msg
    return msg


def get_sc_ibounds(k15_data, sc_data):
    start_k15_time = k15_data[0, 0]
    stop_k15_time = k15_data[0, -1]
    assert start_k15_time < sc_data[0, -1], "Error! SlowControl data is written later than K15 data."
    assert stop_k15_time > sc_data[0, 0], \
        "Error! SlowControl data registration finished earlier than K15 data registration started."
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
