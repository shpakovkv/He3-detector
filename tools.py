from k15reader import get_raw_lines, get_k15_data, get_slow_control_data
from he3graph import graph_k15, graph_k15_and_sc

from he3analysis import convert_time
from he3analysis import filter_128
from he3analysis import get_sum_by_number_of_channels
from he3analysis import get_average_by_time_interval
from he3analysis import get_base_output_fname
from he3analysis import write_data
from he3analysis import print_k15_rates
from he3analysis import print_sc_average
from he3analysis import get_sc_ibounds
from he3analysis import get_counting_rate
from he3analysis import get_128_ejections

import os
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import dates as md
import datetime


def file_processing(filename,
                    filter128=True,
                    group_by_4=False,
                    group_by_sec=0,
                    save_data=False,
                    make_graph=False,
                    save_graph=False,
                    show_graph=False,
                    verbose=1):
    if verbose > 0:
        print()
        print("Файл \"{}\"".format(os.path.basename(filename)))

    raw_lines = get_raw_lines(filename)
    data = get_k15_data(raw_lines)

    if filter128:
        filter_128(data)

    if group_by_4:
        data = get_sum_by_number_of_channels(data, 4)

    rates, err_rates, gaps = get_counting_rate(data)
    if verbose > 0:
        print_k15_rates(data, rates, err_rates, gaps, group_by_4, verbose)

    if group_by_sec > 0:
        data = get_average_by_time_interval(data, group_by_sec, include_tail=True, verbose=verbose)

    base_out_name = get_base_output_fname(filename, group_by_4, group_by_sec)
    if save_data:
        write_data(data, base_out_name, group_by_sec, verbose)

    if make_graph:
        make_k15_graph(data, group_by_4, group_by_sec, base_out_name, save_graph, show_graph, verbose)

    # graph_name = os.path.join(base_out_name, "graph_1-4_and_9-12", base_out_name)
    # graph_all(data_grouped_by_4, [1, 0, 1], labels=["Sum 1+2+3+4", "", "Sum 9+10+11+12"])
    # graph_all(data_grouped_by_4, [1, 0, 1], labels=["Ch1-4", "", "Ch9-12"])


def process_k15_and_sc(k15_file,
                       sc_file,
                       filter128=True,
                       group_by_4=False,
                       group_by_sec=0,
                       save_data=False,
                       make_graph=False,
                       save_graph=False,
                       show_graph=False,
                       verbose=1,
                       shift_k15_seconds=0
                       ):
    if verbose > 0:
        print()
        print("Файл \"{}\"".format(os.path.basename(k15_file)))
        print("Файл \"{}\"".format(os.path.basename(sc_file)))

    raw_lines = get_raw_lines(k15_file)
    data_k15 = get_k15_data(raw_lines, shift_k15_seconds)
    raw_lines = get_raw_lines(sc_file)
    data_sc = get_slow_control_data(raw_lines)

    if filter128:
        filter_128(data_k15)

    if group_by_4:
        data_k15 = get_sum_by_number_of_channels(data_k15, 4)

    start_sc_idx, stop_sc_idx = get_sc_ibounds(data_k15, data_sc)
    # DEBUG !!!!!!!!!!!!!!!!!!!!!!!!!!
    # start_sc_idx, stop_sc_idx = 4640, 5911

    data_sc = data_sc[:, start_sc_idx: stop_sc_idx]
    # print("SC data used from ")
    # print(datetime.datetime.fromtimestamp(int(data_sc[0, 0])).strftime('%H:%M:%S'), end="")
    # print(" until ")
    # print(datetime.datetime.fromtimestamp(int(data_sc[0, -1])).strftime('%H:%M:%S'))

    rates_sc, err_rates_sc, gaps_sc = get_counting_rate(data_sc)
    rates_k15, err_rates_k15, gaps_k15 = get_counting_rate(data_k15)
    if verbose > 0:
        print_k15_rates(data_k15, rates_k15, err_rates_k15, gaps_k15, group_by_4, verbose)
        print_sc_average(data_sc, rates_sc, err_rates_sc, gaps_sc, verbose)

    data_sc_average = None
    if group_by_sec > 0:
        data_k15 = get_average_by_time_interval(data_k15, group_by_sec, include_tail=True, verbose=verbose)
        data_sc_average = get_average_by_time_interval(data_sc, group_by_sec, include_tail=True, verbose=0)

    base_out_name = get_base_output_fname(k15_file, group_by_4, group_by_sec)
    if save_data:
        write_data(data_k15, base_out_name, group_by_sec, verbose)

    if make_graph:
        make_k15_graph(data_k15, group_by_4, group_by_sec, base_out_name, save_graph, show_graph, verbose)
        if data_sc_average is None:
            # rates == list of avgs for all channels
            cols = data_sc.shape[0]
            data_sc_average = np.ndarray(shape=(cols, 2), dtype=data_sc.dtype)
            data_sc_average[0, 0] = data_sc[0, 0]
            data_sc_average[0, 1] = data_sc[0, -1]
            # ch is the index of channel (curve)
            for ch in range(cols - 1):
                data_sc_average[ch + 1, 0] = rates_sc[ch]
                data_sc_average[ch + 1, 1] = rates_sc[ch]
        make_k15_and_sc_graph(data_k15, data_sc, data_sc_average,
                              group_by_4=group_by_4,
                              base_out_name=base_out_name,
                              save_graph=save_graph,
                              show_graph=show_graph,
                              verbose=verbose)


def make_k15_graph(data, group_by_4, group_by_sec, base_out_name, save_graph, show_graph, verbose):
    save_graph_as = None
    if save_graph:
        save_graph_as = base_out_name
    scatter_graph = False
    if group_by_sec:
        scatter_graph = True
    if group_by_4:
        if save_graph:
            save_graph_as += ".png"
        graph_k15(data, [1, 0, 1], labels=["Ch1-4", "", "Ch9-12"],
                  save_as=save_graph_as, scatter=scatter_graph, show_graph=show_graph)
    else:
        save_as = None
        if save_graph_as is not None:
            save_as = save_graph_as + "_Ch1-4.png"
        graph_k15(data, [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                  labels=["Ch1", "Ch2", "Ch3", "Ch4", "", "", "", "", "", "", "", ""],
                  save_as=save_as,
                  scatter=scatter_graph,
                  show_graph=show_graph)
        if save_graph_as is not None:
            save_as = save_graph_as + "_Ch9-12.png"
        graph_k15(data, [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
                  labels=["", "", "", "", "", "", "", "", "Ch9", "Ch10", "Ch11", "Ch12", ],
                  save_as=save_as,
                  scatter=scatter_graph,
                  show_graph=show_graph)
    if verbose > 1 and save_graph_as is not None:
        if group_by_4:
            print("График сохранен: {}".format(os.path.basename(save_graph_as)))
        else:
            print("График сохранен: {}".format(os.path.basename(save_graph_as + "_Ch1-4.png")))
            print("График сохранен: {}".format(os.path.basename(save_graph_as + "_Ch9-12.png")))
    if show_graph:
        plt.show()


def make_k15_and_sc_graph(data_k15, data_sc, data_sc_avg=None, group_by_4=False, base_out_name="", save_graph=False, show_graph=False, verbose=0):
    save_graph_as = None
    if save_graph:
        save_graph_as = os.path.dirname(base_out_name)
        save_graph_as = os.path.join(save_graph_as, "SlowControl_" + os.path.basename(base_out_name))
    if group_by_4:
        if save_graph:
            save_graph_as += ".png"
        # graph_k15(data, [1, 0, 1], labels=["Ch1-4", "", "Ch9-12"], save_as=save_graph_as, scatter=scatter_graph)
        graph_k15_and_sc(data_k15,
                         data_sc,
                         data_sc_avg,
                         labels=["Ch1-4", "", "Ch9-12"],
                         save_as=save_graph_as,
                         show_graph=show_graph)
    else:
        save_as = None
        if save_graph_as is not None:
            save_as = save_graph_as + "_Ch1-4.png"
        graph_k15_and_sc(data_k15, data_sc, data_sc_avg,
                         mask=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                         labels=["Ch1", "Ch2", "Ch3", "Ch4", "", "", "", "", "", "", "", ""],
                         save_as=save_as,
                         show_graph=show_graph)
        if save_graph_as is not None:
            save_as = save_graph_as + "_Ch9-12.png"
        graph_k15_and_sc(data_k15, data_sc, data_sc_avg,
                         mask=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
                         labels=["", "", "", "", "", "", "", "", "Ch9", "Ch10", "Ch11", "Ch12", ],
                         save_as=save_as,
                         show_graph=show_graph)
    if verbose > 1 and save_graph_as is not None:
        if group_by_4:
            print("График сохранен: {}".format(os.path.basename(save_graph_as)))
        else:
            print("График сохранен: {}".format(os.path.basename(save_graph_as + "_Ch1-4.png")))
            print("График сохранен: {}".format(os.path.basename(save_graph_as + "_Ch9-12.png")))
    if show_graph:
        plt.show()


def graph_timeline(data_all_k15, data_sc, mask, save_as, show, title):
    plt.close("all")
    fig, ax = plt.subplots(3, 1, sharex='all')

    if title is not None:
        fig.suptitle(title)

    plt.setp(ax[0], ylabel='Voltage, V')
    plt.setp(ax[1], ylabel='Current, mkA')
    plt.setp(ax[2], ylabel='Counts')

    dates_sc = [datetime.datetime.fromtimestamp(ts) for ts in data_sc[0]]

    ax[0].plot(dates_sc, data_sc[1], '-', label="Voltage", color="deepskyblue")
    ax[1].plot(dates_sc, data_sc[2], '-', label="Current", color="darkorange")

    dates_k15 = [datetime.datetime.fromtimestamp(ts) for ts in data_all_k15[0]]

    if mask & 0b001:
        ax[2].plot(dates_k15, data_all_k15[1], '-', label="Ch1-4", color="red")
    if mask & 0b010:
        ax[2].plot(dates_k15, data_all_k15[2], '-', label="Ch5-8", color="lime")
    if mask & 0b100:
        ax[2].plot(dates_k15, data_all_k15[3], '-', label="Ch9-12", color="blue")

    # Choose your xtick format string
    # date_fmt = '%d-%m-%y %H:%M:%S'
    date_fmt = '%H:%M:%S'

    # Use a DateFormatter to set the data to the correct format.
    local_time_zone = datetime.datetime.now().astimezone().tzinfo
    date_formatter = md.DateFormatter(date_fmt, tz=local_time_zone)
    ax[2].xaxis.set_major_formatter(date_formatter)

    # Sets the tick labels diagonal so they fit easier.
    fig.autofmt_xdate()
    for _ax in ax:
        _ax.grid(True)

    ax[2].legend(loc='best')

    if save_as is not None:
        if not os.path.isdir(os.path.dirname(save_as)):
            os.makedirs(os.path.dirname(save_as))
        plt.savefig(save_as, dpi=300)

    if show:
        plt.show()


def make_timeline_graph_grouped_by_4(k15_files, sc_files, mask=0b101, show=True, save_as=None, shift_k15_seconds=0):
    if not isinstance(k15_files, list):
        assert isinstance(k15_files, str), \
            "Wrong input type. Expected list os string or string, got " \
            "{} instead.".format(type(k15_files))
        k15_files = [k15_files]
    if not isinstance(sc_files, list):
        assert isinstance(sc_files, str), \
            "Wrong input type. Expected list os string or string, got " \
            "{} instead.".format(type(sc_files))
        sc_files = [sc_files]
    for fname in k15_files:
        assert os.path.isfile(fname), "File {} not found.".format(fname)
    for fname in sc_files:
        assert os.path.isfile(fname), "File {} not found.".format(fname)

    list_of_data_k15 = [None] * len(k15_files)
    list_of_128_ejection_k15 = [None] * len(k15_files)
    list_of_data_sc = [None] * len(sc_files)

    for idx in range(len(list_of_data_k15)):
        raw_lines = get_raw_lines(k15_files[idx])
        data = get_k15_data(raw_lines, shift_k15_seconds)
        data = get_sum_by_number_of_channels(data, 4)
        list_of_128_ejection_k15[idx] = get_128_ejections(data)
        filter_128(data)
        list_of_data_k15[idx] = data

    for idx in range(len(list_of_data_sc)):
        raw_lines = get_raw_lines(sc_files[idx])
        data = get_slow_control_data(raw_lines)
        list_of_data_sc[idx] = data

    sc_cols = list_of_data_sc[0].shape[0]
    sc_shape = (sc_cols, 1)
    single_nan_sc = np.zeros(shape=sc_shape, dtype=np.float64)
    single_nan_sc[0, 0] = 0.0
    for idx in range(1, single_nan_sc.shape[0]):
        single_nan_sc[idx, 0] = np.nan
    data_all_sc = list_of_data_sc[0].astype(np.float64)

    for data_sc in list_of_data_sc[1:]:
        data_all_sc = np.concatenate((data_all_sc, single_nan_sc, data_sc), axis=1)

    k15_cols = list_of_data_k15[0].shape[0]
    k15_shape = (k15_cols, 1)
    single_nan_k15 = np.empty(shape=k15_shape, dtype=np.float64)
    single_nan_k15[0, 0] = 0.0
    for idx in range(1, single_nan_k15.shape[0]):
        single_nan_k15[idx, 0] = np.nan
    data_all_k15 = list_of_data_k15[0].astype(np.float64)
    err_128_data_all_k15 = list_of_128_ejection_k15[0].astype(np.float64)

    for data_k15 in list_of_data_k15[1:]:
        data_all_k15 = np.concatenate((data_all_k15, single_nan_k15, data_k15), axis=1)

    for err_128_data in list_of_128_ejection_k15[1:]:
        err_128_data_all_k15 = np.concatenate((err_128_data_all_k15, single_nan_k15, err_128_data), axis=1)

    graph_timeline(data_all_k15, data_all_sc, mask, save_as, show, title="Experiment timeline")
    graph_timeline(err_128_data_all_k15, data_all_sc, mask, save_as + "_128", show, title="128 ejections only")


def time_step_graph(filename, datatype, show=True, save=False):
    raw_lines = get_raw_lines(filename)
    if datatype == "k15":
        data = get_k15_data(raw_lines)
    elif datatype == "sc":
        data = get_slow_control_data(raw_lines)
    else:
        raise TypeError("Use 'k15' type or 'sc' type. Unexpected "
                        "type '{}'.".format(datatype))
    assert os.path.isfile(filename), \
        "File not found ({})".format(filename)
    time_data = data[0, :]
    res = np.zeros(shape=time_data.shape, dtype=float)
    res[0] = np.nan
    prev = time_data[0]
    for idx in range(1, time_data.shape[0]):
        res[idx] = time_data[idx] - prev
        prev = time_data[idx]

    plt.close('all')
    plt.title(os.path.basename(filename), loc='center')
    plt.plot(res, '-', color="blue")
    if save:
        save_as = os.path.dirname(filename)
        if datatype == "k15":
            save_as = os.path.join(save_as, "Time_Step_" + os.path.basename(filename) + ".png")
        elif datatype == "sc":
            save_as = os.path.join(save_as, "Time_Step_SlowControl_" + os.path.basename(filename) + ".png")
        plt.savefig(save_as, dpi=300)
    if show:
        plt.show()


def check_time_graph_k15(file_list, show=True, save=False):
    if isinstance(file_list, str):
        tmp = list()
        tmp.append(file_list)
        file_list = tmp
    assert isinstance(file_list, list), \
        "Wrong input type. Expected list of st or str, got {} instead." \
        "".format(type(file_list))

    for fname in file_list:
        time_step_graph(fname, "k15", show=show, save=save)


def check_time_graph_slow_control(file_list, show=True, save=False):
    if isinstance(file_list, str):
        tmp = list()
        tmp.append(file_list)
        file_list = tmp
    assert isinstance(file_list, list), \
        "Wrong input type. Expected list of st or str, got {} instead." \
        "".format(type(file_list))

    for fname in file_list:
        time_step_graph(fname, "sc", show=show, save=save)
