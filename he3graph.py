import sys
import os
import numpy as np

from matplotlib import pyplot as plt
from matplotlib import dates as md
import datetime


def convert_time(time, unixtime):
    if unixtime:
        return md.epoch2num(time)
    return time


def graph_all(data, mask=None, unixtime=True, labels=None, save_as=None, show_graph=False, scatter=False):
    if data.shape[1] == 1:
        scatter = True

    line_fmt = '-'
    if scatter:
        line_fmt = 's-'

    number_of_channels = data.shape[0] - 1
    seconds = convert_time(data[0], unixtime=unixtime)
    if mask is None:
        mask = [1] * number_of_channels

    if labels is None:
        labels = ["Ch{}".format(val + 1) for val in range(number_of_channels)]
    fig, ax = plt.subplots()

    for ch in range(number_of_channels):
        if mask[ch]:
            ax.plot(seconds, data[ch + 1], line_fmt, label=labels[ch].format(ch + 1))
            # ax.plot_date(seconds, data[ch + 1], fmt=line_fmt, tz=currentTimeZone, label=labels[ch].format(ch + 1))

    # Choose your xtick format string
    # date_fmt = '%d-%m-%y %H:%M:%S'
    date_fmt = '%H:%M:%S'

    # Use a DateFormatter to set the data to the correct format.
    local_time_zone = datetime.datetime.now().astimezone().tzinfo
    date_formatter = md.DateFormatter(date_fmt, tz=local_time_zone)
    ax.xaxis.set_major_formatter(date_formatter)

    # Sets the tick labels diagonal so they fit easier.
    fig.autofmt_xdate()

    ax.grid(True)
    plt.xlabel("Time")
    plt.ylabel("Counts")
    channels_list = [idx for idx in range(number_of_channels) if mask[idx]]
    plt.title("He³ detectors")

    ax.legend()  # легенда для всего рисунка fig
    if save_as is not None:
        # assert not os.path.isfile(save_as), "A file with name {} already exists.".format(os.path.basename(save_as))
        if not os.path.isdir(os.path.dirname(save_as)):
            os.makedirs(os.path.dirname(save_as))
        plt.savefig(save_as, dpi=300)
    if show_graph:
        plt.show()
    else:
        plt.close()
