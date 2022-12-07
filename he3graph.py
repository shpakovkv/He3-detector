import os
from he3analysis import convert_time
from matplotlib import pyplot as plt
from matplotlib import dates as md
import datetime
import pytz


TIMEZONE = pytz.timezone("Etc/GMT+3")


def graph_k15(data, mask=None, unixtime=True, labels=None, save_as=None, show_graph=False, scatter=False):
    plt.close('all')
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
    date_formatter = md.DateFormatter(date_fmt, tz=TIMEZONE)
    ax.xaxis.set_major_formatter(date_formatter)

    # Sets the tick labels diagonal so they fit easier.
    fig.autofmt_xdate()

    ax.grid(True)
    plt.xlabel("Time")
    plt.ylabel("Counts")
    channels_list = [idx for idx in range(number_of_channels) if mask[idx]]
    plt.title("He³ detectors")

    ax.legend(loc='best')  # легенда для всего рисунка fig
    if save_as is not None:
        # assert not os.path.isfile(save_as), "A file with name {} already exists.".format(os.path.basename(save_as))
        if not os.path.isdir(os.path.dirname(save_as)):
            os.makedirs(os.path.dirname(save_as))
        plt.savefig(save_as, dpi=300)
    if show_graph:
        plt.show()
    else:
        plt.close()


def graph_k15_and_sc(data_k15, data_sc, data_sc_avg=None, mask=None, unixtime=True, labels=None, save_as=None, show_graph=False, scatter=False):
    subplot_title_size = 8
    plt.close('all')
    if data_k15.shape[1] == 1:
        scatter = True

    line_fmt = '-'
    if scatter:
        line_fmt = 's-'

    k15_channels = data_k15.shape[0] - 1
    if mask is None and labels is not None:
        mask = [0] * k15_channels
        for idx, label in enumerate(labels):
            if label:
                mask[idx] = 1

    if labels is None:
        labels = ["Ch{}".format(val + 1) for val in range(k15_channels)]

    fig, ax = plt.subplots(3, 1, sharex='all')

    ax[0].plot(convert_time(data_sc[0], unixtime=unixtime), data_sc[1], '-')
    if data_sc_avg is not None:
        ax[0].plot(convert_time(data_sc_avg[0], unixtime=unixtime), data_sc_avg[1], 's-', label='Voltage, V')
        avg_text = "Average {:.1f} V".format(data_sc_avg[1, 0])
        ax[0].set_title(avg_text, size=subplot_title_size)
    plt.setp(ax[0], ylabel='Voltage, V')

    ax[1].plot(convert_time(data_sc[0], unixtime=unixtime), data_sc[2], '-')
    if data_sc_avg is not None:
        ax[1].plot(convert_time(data_sc_avg[0], unixtime=unixtime), data_sc_avg[2], 's-', label='Current, mkA')
        avg_text = "Average {:.4f} mkA".format(data_sc_avg[2, 0])
        ax[1].set_title(avg_text, size=subplot_title_size)
    plt.setp(ax[1], ylabel='Current, mkA')

    seconds = convert_time(data_k15[0], unixtime=unixtime)
    for ch in range(k15_channels):
        if mask[ch]:
            ax[2].plot(seconds, data_k15[ch + 1], line_fmt, label=labels[ch].format(ch + 1))
            # ax.plot_date(seconds, data[ch + 1], fmt=line_fmt, tz=currentTimeZone, label=labels[ch].format(ch + 1))

    # Choose your xtick format string
    # date_fmt = '%d-%m-%y %H:%M:%S'
    date_fmt = '%H:%M:%S'

    # Use a DateFormatter to set the data to the correct format.
    local_time_zone = datetime.datetime.now().astimezone().tzinfo
    date_formatter = md.DateFormatter(date_fmt, tz=TIMEZONE)
    ax[2].xaxis.set_major_formatter(date_formatter)

    # Sets the tick labels diagonal so they fit easier.
    fig.autofmt_xdate()

    ax[0].grid(True)
    ax[1].grid(True)
    ax[2].grid(True)
    plt.xlabel("Time")
    plt.ylabel("Counts")
    channels_list = [idx for idx in range(k15_channels) if mask[idx]]
    fig.suptitle('He³ detectors', fontsize=14)
    # plt.title("He³ detectors")

    ax[2].legend(loc='best')
    # plt.subplots_adjust(hspace=0)
    if save_as is not None:
        # assert not os.path.isfile(save_as), "A file with name {} already exists.".format(os.path.basename(save_as))
        if not os.path.isdir(os.path.dirname(save_as)):
            os.makedirs(os.path.dirname(save_as))
        plt.savefig(save_as, dpi=300)
    if show_graph:
        plt.show()
    else:
        plt.close()
