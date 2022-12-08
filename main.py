import os

from tools import process_k15_and_sc
from tools import file_processing
from tools import make_timeline_graph_grouped_by_4
from tools import check_time_graph_k15
from tools import check_time_graph_slow_control

from file_handler import get_file_list

from template import process01


PLUS_4_HOURS_SEC = 14400
PLUS_3_HOURS_SEC = 10800


def make_2021_01_01_tmp01():
    """
        Process k15 data with SlowControl data,
        make k15 graphs,
        make k15 + SlowControl graphs,
        make TimeLine graph (all data except background on one graph),
        (optional) make data skipping and record interval stability check graph.

        HOW TO USE:

        - make a copy of this function with different name (use the data acquisition date)

        - you may delete this docstring in your copy of this function to save space

        - specify path_k15 (there should be no other files in the folder except k15 files with He3 data)

        - specify path_sc (there should be no other files in the folder except SlowControl files)

        - specify save_timeline_as (path to folder and file name)

        - specify shift_k15_seconds - k15 and SLowControl records system time difference (in seconds)

        - (optional) specify any other option if the default settings do not suit you (see process01 docs)

        :return: None
        :rtype: None
    """

    path_k15 = "PATH\\TO\\K15_FILES\\FOLDER"
    path_sc = "PATH\\TO\\SLOW_CONTROL_FILES\\FOLDER"
    save_timeline_to = "PATH\\TO\\FOLDER\\2021_01_01_TimeLine"

    file_list_k15 = get_file_list(path_k15, sort_by_name=True)
    file_list_sc = get_file_list(path_sc, sort_by_ctime=True)

    make_timeline_graph_grouped_by_4(file_list_k15[1:],
                                     file_list_sc,
                                     mask=0b101,
                                     show=True,
                                     save_as=save_timeline_to,
                                     shift_k15_seconds=PLUS_3_HOURS_SEC)

    for filename in file_list_k15[:1]:
        file_processing(filename,
                        filter128=True,
                        group_by_4=True,
                        group_by_sec=0,
                        save_data=False,
                        make_graph=True,
                        save_graph=True,
                        show_graph=False,
                        verbose=1)

    for filename in file_list_k15[1:]:
        process_k15_and_sc(filename,
                           file_list_sc,
                           filter128=True,
                           group_by_4=True,
                           group_by_sec=0,
                           save_data=False,
                           make_graph=True,
                           save_graph=True,
                           show_graph=False,
                           verbose=1,
                           shift_k15_seconds=PLUS_3_HOURS_SEC)

    check_time_graph_k15(file_list_k15, show=False, save=True)
    check_time_graph_slow_control(file_list_sc, show=False, save=True)


def make_2021_01_01_tmp02():
    """
    Process k15 data with SlowControl data,
    make k15 graphs,
    make k15 + SlowControl graphs,
    make TimeLine graph (all data except background on one graph),
    (optional) make data skipping and record interval stability check graph.

    HOW TO USE:

    - make a copy of this function with different name (use the data acquisition date)

    - you may delete this docstring in your copy of this function to save space

    - specify path_k15 (there should be no other files in the folder except k15 files with He3 data)

    - specify path_sc (there should be no other files in the folder except SlowControl files)

    - specify save_timeline_as (path to folder and file name)

    - specify k15_time_shift - k15 and SLowControl records system time difference (in seconds)

    - (optional) specify any other option if the default settings do not suit you (see process01 docs)

    :return: None
    :rtype: None
    """
    path_k15 = "PATH\\TO\\K15_FILES\\FOLDER"
    path_sc = "PATH\\TO\\SLOW_CONTROL_FILES\\FOLDER"
    save_timeline_as = "PATH\\TO\\FOLDER\\2021_01_01_TimeLine"
    k15_time_shift = PLUS_3_HOURS_SEC

    verbose = 2
    force_group_ch_by_4 = True
    force_filter128 = True
    force_show_graph = False
    force_save_graph = True

    force_replace_timeline_graph = False

    # cut_out = [["", ""],
    #            ]

    cut_out = None

    make_checks = False

    # ============================================================================

    # mask for plotting each channel individually
    col_mask = [1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1]
    if force_group_ch_by_4:
        # mask for plotting a sum of 4 channels
        col_mask = [1, 0, 1]

    file_list_k15 = get_file_list(path_k15, sort_by_name=True)
    file_list_sc = get_file_list(path_sc, sort_by_ctime=True)

    if not os.path.isfile(save_timeline_as) or force_replace_timeline_graph:
        make_timeline_graph_grouped_by_4(file_list_k15[1:],
                                         file_list_sc,
                                         mask=0b101,
                                         show=True,
                                         save_as=save_timeline_as,
                                         shift_k15_seconds=k15_time_shift)

    for filename in file_list_k15[:1]:
        file_processing(filename,
                        filter128=force_filter128,
                        group_by_4=force_group_ch_by_4,
                        group_by_sec=0,
                        save_data=False,
                        make_graph=True,
                        save_graph=force_save_graph,
                        show_graph=force_show_graph,
                        col_mask=col_mask,
                        cut_intervals=cut_out,
                        verbose=verbose)

    for filename in file_list_k15[1:]:
        process_k15_and_sc(filename,
                           file_list_sc,
                           filter128=force_filter128,
                           group_by_4=force_group_ch_by_4,
                           group_by_sec=0,
                           save_data=False,
                           make_graph=True,
                           save_graph=force_save_graph,
                           show_graph=force_show_graph,
                           verbose=verbose,
                           col_mask=col_mask,
                           cut_intervals=cut_out,
                           shift_k15_seconds=k15_time_shift)

    if make_checks:
        check_time_graph_k15(file_list_k15, show=False, save=True)
        check_time_graph_slow_control(file_list_sc, show=False, save=True)


def make_2022_22_22():
    """Process k15 data with SlowControl data,
    make k15 graphs,
    make k15 + SlowControl graphs,
    make TimeLine graph (all data except background on one graph),
    (optional) make data skipping and record interval stability check graph.

    HOW TO USE:

    - make a copy of this function with different name (use the data acquisition date)

    - you may delete this docstring in your copy of this function to save space

    - specify path_k15 (there should be no other files in the folder except k15 files with He3 data)

    - specify path_sc (there should be no other files in the folder except SlowControl files)

    - specify save_timeline_as (path to folder and file name)

    - specify k15_time_shift - k15 and SLowControl records system time difference (in seconds)

    - (optional) specify any other option if the default settings do not suit you (see process01 docs)

    :return: None
    :rtype: None
    """
    path_k15 = "PATH\\TO\\K15_FILES\\FOLDER"
    path_sc = "PATH\\TO\\SLOW_CONTROL_FILES\\FOLDER"
    save_timeline_as = "PATH\\TO\\FOLDER\\2022_22_22_TimeLine"
    k15_time_shift = PLUS_4_HOURS_SEC

    verbose = 2
    k15_force_group = False
    sc_force_group = True
    force_filter128 = True
    force_show_graph = False
    force_save_graph = True
    force_replace_timeline_graph = False

    # cut_out = [["", ""],
    #            ]

    cut_out = None

    make_checks = False

    process01(path_k15,
              path_sc,
              save_timeline_as,
              k15_time_shift,
              k15_group4=k15_force_group,
              sc_group4=sc_force_group,
              filter128=force_filter128,
              cut_list=cut_out,
              force_timeline=force_replace_timeline_graph,
              show_graph=force_show_graph,
              save_graph=force_save_graph,
              check=make_checks,
              verbose=verbose)


if __name__ == "__main__":
    # make_2021_01_01_tmp02()
    # make_2021_01_01_tmp02()
    make_2022_22_22()
