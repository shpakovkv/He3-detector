import os
from tools import file_processing
from tools import process_k15_and_sc
from tools import make_timeline_graph_grouped_by_4
from tools import check_time_graph_k15
from tools import check_time_graph_slow_control
from file_handler import get_file_list


PLUS_3_HOURS_SEC = 10800
PLUS_4_HOURS_SEC = 14400


def make_k15_only_2001_01_01():
    """
    Process k15 data,
    make k15 graphs,

    HOW TO USE:
    - change k15_dir (there should be no other files in the folder except k15 files with He3 data)

    Options:
    - you may change any function parameters as you need
    """
    k15_dir = "PATH\\TO\\K15_FILES\\FOLDER"

    k15_files = get_file_list(k15_dir, sort_by_name=True)

    # process k15 files
    for k15_file in k15_files:
        file_processing(k15_file,
                        filter128=True,
                        group_by_4=True,
                        group_by_sec=0,
                        save_data=False,
                        make_graph=True,
                        save_graph=True,
                        verbose=2)

    # time-per-record stability check graph
    check_time_graph_k15(k15_files, show=False, save=True)


def make_all_2001_01_01():
    """
    Process k15 data with SlowControl data,
    make k15 graphs,
    make k15 + SlowControl graphs,
    make TimeLine graph (all data except background on one graph)

    HOW TO USE:
    - change k15_dir (there should be no other files in the folder except k15 files with He3 data)
    - change sc_dir (there should be no other files in the folder except SlowControl files)
    - change save_timeline_as (path to folder and file name)

    Options:
    - change shift_k15_seconds if k15 data is shifted by a non-standart number of seconds
    - change loops borders ([:1] / [1:]) if the first file is not background data
    - you may change any other function parameters as you need
    """
    k15_dir = "PATH\\TO\\K15_FILES\\FOLDER"
    sc_dir = "PATH\\TO\\SLOW_CONTROL_FILES\\FOLDER"
    save_timeline_as = "PATH\\TO\\FOLDER\\2001_01_01_TimeLine"

    k15_files = get_file_list(k15_dir, sort_by_name=True)
    sc_files = get_file_list(sc_dir, sort_by_ctime=True)

    make_timeline_graph_grouped_by_4(k15_files[1:],
                                     sc_files,
                                     mask=0b101,        # use only first and third data column
                                     show=True,
                                     save_as=save_timeline_as,
                                     shift_k15_seconds=PLUS_3_HOURS_SEC)

    # process only first k15 file (usually background)
    for k15_file in k15_files[:1]:
        file_processing(k15_file,
                        filter128=True,
                        group_by_4=True,
                        group_by_sec=0,
                        save_data=False,
                        make_graph=True,
                        save_graph=True,
                        verbose=2)

    # process all files except first (usually background)
    print("============================================")
    for k15_file in k15_files[1:]:
        process_k15_and_sc(k15_file,
                           sc_files,
                           filter128=True,
                           group_by_4=True,
                           group_by_sec=0,
                           save_data=False,
                           make_graph=True,
                           save_graph=True,
                           show_graph=True,
                           verbose=1,
                           shift_k15_seconds=PLUS_3_HOURS_SEC
                           )

    # time-per-record stability check graph
    check_time_graph_slow_control(sc_files, show=False, save=True)
    check_time_graph_k15(k15_files, show=False, save=True)


def make_2022_10_03():
    path_k15 = "PATH\\TO\\K15_FILES\\FOLDER"
    path_sc = "PATH\\TO\\SLOW_CONTROL_FILES\\FOLDER"
    save_timeline_as = "PATH\\TO\\FOLDER\\2001_01_01_TimeLine"

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
                                         shift_k15_seconds=PLUS_4_HOURS_SEC)

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
                           shift_k15_seconds=PLUS_4_HOURS_SEC)

    if make_checks:
        check_time_graph_k15(file_list_k15, show=False, save=True)
        check_time_graph_slow_control(file_list_sc, show=False, save=True)


def process01(path_k15, path_sc, save_timeline_as, k15_time_shift,
              k15_group4=True, sc_group4=False, filter128=True, cut_list=None,
              force_timeline=False, show_graph=False, save_graph=True,
              check=True, number_of_bg_files=1, verbose=2):
    """
    Process k15 data with SlowControl data,
    make k15 graphs,
    make k15 + SlowControl graphs,
    make TimeLine graph (all data except background on one graph),
    (optional) make data skipping and record interval stability check graph.

    HOW TO USE:

    - specify path_k15 (there should be no other files in the folder except k15 files with He3 data)

    - specify path_sc (there should be no other files in the folder except SlowControl files)

    - specify save_timeline_as (path to folder and file name)

    - specify k15_time_shift - k15 and SLowControl records system time difference (in seconds)

    - (optional) specify any other option if the default settings do not suit you

    Options:

    :param path_k15: folder with only k15 data files (there must be no other files)
    :type path_k15: str
    :param path_sc: folder with only SlowControl data file(s) (there must be no other files)
    :type path_sc: str
    :param save_timeline_as: the filename (full path) under which you want to save the TImeLine graph
    :type save_timeline_as: str
    :param k15_time_shift: the difference in system time between linux machine with k15 registration system and
                           SlowControl registration machine
    :type k15_time_shift: int or float
    :param k15_group4: if true groups k15 registration channels by 4
                       (from first to last) for k15 graphs and stats
    :type k15_group4: bool
    :param sc_group4: if true groups k15 registration channels by 4
                      (from first to last) for k15 + SlowControl graphs and stats
    :type sc_group4: bool
    :param filter128: if true replaces all values (by channels)
                      with the remainder of the division by 128.
                      Calculates the error that the filtering process makes in channel statistics.
    :type filter128: bool
    :param cut_list: list of time intervals to be cut from the data
                     where each element is a list of string (interval example: ["09:00:00", "14:15:08"])
    :type cut_list: list or None
    :param force_timeline: force save TimeLine graph (even if it already exists)
    :type force_timeline: bool
    :param show_graph: shows all graph (blocking process) one by one.
                     You may zoom in zoom out this graph and save any zooming manual.
    :type show_graph: bool
    :param save_graph: saves all graphs if true
    :type save_graph: bool
    :param check: if true makes data skipping and record interval stability check graph
    :type check: bool
    :param number_of_bg_files: number of files with background data
                               (will be processed without SlowControl),
                               all bg-files must be at the beginning of the file list
    :type number_of_bg_files: int
    :param verbose: level of verbosity
    :type verbose: int
    :return: None
    :rtype: None
    """
    # mask for plotting each channel individually (channels 5-8 not used)
    k15_col_mask = [1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1]
    sc_col_mask = [1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1]

    if k15_group4:
        # mask for plotting k15 data
        # as a sum of 4 channels (channels 5-8 not used)
        k15_col_mask = [1, 0, 1]
    if sc_group4:
        # mask for plotting k15 with SlowControl data
        # as a sum of 4 channels (channels 5-8 not used)
        sc_col_mask = [1, 0, 1]

    file_list_k15 = get_file_list(path_k15, sort_by_name=True)
    file_list_sc = get_file_list(path_sc, sort_by_ctime=True)

    # make TimeLine graph once if not found
    timeline_exist = any(fname.startswith(save_timeline_as) for fname in get_file_list(os.path.dirname(save_timeline_as)))
    if not timeline_exist or force_timeline:
        make_timeline_graph_grouped_by_4(file_list_k15[1:],
                                         file_list_sc,
                                         mask=0b101,
                                         show=True,
                                         save_as=save_timeline_as,
                                         shift_k15_seconds=k15_time_shift)

    for filename in file_list_k15:
        file_processing(filename,
                        filter128=filter128,
                        group_by_4=k15_group4,
                        group_by_sec=0,
                        save_data=False,
                        make_graph=True,
                        save_graph=save_graph,
                        show_graph=show_graph,
                        col_mask=k15_col_mask,
                        cut_intervals=cut_list,
                        shift_k15_seconds=k15_time_shift,
                        verbose=0)

    for filename in file_list_k15[:number_of_bg_files]:
        file_processing(filename,
                        filter128=filter128,
                        group_by_4=sc_group4,
                        group_by_sec=0,
                        save_data=False,
                        make_graph=False,
                        save_graph=False,
                        show_graph=False,
                        col_mask=k15_col_mask,
                        cut_intervals=cut_list,
                        shift_k15_seconds=k15_time_shift,
                        verbose=verbose)

    print()
    print("=" * 50)

    # make k15 with SlowControl data graph
    # skip first file (usually background without SlowControl data)
    for filename in file_list_k15[number_of_bg_files:]:
        process_k15_and_sc(filename,
                           file_list_sc,
                           filter128=filter128,
                           group_by_4=sc_group4,
                           group_by_sec=0,
                           save_data=False,
                           make_graph=True,
                           save_graph=save_graph,
                           show_graph=show_graph,
                           verbose=verbose,
                           col_mask=sc_col_mask,
                           cut_intervals=cut_list,
                           shift_k15_seconds=k15_time_shift)

    if check:
        check_time_graph_k15(file_list_k15, show=False, save=True)
        check_time_graph_slow_control(file_list_sc, show=False, save=True)
