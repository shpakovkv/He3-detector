from tools import file_processing
from tools import process_k15_and_sc
from tools import make_timeline_graph_grouped_by_4
from tools import check_time_graph_k15
from tools import check_time_graph_slow_control
from k15reader import get_file_list


DEFAULT_K15_TIME_SHIFT_SEC = 10800


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
                                     shift_k15_seconds=DEFAULT_K15_TIME_SHIFT_SEC)

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
                           shift_k15_seconds=DEFAULT_K15_TIME_SHIFT_SEC
                           )

    # time-per-record stability check graph
    check_time_graph_slow_control(sc_files, show=False, save=True)
    check_time_graph_k15(k15_files, show=False, save=True)
