import re
import os


def get_file_list(path, sort_by_name=False, sort_by_ctime=False):
    """Returns a list of all files
    contained in the folder (path).
    Each element of the returned list is a full path to the file

    path -- target directory.
    sort -- by default, the list of results is sorted
            by creation time.
    """
    target_files = [os.path.join(path, x) for x in os.listdir(path)
                    if os.path.isfile(os.path.join(path, x))]
    if sort_by_name:
        target_files.sort()
    if sort_by_ctime:
        target_files.sort(key=lambda x: os.path.getctime(x))
    return target_files


def save_signals_csv(filename, data, delimiter=",", integer=True):
    """Saves SignalsData to a CSV file.
    First three lines will be filled with header:
        1) the labels
        2) the curves units
        3) the time unit (only 1 column at this row)

    filename  -- the full path
    signals   -- SignalsData instance
    delimiter -- the CSV file delimiter
    precision -- the precision of storing numbers
    """
    # value_format = '%0.' + str(precision) + 'e'
    fmt = ':i'

    # turn row-oriented array to column-oriented
    # data = data.transpose()

    points = data.shape[1]
    rows = data.shape[0]

    with open(filename, 'w') as fid:
        lines = []
        # add data
        for point in range(points):
            if integer:
                s = delimiter.join(["{:d}".format(data[row, point]) for
                                    row in range(rows)])
            else:
                s = delimiter.join(["{:f}".format(data[row, point]) for
                                    row in range(rows)])
            s += "\n"
            s = re.sub(r'nan', '', s)
            lines.append(s)
        fid.writelines(lines)
