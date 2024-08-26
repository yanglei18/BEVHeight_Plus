import os
from tqdm import tqdm
import sys

def split_data(data, size):
    """ Split data according to world size

    Args:
        data (list): list of data to be split
        size (int): world

    Returns:
        data_list: list of data with length size
    """

    data_list = []
    slen = len(data) // size
    sleft = len(data) % size
    for idx in range(size):
        start = idx * slen
        if idx == size - 1:
            end = len(data) - sleft
        else:            end = (idx + 1) * slen
        data_list.append(data[start:end])
    for lidx, lval in enumerate(range(len(data) - sleft, len(data))):
        data_list[lidx].append(data[lval])
    return data_list


def read_txt(txt_path):
    with open(txt_path, "r") as f:
        lines = f.read().splitlines()
    return lines


def write_txt_by_lines(txt_path, lines, verbose=False):
    with open(txt_path, "w") as f:
        for idx, line in enumerate(lines):

            if idx < len(lines) - 1:
                f.write("{}\n".format(line))
            else:
                f.write("{}".format(line))
    if verbose:
        print("Writed to {} with {} lines".format(txt_path, len(lines)))

for dirname in ['training', 'testing']:
    data_dir = "data/kitti_360/testing/label"
    out_dir = "data/kitti_360/testing/label_2"
    os.makedirs(out_dir, exist_ok=True)
    local_rank = int(sys.argv[2])
    global_rank = int(sys.argv[4])
    anno_names = [x for x in os.listdir(data_dir) if x.endswith(".txt")]
    anno_names = split_data(anno_names, global_rank)[local_rank]


    # list through data_dir
    for anno_file in tqdm(anno_names):
        # make anno_path
        anno_path =  os.path.join(data_dir, anno_file)
        # read anno_path
        anno_lines = read_txt(anno_path)
        # make new_anno_lines
        new_anno_lines = []
        # iterate through anno_lines
        for anno_line in anno_lines:
            # split anno_line

            # fieldnames = [
            #     'type', 'truncated', 'occluded', 'alpha',
            #     'xmin', 'ymin', 'xmax', 'ymax',
            #     'dh', 'dw', 'dl', 'lx',
            #     'ly', 'lz', 'ry']

            # [{'class': 'Car',
            # 'label': 'Car',
            # 'truncated': 0.13,
            # 'occluded': 0.0,
            # 'alpha': -1.67,
            # 'dim': [-1.84, 2.05, 4.63],
            # 'loc': [2.14, -3.06, 30.69],
            # 'rot_y': -1.6}]

            anno_line = anno_line.split(" ")

            # make new_anno_line
            # anno_line[8]  = -float(anno_line[8])
            # anno_line[8]  = 0
            anno_line[12]  = -float(anno_line[12]) + float(anno_line[8])
            new_anno_lines.append(" ".join(map(str, anno_line)))

            # append new_anno_line to
        write_txt_by_lines(os.path.join(out_dir, anno_file), new_anno_lines)
print("finished.")
