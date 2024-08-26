import os


def read_txt(txt_path):
    with open(txt_path, "r") as f:
        lines = f.read().splitlines()
    return lines


def write_txt_by_lines(txt_path, lines, verbose=False):
    with open(txt_path, "w") as f:
        for line in lines:
            f.write("{}\n".format(line))
    if verbose:
        print("Writed to {} with {} lines".format(txt_path, len(lines)))


label_dir = "data/kitti_360/training/label_2"
out_dir = "data/kitti_360/training/label_2_converted"
os.makedirs(out_dir, exist_ok=True)

for txt_name in os.listdir(label_dir):
    txt_path = os.path.join(label_dir, txt_name)
    txt_lines = read_txt(txt_path)
    new_txt_lines = []
    for txt_line in txt_lines:
        new_txt_lines.append(txt_line.replace("Building", "Bus"))
    write_txt_by_lines(os.path.join(out_dir, txt_name), new_txt_lines, verbose=True)
