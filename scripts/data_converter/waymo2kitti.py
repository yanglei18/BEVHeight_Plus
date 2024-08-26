import numpy as np
import sys
import os
import shutil
import re
import argparse

# stop python from writing so much bytecode
sys.dont_write_bytecode = True
sys.path.append(os.getcwd())
np.set_printoptions(suppress=True)

def parse_option():
    parser = argparse.ArgumentParser('Convert waymo dataset to standard kitti format', add_help=False)
    parser.add_argument('--source-root', type=str, default="data/waymo-org", help='root path to waymo dataset that has been parsed')
    parser.add_argument('--target-root', type=str, default="data/waymo-kitti", help='root path to waymo dataset in kitti format')
    args = parser.parse_args()
    return args

def mkdir_if_missing(directory, delete_if_exist=False):
    """
    Recursively make a directory structure even if missing.

    if delete_if_exist=True then we will delete it first
    which can be useful when better control over initialization is needed.
    """

    if delete_if_exist and os.path.exists(directory): shutil.rmtree(directory)

    # check if not exist, then make
    if not os.path.exists(directory):
        os.makedirs(directory)

def make_symlink_or_copy(src_path, intended_path, MAKE_SYMLINK = False):
    if not os.path.exists(intended_path):
        if MAKE_SYMLINK:
            os.symlink(src_path, intended_path)
        else:
            command = "cp " + src_path + " " + intended_path
            os.system(command)

def main(src_root, dest_root, split='train', imind=0):
    if split == 'train':
        split_name = 'training'
        org_file = 'data/waymo/waymo_train_org.txt'
        dst_file = os.path.join(dest_root, 'ImageSets/train.txt')
    else:
        split_name = 'validation'
        org_file = 'data/waymo/waymo_val_org.txt'
        dst_file = os.path.join(dest_root, 'ImageSets/val.txt')

    kitti_org = dict()
    kitti_org['cal'] = os.path.join(src_root, 'replace', 'calib')
    kitti_org['ims'] = os.path.join(src_root, 'replace', 'image_0')
    kitti_org['lab'] = os.path.join(src_root, 'replace', 'label_0')
    kitti_org['vel'] = os.path.join(src_root, 'replace', 'velodyne')
    
    kitti_dst = dict()
    kitti_dst['cal'] = os.path.join(dest_root, split_name, 'calib')
    kitti_dst['ims'] = os.path.join(dest_root, split_name, 'image_2')
    kitti_dst['lab'] = os.path.join(dest_root, split_name, 'label_2')
    kitti_dst['vel'] = os.path.join(dest_root, split_name, 'velodyne')

    # mkdirs
    mkdir_if_missing(kitti_dst['cal'])
    mkdir_if_missing(kitti_dst['ims'])
    mkdir_if_missing(kitti_dst['lab'])
    mkdir_if_missing(kitti_dst['vel'])
    mkdir_if_missing(os.path.join(dest_root, 'ImageSets'))

    # copy or link
    text_file_new = open(dst_file, 'w')
    text_file     = open(org_file, 'r')
    text_lines = text_file.readlines()
    text_file.close()

    for line in text_lines:
        parsed = line.strip().split(' ')#re.search('(\d+)', line)

        if parsed is not None:
            seg, id = parsed
            new_id = '{:06d}'.format(imind)

            org_calib_path = os.path.join(kitti_org['cal'].replace('replace', seg), id + '.txt')
            org_label_path = os.path.join(kitti_org['lab'].replace('replace', seg), id + '.txt')
            org_image_path = os.path.join(kitti_org['ims'].replace('replace', seg), id + '.png')
            org_velodyne_path = os.path.join(kitti_org['vel'].replace('replace', seg), id + '.bin')

            # If any of the calib/label/image is missing
            if not os.path.exists(org_calib_path) or not os.path.exists(org_label_path) or not os.path.exists(org_image_path) or not os.path.exists(org_velodyne_path):
                print("{}/{} not found ...".format(seg, id))
                imind += 1
                continue
            
            new_calib_path = os.path.join(kitti_dst['cal'], str(new_id) + '.txt')
            new_label_path = os.path.join(kitti_dst['lab'], str(new_id) + '.txt')
            new_image_path = os.path.join(kitti_dst['ims'], str(new_id) + '.png')
            new_velodyne_path = os.path.join(kitti_dst['vel'], str(new_id) + '.bin')

            make_symlink_or_copy(org_calib_path, new_calib_path)
            make_symlink_or_copy(org_label_path, new_label_path)
            make_symlink_or_copy(org_image_path, new_image_path)
            make_symlink_or_copy(org_velodyne_path, new_velodyne_path)

            # Labels are duplicated. Make them unique by writing to the same file
            # command = "sort -u " + new_label_path + " -o " + new_label_path
            # os.system(command)

            text_file_new.write(new_id + '\n')
            imind += 1

        if imind % 5000 == 0 or line == text_lines[-1]:
            print("{} images done...".format(imind))

    text_file_new.close()


if __name__ == "__main__":
    args = parse_option()
    source_root, target_root = args.source_root, args.target_root
    main(os.path.join(source_root, 'training_org'), target_root, 'train')
    main(os.path.join(source_root, 'validation_org'), target_root, 'val')
    print('Done')
