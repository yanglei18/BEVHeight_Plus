import numpy as np
import sys
import os
import shutil
import re

# stop python from writing so much bytecode
sys.dont_write_bytecode = True
sys.path.append(os.getcwd())
np.set_printoptions(suppress=True)

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

curr_folder = os.getcwd()
kitti_org_train = dict()
# kitti_org_train['cal'] = os.path.join(curr_folder, 'training_org', 'replace', 'calib')
# kitti_org_train['ims'] = os.path.join(curr_folder, 'training_org', 'replace', 'image_0')
# kitti_org_train['lab'] = os.path.join(curr_folder, 'training_org', 'replace', 'label_0')
kitti_org_train['cal'] = os.path.join(curr_folder, 'waymo-kitti', 'training_org', 'replace', 'calib')
kitti_org_train['ims'] = os.path.join(curr_folder, 'waymo-kitti', 'training_org', 'replace', 'image_0')
kitti_org_train['lab'] = os.path.join(curr_folder, 'waymo-kitti', 'training_org', 'replace', 'label_0')
kitti_org_train['vel'] = os.path.join(curr_folder, 'waymo-kitti', 'training_org', 'replace', 'velodyne')
# kitti_org_train['cal'] = os.path.join(curr_folder, 'waymo-kitti', 'excess_org', 'training_org', 'replace', 'calib')
# kitti_org_train['ims'] = os.path.join(curr_folder, 'waymo-kitti', 'excess_org', 'training_org', 'replace', 'image_0')
# kitti_org_train['lab'] = os.path.join(curr_folder, 'waymo-kitti', 'excess_org', 'training_org', 'replace', 'label_0')
# kitti_org_train['vel'] = os.path.join(curr_folder, 'waymo-kitti', 'excess_org', 'training_org', 'replace', 'velodyne')
# kitti_org_train['dep'] = os.path.join(curr_folder, 'training_org','replace', 'projected_points_0')


kitti_tra = dict()
split = "training"
# kitti_tra['cal'] = os.path.join(curr_folder, split, 'calib')
# kitti_tra['ims'] = os.path.join(curr_folder, split, 'image')
# kitti_tra['lab'] = os.path.join(curr_folder, split, 'label')
kitti_tra['cal'] = os.path.join(curr_folder, 'waymo-kitti', split, 'calib')
kitti_tra['ims'] = os.path.join(curr_folder, 'waymo-kitti', split, 'image')
kitti_tra['lab'] = os.path.join(curr_folder, 'waymo-kitti', split, 'label')
kitti_tra['vel'] = os.path.join(curr_folder, 'waymo-kitti', split, 'velodyne')
# kitti_tra['dep'] = os.path.join(curr_folder, split, 'depth')

kitti_org_val = dict()
# kitti_org_val['cal'] = os.path.join(curr_folder, 'validation_org', 'replace', 'calib')
# kitti_org_val['ims'] = os.path.join(curr_folder, 'validation_org', 'replace', 'image_0')
# kitti_org_val['lab'] = os.path.join(curr_folder, 'validation_org', 'replace', 'label_0')
kitti_org_val['cal'] = os.path.join(curr_folder, 'waymo-kitti', 'validation_org', 'replace', 'calib')
kitti_org_val['ims'] = os.path.join(curr_folder, 'waymo-kitti', 'validation_org', 'replace', 'image_0')
kitti_org_val['lab'] = os.path.join(curr_folder, 'waymo-kitti', 'validation_org', 'replace', 'label_0')
kitti_org_val['vel'] = os.path.join(curr_folder, 'waymo-kitti', 'validation_org', 'replace', 'velodyne')
# kitti_org_val['dep'] = os.path.join(curr_folder, 'validation_org','replace', 'projected_points_0')


kitti_val = dict()
split = "validation"
# kitti_val['cal'] = os.path.join(curr_folder, split, 'calib')
# kitti_val['ims'] = os.path.join(curr_folder, split, 'image')
# kitti_val['lab'] = os.path.join(curr_folder, split, 'label')
kitti_val['cal'] = os.path.join(curr_folder, 'waymo-kitti', split, 'calib')
kitti_val['ims'] = os.path.join(curr_folder, 'waymo-kitti', split, 'image')
kitti_val['lab'] = os.path.join(curr_folder, 'waymo-kitti', split, 'label')
kitti_val['vel'] = os.path.join(curr_folder, 'waymo-kitti', split, 'velodyne')

tra_org_file = os.path.join(curr_folder, 'ImageSets/train_org.txt')
# tra_org_file = os.path.join(curr_folder, 'excess_org.txt')
val_org_file = os.path.join(curr_folder, 'ImageSets/val_org.txt')
# tra_file     = os.path.join(curr_folder, 'ImageSets/train.txt')
# val_file     = os.path.join(curr_folder, 'ImageSets/val.txt')
tra_file = os.path.join(curr_folder, 'waymo-kitti', 'ImageSets/train.txt')
val_file = os.path.join(curr_folder, 'waymo-kitti', 'ImageSets/val.txt')

# mkdirs
mkdir_if_missing(kitti_tra['cal'])
mkdir_if_missing(kitti_tra['ims'])
mkdir_if_missing(kitti_tra['lab'])
# mkdir_if_missing(kitti_tra['dep'])
mkdir_if_missing(kitti_tra['vel'])

mkdir_if_missing(kitti_val['cal'])
mkdir_if_missing(kitti_val['ims'])
mkdir_if_missing(kitti_val['lab'])
# mkdir_if_missing(kitti_val['dep'])
mkdir_if_missing(kitti_val['vel'])
mkdir_if_missing(os.path.join(curr_folder, 'waymo-kitti', 'ImageSets'))

#===================================================================================================
# compare train
#===================================================================================================
def read_file_strip_lines(file_path):  
    with open(file_path, 'r', encoding='utf-8') as file:  
        lines = [line.strip() for line in file]  
    return lines

path1 = './training_files.txt'
path2 = './ImageSets/train_tfrecord.txt'
path3 = './validation_files.txt'
path4 = './ImageSets/val_tfrecord.txt'

files1 = read_file_strip_lines(path1)
# print(files1)
files2 = read_file_strip_lines(path2)
n = 0
m = 0
unexist_tra = []
for file in files2:
    if file in files1:
        n += 1
    else:
        # print('Existed error in training : ', file)
        unexist_tra.append(file)
        m += 1
print('Existed files num in training : ', n)
print('Not existed files num in training : ', m)

files3 = read_file_strip_lines(path3)
# print(files1)
files4 = read_file_strip_lines(path4)
n = 0
m = 0
unexist_val = []
for file in files4:
    if file in files3:
        n += 1
    else:
        # print('Existed error in validation : ', file)
        unexist_val.append(file)
        m += 1
print('Existed files num in validation : ', n)
print('Not existed files num in validation : ', m)


#===================================================================================================
# Link train
#===================================================================================================
print('=============== Linking train =======================')
print("=> Reading {}".format(tra_org_file))
print("=> Writing {}".format(tra_file))
text_file_new = open(tra_file, 'w')
# text_file_new = open(tra_file, 'a')
text_file     = open(tra_org_file, 'r')
text_lines = text_file.readlines()
text_file.close()


imind = 0
# imind = 31015
num = 0
num1 = 0
error_files = {}
for line in text_lines:
    parsed = line.strip().split(' ')#re.search('(\d+)', line)

    if parsed is not None:
        seg, id = parsed
        new_id = '{:06d}'.format(imind)

        org_calib_path = os.path.join(kitti_org_train['cal'].replace('replace', seg), id + '.txt')
        org_label_path = os.path.join(kitti_org_train['lab'].replace('replace', seg), id + '.txt')
        org_image_path = os.path.join(kitti_org_train['ims'].replace('replace', seg), id + '.png')
        org_velodyne_path = os.path.join(kitti_org_train['vel'].replace('replace', seg), id + '.bin')

        # If any of the calib/label/image is missing
        if not os.path.exists(org_calib_path) or not os.path.exists(org_label_path) or not os.path.exists(org_image_path) or not os.path.exists(org_velodyne_path):
            # print("{}/{} not found ...".format(seg, id))
            num += 1
            imind += 1
            if seg+'.tfrecord' not in unexist_tra:
                if seg not in error_files:
                    error_files[seg] = []
                error_files[seg].append(id)
                # print('Error: ', seg+'.tfrecord', unexist_tra)
                num1 += 1
            continue

        new_calib_path = os.path.join(kitti_tra['cal'], str(new_id) + '.txt')
        new_label_path = os.path.join(kitti_tra['lab'], str(new_id) + '.txt')
        new_image_path = os.path.join(kitti_tra['ims'], str(new_id) + '.png')
        new_velodyne_path = os.path.join(kitti_tra['vel'], str(new_id) + '.bin')

        make_symlink_or_copy(org_calib_path, new_calib_path)
        # make_symlink_or_copy(org_label_path, new_label_path, MAKE_SYMLINK = True)
        make_symlink_or_copy(org_label_path, new_label_path)
        make_symlink_or_copy(org_image_path, new_image_path)
        make_symlink_or_copy(org_velodyne_path, new_velodyne_path)
        # make_symlink_or_copy(os.path.join(kitti_org_train['dep'].replace('replace', seg), id + '.npy'), os.path.join(kitti_tra['dep'], str(new_id) + '.npy'))

        # Labels are duplicated. Make them unique by writing to the same file
        # command = "sort -u " + new_label_path + " -o " + new_label_path
        # os.system(command)

        text_file_new.write(new_id + '\n')
        imind += 1

    if imind % 5000 == 0 or line == text_lines[-1]:
        print("{} images done...".format(imind))

text_file_new.close()
print('no exsisted files num in train: ', num)
print('[Error] no exsisted files num in train: ', num1)
error_txt = open('error_train.txt', 'w')
# error_txt = open('error_excess.txt', 'w')
for file,ids in error_files.items():
    error_txt.write(file +' ' + str(ids) + '\n')
error_txt.close()
train_parsed_txt = open('train_parsed.txt', 'w')
# train_parsed_txt = open('excess_parsed.txt', 'w')
for file in os.listdir(os.path.join(curr_folder, 'waymo-kitti', 'training_org')):
# for file in os.listdir(os.path.join(curr_folder, 'waymo-kitti', 'excess_org', 'training_org')):
    train_parsed_txt.write(file + '\n')
train_parsed_txt.close()

#===================================================================================================
# Link val
#===================================================================================================
print('=============== Linking val =======================')
print("=> Reading {}".format(val_org_file))
print("=> Writing {}".format(val_file))
text_file_new = open(val_file, 'w')
text_file     = open(val_org_file, 'r')
text_lines    = text_file.readlines()
text_file.close()

imind = 0
num = 0
num1 = 0
error_files = {}
for line in text_lines:
    parsed = line.strip().split(' ')#re.search('(\d+)', line)

    if parsed is not None:
        seg, id = parsed
        new_id = '{:06d}'.format(imind)

        org_calib_path = os.path.join(kitti_org_val['cal'].replace('replace', seg), id + '.txt')
        org_label_path = os.path.join(kitti_org_val['lab'].replace('replace', seg), id + '.txt')
        org_image_path = os.path.join(kitti_org_val['ims'].replace('replace', seg), id + '.png')
        org_velodyne_path = os.path.join(kitti_org_val['vel'].replace('replace', seg), id + '.bin')

        # If any of the calib/label/image is missing
        if not os.path.exists(org_calib_path) or not os.path.exists(org_label_path) or not os.path.exists(org_image_path) or not os.path.exists(org_velodyne_path):
            # print("{}/{} not found ...".format(seg, id))
            num += 1
            imind += 1
            if seg+'.tfrecord' not in unexist_val:
                if seg not in error_files:
                    error_files[seg] = []
                error_files[seg].append(id)
                # print('Error: ', seg+'.tfrecord', unexist_val)
                num1 += 1
            continue

        new_calib_path = os.path.join(kitti_val['cal'], str(new_id) + '.txt')
        new_label_path = os.path.join(kitti_val['lab'], str(new_id) + '.txt')
        new_image_path = os.path.join(kitti_val['ims'], str(new_id) + '.png')
        new_velodyne_path = os.path.join(kitti_val['vel'], str(new_id) + '.bin')

        make_symlink_or_copy(org_calib_path, new_calib_path)
        # make_symlink_or_copy(org_label_path, new_label_path, MAKE_SYMLINK = True)
        make_symlink_or_copy(org_label_path, new_label_path)
        make_symlink_or_copy(org_image_path, new_image_path)
        make_symlink_or_copy(org_velodyne_path, new_velodyne_path)
        # make_symlink_or_copy(os.path.join(kitti_org_val['dep'].replace('replace', seg), id + '.npy'), os.path.join(kitti_val['dep'], str(new_id) + '.npy'))

        # Labels are duplicated. Make them unique by writing to the same file
        # command = "sort -u " + new_label_path + " -o " + new_label_path
        # os.system(command)

        text_file_new.write(new_id + '\n')
        imind += 1

    if imind % 5000 == 0 or line == text_lines[-1]:
        print("{} images done...".format(imind))

text_file_new.close()
print('Done')
print('no exsisted files num in val: ', num)
print('[Error] no exsisted files num in val: ', num1)
error_txt = open('error_val.txt', 'w')
for file,ids in error_files.items():
    error_txt.write(file +' ' + str(ids) + '\n')
error_txt.close()
val_parsed_txt = open('val_parsed.txt', 'w')
for file in os.listdir(os.path.join(curr_folder, 'waymo-kitti', 'validation_org')):
    val_parsed_txt.write(file + '\n')
val_parsed_txt.close()