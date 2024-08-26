import os

def read_file_strip_lines(file_path):  
    with open(file_path, 'r', encoding='utf-8') as file:  
        lines = [line.strip() for line in file]  
    return lines

def read_file(dir_path):
    files = []
    for file in os.listdir(dir_path):
        files.append(file)
    return files

def gen_waymo_org_txt(path_excess, txt_file, interval=2):
    file_excess = read_file(path_excess)
    excess_dict = {}
    for file in file_excess:
        ids = []
        for i in os.listdir(os.path.join(path_excess, file, 'label_0')):
            ids.append(i.split('.')[0])
        ids = sorted(ids)
        # print(ids)
        idx = []
        for i, id in enumerate(ids):
            if i == 0:
                idx.append(id)
            else:
                if int(id) - int(idx[-1]) > interval:
                    idx.append(id)
        excess_dict[file] = idx
        # print(ids)
        # print(file, excess_dict[file])
    print('excess files num in training: ', len(excess_dict))
    excess_dict = dict(sorted(excess_dict.items(), key=lambda item: item[0]))
    excess_txt = open(txt_file, 'w')
    for file, ids in excess_dict.items():
        for id in ids:
            excess_txt.write(file +' ' + str(id) + '\n')
    excess_txt.close()

if __name__ == "__main__":
    # path1_org = '/media/tsinghua-adept-03/898e2902-afae-441b-9a43-8c3bbde01321/waymo-2020/training'
    train_path = 'data/waymo-kitti/training_org'
    train_txt_file = 'data/waymo/waymo_train_org.txt'
    gen_waymo_org_txt(train_path, train_txt_file)
