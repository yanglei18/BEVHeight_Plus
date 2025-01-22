import pickle
import random

def read_pkl(pkl_path):
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    return data

if __name__ == "__main__":
    train_pkl = "data/nuscenes/bevheight_plus_nuscenes_infos_train.pkl"
    val_pkl = "data/nuscenes/bevheight_plus_nuscenes_infos_val.pkl"
    test_pkl = "data/nuscenes/bevheight_plus_nuscenes_infos_test.pkl"    
    train_data = read_pkl(train_pkl)
    val_data = read_pkl(val_pkl)
    test_data = read_pkl(test_pkl)
    
    print(train_data.keys())
    print(train_data["metadata"])
    
    print(len(train_data["infos"]), train_data["infos"][0].keys())
    print(len(test_data["infos"]), test_data["infos"][0].keys())
    
    all_infos = train_data["infos"][6008:] + val_data["infos"] + test_data["infos"]
    random.shuffle(all_infos)

    all_data = {
        "metadata": train_data["metadata"],
        "infos": all_infos
    }
    
    all_pkl = "data/nuscenes/bevheight_plus_nuscenes_infos_random_all.pkl"
    
    with open(all_pkl,'wb') as fid:
       pickle.dump(all_data, fid)
    
    
    

    print("lidar_path: ")
    '''
    for tag in train_data["infos"][0].keys():
        print("tag: ", tag)
        print(train_data["infos"][0][tag])
        print(test_data["infos"][0][tag])
    '''
    
    
    