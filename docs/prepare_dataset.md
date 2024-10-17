# Dataset Setup

## 1. DAIR-V2X-I Dataset
#### 1.1. Download DAIR-V2X-I dataset from official [website](https://thudair.baai.ac.cn/index).
#### 1.2. Convert the dataset to KITTI format.
```
ln -s [single-infrastructure-side root] ./data/dair-v2x
python scripts/data_converter/dair2kitti.py --source-root data/dair-v2x-i --target-root data/dair-v2x-i-kitti
```

The directory will be as follows.
```
BEVHeight_Plus
├── data
│   ├── dair-v2x-i
│   │   ├── velodyne
│   │   ├── image
│   │   ├── calib
│   │   ├── label
|   |   └── data_info.json
|   ├── dair-v2x-i-kitti
|   |   ├── training
|   |   |   ├── calib
|   |   |   ├── label_2
|   |   |   └── images_2
|   |   └── ImageSets
|   |        ├── train.txt
|   |        └── val.txt
|   |...
|...
```
#### 1.3. Prepare infos for DAIR-V2X-I dataset.
```
python scripts/gen_info_dair.py
```


## 2. Rope3D Dataset
#### 2.1. Download Rope3D dataset from official [website](https://thudair.baai.ac.cn/index).
#### 2.2. Convert the dataset to KITTI format.
```
ln -s [rope3d root] ./data/rope3d
python scripts/data_converter/rope2kitti.py --source-root data/rope3d --target-root data/rope3d-kitti
```
The directory will be as follows.
```
BEVHeight_Plus
├── data
|   ├── rope3d
|   |   ├── training
|   |   ├── validation
|   |   ├── training-image_2a
|   |   ├── training-image_2b
|   |   ├── training-image_2c
|   |   ├── training-image_2d
|   |   └── validation-image_2
|   ├── rope3d-kitti
|   |   ├── training
|   |   |   ├── calib
|   |   |   ├── denorm
|   |   |   ├── label_2
|   |   |   └── images_2
|   |   └── map_token2id.json
|   |...  
├── ...
```
#### 2.3. Prepare infos for Rope3D dataset.
```
python scripts/gen_info_rope3d.py
```

## 3. KITTI Dataset
#### 3.1. Download KITTI dataset from official [website](https://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d).

#### 3.2. Prepare infos for KITTI dataset.
```
python scripts/gen_info_kitti.py --data_root data/kitti
```

## 4. KITTI-360 Dataset
#### 4.1. Download KITTI-360 dataset from official [website](https://www.cvlibs.net/datasets/kitti-360/).
#### 4.2. Download the processed KITTI-360 `train_val` and dummy `testing` [labels](https://drive.google.com/file/d/1h1VmHNdoIKRecJKANt1Wj_-nDNX_HCQG/view?usp=sharing). Extract them.

#### 4.3. Convert the dataset to KITTI format.
```
python data/kitti-360/calib_converter.py
python data/kitti-360/bbox_converter.py

# creat soft link
ln -s data/kitti-360/train_val data/kitti-360/training
ln -s data/kitti-360/training/label_2_converted data/kitti-360/training/label_2
```
The directory will be as follows.
```
BEVHeight_Plus
├── data
│      └── kitti-360
│             ├── ImageSets
│             ├── KITTI-360
│             │      ├── calibration
│             │      ├── data_2d_raw
│             │      ├── data_2d_semantics
│             │      ├── data_3d_boxes
│             │      └── data_poses
│             ├── train_val
│             │      ├── calib
│             │      ├── label
│             │      └── images_2
│             ├── training
│             │      ├── calib
│             │      ├── label
│             │      └── images_2
│             └── testing
│                    ├── calib
│                    ├── label
│                    └── images_2
│ ...
```

#### 4.4. Prepare infos for KITTI-360 dataset.
```
python scripts/gen_info_kitti.py --data_root data/kitti-360
```

## 5. Waymo Dataset
#### 5.1. Download Waymo dataset from official [website](https://waymo.com/open/download/).

#### 5.2. Convert the dataset to KITTI format.

Set up environment
```
# Decompress the Waymo zip files into their corresponding directories
ls *.tar | xargs -i tar xvf {} -C your_target_dir
# Set up environment
conda create -n py36_waymo_tf python=3.7
conda activate py36_waymo_tf
conda install cudatoolkit=11.3 -c pytorch
# Newer versions of tf are not in conda. tf>=2.4.0 is compatible with conda.
pip install tensorflow-gpu==2.4
conda install pandas
pip3 install waymo-open-dataset-tf-2-4-0 --user
```

Parse Waymo dataset 
```
ln -s [waymo root] ./data/waymo/raw_data
conda activate py36_waymo_tf
python scripts/data_converter/converter.py --load_dir data/waymo --save_dir data/waymo/parse_data --split training   --num_proc 10
python scripts/data_converter/converter.py --load_dir data/waymo --save_dir data/waymo/parse_data --split validation   --num_proc 10
```

Convert Waymo dataset parsed to KITTI format
```
python scripts/data_converter/waymo2kitti.py --source-root data/waymo/parse_data --target-root data/waymo-kitti
```
```
BEVHeight_Plus
├── data
│   ├── waymo
│   │   ├── ImageSets
│   │   │   ├── train_tfrecord.txt
│   │   │   ├── val_tfrecord.txt
│   │   ├── raw_data
│   │   │   ├── training
│   │   │   │   ├── segment-10017090168044687777_6380_000_6400_000_with_camera_labels.tfrecord
│   │   │   ├── validation
│   │   │   │   ├── segment-10203656353524179475_7625_000_7645_000_with_camera_labels.tfrecord
│   │   ├── parse_data
│   │   │   ├── training_org
│   │   │   │   ├── segment-id
│   │   │   ├── validation_org
│   │   │   │   ├── segment-id
│   │   ├── waymo_train_org.txt
│   │   ├── waymo_val_org.txt
│   ├── waymo-kitti
│   │   ├── ImageSets
│   │   ├── training
│   │   ├── validation
```
#### 5.3. Prepare infos for Waymo dataset.
```
python scripts/gen_info_kitti.py --data_root data/waymo-kitti
```

## 6. Visualize the dataset in KITTI format
```
python scripts/data_converter/visual_tools.py --data_root data/waymo-kitti --demo_dir ./demo
```
