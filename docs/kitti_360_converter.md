# Data converter to kitti360 format

## Prepare
1. Download the [KITTI-360](http://www.cvlibs.net/datasets/kitti-360/) dataset
2. Download the processed KITTI-360 `train_val` and dummy `testing` [labels](https://drive.google.com/file/d/1h1VmHNdoIKRecJKANt1Wj_-nDNX_HCQG/view?usp=sharing). Extract them.
3. Arrange datasets as

```bash
├── data
│      └── kitti_360
│             ├── ImageSets
│             ├── KITTI-360
│             │      ├── calibration
│             │      ├── data_2d_raw
│             │      ├── data_3d_raw
│             │      ├── data_3d_boxes
│             │      └── data_poses
│             ├── train_val
│             │      ├── calib
│             │      ├── label
│             │      └── label_dota
│             └── testing
│                    ├── calib
│                    ├── label
│                    └── label_dota
│ ...
```

Next, link the corresponding images and split.

```bash
python data/kitti_360/setup_split.py
```


You should see the following structure with `61056` samples in each sub-folder of `train_val` split, and `910` samples in each
sub-folder of `testing` split.

```bash
SeaBird/PanopticBEV
├── data
│      └── kitti_360
│             ├── ImageSets
│             ├── train_val
│             │      ├── calib
│             │      ├── image
│             │      ├── label
│             │      ├── label_dota
│             ├── testing
│             │      ├── calib
│             │      ├── image
│             │      ├── label
│             │      ├── label_dota
```


## Usage
1. Convert calib

这个文件会将 data/kitti_360/training 和 data/kitti_360/testing 下的 calib 文件转换为 calib_2 文件。
```bash
python3 data/kitti_360/calib_converter.py
```

2. Convert bbox3d
这个文件会将label转换为label_2
```bash
python3 data/kitti_360/bbox_converter.py
```

3. 如果想将building转换为bus
   会将label_2的文件转换为 label_2_converted，自己重命名下使用
```bash
python3 data/kitti_360/building_to_bus.py
```

4. 软链到使用的名字
```bash
ln -s data/kitti_360/train_val data/kitti_360/training
ln -s data/kitti_360/training/label_2_converted data/kitti_360/training/label_2
```
