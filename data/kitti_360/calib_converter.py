import numpy as np
import os
from tqdm import tqdm
# P2: 7.070493000000e+02 0.000000000000e+00 6.040814000000e+02 4.575831000000e+01 0.000000000000e+00 7.070493000000e+02 1.805066000000e+02 -3.454157000000e-01 0.000000000000e+00 0.000000000000e+00 1.000000000000e+00 4.981016000000e-03
# Tr_velo_to_cam: 6.927964000000e-03 -9.999722000000e-01 -2.757829000000e-03 -2.457729000000e-02 -1.162982000000e-03 2.749836000000e-03 -9.999955000000e-01 -6.127237000000e-02 9.999753000000e-01 6.931141000000e-03 -1.143899000000e-03 -3.321029000000e-01

def read_txt(txt_path):
    with open(txt_path, "r") as f:
        lines = f.read().splitlines()
    return lines



for dir_name in ["training", "testing"]:
    ori_dir = f"./data/kitti_360/{dir_name}/calib"
    out_dir = f"./data/kitti_360/{dir_name}/calib2"
    os.makedirs(out_dir, exist_ok=True)

    P2 = "552.554261 0.000000 682.049453 0.000000 0.000000 552.554261 238.769549 0.000000 0.000000 0.000000 1.000000 0.000000"
    Tr_velo_to_cam = np.array([[ 4.36118536e-02, -9.99037896e-01, -4.50446890e-03, 2.63182558e-01],
    [-9.12138516e-02,  5.07848386e-04, -9.95830966e-01, -1.03064952e-01],
    [ 9.94876090e-01,  4.38411379e-02, -9.11047496e-02, -8.29521210e-01]])
    Tr_velo_to_cam = Tr_velo_to_cam.reshape([-1]).tolist()
    Tr_velo_to_cam = " ".join([str(x) for x in Tr_velo_to_cam])
    calib_content =  "P2: {}\nTr_velo_to_cam: {}".format(P2, Tr_velo_to_cam)

    for calib_name in tqdm(os.listdir(ori_dir)):
        out_path = os.path.join(out_dir, calib_name)
        with open(out_path, "w") as f:
            f.write("{}".format(calib_content))
print("finished")
