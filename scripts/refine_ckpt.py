import  torch
from collections import OrderedDict

if __name__ == "__main__":
    ckpt_path = "ckpt/dair-v2x/bevheight_34.ckpt"
    checkpoint = torch.load(ckpt_path)
    print(checkpoint.keys())
    print(type(checkpoint["state_dict"]))

    refined_state_dict = OrderedDict()
    for k, v in  checkpoint["state_dict"].items():
        if "frustum" not in k:
            refined_state_dict[k] = v
    checkpoint = {
        "state_dict": refined_state_dict
    }
    torch.save(checkpoint, "ckpt/dair-v2x/bevheight_34_refined.ckpt")