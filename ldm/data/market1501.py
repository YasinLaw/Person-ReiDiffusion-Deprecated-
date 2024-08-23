import os

import cv2
import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class Market1501Base(Dataset):
    def __init__(self, data_path, is_train=True, remap_id=False):
        super().__init__()
        self.is_train = is_train
        self.data_path = data_path
        self.img_paths = [
            el for el in os.listdir(data_path) if "0000" not in el and "-1" not in el
        ]
        self.img_paths = [
            el for el in self.img_paths if os.path.splitext(el)[1] == ".jpg"
        ]

        self.lb_ids = [int(el.split("_")[0]) for el in self.img_paths]
        self.lb_cams = [int(el.split("_")[1][1]) for el in self.img_paths]
        self.img_paths = [os.path.join(data_path, el) for el in self.img_paths]

        # useful for sampler
        self.lb_img_dict = dict()
        self.lb_ids_uniq = set(self.lb_ids)
        lb_array = np.array(self.lb_ids)
        for lb in self.lb_ids_uniq:
            idx = np.where(lb_array == lb)[0]
            self.lb_img_dict.update({lb: idx})

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        return (
            self.img_paths[idx],
            sorted(self.lb_ids_uniq).index(self.lb_ids[idx]),
            self.lb_cams[idx],
        )


class Market1501Train(Market1501Base):
    def __init__(self):
        super().__init__(
            data_path="/home/luo/data/Market-1501-v15.09.15/bounding_box_train",
            is_train=True,
        )


class Market1501Test(Market1501Base):
    def __init__(self):
        super().__init__(
            data_path="/home/luo/data/Market-1501-v15.09.15/bounding_box_test",
            is_train=False,
        )


if __name__ == "__main__":
    ds = Market1501Base(
        "/home/luo/data/Market-1501-v15.09.15/bounding_box_train",
        is_train=True,
    )
    im, _, _ = ds[1]
    im = Image.open(im)
    im = np.array(im)
    cv2.imshow("market1501", im)
    cv2.waitKey(0)
