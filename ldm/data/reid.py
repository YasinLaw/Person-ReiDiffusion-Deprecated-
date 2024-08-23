import copy
import json
import os
import random
from collections import defaultdict

import PIL
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, Sampler
from torchvision.transforms import transforms


class UniPedBase(Dataset):
    def __init__(self, json_file_path, data_root, flip_p=0.5, size=None):
        self.json_file_path = json_file_path
        self.data_root = data_root
        self.size = size

        with open(self.json_file_path, "r") as f:
            data = json.load(f)
            self.image_paths = [e["file_path"] for e in data]
            self.ids = [e["id"] for e in data]
            self.captions = [e["captions"] for e in data]

        self._length = len(self.image_paths)
        self.labels = {
            "id": [i for i in self.ids],
            "file_path": [os.path.join(self.data_root, l) for l in self.image_paths],
            "caption_1": [cap[0] for cap in self.captions],
            "caption_2": [cap[1] for cap in self.captions],
        }
        self.flip = transforms.RandomHorizontalFlip(p=flip_p)

    def __len__(self):
        return self._length

    def __getitem__(self, idx):
        item = dict((k, self.labels[k][idx]) for k in self.labels)
        image = Image.open(str(item["file_path"]))
        # default to score-sde preprocessing
        img = np.array(image).astype(np.uint8)
        image = Image.fromarray(img)
        if self.size is not None:
            image = image.resize((self.size, self.size), resample=PIL.Image.BICUBIC)
        else:
            image = image.resize((128, 384), resample=PIL.Image.BICUBIC)
        image = self.flip(image)
        image = np.array(image).astype(np.uint8)
        item["image"] = (image / 127.5 - 1.0).astype(np.float32)

        return item


class UniPedTrain(UniPedBase):
    def __init__(self, **kwargs):
        super().__init__(
            json_file_path="/home/luo/data/reid/uniped_train.json",
            data_root="/home/luo/data/reid/",
            **kwargs
        )


class UniPedValidation(UniPedBase):
    def __init__(self, **kwargs):
        super().__init__(
            json_file_path="/home/luo/data/reid/uniped_val.json",
            data_root="/home/luo/data/reid/",
            **kwargs
        )


class UniPedTest(UniPedBase):
    def __init__(self, **kwargs):
        super().__init__(
            json_file_path="/home/luo/data/reid/uniped_test.json",
            data_root="/home/luo/data/reid/",
            **kwargs
        )


class UniPedTrainAll(UniPedBase):
    def __init__(self, **kwargs):
        super().__init__(
            json_file_path="/home/luo/data/reid/uniped_all.json",
            data_root="/home/luo/data/reid/",
            **kwargs
        )


class CUHKTrain(UniPedBase):
    def __init__(self, **kwargs):
        super().__init__(
            json_file_path="/home/luo/data/reid/cuhk_train.json",
            data_root="/home/luo/data/reid/",
            **kwargs
        )


class CUHKTest(UniPedBase):
    def __init__(self, **kwargs):
        super().__init__(
            json_file_path="/home/luo/data/reid/cuhk_test.json",
            data_root="/home/luo/data/reid/",
            **kwargs
        )


class CUHKValidation(UniPedBase):
    def __init__(self, **kwargs):
        super().__init__(
            json_file_path="/home/luo/data/reid/cuhk_val.json",
            data_root="/home/luo/data/reid/",
            **kwargs
        )


class ICFGTrain(UniPedBase):
    def __init__(self, **kwargs):
        super().__init__(
            json_file_path="/home/luo/data/reid/icfg_train.json",
            data_root="/home/luo/data/reid/",
            **kwargs
        )


class ICFGTest(UniPedBase):
    def __init__(self, **kwargs):
        super().__init__(
            json_file_path="/home/luo/data/reid/icfg_test.json",
            data_root="/home/luo/data/reid/",
            **kwargs
        )


class RSTPTrain(UniPedBase):
    def __init__(self, **kwargs):
        super().__init__(
            json_file_path="/home/luo/data/reid/rstp_train.json",
            data_root="/home/luo/data/reid/",
            **kwargs
        )


class RSTPTest(UniPedBase):
    def __init__(self, **kwargs):
        super().__init__(
            json_file_path="/home/luo/data/reid/rstp_test.json",
            data_root="/home/luo/data/reid/",
            **kwargs
        )


class RSTPValidation(UniPedBase):
    def __init__(self, **kwargs):
        super().__init__(
            json_file_path="/home/luo/data/reid/rstp_val.json",
            data_root="/home/luo/data/reid/",
            **kwargs
        )


class RandomIdentitySampler(Sampler):
    """
    Randomly sample N identities, then for each identity,
    randomly sample K instances, therefore batch size is N*K.
    Args:
    - data_source (list): list of (img_path, pid, camid).
    - num_instances (int): number of instances per identity in a batch.
    - batch_size (int): number of examples in a batch.
    """

    def __init__(self, data_source, batch_size, num_instances=4):
        super(RandomIdentitySampler, self).__init__()
        self.data_source = data_source
        self.batch_size = batch_size
        self.num_instances = num_instances
        self.num_pids_per_batch = self.batch_size // self.num_instances
        self.index_dic = defaultdict(list)  # dict with list value
        # {783: [0, 5, 116, 876, 1554, 2041],...,}
        for index, p in enumerate(self.data_source):
            pid = p["id"]
            self.index_dic[pid].append(index)
        self.pids = list(self.index_dic.keys())

        # estimate number of examples in an epoch
        self.length = 0
        for pid in self.pids:
            idxs = self.index_dic[pid]
            num = len(idxs)
            if num < self.num_instances:
                num = self.num_instances
            self.length += num - num % self.num_instances

    def __iter__(self):
        batch_idxs_dict = defaultdict(list)

        for pid in self.pids:
            idxs = copy.deepcopy(self.index_dic[pid])
            if len(idxs) < self.num_instances:
                idxs = np.random.choice(idxs, size=self.num_instances, replace=True)
            random.shuffle(idxs)
            batch_idxs = []
            for idx in idxs:
                batch_idxs.append(idx)
                if len(batch_idxs) == self.num_instances:
                    batch_idxs_dict[pid].append(batch_idxs)
                    batch_idxs = []

        avai_pids = copy.deepcopy(self.pids)
        final_idxs = []

        while len(avai_pids) >= self.num_pids_per_batch:
            selected_pids = random.sample(avai_pids, self.num_pids_per_batch)
            for pid in selected_pids:
                batch_idxs = batch_idxs_dict[pid].pop(0)
                final_idxs.extend(batch_idxs)
                if len(batch_idxs_dict[pid]) == 0:
                    avai_pids.remove(pid)

        return iter(final_idxs)

    def __len__(self):
        return self.length
