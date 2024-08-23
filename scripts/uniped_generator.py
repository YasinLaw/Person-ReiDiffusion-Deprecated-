import itertools
import json
import os

datasets = ["cuhk", "icfg", "rstp"]
base_path = "/home/luo/data/reid"
cuhk_base_path = os.path.join(base_path, "CUHK-PEDES/imgs")
icfg_base_path = os.path.join(base_path, "ICFG-PEDES/imgs")
rstp_base_path = os.path.join(base_path, "RSTPReid/imgs")
cuhk_json_path = os.path.join(base_path, "CUHK-PEDES/reid_raw.json")
icfg_json_path = os.path.join(base_path, "ICFG-PEDES/ICFG-PEDES.json")
rstp_json_path = os.path.join(base_path, "RSTPReid/data_captions.json")
train_json_path = os.path.join(base_path, "uniped_train.json")
val_json_path = os.path.join(base_path, "uniped_val.json")
test_json_path = os.path.join(base_path, "uniped_test.json")
all_json_path = os.path.join(base_path, "uniped_all.json")


def dataset_json_path(dataset: str, phase: str):
    return os.path.join(base_path, f"{dataset}_{phase}.json")


num_cuhk = 13003
num_icfg = 4102
num_rstp = 4101

train_data = []
test_data = []
val_data = []

cuhk_train_data = []
cuhk_test_data = []
cuhk_val_data = []
icfg_train_data = []
icfg_test_data = []
rstp_train_data = []
rstp_test_data = []
rstp_val_data = []

all_data = []

train_count = 0

for dataset in datasets:
    if dataset == "cuhk":
        with open(cuhk_json_path, "r") as f:
            data = json.load(f)
            for person in data:
                split = person["split"]
                del person["processed_tokens"]
                all_data.append(person)
                person["file_path"] = os.path.join(cuhk_base_path, person["file_path"])
                if split == "train":
                    train_count += 1
                    train_data.append(person)
                    cuhk_train_data.append(person)
                elif split == "val":
                    val_data.append(person)
                    cuhk_val_data.append(person)
                elif split == "test":
                    test_data.append(person)
                    cuhk_test_data.append(person)
    elif dataset == "icfg":
        with open(icfg_json_path, "r") as f:
            data = json.load(f)
            for person in data:
                person["id"] += num_cuhk + 1
                split = person["split"]
                del person["processed_tokens"]
                person["captions"].append(person["captions"][0])
                person["file_path"] = os.path.join(icfg_base_path, person["file_path"])
                all_data.append(person)
                if split == "train":
                    train_count += 1
                    train_data.append(person)
                    icfg_train_data.append(person)
                elif split == "test":
                    test_data.append(person)
                    icfg_test_data.append(person)
    elif dataset == "rstp":
        with open(rstp_json_path, "r") as f:
            data = json.load(f)
            for person in data:
                person["id"] += num_icfg + num_cuhk + 1
                split = person["split"]
                person["file_path"] = os.path.join(rstp_base_path, person["img_path"])
                del person["img_path"]
                all_data.append(person)
                if split == "train":
                    train_count += 1
                    train_data.append(person)
                    rstp_train_data.append(person)
                elif split == "test":
                    test_data.append(person)
                    rstp_test_data.append(person)
                elif split == "val":
                    val_data.append(person)
                    rstp_val_data.append(person)

    else:
        raise NotImplementedError(dataset)

with open(train_json_path, "w") as f:
    json.dump(train_data, f, indent=4)
with open(val_json_path, "w") as f:
    json.dump(val_data, f, indent=4)
with open(test_json_path, "w") as f:
    json.dump(test_data, f, indent=4)

with open(dataset_json_path("cuhk", "train"), "w") as f:
    json.dump(cuhk_train_data, f, indent=4)
with open(dataset_json_path("cuhk", "test"), "w") as f:
    json.dump(cuhk_test_data, f, indent=4)
with open(dataset_json_path("cuhk", "val"), "w") as f:
    json.dump(cuhk_val_data, f, indent=4)

with open(dataset_json_path("icfg", "train"), "w") as f:
    json.dump(icfg_train_data, f, indent=4)
with open(dataset_json_path("icfg", "test"), "w") as f:
    json.dump(icfg_test_data, f, indent=4)

with open(dataset_json_path("rstp", "train"), "w") as f:
    json.dump(rstp_train_data, f, indent=4)
with open(dataset_json_path("rstp", "test"), "w") as f:
    json.dump(rstp_test_data, f, indent=4)
with open(dataset_json_path("rstp", "val"), "w") as f:
    json.dump(rstp_val_data, f, indent=4)

with open(all_json_path, "w") as f:
    json.dump(all_data, f, indent=4)

# print(len(list({v["id"]: v for v in train_data}.values())))
