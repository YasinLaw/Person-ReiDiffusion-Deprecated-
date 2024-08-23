import argparse

import numpy as np
import torch
from PIL import Image
from omegaconf import OmegaConf
from tqdm.auto import tqdm

from ldm.modules.encoders.modules import CLIPReIDEmbedder
from ldm.util import instantiate_from_config


def evaluate(query_feature, query_label, gallery_features, gallery_labels):
    gallery_labels = np.array(gallery_labels)

    score = np.dot(gallery_features, query_feature)

    # index descending
    index = np.argsort(score)

    index = index[::-1]
    good_idx = np.argwhere(gallery_labels == query_label)

    junk_idx = np.argwhere(gallery_labels == -1)

    cmc_tmp = compute_mAP(index, good_idx, junk_idx)
    return cmc_tmp


def compute_mAP(index, good_index, junk_index):
    ap = 0
    cmc = torch.IntTensor(len(index)).zero_()
    if good_index.size == 0:  # if empty
        cmc[0] = -1
        return ap, cmc

    mask = np.in1d(index, junk_index, invert=True)
    index = index[mask]

    ngood = len(good_index)
    mask = np.in1d(index, good_index)
    rows_good = np.argwhere(mask == True)
    rows_good = rows_good.flatten()

    cmc[rows_good[0] :] = 1
    for i in range(ngood):
        d_recall = 1.0 / ngood
        precision = (i + 1) * 1.0 / (rows_good[i] + 1)
        if rows_good[i] != 0:
            old_precision = i * 1.0 / rows_good[i]
        else:
            old_precision = 1.0
        ap = ap + d_recall * (old_precision + precision) / 2.0

    return ap, cmc


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--reid",
        action="store_true",
        help="uses the reid model",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=64,
        help="how many samples to produce for each given prompt. A.k.a. batch size",
    )
    parser.add_argument("--dataset", "-d", type=str, default="cuhk")

    opt = parser.parse_args()

    if opt.reid:
        opt.config = "configs/latent-diffusion/eval.yaml"

    assert opt.dataset is not None and opt.dataset in ["cuhk", "icfg", "rstp"]

    config = OmegaConf.load(f"{opt.config}")

    data_module = None
    if opt.dataset == "cuhk":
        data_module = instantiate_from_config(config.cuhk)
    elif opt.dataset == "icfg":
        data_module = instantiate_from_config(config.icfg)
    elif opt.dataset == "rstp":
        data_module = instantiate_from_config(config.rstp)
    data_module.prepare_data()
    data_module.setup()

    # use test dataset to calculate mAP, rank1, rank5, rank10 and FID

    query_ids = []
    query_id_idx = []
    gallery_ids = []
    gallery_id_idx = []

    with torch.no_grad():
        dataloader = data_module._test_dataloader()

        # use only center loss
        # ckpt_path = "/home/luo/repo/stable-diffusion/logs/2024-03-27T17-59-54_only-text-center/checkpoints/last.ckpt"

        # use only triplet loss
        ckpt_path = "/home/luo/repo/stable-diffusion/logs/2024-04-24T12-12-56_clip_all/checkpoints/last.ckpt"

        embedder = CLIPReIDEmbedder.load_from_checkpoint(ckpt_path).to("cuda")
        # embedder = CLIPReIDEmbedder().to("cuda")
        embedder.eval()
        features = torch.zeros([len(dataloader.dataset), 512])

        for batch_idx, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
            base_idx = batch_idx * opt.n_samples

            for idx, i in enumerate(batch["id"]):
                i = i.item()
                if i not in query_ids:
                    query_ids.append(i)
                    query_id_idx.append((i, base_idx + idx))
                else:
                    gallery_ids.append(i)
                    gallery_id_idx.append((i, base_idx + idx))

            # c = embedder.get_image_features(batch)[0]
            c = embedder.get_text_features(batch)[0]

            ff = c
            features[
                base_idx : min(base_idx + opt.n_samples, len(dataloader.dataset)),
                :,
            ] = torch.flatten(ff, start_dim=1)

        query_idx = [t[-1] for t in query_id_idx]
        gallery_idx = [t[-1] for t in gallery_id_idx]
        query_features = torch.index_select(
            features, dim=0, index=torch.tensor(query_idx)
        ).numpy()
        gallery_features = torch.index_select(
            features, dim=0, index=torch.tensor(gallery_idx)
        ).numpy()

        cmc = torch.IntTensor(len(gallery_ids)).zero_()
        ap = 0.0

        for i in range(len(query_ids)):
            ap_tmp, cmc_tmp = evaluate(
                query_features[i],
                query_ids[i],
                gallery_features,
                gallery_ids,
            )
            if cmc_tmp[0] == -1:
                continue
            cmc = cmc + cmc_tmp
            ap += ap_tmp
            # print(i, torch.argwhere(1 == cmc_tmp))

        cmc = cmc.float()
        cmc = cmc / len(query_ids)  # average CMC

        print(
            f"RANK 1: {cmc[0]} RANK 5: {cmc[4]} RANK 10: {cmc[9]} mAP: {ap/len(query_ids)}"
        )
        # calculate indicators


if __name__ == "__main__":
    main()
