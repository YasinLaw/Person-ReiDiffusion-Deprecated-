from functools import partial

import clip
import pytorch_lightning as pl
import schedulefree
import torch
import torch.nn as nn
from PIL import Image
from torch.nn import CrossEntropyLoss
from torch.optim.lr_scheduler import LambdaLR
from transformers import CLIPTokenizer, CLIPTextModel

from ldm.modules.x_transformer import (
    Encoder,
    TransformerWrapper,
)
from ldm.util import instantiate_from_config


class AbstractEncoder(pl.LightningModule):
    def __init__(self):
        super().__init__()

    def encode(self, *args, **kwargs):
        raise NotImplementedError


class ClassEmbedder(nn.Module):
    def __init__(self, embed_dim, n_classes=1000, key="class"):
        super().__init__()
        self.key = key
        self.embedding = nn.Embedding(n_classes, embed_dim)

    def forward(self, batch, key=None):
        if key is None:
            key = self.key
        # this is for use in crossattn
        c = batch[key][:, None]
        c = self.embedding(c)
        return c


class TransformerEmbedder(AbstractEncoder):
    """Some transformer encoder layers"""

    def __init__(self, n_embed, n_layer, vocab_size, max_seq_len=77, device="cuda"):
        super().__init__()
        self.device = device
        self.transformer = TransformerWrapper(
            num_tokens=vocab_size,
            max_seq_len=max_seq_len,
            attn_layers=Encoder(dim=n_embed, depth=n_layer),
        )

    def forward(self, tokens):
        tokens = tokens.to(self.device)  # meh
        z = self.transformer(tokens, return_embeddings=True)
        return z

    def encode(self, x):
        return self(x)


class BERTTokenizer(AbstractEncoder):
    """Uses a pretrained BERT tokenizer by huggingface. Vocab size: 30522 (?)"""

    def __init__(self, vq_interface=True, max_length=77):
        super().__init__()
        from transformers import BertTokenizerFast

        self.tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
        self.vq_interface = vq_interface
        self.max_length = max_length

    def forward(self, text):
        batch_encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            return_length=True,
            return_overflowing_tokens=False,
            padding="max_length",
            return_tensors="pt",
        )
        tokens = batch_encoding["input_ids"].to(self.device)
        return tokens

    @torch.no_grad()
    def encode(self, text):
        tokens = self(text)
        if not self.vq_interface:
            return tokens
        return None, None, [None, None, tokens]

    def decode(self, text):
        return text


class BERTEmbedder(AbstractEncoder):
    """Uses the BERT tokenizr model and add some transformer encoder layers"""

    def __init__(
        self,
        n_embed,
        n_layer,
        vocab_size=30522,
        max_seq_len=77,
        use_tokenizer=True,
        embedding_dropout=0.0,
    ):
        super().__init__()
        self.use_tknz_fn = use_tokenizer
        if self.use_tknz_fn:
            self.tknz_fn = BERTTokenizer(vq_interface=False, max_length=max_seq_len)
        self.transformer = TransformerWrapper(
            num_tokens=vocab_size,
            max_seq_len=max_seq_len,
            attn_layers=Encoder(dim=n_embed, depth=n_layer),
            emb_dropout=embedding_dropout,
        )

    def forward(self, text):
        if self.use_tknz_fn:
            tokens = self.tknz_fn(text)  # .to(self.device)
        else:
            tokens = text
        z = self.transformer(tokens, return_embeddings=True)
        return z

    def encode(self, text):
        # output of length 77
        return self(text)


class SpatialRescaler(nn.Module):
    def __init__(
        self,
        n_stages=1,
        method="bilinear",
        multiplier=0.5,
        in_channels=3,
        out_channels=None,
        bias=False,
    ):
        super().__init__()
        self.n_stages = n_stages
        assert self.n_stages >= 0
        assert method in [
            "nearest",
            "linear",
            "bilinear",
            "trilinear",
            "bicubic",
            "area",
        ]
        self.multiplier = multiplier
        self.interpolator = partial(torch.nn.functional.interpolate, mode=method)
        self.remap_output = out_channels is not None
        if self.remap_output:
            print(
                f"Spatial Rescaler mapping from {in_channels} to {out_channels} channels after resizing."
            )
            self.channel_mapper = nn.Conv2d(in_channels, out_channels, 1, bias=bias)

    def forward(self, x):
        for stage in range(self.n_stages):
            x = self.interpolator(x, scale_factor=self.multiplier)

        if self.remap_output:
            x = self.channel_mapper(x)
        return x

    def encode(self, x):
        return self(x)


class FrozenCLIPEmbedder(AbstractEncoder):
    """Uses the CLIP transformer encoder for text (from Hugging Face)"""

    def __init__(
        self, version="openai/clip-vit-large-patch14", device="cuda", max_length=77
    ):
        super().__init__()
        self.tokenizer = CLIPTokenizer.from_pretrained(version)
        self.transformer = CLIPTextModel.from_pretrained(version)
        self.device = device
        self.max_length = max_length
        self.freeze()

    def freeze(self):
        self.transformer = self.transformer.eval()
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, text):
        batch_encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            return_length=True,
            return_overflowing_tokens=False,
            padding="max_length",
            return_tensors="pt",
        )
        tokens = batch_encoding["input_ids"].to(self.device)
        outputs = self.transformer(input_ids=tokens)

        z = outputs.last_hidden_state
        return z

    def encode(self, text):
        return self(text)


def normalize(x, axis=-1):
    """Normalizing to unit length along the specified dimension.
    Args:
      x: pytorch Variable
    Returns:
      x: pytorch Variable, same shape as input
    """
    x = 1.0 * x / (torch.norm(x, 2, axis, keepdim=True).expand_as(x) + 1e-12)
    return x


def euclidean_dist(x, y):
    """
    Args:
      x: pytorch Variable, with shape [m, d]
      y: pytorch Variable, with shape [n, d]
    Returns:
      dist: pytorch Variable, with shape [m, n]
    """
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)  # B, B
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    dist = dist - 2 * torch.matmul(x, y.t())
    # dist.addmm_(1, -2, x, y.t())
    dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
    return dist


def hard_example_mining(dist_mat, labels, return_inds=False):
    """For each anchor, find the hardest positive and negative sample.
    Args:
      dist_mat: pytorch Variable, pair wise distance between samples, shape [N, N]
      labels: pytorch LongTensor, with shape [N]
      return_inds: whether to return the indices. Save time if `False`(?)
    Returns:
      dist_ap: pytorch Variable, distance(anchor, positive); shape [N]
      dist_an: pytorch Variable, distance(anchor, negative); shape [N]
      p_inds: pytorch LongTensor, with shape [N];
        indices of selected hard positive samples; 0 <= p_inds[i] <= N - 1
      n_inds: pytorch LongTensor, with shape [N];
        indices of selected hard negative samples; 0 <= n_inds[i] <= N - 1
    NOTE: Only consider the case in which all labels have same num of samples,
      thus we can cope with all anchors in parallel.
    """
    assert len(dist_mat.size()) == 2
    assert dist_mat.size(0) == dist_mat.size(1)
    N = dist_mat.size(0)

    # shape [N, N]
    is_pos = labels.expand(N, N).eq(labels.expand(N, N).t())
    is_neg = labels.expand(N, N).ne(labels.expand(N, N).t())

    # `dist_ap` means distance(anchor, positive)
    # both `dist_ap` and `relative_p_inds` with shape [N, 1]
    dist_ap, relative_p_inds = torch.max(
        dist_mat[is_pos].contiguous().view(N, -1), 1, keepdim=True
    )
    # print(dist_mat[is_pos].shape)
    # `dist_an` means distance(anchor, negative)
    # both `dist_an` and `relative_n_inds` with shape [N, 1]
    dist_an, relative_n_inds = torch.min(
        dist_mat[is_neg].contiguous().view(N, -1), 1, keepdim=True
    )
    # shape [N]
    dist_ap = dist_ap.squeeze(1)
    dist_an = dist_an.squeeze(1)

    if return_inds:
        # shape [N, N]
        ind = (
            labels.new()
            .resize_as_(labels)
            .copy_(torch.arange(0, N).long())
            .unsqueeze(0)
            .expand(N, N)
        )
        # shape [N, 1]
        p_inds = torch.gather(
            ind[is_pos].contiguous().view(N, -1), 1, relative_p_inds.data
        )
        n_inds = torch.gather(
            ind[is_neg].contiguous().view(N, -1), 1, relative_n_inds.data
        )
        # shape [N]
        p_inds = p_inds.squeeze(1)
        n_inds = n_inds.squeeze(1)
        return dist_ap, dist_an, p_inds, n_inds

    return dist_ap, dist_an


class TripletLoss(object):
    """
    Triplet loss using HARDER example mining,
    modified based on original triplet loss using hard example mining
    """

    def __init__(self, margin=None, hard_factor=0.0):
        self.margin = margin
        self.hard_factor = hard_factor
        if margin is not None:
            self.ranking_loss = nn.MarginRankingLoss(margin=margin)
        else:
            self.ranking_loss = nn.SoftMarginLoss()

    def __call__(self, global_feat, labels, normalize_feature=False):
        if normalize_feature:
            global_feat = normalize(global_feat, axis=-1)
        dist_mat = euclidean_dist(global_feat, global_feat)  # B,B
        dist_ap, dist_an = hard_example_mining(dist_mat, labels)

        dist_ap *= 1.0 + self.hard_factor
        dist_an *= 1.0 - self.hard_factor

        y = dist_an.new().resize_as_(dist_an).fill_(1)
        if self.margin is not None:
            loss = self.ranking_loss(dist_an, dist_ap, y)
        else:
            loss = self.ranking_loss(dist_an - dist_ap, y)
        return loss, dist_ap, dist_an


class SupConLoss(pl.LightningModule):
    def __init__(self, device=None):
        super(SupConLoss, self).__init__()
        self.temperature = 1.0

    def forward(self, text_features, image_features, t_label, i_labels):
        batch_size = text_features.shape[0]
        batch_size_N = image_features.shape[0]
        mask = torch.eq(
            t_label.unsqueeze(1).expand(batch_size, batch_size_N),
            i_labels.unsqueeze(0).expand(batch_size, batch_size_N),
        ).to(self.device)

        logits = torch.div(
            torch.matmul(text_features, image_features.T), self.temperature
        )
        # for numerical stability
        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits = logits - logits_max.detach()
        exp_logits = torch.exp(logits)
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
        loss = -mean_log_prob_pos.mean()
        return loss


class CenterLoss(pl.LightningModule):
    """Center loss.

    Reference:
    Wen et al. A Discriminative Feature Learning Approach for Deep Face Recognition. ECCV 2016.

    Args:
        num_classes (int): number of classes.
        feat_dim (int): feature dimension.
    """

    def __init__(self, num_classes=21206, feat_dim=512):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim

        self.centers = nn.Parameter(
            torch.randn(self.num_classes, self.feat_dim).to(self.device)
        ).to(self.device)

    def forward(self, x, labels):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (num_classes).
        """
        assert x.size(0) == labels.size(
            0
        ), "features.size(0) is not equal to labels.size(0)"

        batch_size = x.size(0)
        distmat = (
            torch.pow(x, 2)
            .sum(dim=1, keepdim=True)
            .expand(batch_size, self.num_classes)
            + torch.pow(self.centers, 2)
            .sum(dim=1, keepdim=True)
            .expand(self.num_classes, batch_size)
            .t()
        )
        # distmat.addmm_(1, -2, x, self.centers.t())
        distmat.addmm_(x, self.centers.t(), beta=1, alpha=-2)

        classes = torch.arange(self.num_classes).to(self.device)
        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels.eq(classes.expand(batch_size, self.num_classes))

        dist = []
        for i in range(batch_size):
            value = distmat[i][mask[i]]
            value = value.clamp(min=1e-12, max=1e12)  # for numerical stability
            dist.append(value)
        dist = torch.cat(dist)
        loss = dist.mean()
        return loss


class CLIPReIDEmbedder(pl.LightningModule):
    def __init__(
        self,
        version="ViT-B/16",
        clip_ckpt_path: str = None,
        normalize=True,
        use_scheduler=False,
        scheduler_config=None,
        monitor=None,
        **kwargs,
    ):
        super().__init__()
        self.version = version
        self.ckpt_path = clip_ckpt_path
        self.preprocess = clip.load(self.version, jit=False, device=self.device)[1]
        if self.ckpt_path is None:
            self.model = clip.load(self.version, jit=False, device=self.device)[0]
        else:
            self.model = CLIPReIDEmbedder.load_from_checkpoint(clip_ckpt_path).model.to(
                self.device
            )
        self.normalize = normalize

        self.use_scheduler = use_scheduler
        self.scheduler_config = scheduler_config

        self.supcon_loss = SupConLoss()
        self.triplet_loss = TripletLoss()
        self.cross_entropy = CrossEntropyLoss()

        self.triplet_loss_weight = 10
        self.center_loss_weight = 0.01

        self.center_loss = CenterLoss(num_classes=21206)

        self.monitor = monitor

        # Caution: total ground up model
        # self.model.initialize_parameters()

        self.save_hyperparameters()

    def forward(self, text):
        raise NotImplementedError

    def encode(self, batch):
        # text_1 = batch["caption_1"]
        # text_2 = batch["caption_2"]
        #
        # tokens_1 = clip.tokenize(text_1, truncate=True).to(self.device)
        # x_1 = self.model.token_embedding(tokens_1).type(
        #     self.model.dtype
        # )  # [batch_size, n_ctx, d_model]
        # x_1 = x_1 + self.model.positional_embedding.type(self.model.dtype)
        # x_1 = x_1.permute(1, 0, 2)  # NLD -> LND
        # x_1 = self.model.transformer(x_1)
        # x_1 = x_1.permute(1, 0, 2)  # LND -> NLD
        #
        # tokens_2 = clip.tokenize(text_2, truncate=True).to(self.device)
        # x_2 = self.model.token_embedding(tokens_2).type(
        #     self.model.dtype
        # )  # [batch_size, n_ctx, d_model]
        # x_2 = x_2 + self.model.positional_embedding.type(self.model.dtype)
        # x_2 = x_2.permute(1, 0, 2)  # NLD -> LND
        # x_2 = self.model.transformer(x_2)
        # x_2 = x_2.permute(1, 0, 2)  # LND -> NLD
        #
        # z_1 = self.model.ln_final(x_1).type(self.model.dtype)  # #  [batch, 77, 512]
        # z_2 = self.model.ln_final(x_2).type(self.model.dtype)

        z = self.get_text_features(batch, feat_normalize=self.normalize)[0]
        return z.unsqueeze(1)

    def shared_step(
        self,
        batch,
        batch_idx,
        use_pair_loss=True,
        use_supcon_loss=True,
        use_triplet_loss=True,
        use_center_loss=True,
    ):
        loss_dict = {}
        log_prefix = "train" if self.training else "val"

        labels = batch["id"]
        text_features, tokenized_texts = self.get_text_features(
            batch, feat_normalize=self.normalize
        )
        image_features, preprocessed_images = self.get_image_features(
            batch, feat_normalize=self.normalize
        )

        loss = 0

        # pair loss
        if use_pair_loss:
            logits = text_features @ image_features.T
            images_similarity = image_features @ image_features.T
            texts_similarity = text_features @ text_features.T
            targets = nn.functional.softmax(
                (images_similarity + texts_similarity) / 2, dim=-1
            )

            texts_loss = self.cross_entropy(logits, targets)
            images_loss = self.cross_entropy(logits.T, targets.T)
            pair_loss = (images_loss + texts_loss) / 2.0  # shape: (batch_size)
            loss += pair_loss.mean()
            self.log_dict({f"{log_prefix}/pair_loss": pair_loss})

        if use_supcon_loss:
            supcon_loss_t2i = self.supcon_loss(
                text_features, image_features, labels, labels
            )
            supcon_loss_i2t = self.supcon_loss(
                image_features, text_features, labels, labels
            )
            supcon_loss = (supcon_loss_t2i + supcon_loss_i2t) / 2.0
            loss += supcon_loss
            loss_dict.update({f"{log_prefix}/supcon_loss": supcon_loss})

        if use_triplet_loss:
            triplet_loss_t = self.triplet_loss(text_features, labels)[0]
            triplet_loss_i = self.triplet_loss(image_features, labels)[0]
            triplet_loss = (triplet_loss_t + triplet_loss_i) / 2.0
            loss += triplet_loss * self.triplet_loss_weight
            loss_dict.update(
                {f"{log_prefix}/triplet_loss": triplet_loss * self.triplet_loss_weight}
            )

        if use_center_loss:
            center_loss_t = self.center_loss(text_features, labels)
            center_loss_i = self.center_loss(image_features, labels)
            center_loss = (center_loss_t + center_loss_i) / 2.0
            loss += center_loss * self.center_loss_weight
            loss_dict.update(
                {
                    f"{log_prefix}/center_loss_scaled": center_loss
                    * self.center_loss_weight
                }
            )

        loss_dict.update({f"{log_prefix}/loss": loss})

        return loss, loss_dict

    def training_step(self, batch, batch_idx):
        lr = self.optimizers().param_groups[0]["lr"]
        # self.log("lr_abs", lr, on_step=True, logger=True, on_epoch=False, prog_bar=True)
        loss, loss_dict = self.shared_step(batch, batch_idx)
        self.log_dict(
            loss_dict, prog_bar=True, logger=True, on_step=True, on_epoch=False
        )
        self.log(
            "global_step",
            self.global_step,
            prog_bar=True,
            logger=True,
            on_step=True,
            on_epoch=False,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        loss, loss_dict = self.shared_step(
            batch, batch_idx, use_supcon_loss=False, use_triplet_loss=False
        )
        self.log_dict(
            loss_dict, prog_bar=True, logger=True, on_step=False, on_epoch=True
        )
        return loss

    def test_step(self, batch, batch_idx):
        loss, loss_dict = self.shared_step(
            batch, batch_idx, use_supcon_loss=False, use_triplet_loss=False
        )
        self.log_dict(
            loss_dict, prog_bar=True, logger=True, on_step=False, on_epoch=True
        )
        return loss

    def on_train_epoch_start(self) -> None:
        opt = self.optimizers()
        opt = opt.optimizer
        opt.train()

    def on_validation_epoch_start(self) -> None:
        opt = self.optimizers()
        if opt is not None:
            opt = opt.optimizer
            opt.eval()

    def on_test_epoch_start(self) -> None:
        opt = self.optimizers()
        opt = opt.optimizer
        opt.eval()

    def configure_optimizers(self):
        lr = self.learning_rate
        optim = schedulefree.AdamWScheduleFree(self.parameters(), lr=lr)
        if self.use_scheduler:
            assert "target" in self.scheduler_config
            scheduler = instantiate_from_config(self.scheduler_config)

            print("Setting up LambdaLR scheduler...")
            scheduler = [
                {
                    "scheduler": LambdaLR(optim, lr_lambda=scheduler.schedule),
                    "interval": "step",
                    "frequency": 1,
                }
            ]
            return [optim], scheduler
        return optim

    def get_text_features(self, batch, feat_normalize=True):
        if isinstance(batch, list):
            captions = batch
            if len(captions) == 1:
                captions.append(captions[-1])
        else:
            captions = [
                batch["caption_1"],
                batch["caption_2"],
            ]
        tokenized = [
            clip.tokenize(captions[0], truncate=True).to(self.device),
            clip.tokenize(captions[1], truncate=True).to(self.device),
        ]
        if captions[0] != captions[1]:
            embeddings = [
                self.model.encode_text(tokenized[0]),
                self.model.encode_text(tokenized[1]),
            ]
        else:
            embedding = self.model.encode_text(tokenized[0])
            embeddings = [embedding, embedding]
        text_features = torch.mean(torch.stack(embeddings, dim=0), dim=0)
        if feat_normalize:
            text_features = text_features / text_features.norm(dim=1, keepdim=True)
        return text_features, tokenized

    def get_image_features(self, batch, feat_normalize=True):
        images = torch.stack(
            [self.preprocess(Image.open(k)) for k in batch["file_path"]]
        ).to(self.device)
        image_features = self.model.encode_image(images)
        if feat_normalize:
            image_features = image_features / image_features.norm(dim=1, keepdim=True)
        return image_features, images


class FrozenLLM2VecEmbedder(pl.LightningModule):
    # TODO
    pass
