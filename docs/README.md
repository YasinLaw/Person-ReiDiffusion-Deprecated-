# How Stable Diffusion for ReID Works?

# SD for ReID 如何工作？

## 整体架构：

模型分为两大部分，VAE 与 LDM（latent diffusion model），LDM 嵌在 VAE 的 encoder 与 decoder 之间。

训练流程为：

1. 训练 VAE，通过 encoder 提取图像特征，特征交予 decoder 解码。这一步可极大减少处理数据量（图像 -> 特征）
2. 训练 LDM，通过去噪过程还原上一步得到的特征。在这一步骤中，交叉注意力机制被应用，以融合图像描述与图像内容

Todo:

1. 寻找过拟合原因
2. 提取特征，判断特征有效性
3. 排除 CLIP 特征区分度影响