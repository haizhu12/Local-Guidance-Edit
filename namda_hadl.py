# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import matplotlib.pyplot as plt
# from PIL import Image
# import torchvision.transforms as T
# import numpy as np
#
#
# ########################
# # 1. 玩具文本嵌入器
# ########################
# class ToyTextEmbedder(nn.Module):
#     """
#     一个极简的文本编码器示例，将文本 token 映射到固定维度的向量。
#     这里用随机整数代替 token ID，再通过嵌入层获得对应向量。
#     """
#
#     def __init__(self, embed_dim=64, vocab_size=10000):
#         super().__init__()
#         self.embedding = nn.Embedding(vocab_size, embed_dim)
#         nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)
#
#     def forward(self, text_tokens):
#         # text_tokens: (batch, seq_len)
#         return self.embedding(text_tokens)
#
#
# ###############################
# # 2. 图像特征提取（按 patch 划分）
# ###############################
# # def extract_image_patches(image, patch_size=32):
# #     """
# #     将输入图像（PIL Image）转换为 tensor 后划分为不重叠的 patch，
# #     并将每个 patch flatten 成向量。返回 patch 特征和网格形状 (H, W)。
# #     """
# #     transform = T.ToTensor()
# #     img_tensor = transform(image)  # shape: (3, H_img, W_img)
# #     _, H_img, W_img = img_tensor.shape
# #
# #     # 计算行列方向上的 patch 数量
# #     H = H_img // patch_size
# #     W = W_img // patch_size
# #
# #     patches = []
# #     for i in range(H):
# #         for j in range(W):
# #             patch = img_tensor[:, i * patch_size:(i + 1) * patch_size, j * patch_size:(j + 1) * patch_size]
# #             patch_feat = patch.view(-1)  # flatten成一个向量
# #             patches.append(patch_feat)
# #
# #     img_feats = torch.stack(patches, dim=0)  # shape: (num_patches, patch_feat_dim)
# #     return img_feats, H, W
#
# def extract_image_patches(image, patch_size=32):
#     """
#     将输入图像（PIL Image）转换为 tensor 后划分为不重叠的 patch，
#     并将每个 patch reshape 成一维向量。返回 patch 特征和网格形状 (H, W)。
#     """
#     transform = T.ToTensor()
#     img_tensor = transform(image)  # shape: (3, H_img, W_img)
#     _, H_img, W_img = img_tensor.shape
#
#     # 计算行列方向上的 patch 数量
#     H = H_img // patch_size
#     W = W_img // patch_size
#
#     patches = []
#     for i in range(H):
#         for j in range(W):
#             patch = img_tensor[:, i * patch_size:(i + 1) * patch_size, j * patch_size:(j + 1) * patch_size]
#             patch_feat = patch.reshape(-1)  # 用 reshape 替换 view
#             patches.append(patch_feat)
#
#     img_feats = torch.stack(patches, dim=0)  # shape: (num_patches, patch_feat_dim)
#     return img_feats, H, W
#
#
# ###############################
# # 3. 简单的交叉注意力模块
# ###############################
# class SimpleCrossAttention(nn.Module):
#     """
#     计算交叉注意力：
#         Attention = Softmax( λ * (QK^T / sqrt(d)) )
#     其中 Q 来自图像特征，K、V 来自文本嵌入。
#     """
#
#     def __init__(self, dim_img, dim_txt, dim_out):
#         super().__init__()
#         self.W_Q = nn.Linear(dim_img, dim_out)
#         self.W_K = nn.Linear(dim_txt, dim_out)
#         self.W_V = nn.Linear(dim_txt, dim_out)
#
#     def forward(self, img_feats, txt_feats, lambda_value):
#         # img_feats: (num_patches, dim_img)
#         # txt_feats: (seq_len, dim_txt)
#         Q = self.W_Q(img_feats)  # (num_patches, dim_out)
#         K = self.W_K(txt_feats)  # (seq_len, dim_out)
#         V = self.W_V(txt_feats)  # (seq_len, dim_out)
#
#         # 计算点积并缩放： (num_patches, seq_len)
#         d = K.shape[-1]
#         scores = torch.matmul(Q, K.transpose(0, 1)) / (d ** 0.5)
#         # 乘以 lambda 调整注意力分布
#         scores = lambda_value * scores
#         attention_map = F.softmax(scores, dim=-1)  # (num_patches, seq_len)
#
#         # 通过注意力加权值向量（在本例中主要用于展示，不用于后续编辑）
#         updated_feats = torch.matmul(attention_map, V)  # (num_patches, dim_out)
#
#         return attention_map, updated_feats
#
#
# ###############################
# # 4. 主流程：不同 λ 下的注意力可视化
# ###############################
# def main_visualize_attention(
#         image_path="test.jpg",
#         text="Make the dog bigger",
#         lambda_values=[0.1, 1.0, 5.0],
#         patch_size=32
# ):
#     # 1. 加载图像并提取 patch 特征
#     image = Image.open(image_path).convert("RGB")
#     img_feats, H, W = extract_image_patches(image, patch_size=patch_size)
#     dim_img = img_feats.shape[-1]
#
#     # 2. 文本处理：将文本简单分词，并转成 token ID（随机模拟）
#     tokens = text.split()
#     seq_len = len(tokens)
#     # 随机生成 token ID（范围 0 到 9999）
#     text_tokens = torch.randint(low=0, high=10000, size=(1, seq_len))
#
#     # 初始化文本嵌入器
#     embed_dim = 64
#     text_embedder = ToyTextEmbedder(embed_dim=embed_dim)
#     # 得到文本嵌入，形状: (1, seq_len, embed_dim)，取第一个 batch
#     txt_feats = text_embedder(text_tokens)[0]  # (seq_len, embed_dim)
#     dim_txt = txt_feats.shape[-1]
#
#     # 3. 初始化交叉注意力模块
#     dim_out = 64
#     cross_attn = SimpleCrossAttention(dim_img, dim_txt, dim_out)
#
#     # 4. 对每个 lambda 值计算注意力，并可视化
#     num_lambdas = len(lambda_values)
#     fig, axs = plt.subplots(1, num_lambdas, figsize=(4 * num_lambdas, 4))
#     if num_lambdas == 1:
#         axs = [axs]
#
#     for idx, lam in enumerate(lambda_values):
#         attention_map, _ = cross_attn(img_feats, txt_feats, lambda_value=lam)
#         # attention_map: (num_patches, seq_len)
#         # 为简单起见，取每个 patch 上最大注意力值（即最相关 token 的分数）作为该 patch 的注意力强度
#         patch_attn = attention_map.max(dim=-1)[0].detach().cpu().numpy()  # (num_patches,)
#
#         # 将一维 patch 注意力数组重塑为二维 (H, W)
#         attn_2d = patch_attn.reshape(H, W)
#         # 归一化到 [0, 1]
#         attn_norm = (attn_2d - attn_2d.min()) / (attn_2d.max() - attn_2d.min() + 1e-8)
#
#         axs[idx].imshow(attn_norm, cmap="jet")
#         axs[idx].set_title(f"λ = {lam}")
#         axs[idx].axis("off")
#
#     plt.tight_layout()
#     plt.show()
#
#
# if __name__ == "__main__":
#     # 请确保 test.jpg 存在，或修改 image_path 为正确路径
#     main_visualize_attention(
#         image_path="data/cat_212.jpg",
#         text="A cat wearscloth which has black and white colors",
#         lambda_values=[0.01, 0.5, 1.0],
#         patch_size=32
#     )

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import matplotlib.pyplot as plt
# import torchvision.transforms as T
# from PIL import Image
# import numpy as np
#
#
# ########################
# # 1. 玩具文本嵌入器
# ########################
# class ToyTextEmbedder(nn.Module):
#     """
#     一个极简的文本编码器示例，将文本 token 映射到固定维度的向量。
#     这里用随机整数代替 token ID，再通过嵌入层获得对应向量。
#     """
#
#     def __init__(self, embed_dim=64, vocab_size=10000):
#         super().__init__()
#         self.embedding = nn.Embedding(vocab_size, embed_dim)
#         nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)
#
#     def forward(self, text_tokens):
#         # text_tokens: (batch, seq_len)
#         return self.embedding(text_tokens)
#
#
# ###############################
# # 2. 将图像划分为不重叠的 patch
# ###############################
# def extract_image_patches(image, patch_size=32):
#     """
#     将输入图像（PIL Image）转换为 tensor 后划分为不重叠的 patch，
#     并将每个 patch flatten 成向量。返回 patch 特征和网格形状 (H, W) 以及原始图像尺寸。
#     """
#     transform = T.ToTensor()
#     img_tensor = transform(image)  # shape: (3, H_img, W_img)
#     _, H_img, W_img = img_tensor.shape
#
#     # 计算行列方向上的 patch 数量
#     H = H_img // patch_size
#     W = W_img // patch_size
#
#     patches = []
#     for i in range(H):
#         for j in range(W):
#             patch = img_tensor[:, i * patch_size:(i + 1) * patch_size, j * patch_size:(j + 1) * patch_size]
#             # 用 reshape 代替 view，兼容非连续内存
#             patch_feat = patch.reshape(-1)
#             patches.append(patch_feat)
#
#     img_feats = torch.stack(patches, dim=0)  # shape: (num_patches, patch_feat_dim)
#     return img_feats, H, W, H_img, W_img
#
#
# ###############################
# # 3. 简单的交叉注意力模块
# ###############################
# class SimpleCrossAttention(nn.Module):
#     """
#     计算交叉注意力：
#         Attention = Softmax( λ * (QK^T / sqrt(d)) )
#     其中 Q 来自图像特征，K、V 来自文本嵌入。
#     """
#
#     def __init__(self, dim_img, dim_txt, dim_out):
#         super().__init__()
#         self.W_Q = nn.Linear(dim_img, dim_out)
#         self.W_K = nn.Linear(dim_txt, dim_out)
#         self.W_V = nn.Linear(dim_txt, dim_out)
#
#     def forward(self, img_feats, txt_feats, lambda_value):
#         # img_feats: (num_patches, dim_img)
#         # txt_feats: (seq_len, dim_txt)
#         Q = self.W_Q(img_feats)  # (num_patches, dim_out)
#         K = self.W_K(txt_feats)  # (seq_len, dim_out)
#         V = self.W_V(txt_feats)  # (seq_len, dim_out)
#
#         d = K.shape[-1]
#         # 计算点积并进行缩放
#         scores = torch.matmul(Q, K.transpose(0, 1)) / (d ** 0.5)
#         # 乘以 λ 调整注意力分布
#         scores = lambda_value * scores
#         # 计算注意力图
#         attention_map = F.softmax(scores, dim=-1)  # (num_patches, seq_len)
#
#         # 通过注意力加权 V（本示例仅用 attention_map 来可视化）
#         updated_feats = torch.matmul(attention_map, V)  # (num_patches, dim_out)
#
#         return attention_map, updated_feats
#
#
# ###############################
# # 4. 将 Patch Attention 上采样到原图大小
# ###############################
# def upscale_attention_to_image(attn_2d, H_img, W_img, mode='bilinear'):
#     """
#     将 (H, W) 大小的注意力图放大到 (H_img, W_img)。
#     attn_2d: numpy array，形状 (H, W)
#     H_img, W_img: 原图像分辨率
#     mode: 上采样模式，可选 'nearest', 'bilinear' 等
#     返回:
#         upsampled_attn: shape (H_img, W_img) 的 numpy array
#     """
#     attn_tensor = torch.from_numpy(attn_2d).float().unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
#     attn_up = F.interpolate(attn_tensor, size=(H_img, W_img), mode=mode, align_corners=False)
#     attn_up = attn_up.squeeze().numpy()  # (H_img, W_img)
#     return attn_up
#
#
# ###############################
# # 5. 主流程：可视化灰度注意力图 (与原图同分辨率)
# ###############################
# def main_visualize_attention(
#         image_path="test.jpg",
#         text="Make the cat bigger",
#         lambda_value=5.0,
#         patch_size=32
# ):
#     """
#     通过交叉注意力生成玩具级注意力图，并上采样到与原图相同分辨率，
#     使其看上去更类似于“模糊灰度图”。
#     """
#     # 1. 加载图像并提取 patch 特征
#     image = Image.open(image_path).convert("RGB")
#     img_feats, H, W, H_img, W_img = extract_image_patches(image, patch_size=patch_size)
#     dim_img = img_feats.shape[-1]
#
#     # 2. 文本处理：将文本拆分为 token（此处仅模拟）
#     tokens = text.split()
#     seq_len = len(tokens)
#     # 用随机 ID 模拟 token
#     text_tokens = torch.randint(low=0, high=10000, size=(1, seq_len))
#
#     # 初始化文本嵌入器
#     embed_dim = 64
#     text_embedder = ToyTextEmbedder(embed_dim=embed_dim)
#     # 得到文本嵌入，形状: (1, seq_len, embed_dim)，取 batch 维度 0
#     txt_feats = text_embedder(text_tokens)[0]  # (seq_len, embed_dim)
#     dim_txt = txt_feats.shape[-1]
#
#     # 3. 初始化交叉注意力模块
#     dim_out = 64
#     cross_attn = SimpleCrossAttention(dim_img, dim_txt, dim_out)
#
#     # 4. 计算注意力
#     attention_map, _ = cross_attn(img_feats, txt_feats, lambda_value=lambda_value)
#
#     # (num_patches, seq_len) -> 每个 patch 在所有 token 上的平均注意力
#     patch_attn = attention_map.mean(dim=-1).detach().cpu().numpy()  # (num_patches,)
#     # 变形为 (H, W)
#     attn_2d = patch_attn.reshape(H, W)
#
#     # 归一化到 [0, 1]
#     attn_min, attn_max = attn_2d.min(), attn_2d.max()
#     attn_norm = (attn_2d - attn_min) / (attn_max - attn_min + 1e-8)
#
#     # 反转让高注意力为黑色
#     attn_invert = 1.0 - attn_norm
#
#     # 5. 上采样到原图大小
#     attn_up = upscale_attention_to_image(attn_invert, H_img, W_img, mode='bilinear')
#
#     # 6. 显示结果
#     # (1) 原图
#     plt.figure(figsize=(12, 6))
#     plt.subplot(1, 2, 1)
#     plt.imshow(image)
#     plt.title("Original Image")
#     plt.axis("off")
#
#     # (2) 上采样后的灰度注意力图
#     plt.subplot(1, 2, 2)
#     plt.imshow(attn_up, cmap="gray", vmin=0.0, vmax=1.0)
#     plt.title(f"Upscaled Attention (λ={lambda_value})")
#     plt.axis("off")
#
#     plt.tight_layout()
#     plt.show()
#
#
# if __name__ == "__main__":
#     # 你可以修改 image_path 和 text、lambda_value 等参数
#     main_visualize_attention(
#         image_path="data/cat_212.jpg",
#         text="A cat wearscloth which has black and white colors",
#         lambda_value=0.1,  # 可试试 0.1, 1, 10, 100, 等
#         patch_size=32
#     )


# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import matplotlib.pyplot as plt
# import torchvision.transforms as T
# from PIL import Image
# import numpy as np
#
#
# #######################################
# # 1. 玩具文本嵌入器 (ToyTextEmbedder)
# #######################################
# class ToyTextEmbedder(nn.Module):
#     """
#     一个极简的文本编码器示例，将文本 token 映射到固定维度的向量。
#     这里用随机整数代替 token ID，再通过 Embedding 层获得对应向量。
#     真实应用中可替换为 CLIP/BERT 等。
#     """
#
#     def __init__(self, embed_dim=64, vocab_size=10000):
#         super().__init__()
#         self.embedding = nn.Embedding(vocab_size, embed_dim)
#         nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)
#
#     def forward(self, text_tokens):
#         # text_tokens: (batch_size, seq_len)
#         return self.embedding(text_tokens)  # (batch_size, seq_len, embed_dim)
#
#
# ###################################################
# # 2. 将图像拆分为Patch并转换为向量特征 (extract_image_patches)
# ###################################################
# def extract_image_patches(image, patch_size=32):
#     """
#     将输入图像 (PIL Image) 转换为tensor后，划分为不重叠patch，
#     并将每个patch flatten为一维向量。返回:
#       - img_feats: shape (num_patches, patch_feat_dim)
#       - H, W: patch网格形状
#       - H_img, W_img: 原图像分辨率
#     """
#     transform = T.ToTensor()
#     img_tensor = transform(image)  # (3, H_img, W_img)
#     _, H_img, W_img = img_tensor.shape
#
#     H = H_img // patch_size
#     W = W_img // patch_size
#
#     patches = []
#     for i in range(H):
#         for j in range(W):
#             patch = img_tensor[:, i * patch_size:(i + 1) * patch_size, j * patch_size:(j + 1) * patch_size]
#             patch_feat = patch.reshape(-1)  # 用 reshape 代替 view
#             patches.append(patch_feat)
#
#     img_feats = torch.stack(patches, dim=0)  # (num_patches, patch_feat_dim)
#     return img_feats, H, W, H_img, W_img
#
#
# ##########################################################
# # 3. 多头交叉注意力模块 (MultiHeadCrossAttention)
# ##########################################################
# class MultiHeadCrossAttention(nn.Module):
#     """
#     使用 nn.MultiheadAttention 来实现多头交叉注意力。
#     - 图像特征  -> Query
#     - 文本特征  -> Key, Value
#     - 通过 λ 调整注意力分数分布的“锐化”或“平滑”。
#     """
#
#     def __init__(self, dim_img, dim_txt, num_heads=4):
#         super().__init__()
#         # 先将图像和文本特征投影到相同维度 (embed_dim)，以便做多头注意力
#         # 你也可以在外部保证维度一致，这里演示常见做法
#         embed_dim = 64  # 你可以根据需要调整
#
#         self.img_proj = nn.Linear(dim_img, embed_dim)
#         self.txt_proj = nn.Linear(dim_txt, embed_dim)
#
#         # 多头注意力模块: embed_dim=64, num_heads=4, batch_first=False(默认)
#         self.mha = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=False)
#
#         # 额外可选：在多头注意力后再做一次投影 (常见于Transformer结构)
#         self.out_proj = nn.Linear(embed_dim, embed_dim)
#
#     def forward(self, img_feats, txt_feats, lambda_value=1.0):
#         """
#         img_feats: (num_patches, dim_img)
#         txt_feats: (seq_len, dim_txt)
#         lambda_value: 缩放注意力分数
#         返回:
#           - attn_weights: (num_patches, seq_len) 的注意力分布
#           - out: (num_patches, embed_dim) 更新后的图像特征
#         """
#         # 投影到相同的 embed_dim
#         q = self.img_proj(img_feats)  # (num_patches, embed_dim)
#         k = self.txt_proj(txt_feats)  # (seq_len, embed_dim)
#         v = k.clone()  # 这里 Key, Value 都来自 txt_proj(txt_feats)
#
#         # PyTorch的MultiheadAttention期望输入形状: (seq_len, batch_size, embed_dim)
#         # 这里我们没有显式的 batch 维度，因此可以将 batch_size=1 并把序列维度放在前面
#         # 对于图像特征: seq_len = num_patches
#         q = q.unsqueeze(1)  # (num_patches, 1, embed_dim)
#         k = k.unsqueeze(1)  # (seq_len, 1, embed_dim)
#         v = v.unsqueeze(1)  # (seq_len, 1, embed_dim)
#
#         # 计算多头注意力
#         # attn_output: (num_patches, 1, embed_dim)
#         # attn_weights: (1, num_patches, seq_len)
#         attn_output, attn_weights = self.mha(q, k, v)
#
#         # PyTorch不会直接提供 λ 缩放，但我们可以自己对 attn_weights 做乘法
#         # attn_weights: (1, num_patches, seq_len)
#         attn_weights = attn_weights * lambda_value
#
#         # 重新归一化
#         attn_weights = F.softmax(attn_weights, dim=-1)  # (1, num_patches, seq_len)
#
#         # 基于新的 attn_weights 重新计算 attn_output
#         # 由于PyTorch的MultiheadAttention不支持动态替换权重，这里用手动加权方式
#         # (num_patches, 1, seq_len) x (seq_len, 1, embed_dim) -> (num_patches, 1, embed_dim)
#         attn_weights_t = attn_weights.transpose(0, 1)  # (num_patches, 1, seq_len)
#         v_t = v.transpose(0, 1)  # (1, seq_len, embed_dim) -> (seq_len, embed_dim) with batch=1
#         v_t = v_t.unsqueeze(0)  # (1, seq_len, embed_dim)
#
#         # 注意: 这里简化处理, 你也可以自己写更灵活的多头加权
#         # out: (num_patches, 1, embed_dim)
#         out = torch.matmul(attn_weights_t, v_t).squeeze(1)  # -> (num_patches, embed_dim)
#
#         # 再做一次投影 (可选)
#         out = self.out_proj(out)  # (num_patches, embed_dim)
#
#         # 最终的 attn_weights 形状转成 (num_patches, seq_len)
#         attn_weights = attn_weights.squeeze(0)  # (num_patches, seq_len)
#
#         return attn_weights, out
#
#
# ##############################################
# # 4. 上采样到原图大小，便于可视化
# ##############################################
# def upscale_attention_to_image(attn_2d, H_img, W_img, mode='bilinear'):
#     """
#     将 (H, W) 大小的注意力图放大到 (H_img, W_img)。
#     attn_2d: numpy array，形状 (H, W)
#     H_img, W_img: 原图分辨率
#     """
#     attn_tensor = torch.from_numpy(attn_2d).float().unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
#     attn_up = F.interpolate(attn_tensor, size=(H_img, W_img), mode=mode, align_corners=False)
#     attn_up = attn_up.squeeze().numpy()  # (H_img, W_img)
#     return attn_up
#
#
# ############################################################
# # 5. 主流程：演示多头交叉注意力并可视化灰度注意力图
# ############################################################
# def main_visualize_attention(
#         image_path="test.jpg",
#         text="Make the cat bigger",
#         lambda_value=5.0,
#         patch_size=32,
#         num_heads=4
# ):
#     """
#     使用多头交叉注意力实现更高级的跨模态注意力，并可视化与原图同分辨率的灰度注意力图。
#     """
#     # 1. 加载图像
#     image = Image.open(image_path).convert("RGB")
#     img_feats, H, W, H_img, W_img = extract_image_patches(image, patch_size=patch_size)
#     dim_img = img_feats.shape[-1]  # patch_feat_dim
#
#     # 2. 生成文本 token (随机模拟)
#     tokens = text.split()  # e.g. ["Make", "the", "cat", "bigger"]
#     seq_len = len(tokens)
#     text_tokens = torch.randint(low=0, high=10000, size=(1, seq_len))  # (1, seq_len)
#
#     # 3. 文本嵌入
#     embed_dim = 64
#     text_embedder = ToyTextEmbedder(embed_dim=embed_dim)
#     txt_feats = text_embedder(text_tokens)[0]  # (seq_len, embed_dim)
#     dim_txt = txt_feats.shape[-1]
#
#     # 4. 多头交叉注意力
#     cross_attn = MultiHeadCrossAttention(dim_img, dim_txt, num_heads=num_heads)
#     attn_weights, _ = cross_attn(img_feats, txt_feats, lambda_value=lambda_value)
#     # attn_weights: (num_patches, seq_len)
#
#     # 对 seq_len 做平均，得到每个 patch 的单通道注意力
#     patch_attn = attn_weights.mean(dim=-1).detach().cpu().numpy()  # (num_patches,)
#     attn_2d = patch_attn.reshape(H, W)  # (H, W)
#
#     # 归一化到 [0,1]
#     attn_min, attn_max = attn_2d.min(), attn_2d.max()
#     attn_norm = (attn_2d - attn_min) / (attn_max - attn_min + 1e-8)
#     # 反转让高注意力变黑色
#     attn_invert = 1.0 - attn_norm
#
#     # 上采样到原图大小
#     attn_up = upscale_attention_to_image(attn_invert, H_img, W_img, mode='bilinear')
#
#     # 5. 可视化
#     plt.figure(figsize=(12, 6))
#
#     # 原图
#     plt.subplot(1, 2, 1)
#     plt.imshow(image)
#     plt.title("Original Image")
#     plt.axis("off")
#
#     # 注意力图
#     plt.subplot(1, 2, 2)
#     plt.imshow(attn_up, cmap="gray", vmin=0.0, vmax=1.0)
#     plt.title(f"MultiHeadCrossAttention (λ={lambda_value}, heads={num_heads})")
#     plt.axis("off")
#
#     plt.tight_layout()
#     plt.show()
#
#
# if __name__ == "__main__":
#     # 示例用法
#     main_visualize_attention(
#         image_path="data/cat_212.jpg",  # 请准备好 test.jpg 或替换路径
#         text="Make the cat bigger",  # 玩具文本
#         lambda_value=5,  # 可尝试 0.1, 1, 10, 100, 等
#         patch_size=32,
#         num_heads=4  # 多头数量
#     )


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

# 尝试从不同路径导入 CrossAttention
try:
    from diffusers.models.attention import CrossAttention
except ImportError:
    from diffusers.models.cross_attention import CrossAttention

##############################################################################
# 1. 自定义 CrossAttention，保存注意力到 self.attn_weights
##############################################################################
class CrossAttentionSaveAttn(CrossAttention):
    """
    基于 diffusers 的 CrossAttention，实现时额外将注意力矩阵存到 self.attn_weights，
    以便后续 Hook 捕获注意力信息。
    """
    def forward(
        self,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        **cross_attention_kwargs,
    ):
        # 如果没有 encoder_hidden_states，则使用 hidden_states（自注意力）
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states

        batch_size, sequence_length, _ = hidden_states.shape
        cross_batch_size, cross_sequence_length, _ = encoder_hidden_states.shape

        if cross_batch_size != batch_size:
            cross_attention_kwargs["cross_attn_expand_dim"] = cross_batch_size // batch_size

        query = self.to_q(hidden_states)
        key = self.to_k(encoder_hidden_states)
        value = self.to_v(encoder_hidden_states)

        # 将各头重排： (batch, seq_len, num_heads, head_dim) -> (batch*num_heads, seq_len, head_dim)
        def reshape_heads_to_batch_dim(tensor, num_heads):
            return tensor.reshape(batch_size, -1, num_heads, self.head_dim).permute(0, 2, 1, 3).reshape(
                batch_size * num_heads, -1, self.head_dim
            )

        query = reshape_heads_to_batch_dim(query, self.num_heads)
        key = reshape_heads_to_batch_dim(key, self.num_heads)
        value = reshape_heads_to_batch_dim(value, self.num_heads)

        scale = 1 / np.sqrt(self.head_dim)
        attention_scores = torch.baddbmm(
            torch.empty(query.shape[0], query.shape[1], key.shape[1], dtype=query.dtype, device=query.device),
            query,
            key.transpose(1, 2),
            beta=0,
            alpha=scale,
        )

        if attention_mask is not None:
            if attention_mask.shape[0] < attention_scores.shape[0]:
                attention_mask = attention_mask.repeat_interleave(
                    attention_scores.shape[0] // attention_mask.shape[0], dim=0
                )
            attention_scores = attention_scores + attention_mask

        attn_probs = attention_scores.softmax(dim=-1)

        # 保存注意力权重
        self.attn_weights = attn_probs.detach()

        hidden_states = torch.bmm(attn_probs, value)

        def reshape_batch_dim_to_heads(tensor, num_heads):
            batch_size_times_num_heads, seq_len, head_dim = tensor.shape
            tensor = tensor.reshape(batch_size, num_heads, seq_len, head_dim)
            tensor = tensor.permute(0, 2, 1, 3).reshape(batch_size, seq_len, num_heads * head_dim)
            return tensor

        hidden_states = reshape_batch_dim_to_heads(hidden_states, self.num_heads)
        hidden_states = self.to_out[0](hidden_states)
        hidden_states = self.to_out[1](hidden_states)
        return hidden_states

##############################################################################
# 2. 猴子补丁：将原始 CrossAttention 替换为我们的 CrossAttentionSaveAttn
##############################################################################
def monkey_patch_crossattention():
    import diffusers.models.attention
    diffusers.models.attention.CrossAttention = CrossAttentionSaveAttn
    # 如果上面未能替换，也尝试替换 diffusers.models.cross_attention
    try:
        import diffusers.models.cross_attention
        diffusers.models.cross_attention.CrossAttention = CrossAttentionSaveAttn
    except ImportError:
        pass

##############################################################################
# 3. 注册 forward_hook，收集 attn_weights
##############################################################################
attention_maps = {}  # 全局字典

def cross_attention_hook(module, input, output):
    """
    Hook：每次 CrossAttention.forward 后被调用，将 module.attn_weights 保存到 attention_maps 字典中。
    """
    if hasattr(module, "attn_weights"):
        # 用模块 id 作为 key，可自行修改为更易识别的名称
        layer_id = str(id(module))
        attention_maps[layer_id] = module.attn_weights.cpu()

def register_hooks_for_unet(unet):
    """
    遍历 unet 的所有子模块，为 CrossAttentionSaveAttn 模块注册 forward_hook。
    """
    for name, submodule in unet.named_modules():
        if isinstance(submodule, CrossAttentionSaveAttn):
            submodule.register_forward_hook(cross_attention_hook)

##############################################################################
# 4. 上采样并可视化注意力
##############################################################################
def upscale_and_visualize(attn_2d, out_h=512, out_w=512, title=""):
    """
    将 (h, w) 的注意力图上采样到 (out_h, out_w)，归一化后反转（使高注意力为黑色），并显示灰度图。
    """
    a_min, a_max = attn_2d.min(), attn_2d.max()
    attn_norm = (attn_2d - a_min) / (a_max - a_min + 1e-8)
    attn_invert = 1.0 - attn_norm

    t = torch.from_numpy(attn_invert).unsqueeze(0).unsqueeze(0)  # (1,1,h,w)
    t_up = F.interpolate(t, size=(out_h, out_w), mode='bilinear', align_corners=False)
    attn_up = t_up.squeeze().numpy()

    plt.imshow(attn_up, cmap='gray', vmin=0, vmax=1)
    plt.title(title)
    plt.axis("off")

##############################################################################
# 5. 主流程：加载 Stable Diffusion、生成图像、并可视化注意力
##############################################################################
def main():
    # a) 先进行猴子补丁
    monkey_patch_crossattention()

    # b) 加载 Stable Diffusion Pipeline（text2img），关闭安全检查
    from diffusers import StableDiffusionPipeline
    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16
    ).to("cuda")
    pipe.safety_checker = None

    # c) 注册 hook 到 UNet 上
    register_hooks_for_unet(pipe.unet)

    # d) 生成图像
    prompt = "A cute cat wearing a red hat, photorealistic"
    with torch.autocast("cuda"):
        image = pipe(prompt, num_inference_steps=20, guidance_scale=7.5).images[0]

    # e) 打印捕获到的注意力层信息
    if not attention_maps:
        raise RuntimeError("没有捕获到任何 Cross-Attention，请检查 monkey patch 和 hook 注册。")
    print("Captured attention layers:")
    for k, v in attention_maps.items():
        print(f"  layer_id={k}, shape={tuple(v.shape)}")

    # f) 从 attention_maps 中任选一层进行可视化
    # 假设注意力张量形状为 (batch*num_heads, query_len, key_len)
    # 这里对 batch*num_heads、以及 key_len（文本 token 维度）做平均，仅保留 query_len 部分
    last_layer_id = list(attention_maps.keys())[-1]
    attn_tensor = attention_maps[last_layer_id]  # 例如 shape (8, 64, 77)
    attn_mean = attn_tensor.mean(dim=0).mean(dim=1)  # 先平均头部 (64,77) 再对 token 平均 -> (64,)
    # 根据实际 query_len 来恢复 2D 空间尺寸
    query_len = attn_mean.shape[0]
    side = int(np.sqrt(query_len))
    if side * side != query_len:
        raise ValueError(f"无法将 query_len={query_len} reshape 成正方形，请检查该层输出。")
    attn_2d = attn_mean.reshape(side, side).numpy()

    # g) 上采样注意力图到 512x512 并可视化
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title("Generated Image")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    upscale_and_visualize(attn_2d, out_h=512, out_w=512, title="Cross-Attention Map")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()


