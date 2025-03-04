import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from transformers import pipeline


def estimate_depth(image_path, model_name="Intel/dpt-hybrid-midas"):
    """
    使用给定的深度估计模型，对单张图像进行深度估计并返回深度图。
    :param image_path: 原图路径
    :param model_name: Transformers 上可用的深度估计模型
    :return: depth_map_normalized (numpy array)，归一化后的深度值
    """
    # 初始化深度估计 pipeline
    depth_estimator = pipeline(
        task="depth-estimation",
        model=model_name,
        device=0 if torch.cuda.is_available() else -1
    )

    # 加载图像
    image = Image.open(image_path).convert("RGB")

    # 进行深度预测
    prediction = depth_estimator(image)
    depth = prediction[0]["predicted_depth"]

    # 转换成 numpy array
    depth_map = depth.squeeze().numpy()

    # 归一化到 [0, 1]
    depth_min, depth_max = depth_map.min(), depth_map.max()
    depth_map_normalized = (depth_map - depth_min) / (depth_max - depth_min + 1e-6)

    return depth_map_normalized


def visualize_depth_map(depth_map, cmap='magma'):
    """
    将深度图进行可视化
    :param depth_map: (H, W) 的深度值 numpy 数组
    :param cmap: matplotlib 的色表，用于控制可视化风格
    """
    plt.figure(figsize=(8, 6))
    plt.imshow(depth_map, cmap=cmap)
    plt.colorbar(label='Depth')
    plt.axis('off')
    plt.show()


if __name__ == "__main__":
    # 输入图像路径
    image_path = "cat.jpg"

    # 生成深度图
    depth_map = estimate_depth(image_path, model_name="Intel/dpt-hybrid-midas")

    # 可视化深度图
    visualize_depth_map(depth_map)
