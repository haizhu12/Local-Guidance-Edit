import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def generate_diffusion_noise_image(width, height, mean=0.5, std=0.2, save_path=None):
    """
    生成类似扩散模型中的高斯噪声图。

    :param width: 图像宽度
    :param height: 图像高度
    :param mean: 噪声的均值 (通常为 0.5)
    :param std: 噪声的标准差 (控制噪声强度)
    :param save_path: 如果提供路径，将图像保存为文件
    :return: 噪声图像 (PIL.Image 对象)
    """
    # 生成高斯分布的噪声图像 (范围在 0-1 之间)
    noise = np.random.normal(mean, std, (height, width, 3))
    noise = np.clip(noise, 0, 1)  # 限制范围在 [0, 1]

    # 转换为 0-255 的整数值
    noise = (noise * 255).astype(np.uint8)

    # 转换为 PIL 图像
    noise_image = Image.fromarray(noise)

    # 保存图像（如果提供了保存路径）
    if save_path:
        noise_image.save(save_path)
        print(f"噪声图已保存到 {save_path}")

    return noise_image


# 可视化和保存
if __name__ == "__main__":
    # 设置图像大小
    width, height = 512, 512

    # 生成噪声图像
    noise_image = generate_diffusion_noise_image(width, height, mean=0.5, std=0.2, save_path="diffusion_noise.png")

    # 显示图像
    plt.imshow(noise_image)
    plt.axis('off')
    plt.title("Gaussian Diffusion Noise")
    plt.show()
