import torch
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
from PIL import Image
import numpy as np
import cv2

# 加载 Stable Diffusion 和 ControlNet 模型
def load_models():
    # 加载 ControlNet 模型（以 Canny 边缘为例）
    controlnet = ControlNetModel.from_pretrained(
        "lllyasviel/sd-controlnet-canny", torch_dtype=torch.float16
    )

    # 加载 Stable Diffusion 模型
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        controlnet=controlnet,
        torch_dtype=torch.float16,
    )

    # 设置调度器（支持更快的生成）
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.to("cuda")
    return pipe

# 提取目标区域的边缘
def extract_edges(image_path):
    # 读取输入图像
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Failed to load image from {image_path}")

    # 使用 Canny 边缘检测
    edges = cv2.Canny(image, 100, 200)

    # 转换为 RGB 格式以适配模型输入
    edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)

    # 转换为 PIL 图像
    return Image.fromarray(edges_rgb)

# 图像编辑或生成
def generate_with_controlnet(pipe, prompt, edge_image, guidance_scale=7.5):
    # 预处理边缘图像
    edge_image = edge_image.resize((512, 512))  # 确保图像大小匹配模型输入
    edge_image = np.array(edge_image).astype(np.float32) / 255.0
    edge_image = torch.tensor(edge_image).permute(2, 0, 1).unsqueeze(0).to("cuda")

    # 生成图像
    result = pipe(
        prompt,
        image=edge_image,  # ControlNet 的条件输入
        guidance_scale=guidance_scale,
        num_inference_steps=20,
    )

    # 返回生成结果
    return result.images[0]

if __name__ == "__main__":
    # 1. 加载模型
    pipe = load_models()

    # 2. 提取边缘
    input_image_path = "data/car.jpg"  # 输入图像路径
    edge_image = extract_edges(input_image_path)

    # 3. 文本描述
    prompt = "Remove the car, everything else stays the same"

    # 4. 生成图像
    output_image = generate_with_controlnet(pipe, prompt, edge_image)

    # 5. 保存结果
    output_image.save("data/controlnet_car_remove.png")
    print("Image saved as controlnet_car_remove.png")
