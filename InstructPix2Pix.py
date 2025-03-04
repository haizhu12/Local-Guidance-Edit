import torch
from diffusers import StableDiffusionInstructPix2PixPipeline
from PIL import Image
import requests
from io import BytesIO
import os

# 设置输入图像路径、指令和输出路径
input_image_path = 'data/car.jpg'  # 替换为您的输入图像路径或URL
instruction = "Remove the car, everything else stays the same"  # 替换为您的文本指令
output_image_path = 'data/edited_image_Remove_car.jpg'  # 替换为您希望保存的输出图像路径

def load_image(image_path_or_url):
    """
    从本地路径或URL加载图像，并转换为RGB格式。
    """
    if image_path_or_url.startswith(('http://', 'https://')):
        response = requests.get(image_path_or_url)
        response.raise_for_status()
        img = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        if not os.path.exists(image_path_or_url):
            raise FileNotFoundError(f"未找到图像文件: {image_path_or_url}")
        img = Image.open(image_path_or_url).convert("RGB")
    return img

def main():
    # 加载输入图像
    try:
        input_image = load_image(input_image_path)
        print("成功加载输入图像。")
    except Exception as e:
        print(f"加载图像时出错: {e}")
        return

    # 设置设备为GPU（如果可用），否则使用CPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")

    # 加载 InstructPix2Pix 模型
    model_id = "timbrooks/instruct-pix2pix"
    try:
        pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            safety_checker=None,  # 根据需要禁用安全检查器
        )
        pipe = pipe.to(device)
        print("模型加载成功。")
    except Exception as e:
        print(f"加载模型时出错: {e}")
        return

    # 如果使用GPU，尝试启用内存高效的注意力机制
    if device == "cuda":
        try:
            pipe.enable_xformers_memory_efficient_attention()
            print("已启用内存高效的注意力机制。")
        except Exception as e:
            print(f"无法启用内存高效的注意力机制: {e}")
            print("如果需要启用，请安装 xformers。")
            print("继续运行，不启用内存高效的注意力机制。")

    # 执行图像编辑
    try:
        with torch.autocast(device):
            # 使用 `prompt` 参数传递指令
            edited_image = pipe(
                prompt=instruction,
                image=input_image,
                num_inference_steps=50,   # 根据需要调整步数
                guidance_scale=7.5,       # 根据需要调整引导尺度
            ).images[0]
        print("图像编辑成功。")
    except Exception as e:
        print(f"图像编辑时出错: {e}")
        return

    # 保存编辑后的图像
    try:
        edited_image.save(output_image_path)
        print(f"编辑后的图像已保存至 {output_image_path}")
    except Exception as e:
        print(f"保存图像时出错: {e}")

if __name__ == "__main__":
    main()
