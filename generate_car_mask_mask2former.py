import torch
from transformers import Mask2FormerForInstanceSegmentation, Mask2FormerImageProcessor
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os
import requests
from io import BytesIO

def load_image(image_path_or_url):
    """
    从本地路径或URL加载图像，并转换为RGB格式。
    """
    try:
        if image_path_or_url.startswith(('http://', 'https://')):
            response = requests.get(image_path_or_url)
            response.raise_for_status()
            img = Image.open(BytesIO(response.content)).convert("RGB")
        else:
            if not os.path.exists(image_path_or_url):
                raise FileNotFoundError(f"未找到图像文件: {image_path_or_url}")
            img = Image.open(image_path_or_url).convert("RGB")
        return img
    except Exception as e:
        raise RuntimeError(f"加载图像时出错: {e}")

def main():
    # 设置输入图像路径和输出掩码路径
    input_image_path = 'data/car.jpg'  # 替换为您的输入图像路径或URL
    output_mask_path = 'data/car_mask2former.png'     # 替换为您希望保存的掩码路径

    # 加载输入图像
    try:
        image = load_image(input_image_path)
        print("成功加载输入图像。")
    except Exception as e:
        print(e)
        return

    # 设置设备为GPU（如果可用），否则使用CPU
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"使用设备: {device}")

    # 加载 Mask2Former 模型和处理器
    model_name = "facebook/mask2former-swin-base-coco"
    try:
        processor = Mask2FormerImageProcessor.from_pretrained(model_name)
        model = Mask2FormerForInstanceSegmentation.from_pretrained(model_name)
        model.to(device)
        print("模型和处理器加载成功。")
    except Exception as e:
        print(f"加载模型或处理器时出错: {e}")
        return

    # 处理图像并生成掩码
    try:
        inputs = processor(images=image, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)

        # COCO 数据集中，'car' 的类别编号是 3
        COCO_INSTANCE_CATEGORY_NAMES = [
            '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
            'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
            'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
            'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella',
            'N/A', 'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
            'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
            'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass', 'cup', 'fork',
            'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli',
            'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
            'potted plant', 'bed', 'N/A', 'dining table', 'N/A', 'N/A', 'toilet',
            'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
            'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
            'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
        ]

        # 获取所有预测的类别和掩码
        predicted_classes = [COCO_INSTANCE_CATEGORY_NAMES[i] for i in outputs.logits.argmax(-1).cpu().numpy()]
        predicted_masks = outputs.pred_masks.cpu().numpy()

        # 创建一个空的掩码
        combined_mask = np.zeros(image.size[::-1], dtype=np.uint8)  # PIL Image 的 size 是 (width, height)

        # 遍历所有预测，筛选出类别为 "car" 的掩码
        for cls, mask in zip(predicted_classes, predicted_masks):
            if cls == "car":
                # 将掩码转换为二值图像
                binary_mask = (mask > 0.5).astype(np.uint8) * 255
                combined_mask = np.maximum(combined_mask, binary_mask)

        if np.sum(combined_mask) == 0:
            print("未检测到汽车。")
            return

        # 将掩码转换为 PIL 图像
        mask_image = Image.fromarray(combined_mask).convert("L")

        # 保存掩码
        mask_image.save(output_mask_path)
        print(f"汽车分割掩码已保存至 {output_mask_path}")

        # 可视化原图和掩码
        fig, axs = plt.subplots(1, 2, figsize=(15, 10))
        axs[0].imshow(image)
        axs[0].set_title('原始图像')
        axs[0].axis('off')

        axs[1].imshow(mask_image, cmap='gray')
        axs[1].set_title('汽车分割掩码')
        axs[1].axis('off')

        plt.show()

    except Exception as e:
        print(f"生成分割掩码时出错: {e}")
        return

if __name__ == "__main__":
    main()
