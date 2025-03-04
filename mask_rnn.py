import torch
import torchvision
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os


def load_image(image_path):
    """
    从本地路径加载图像，并转换为 RGB 格式。

    参数:
        image_path (str): 图像文件的路径。

    返回:
        PIL.Image.Image: 加载的 RGB 图像。
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"未找到图像文件: {image_path}")
    image = Image.open(image_path).convert("RGB")
    return image


def get_car_masks(image, model, device, threshold=0.5):
    """
    使用 Mask R-CNN 模型检测图像中的汽车，并生成分割掩码。

    参数:
        image (PIL.Image.Image): 输入图像。
        model (torch.nn.Module): 预训练的 Mask R-CNN 模型。
        device (torch.device): 设备（CPU 或 GPU）。
        threshold (float): 置信度阈值，低于此值的检测结果将被忽略。

    返回:
        PIL.Image.Image: 二值分割掩码，汽车区域为白色，其他区域为黑色。
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    image_tensor = transform(image).to(device)

    model.eval()
    with torch.no_grad():
        predictions = model([image_tensor])

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

    masks = []
    for idx, score in enumerate(predictions[0]['scores']):
        label = predictions[0]['labels'][idx].item()
        if COCO_INSTANCE_CATEGORY_NAMES[label] == 'car' and score > threshold:
            mask = predictions[0]['masks'][idx, 0].mul(255).byte().cpu().numpy()
            masks.append(mask)

    if not masks:
        print("未检测到汽车。")
        return None

    # 合并所有汽车的掩码
    combined_mask = np.zeros(masks[0].shape, dtype=np.uint8)
    for mask in masks:
        combined_mask = np.maximum(combined_mask, mask)

    # 将掩码转换为 PIL 图像
    mask_image = Image.fromarray(combined_mask).convert("L")
    return mask_image


def main():
    # 设置输入图像路径和输出掩码路径
    input_image_path = 'data/cat_114.jpg'  # 替换为您的输入图像路径
    output_mask_path = 'data/cat_114_mask.jpg'  # 替换为您希望保存的掩码路径

    # 加载图像
    try:
        image = load_image(input_image_path)
        print("成功加载输入图像。")
    except Exception as e:
        print(e)
        return

    # 设置设备
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"使用设备: {device}")

    # 加载预训练的 Mask R-CNN 模型
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
    model.to(device)
    print("加载预训练的 Mask R-CNN 模型。")

    # 获取汽车的分割掩码
    mask = get_car_masks(image, model, device, threshold=0.5)

    if mask:
        # 保存掩码
        mask.save(output_mask_path)
        print(f"汽车分割掩码已保存至 {output_mask_path}")

        # 可视化原图和掩码
        fig, axs = plt.subplots(1, 2, figsize=(15, 10))
        axs[0].imshow(image)
        axs[0].set_title('原始图像')
        axs[0].axis('off')

        axs[1].imshow(mask, cmap='gray')
        axs[1].set_title('汽车分割掩码')
        axs[1].axis('off')

        plt.show()
    else:
        print("未生成任何分割掩码。")


if __name__ == "__main__":
    main()
