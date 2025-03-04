import torch
import numpy as np
import cv2
from PIL import Image
import os
import torch.nn.functional as F
import matplotlib.pyplot as plt
# segment-anything 的核心类和方法
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
from null_text_w_ptp import text2image_instructpix2pix_blend
from msk_from_attn import *
from diffusers import DDIMScheduler,StableDiffusionInstructPix2PixPipeline,DPMSolverMultistepScheduler
from null_text_w_ptp import text2image_instructpix2pix_blend
import argparse
from PIL import ImageEnhance
from msk_from_attn import action_classify

from diffusers import (
    StableDiffusionControlNetPipeline,
    ControlNetModel,
    DDIMScheduler,
    StableDiffusionInpaintPipeline,
)
from transformers import CLIPProcessor, CLIPModel


########################################
# AttentionStore用于捕获和分析注意力
########################################
class AttentionStore:
    def __init__(self):
        self.attention_maps = {}
        self.hooks = []

    def add_hooks(self, unet):
        # 给UNet的交叉注意力层添加hooks
        for name, module in unet.named_modules():
            if hasattr(module, 'transformer_blocks'):
                for idx, blk in enumerate(module.transformer_blocks):
                    attn = blk.attn2  # cross-attention层
                    self.hooks.append(attn.register_forward_hook(self.save_attention(name, idx)))

    def save_attention(self, name, idx):
        def hook(module, input, output):
            attn_weights = output[1]
            self.attention_maps[f"{name}_{idx}"] = attn_weights.detach().cpu()

        return hook

    def remove_hooks(self):
        for h in self.hooks:
            h.remove()

    def visualize_attention(self, save_path, image_height=512, image_width=512):
        os.makedirs(save_path, exist_ok=True)
        if not self.attention_maps:
            print("No attention maps captured.")
            return

        # 获取最后一组捕获的注意力
        last_key = list(self.attention_maps.keys())[-1]
        attn = self.attention_maps[last_key]  # shape: (B, heads, N, M)

        print(f"Attention map shape: {attn.shape}")  # 调试打印

        # 检查维度
        if attn.dim() != 2 or attn.size(0) != 4096 or attn.size(1) != 320:
            raise ValueError(f"Invalid attention map shape: {attn.shape}")

        # 对文本 tokens 求平均
        aggregated_attn = attn.mean(dim=-1)  # 从 [4096, 320] -> [4096]

        # 检查是否可以 reshape 为 64x64
        if aggregated_attn.numel() != 64 * 64:
            raise ValueError(f"Cannot reshape attention to 64x64. Current size: {aggregated_attn.size()}")

        attn_map = aggregated_attn.reshape(64, 64).cpu().numpy()
        attn_map = attn_map / attn_map.max()
        attn_map = (attn_map * 255).astype(np.uint8)

        # 调整到目标分辨率
        attn_map = cv2.resize(attn_map, (image_width, image_height), interpolation=cv2.INTER_CUBIC)
        heatmap = cv2.applyColorMap(attn_map, cv2.COLORMAP_JET)
        cv2.imwrite(os.path.join(save_path, "attention_map.png"), heatmap)

    def enhance_attention_for_token(self, target_token_positions, scale=2.0):
        # 在实际实现中，可在生成过程中对attn_weights进行修改
        # 这里仅提供思路，实际需要在hook中对 attn_weights 做相应修改
        pass


########################################
# 模型加载与辅助函数
########################################

def load_controlnet_pipeline():
    controlnet = ControlNetModel.from_pretrained(
        "lllyasviel/sd-controlnet-canny", torch_dtype=torch.float16
    )
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        controlnet=controlnet,
        torch_dtype=torch.float16
    )
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.to("cuda")
    return pipe


def load_inpaint_pipeline():
    inpaint_pipe = StableDiffusionInpaintPipeline.from_pretrained(
        "runwayml/stable-diffusion-inpainting",
        torch_dtype=torch.float16
    ).to("cuda")
    # 设置 tokenizer 的最大长度
    inpaint_pipe.tokenizer.model_max_length = 77
    return inpaint_pipe


def prepare_control_image(image_path):
    image = cv2.imread(image_path)
    edges = cv2.Canny(image, 50, 150)  # 调整边缘检测阈值
    edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
    control_image = Image.fromarray(edges_rgb)
    control_image = control_image.resize((512, 512))
    return control_image


def generate_mask(image_path):
    # 示例：基于颜色范围生成掩码
    # image = cv2.imread(image_path)
    # mask = cv2.inRange(image, (0, 0, 0), (255, 255, 255))  # 自定义范围
    # mask = cv2.resize(mask, (512, 512))
    # mask = torch.tensor(mask / 255.0).unsqueeze(0).unsqueeze(0).to("cuda")

    # 选择使用的模型文件及其类型
    # 官方仓库提供了 vit_h / vit_l / vit_b 三种，您可根据需要自行下载
    model_type = "vit_l"
    sam_checkpoint = "ckpts/sam_vit_l_0b3195.pth"  # 模型权重，可从官方GitHub获取

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 注册并加载模型
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)

    # 创建自动掩码生成器
    mask_generator = SamAutomaticMaskGenerator(sam)

    # 读取原图
    image_path = image_path
    image_bgr = cv2.imread(image_path)  # BGR格式
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    # 将图像送入 SAM 进行自动分割
    masks = mask_generator.generate(image_rgb)

    # 如果图中只存在一只猫且它是最大的前景，可以简单地按面积最大选取
    max_area = 0
    best_mask = None

    for i, m in enumerate(masks):
        area = m["area"]  # SAM 自动分割返回的 mask 信息中有 "area"
        if area > max_area:
            max_area = area
            best_mask = m["segmentation"]  # bool 型二维数组

    if best_mask is None:
        raise ValueError("SAM did not generate any mask. Check the input image or SAM settings.")

    # 将 bool mask 转成 uint8 (0或255)
    _mask = best_mask.astype(np.uint8) * 255
    # 如果得到的是反色（猫=黑，背景=白），则执行反转：
    _mask = 255 - _mask
    _mask_pil = Image.fromarray(_mask)
    _mask_pil.save("mask_dog.png")
    print("Saved mask.png")
    input_mask1 = Image.open("mask_dog.png")
    input_mask = Image.open("mask_dog.png").convert("L")  # 转为灰度图
    input_mask = np.array(input_mask)
    input_mask = torch.tensor(input_mask).unsqueeze(0).unsqueeze(0) / 255.0  # 归一化到 [0, 1]，形状 (1, 1, H, W)
    # input_mask = np.array(input_mask)
    # input_mask = torch.tensor(input_mask).unsqueeze(0).unsqueeze(0) / 255.0  # 归一化到 [0, 1]，形状 (1, 1, H, W)
    return input_mask


def decode_latents(pipe, latents):
    latents = latents / 0.18215
    image = pipe.vae.decode(latents).sample
    image = (image / 2 + 0.5).clamp(0, 1).cpu().permute(0, 2, 3, 1).numpy()[0]
    image = (image * 255).astype(np.uint8)
    return Image.fromarray(image)


########################################
# CLIP 引导相关函数
########################################

def load_clip_model():
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to("cuda")
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    return clip_model, clip_processor


def clip_similarity(clip_model, clip_processor, image, text):
    inputs = clip_processor(
        text=[text],
        images=image,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=77
    ).to("cuda")

    outputs = clip_model(**inputs)
    image_embeds = outputs.image_embeds
    text_embeds = outputs.text_embeds
    similarity = torch.nn.functional.cosine_similarity(image_embeds, text_embeds).mean()
    return similarity


########################################
# 多阶段扩散 + 局部引导 + ControlNet + AttentionStore + CLIP 优化
########################################

@torch.no_grad()
def multi_stage_local_guidance_clip(
        pipe,
        # clip_model,
        # clip_processor,
        prompt,
        control_image,
        mask,
        attention_store,
        num_inference_steps=50,
        initial_guidance_scale=3.0,
        final_guidance_scale=9.0,
        initial_image_guidance_scale=2.0,
        final_image_guidance_scale=8.0,
        clip_opt_interval=10,
        clip_opt_steps=5,
        clip_lr=0.05
):
    half_steps = num_inference_steps // 2

    text_input = pipe.tokenizer(prompt, return_tensors="pt",
                                padding="max_length",
                                truncation=True,
                                max_length=77).to(pipe.device)
    text_embeddings = pipe.text_encoder(text_input.input_ids)[0]
    uncond_input = pipe.tokenizer("", return_tensors="pt",
                                  padding="max_length",
                                  truncation=True,
                                  max_length=77).to(pipe.device)
    uncond_embeddings = pipe.text_encoder(uncond_input.input_ids)[0]
    text_batch = torch.cat([uncond_embeddings, text_embeddings])

    torch.manual_seed(0)
    latents = torch.randn((1, pipe.unet.in_channels, 64, 64),
                          device="cuda",
                          dtype=torch.float16)

    pipe.scheduler.set_timesteps(num_inference_steps)
    timesteps = pipe.scheduler.timesteps

    control_image = np.array(control_image).astype(np.float32) / 255.0
    control_image = torch.tensor(control_image).permute(2, 0, 1).unsqueeze(0).to(pipe.device, dtype=torch.float16)

    attention_store.add_hooks(pipe.unet)

    for i, t in enumerate(timesteps):
        if i < half_steps:
            guidance_scale = initial_guidance_scale
            image_guidance_scale = initial_image_guidance_scale
        else:
            guidance_scale = final_guidance_scale
            image_guidance_scale = final_image_guidance_scale

        latent_model_input = torch.cat([latents] * 2)
        # 这里可以视需要接入 ControlNet；此处暂时省略 ControlNet 的输出
        unet_out = pipe.unet(
            latent_model_input,
            t,
            encoder_hidden_states=text_batch
        ).sample

        noise_pred_uncond, noise_pred_cond = unet_out.chunk(2)
        guided_noise = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
        guided_noise = noise_pred_uncond + image_guidance_scale * (guided_noise - noise_pred_uncond)

        # 局部扩散指导：仅对 mask 区域应用强引导
        mask_resized = F.interpolate(mask, size=(64, 64), mode='nearest')
        mask_resized = mask_resized.to(pipe.device, dtype=torch.float16)

        guided_noise = guided_noise * mask_resized + noise_pred_uncond * (1 - mask_resized)

        # 更新潜变量
        latents = pipe.scheduler.step(guided_noise, t, latents).prev_sample

        # --- 如果也不需要在生成阶段做 CLIP 微调，可注释掉以下 CLIP 优化部分 ---
        # if (i + 1) % clip_opt_interval == 0:
        #     latents.requires_grad_(True)
        #     optimizer = torch.optim.Adam([latents], lr=clip_lr)
        #     for _ in range(clip_opt_steps):
        #         optimizer.zero_grad()
        #         decoded_image = decode_latents(pipe, latents)
        #         sim = clip_similarity(clip_model, clip_processor, decoded_image,
        #                               pipe.tokenizer.decode(text_input.input_ids[0]))
        #         loss = -sim
        #         loss.backward()
        #         optimizer.step()
        #     latents.requires_grad_(False)
        # --- CLIP 优化部分结束 ---

    attention_store.remove_hooks()

    result_image = decode_latents(pipe, latents)
    if isinstance(result_image, list):
        result_image = result_image[0]
    return result_image


########################################
# 后处理：基于掩码的局部修复 (Inpainting)
########################################

def inpaint_image(inpaint_pipe, init_image, mask, prompt, strength=0.75, guidance_scale=12.0):
    mask_np = (mask.squeeze().cpu().numpy() * 255).astype(np.uint8)
    mask_pil = Image.fromarray(mask_np, mode="L")

    # # 2. 读取原图 (必须是 RGB)
    # original_image = Image.open("data/cat2.jpg").convert("RGB")
    #
    # # 3. 读取上一步保存的 cat_mask.png
    # mask_image = Image.open("cat_mask.png").convert("L")  # L是灰度图
    # # 如果黑白反了，可以再做 255 - mask 或者在 step2 里写反选逻辑
    negative_prompt = "distorted face, unrealistic background, cartoonish style"

    result = inpaint_pipe(
        prompt=prompt,
        image=init_image,
        mask_image=mask_pil,
        strength=strength,
        guidance_scale=guidance_scale,
        negative_prompt=negative_prompt,
        num_inference_steps=50
    ).images[0]

    return result


def model_init_insert(args):
    t1 = time()

    ip2p_pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
        args.instructpix2pix_path,
        safety_checker=None)
    ip2p_pipe.to("cuda")
    ip2p_pipe.scheduler = DPMSolverMultistepScheduler.from_config(ip2p_pipe.scheduler.config)

    sam_predictor = sam_model_registry["vit_h"](
        checkpoint=args.sam_ckpt_path,
    ).to('cuda')

    sd_pipe = None

    mb_pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
        args.magicbrush_path,
        safety_checker=None)
    mb_pipe.to("cuda")
    mb_pipe.scheduler = DPMSolverMultistepScheduler.from_config(ip2p_pipe.scheduler.config)

    t2 = time()
    print(f'>>> Initializing used {t2 - t1:.2f} s.')

    return sam_predictor, sd_pipe, ip2p_pipe, mb_pipe


def parse_args():
    parser = argparse.ArgumentParser(description='Process some parameters.')
    parser.add_argument('--instruction', type=str, default="Make the dog a golden statue", help='The instruction')
    parser.add_argument('--image_path', type=str, default="./data/21.png", help='path to image')
    parser.add_argument('--image_path_style', type=str, default="./data/10.png", help='path to image')
    parser.add_argument('--sam_ckpt_path', type=str, default="./ckpts/sam_vit_h_4b8939.pth",
                        help='The path to SAM checkpoint')
    parser.add_argument('--instructpix2pix_path', type=str, default="timbrooks/instruct-pix2pix",
                        help='The path to InstructPix2Pix checkpoint')
    parser.add_argument('--magicbrush_path', type=str, default="vinesmsuic/magicbrush-jul7",
                        help='The path to MagicBrush checkpoint')
    parser.add_argument('--output_path', type=str, default='./outputs', help='The output path')
    parser.add_argument('--threshold', type=int, default=30, help='The threshold of edge smoother')
    parser.add_argument('--alpha', type=float, default=0.6,
                        help='The alpha value for blending style image and original image')
    parser.add_argument('--blend_beta', type=float, default=0.2, help='The beta value for fusing IP2P and MB')
    parser.add_argument('--image_guidance_scale', type=float, default=7.5, help='The image guidance scale')
    parser.add_argument('--inference_steps', type=int, default=20, help='The inference steps')
    parser.add_argument('--dilate_strength', type=int, default=4, help='The dilate strength')
    parser.add_argument('--seed', type=int, default=42, help='seed')
    parser.add_argument('--brightness', type=float, default=1, help='The brightness of the style image')

    args = parser.parse_args()
    return args



def infernces(args):
    # 准备输入数据和模型
    # 1. initialize pretrained models
    sam_predictor, _, pipe_instru, mb_pipe = model_init_insert(args)
    pipe_controlnet = load_controlnet_pipeline()

    input_image_path = "data/dog.png"
    control_image = prepare_control_image(input_image_path)

    # 2. 读取原图 (必须是 RGB)
    original_image = Image.open("data/dog.png").convert("RGB")


    mode = action_classify(pipe_instru, args.instruction)  # 根据输入文本指令调用 action_classify，确定当前的处理模式（ADD、REMOVE 或 CHANGE）

    mask = generate_mask(input_image_path)
    prompt = "Turn the dog white. Everything else stays the same"
    # clip_model, clip_processor = load_clip_model()
    attention_store = AttentionStore()
    generator = torch.Generator().manual_seed(args.seed)



    # 这里示例生成 n_samples 张图，每张图都做 inpaint 后依次保存
    n_samples = 5
    os.makedirs("generated_images", exist_ok=True)

    for i in range(n_samples):
        result_image = multi_stage_local_guidance_clip(
            pipe_controlnet,
            # clip_model,
            # clip_processor,
            prompt,
            control_image,
            mask,
            attention_store=attention_store,
            num_inference_steps=50,
            initial_guidance_scale=10.0,
            final_guidance_scale=15,
            initial_image_guidance_scale=10.0,
            final_image_guidance_scale=15.0,
            clip_opt_interval=10,
            clip_opt_steps=5,
            clip_lr=0.05
        )

    # style_image, _ = text2image_instructpix2pix_blend(
    #     pipe_instru,
    #     mb_pipe,
    #     mode,
    #     original_image,
    #
    #     prompt,
    #     attention_store,
    #     guidance_scale=args.image_guidance_scale,
    #     generator=generator,
    #     num_inference_steps=args.inference_steps,
    #     beta=args.blend_beta)  # 调用 text2image_instructpix2pix_blend，根据文本指令和输入图片生成风格化图片 style_image，并在 controller_bld 中存储注意力信息。



    #
    #
    #
    #     # 后处理：Inpainting 修复
    #     refined_image = inpaint_image(inpaint_pipe, result_image, mask, prompt)
    #
    #     # 保存结果
    #     save_path = os.path.join("generated_images", f"generated_image_{i}.png")
    #     refined_image.save(save_path)
    #     print(f"[{i+1}/{n_samples}] Saved: {save_path}")
    #
    # # 可视化注意力(最后一次生成的注意力图)
    # attention_store.visualize_attention("attention_visualization")
    # print("Attention map saved in attention_visualization folder.")




    # 3. 读取上一步保存的 cat_mask.png
    # mask_image = Image.open("cat_mask.png").convert("L")  # L是灰度图
    # 如果黑白反了，可以再做 255 - mask 或者在 step2 里写反选逻辑

    # 4. 编写 prompt
    prompt = "Turn the dog white. Everything else stays the same"
    negative_prompt = "extra objects, multiple animals, deformed background"

    # 5. 推理生成
    inpaint_pipe = load_inpaint_pipeline()
    result = inpaint_pipe(
        prompt=prompt,
        image=result_image,
        mask_image=mask,
        strength=0.75,             # 可调
        guidance_scale=12,        # 可调
        negative_prompt=negative_prompt,
        num_inference_steps=500
    ).images[0]

    result.save("cat_replaced.png")
    print("Saved cat_replaced_by_dog.png")
########################################
# 主函数：直接循环生成并保存所有结果
########################################

if __name__ == "__main__":
    args = parse_args()
    infernces(args)

