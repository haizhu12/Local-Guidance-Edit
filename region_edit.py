import torch
import numpy as np
import cv2
from PIL import Image
import os
import torch.nn.functional as F


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

    # def visualize_attention(self, save_path, image_height=512, image_width=512):
    #     os.makedirs(save_path, exist_ok=True)
    #     if not self.attention_maps:
    #         print("No attention maps captured.")
    #         return
    #
    #     # 示例：可视化最后一组捕获的attention maps
    #     last_key = list(self.attention_maps.keys())[-1]
    #     attn = self.attention_maps[last_key]  # shape: (B, heads, N, M)
    #     attn_mean = attn.mean(dim=1).mean(dim=0)  # (N, M)
    #
    #     # 简化：对所有文本token求平均，得到对图像tokens的注意力
    #     aggregated_attn = attn_mean.mean(dim=-1)
    #     attn_map = aggregated_attn.reshape(64, 64)
    #     attn_map = attn_map / attn_map.max()
    #     attn_map = (attn_map.cpu().numpy() * 255).astype(np.uint8)
    #
    #     attn_map = cv2.resize(attn_map, (image_width, image_height), interpolation=cv2.INTER_CUBIC)
    #     heatmap = cv2.applyColorMap(attn_map, cv2.COLORMAP_JET)
    #     cv2.imwrite(os.path.join(save_path, "attention_map.png"), heatmap)
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


# def prepare_control_image(image_path):
#     image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
#     edges = cv2.Canny(image, 100, 200)
#     edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
#     control_image = Image.fromarray(edges_rgb)
#     control_image = control_image.resize((512, 512))
#     return control_image

# 改进后的控制图像生成
def prepare_control_image(image_path):
    image = cv2.imread(image_path)
    edges = cv2.Canny(image, 50, 150)  # 调整边缘检测阈值
    edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
    control_image = Image.fromarray(edges_rgb)
    control_image = control_image.resize((512, 512))
    return control_image


# def generate_mask(image_path):
#     image = cv2.imread(image_path)
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     _, mask = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
#     mask = cv2.resize(mask, (512, 512))
#     mask = torch.tensor(mask / 255.0).unsqueeze(0).unsqueeze(0).to("cuda")
#     return mask

# 改进后的掩码生成
def generate_mask(image_path):
    # 示例：基于颜色范围生成掩码
    image = cv2.imread(image_path)
    mask = cv2.inRange(image, (0, 0, 0), (255, 255, 255))  # 自定义范围
    mask = cv2.resize(mask, (512, 512))
    mask = torch.tensor(mask / 255.0).unsqueeze(0).unsqueeze(0).to("cuda")
    return mask


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
    inputs = clip_processor(text=[text], images=image, return_tensors="pt", padding=True,truncation=True,max_length=77).to("cuda")
    # with torch.no_grad():
    outputs = clip_model(**inputs)
    image_embeds = outputs.image_embeds
    text_embeds = outputs.text_embeds
    similarity = torch.nn.functional.cosine_similarity(image_embeds, text_embeds).mean()
    return similarity


########################################
# 多阶段扩散 + 局部引导 + ControlNet + AttentionStore + CLIP 优化
# Contrastive Fine-Tuning在此不实现，因为需在训练中完成
########################################

@torch.no_grad()
def multi_stage_local_guidance_clip(
        pipe,
        clip_model,
        clip_processor,
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

    text_input = pipe.tokenizer(prompt, return_tensors="pt",padding="max_length",truncation=True,max_length=77).to(pipe.device)
    text_embeddings = pipe.text_encoder(text_input.input_ids)[0]
    uncond_input = pipe.tokenizer("", return_tensors="pt",padding="max_length",truncation=True,max_length=77).to(pipe.device)
    uncond_embeddings = pipe.text_encoder(uncond_input.input_ids)[0]
    text_batch = torch.cat([uncond_embeddings, text_embeddings])
    torch.manual_seed(0)
    latents = torch.randn((1, pipe.unet.in_channels, 64, 64),device="cuda",dtype=torch.float16)
    pipe.scheduler.set_timesteps(num_inference_steps)
    timesteps = pipe.scheduler.timesteps

    control_image = np.array(control_image).astype(np.float32) / 255.0
    control_image = torch.tensor(control_image).permute(2, 0, 1).unsqueeze(0).to(pipe.device, dtype=torch.float16)

    def control_forward(latent_input, t, text_embeds):
        output = pipe.controlnet(
            latent_input, t, encoder_hidden_states=text_embeds, controlnet_cond=control_image, return_dict=False
        )[0]
        return output

    attention_store.add_hooks(pipe.unet)

    for i, t in enumerate(timesteps):
        if i < half_steps:
            guidance_scale = initial_guidance_scale
            image_guidance_scale = initial_image_guidance_scale
        else:
            guidance_scale = final_guidance_scale
            image_guidance_scale = final_image_guidance_scale
        # t = t.to(pipe.device, dtype=torch.float16)
        latent_model_input = torch.cat([latents] * 2)
        # control_output = control_forward(latent_model_input, t, text_batch)
        # unet_out = pipe.unet(latent_model_input, t, encoder_hidden_states=text_batch + control_output).sample  #######################################################====================================================#############################
        unet_out = pipe.unet(latent_model_input, t, encoder_hidden_states=text_batch).sample
        noise_pred_uncond, noise_pred_cond = unet_out.chunk(2)
        guided_noise = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
        guided_noise = noise_pred_uncond + image_guidance_scale * (guided_noise - noise_pred_uncond)

        # 假设mask现在是(1,1,512,512)
        mask = F.interpolate(mask, size=(64, 64), mode='nearest')  # (1,1,64,64)
        mask=mask.to(pipe.device, dtype=torch.float16)
        # 因为mask是(1,1,64,64)，当与(1,4,64,64)相乘时，会自动广播，要求pytorch版本支持广播匹配。
        # 如果仍有问题，可使用mask.expand(-1,4,-1,-1)将mask扩展为(1,4,64,64)

        # 局部扩散指导：仅对mask区域应用强引导
        guided_noise = guided_noise * mask + noise_pred_uncond * (1 - mask)

        # 更新潜变量
        latents = pipe.scheduler.step(guided_noise, t, latents).prev_sample

        # # CLIP优化：每隔 clip_opt_interval 步对latents进行梯度上升以提高文本一致性
        # if (i + 1) % clip_opt_interval == 0:
        #     # 解码当前图像
        #     # with torch.no_grad():
        #     current_image = decode_latents(pipe, latents)
        #     latents.requires_grad_(True)
        #     # optimizer = torch.optim.SGD([latents], lr=clip_lr)
        #     optimizer = torch.optim.Adam([latents], lr=clip_lr)
        #     for _ in range(clip_opt_steps):
        #         optimizer.zero_grad()
        #         decoded_image = decode_latents(pipe, latents)
        #         sim = clip_similarity(clip_model, clip_processor, decoded_image,
        #                               pipe.tokenizer.decode(text_input.input_ids[0]))
        #         loss = -sim
        #         loss.backward()
        #         optimizer.step()
        #
        #     latents.requires_grad_(False)

    attention_store.remove_hooks()

    result_image = decode_latents(pipe, latents)
    if isinstance(result_image, list):
        result_image = result_image[0]
    return result_image


########################################
# 后处理：基于掩码的局部修复 (Inpainting)
########################################

# def inpaint_image(inpaint_pipe, init_image, mask, prompt, strength=0.8, guidance_scale=7.5):
#     mask_np = (mask.squeeze().cpu().numpy() * 255).astype(np.uint8)
#     mask_pil = Image.fromarray(mask_np, mode="L")
#     result = inpaint_pipe(prompt=prompt, image=init_image, mask_image=mask_pil,
#                           strength=strength, guidance_scale=guidance_scale,
#                           num_inference_steps=30).images[0]
#     return result

# # 调整 Inpainting 逻辑
# def inpaint_image(inpaint_pipe, init_image, mask, prompt, strength=0.8, guidance_scale=7.5):
#     # 转换遮罩为灰度图
#     mask_np = (mask.squeeze().cpu().numpy() * 255).astype(np.uint8)
#     mask_pil = Image.fromarray(mask_np, mode="L")
#
#     # 截断过长的 prompt
#     # prompt = prompt[:77]
#
#     # 调用 inpainting 管道
#     result = inpaint_pipe(
#         prompt=prompt,
#         image=init_image,
#         mask_image=mask_pil,
#         strength=strength,
#         guidance_scale=guidance_scale,
#         num_inference_steps=30
#     ).images[0]
#
#     return result

# 改进后的 inpainting
def inpaint_image(inpaint_pipe, init_image, mask, prompt, strength=0.75, guidance_scale=12.0):
    mask_np = (mask.squeeze().cpu().numpy() * 255).astype(np.uint8)
    mask_pil = Image.fromarray(mask_np, mode="L")

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



########################################
# 多次生成与挑选
########################################

def generate_multiple_and_select_best(
        pipe,
        inpaint_pipe,
        clip_model,
        clip_processor,
        prompt,
        control_image,
        mask,
        attention_store,
        n_samples=100
):
    samples = []
    for i in range(n_samples):
        result_image = multi_stage_local_guidance_clip(
            pipe,
            clip_model,
            clip_processor,
            prompt,
            control_image,
            mask,
            attention_store=attention_store,
            num_inference_steps=100,
            initial_guidance_scale=10.0,
            final_guidance_scale=9.0,
            initial_image_guidance_scale=2.0,
            final_image_guidance_scale=15.0,
            clip_opt_interval=10,
            clip_opt_steps=5,
            clip_lr=0.05
        )
        # print(result_image)
        # 确保 result_image 是单张图像
        if isinstance(result_image, list):
            result_image = result_image[0]
        # 后处理：Inpainting修复
        refined_image = inpaint_image(inpaint_pipe, result_image, mask, prompt)

        # 计算CLIP相似度
        sim = clip_similarity(clip_model, clip_processor, refined_image, prompt)
        samples.append((refined_image, sim.item()))

    # 按CLIP相似度从高到低排序，选择最佳结果
    samples.sort(key=lambda x: x[1], reverse=True)
    best_image, best_score = samples[0]
    return best_image, best_score


########################################
# 示例运行
########################################

if __name__ == "__main__":
    # 准备输入数据和模型
    pipe = load_controlnet_pipeline()
    inpaint_pipe = load_inpaint_pipeline()
    input_image_path = "data/cat.jpg"
    control_image = prepare_control_image(input_image_path)
    mask = generate_mask(input_image_path)
    prompt = "Replace the cat in the image with a realistic dog. Keep the background unchanged"
    clip_model, clip_processor = load_clip_model()
    attention_store = AttentionStore()

    # 多次生成与挑选最优结果
    best_image, best_score = generate_multiple_and_select_best(
        pipe,
        inpaint_pipe,
        clip_model,
        clip_processor,
        prompt,
        control_image,
        mask,
        attention_store,
        n_samples=20
    )

    best_image.save("best_image_after_inpaint_selection.png")
    print(f"Image saved as best_image_after_inpaint_selection.png with CLIP similarity score: {best_score:.4f}")

    # 可视化注意力(最后一次生成的注意力图)
    attention_store.visualize_attention("attention_visualization")
    print("Attention map saved in attention_visualization folder.")
