# Local-Guidance-Edit
# Diffusion-Based Zero-Shot Image Editing with Enhanced Cross-Attention Control and Localized Guidance
### [Project Page](https://www.timothybrooks.com/instruct-pix2pix/) | [Paper]() | [Data]()


 XI'AN JIAOTONG UNIVERSITY <br>
  
  ![start](https://github.com/user-attachments/assets/1c8c9379-35eb-4ba1-9f35-e2d403132cd3)


## Auxiliary model

### Hugging Face Hugging Face sam_vit_l_0b3195.pth sam_vit_h_4b8939.pth

Follow the model Hugging Face [Hugging Face sam_vit_l_0b3195.pth](https://huggingface.co/spaces/facebook/ov-seg/tree/f9b1bcfebfafe86b45b0cf16a1797ca5663d81af) 。Putting pth models into the ckpt folder

### Hugging Face sd-controlnet-canny, and download a pretrained model:
Follow the model Hugging Face [sd-controlnet-canny](https://huggingface.co/lllyasviel/sd-controlnet-canny) 。Putting pth models into the Local-Guidance-Edit folder

### Hugging Face clip-vit-base-patch32, and download a pretrained model:
Follow the model  [clip-vit-base-patch32](https://huggingface.co/openai/clip-vit-base-patch32) 。Putting pth models into the Local-Guidance-Edit folder

### Hugging Face stable-diffusion-inpainting, and download a pretrained model:
Follow the model  [stable-diffusion-inpainting](https://huggingface.co/stabilityai/stable-diffusion-2-inpainting) 。Putting pth models into the Local-Guidance-Edit folder

Follow the model  [stable-diffusion-v1-5](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5)

Directory structure
```
Local-Guidance-Edit/
--runwayml/
----stable-diffusion-inpainting
----stable-diffusion-v1-5
```
### Hugging Face stable-diffusion-inpainting, and download a pretrained model:
Follow the model  [timbrooks/instruct-pix2pix](https://huggingface.co/spaces/timbrooks/instruct-pix2pix) 。Putting pth models into the Local-Guidance-Edit folder

### Hugging Face vinesmsuic/magicbrush-jul7, and download a pretrained model:
Follow the model  [magicbrush-jul7](https://huggingface.co/vinesmsuic/magicbrush-jul7) 。Putting pth models into the Local-Guidance-Edit folder

## Setup

Install all dependencies with:
```
ubuntu 20.04  python=3.9  pytorch=1.11.0  cuda=11.6  diffusers 0.31.0
```
conda create -n <your-name> python=3.9
conda activate <your-name>
pip install -r requirements.txt  -i https://pypi.tuna.tsinghua.edu.cn/simple

### Edit a single image:
```
--instruction "Make the cat's featherswhite" --image_path "data/xxxx.jpg"

```

### Results of the comparison with the state of art methodology:
```
results
```
![dingxingyanjiu](https://github.com/user-attachments/assets/4bb3a7a7-6289-45b6-8a5e-1894166ec8ab)


_(For advice on how to get the best results by tuning parameters, see the [Tips](https://github.com/timothybrooks/instruct-pix2pix#tips) section)._

## Setup
virtualized environment：ubuntu 20.04 LTS, GPU:Nvidia RTX 4090 24GB.
Install all dependencies with:
```
conda env create -f environment.yaml
```

Download the pretrained models by running:
```
bash scripts/download_checkpoints.sh
```

## Generated Dataset

All our test images are from the internet. Image size size is 512*512

## Evaluation

To generate plots like the ones in Figures 8 and 10 in the paper, run the following command:

```
python metrics/compute_metrics.py --ckpt /path/to/your/model.ckpt
```

## Tips

If the generated results do not meet your expectations, the following factors may be responsible:

1.Insufficient Image Modification: The image guidance weight might be set too high, which controls the degree of similarity between the output and the input background. If your intended modification requires significant changes, but the image guidance weight restricts such transformations, the output may not align with your expectations. Additionally, a low text weight might reduce the model’s adherence to textual instructions. Try lowering the image guidance weight or increasing the text weight to optimize the generation process.

2.Excessive Image Modification: If the generated image changes too drastically, leading to a loss of important details from the original image, consider:Increasing the image guidance weight or decreasing the text weight.

3.Rewording Text Instructions: The phrasing of instructions can impact the generated results. For example, the prompts "turn him into a dog", "make him a dog", and "as a dog" may yield different outputs. Experimenting with different phrasings may improve the generated image quality.

4.Increasing the Number of Steps: In some cases, increasing the number of inference steps may enhance image quality.

5.Facial Artifacts: If the generated faces appear distorted or unnatural, it may be due to the Stable Diffusion Autoencoder struggling with small faces in the image. Providing a more precise mask can help achieve higher-quality results.

## Comments

- Our codebase is based on the [Stable Diffusion codebase](https://github.com/CompVis/stable-diffusion).

## BibTeX

```
Stay tuned!
```
## acknowledgements
The code for this study references CLIPstyler (2022 CVPR), Yuanze Lin et al. (2024 CVPR), ZONE (2024 CVPR), InstructPix2Pix (2023 CVPR), Segment Anything (2023 ICCV) Stable Diffusion - ControlNet（2023）
