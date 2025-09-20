# dirty_inpaint.py
# Masked img2img + composite-back workflow for base (non-inpainting) SD/SDXL models.
# White in mask = area to change. Black = keep original.

import argparse, os, math, io, random
from typing import Tuple
import numpy as np
from PIL import Image, ImageFilter, ImageOps
import torch
from diffusers import (
    StableDiffusionXLImg2ImgPipeline,
    StableDiffusionImg2ImgPipeline,
    AutoPipelineForImage2Image,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    KDPM2AncestralDiscreteScheduler,
    KDPM2DiscreteScheduler,
    DDIMScheduler,
)

# ---------- helpers ----------

def choose_img2img_pipe(model_path: str, sdxl: bool, dtype, device: str):
    """Choose and validate the appropriate img2img pipeline based on model type.

    If the loaded pipeline appears mismatched (e.g., SDXL weights in SD1.5 pipe),
    reload with the correct pipeline class automatically.
    """
    pipe = None

    def is_sdxl_pipeline(p):
        # Quick heuristic: SDXL pipes have text encoders 1 and 2 and require add_time_ids
        return hasattr(p, "text_encoder") and hasattr(p, "text_encoder_2")

    if sdxl:
        print("Loading as SDXL model...")
        pipe = StableDiffusionXLImg2ImgPipeline.from_single_file(model_path, torch_dtype=dtype)
    else:
        print("Loading as SD 1.5 model...")
        try:
            pipe = StableDiffusionImg2ImgPipeline.from_single_file(model_path, torch_dtype=dtype)
        except Exception as e:
            print(f"Failed to load as SD 1.5, trying SDXL: {e}")
            pipe = StableDiffusionXLImg2ImgPipeline.from_single_file(model_path, torch_dtype=dtype)

    # If user didn't specify --sdxl and the loaded pipe looks like SDXL, switch
    if not sdxl and is_sdxl_pipeline(pipe):
        print("Detected SDXL components in model; reloading SDXL img2img pipeline...")
        pipe = StableDiffusionXLImg2ImgPipeline.from_single_file(model_path, torch_dtype=dtype)

    pipe = pipe.to(device)
    return pipe

def load_rgb(path: str) -> Image.Image:
    img = Image.open(path)
    if img.mode != "RGB":
        img = img.convert("RGB")
    return img

def load_mask_L(path: str) -> Image.Image:
    m = Image.open(path)
    if m.mode != "L":
        m = m.convert("L")
    return m

def to_multiple_of_8(size: Tuple[int, int]) -> Tuple[int, int]:
    w, h = size
    w = (w // 8) * 8
    h = (h // 8) * 8
    return max(8, w), max(8, h)

def resize_if_needed(img: Image.Image, target: Tuple[int, int]) -> Image.Image:
    if target is None:
        return img
    if img.size == target:
        return img
    return img.resize(target, Image.LANCZOS)

def normalize_mask(mask: Image.Image, invert: bool, feather_px: int, target_size=None) -> Image.Image:
    if target_size and mask.size != target_size:
        mask = mask.resize(target_size, Image.LANCZOS)
    if invert:
        mask = ImageOps.invert(mask)
    # optional feather on edges
    if feather_px > 0:
        mask = mask.filter(ImageFilter.GaussianBlur(feather_px))
    return mask

def binarize_mask(mask: Image.Image) -> Image.Image:
    """Convert all non-white pixels to black, keep pure white as white.

    This ensures a strict black/white mask regardless of source artifacts.
    """
    if mask.mode != "L":
        mask = mask.convert("L")
    return mask.point(lambda p: 255 if p == 255 else 0)

def fill_mask_with_noise_or_blur(
    base: Image.Image, mask: Image.Image, mode: str, noise_sigma: float, blur_radius: float
) -> Image.Image:
    """
    Returns a new init image where masked region is replaced by noise or blur,
    unmasked region remains the same.
    """
    w, h = base.size
    if mode == "noise":
        # gaussian noise around local pixel values — simple approach: white noise, then blend by mask
        arr = np.array(base).astype(np.float32)
        noise = np.random.normal(0, noise_sigma, arr.shape).astype(np.float32)
        mixed = np.clip(arr + noise, 0, 255).astype(np.uint8)
        noisy_img = Image.fromarray(mixed)
        # composite: white areas (mask) take from noisy_img, black keep base
        init = Image.composite(noisy_img, base, mask)
    elif mode == "blur":
        blurred = base.filter(ImageFilter.GaussianBlur(blur_radius))
        init = Image.composite(blurred, base, mask)
    else:
        raise ValueError("fill mode must be 'noise' or 'blur'")
    return init

def set_scheduler(pipe, which: str):
    which = which.lower()
    if which in ["dpmpp_2m", "dpm++2m", "dpmpp2m"]:
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    elif which in ["euler_a", "euler-ancestral", "euler_ancestral"]:
        pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
    elif which in ["euler"]:
        pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
    elif which in ["k_dpm2_a", "kdpm2a", "k_dpm2_ancestral"]:
        pipe.scheduler = KDPM2AncestralDiscreteScheduler.from_config(pipe.scheduler.config)
    elif which in ["k_dpm2", "kdpm2"]:
        pipe.scheduler = KDPM2DiscreteScheduler.from_config(pipe.scheduler.config)
    elif which in ["ddim"]:
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    else:
        # leave default
        pass

# ---------- main ----------

def main():
    ap = argparse.ArgumentParser(description="Dirty inpaint via masked img2img + composite")
    ap.add_argument("--model", required=True, help="Path to .safetensors/.ckpt (SD/SDXL)")
    ap.add_argument("--image", required=True, help="Input image (RGB)")
    ap.add_argument("--mask", required=True, help="Mask image (white=inpaint, black=keep)")
    ap.add_argument("--prompt", required=True, help="Positive prompt")
    ap.add_argument("--neg", default="low quality, blurry, artifacts, deformed, oversharp", help="Negative prompt")
    ap.add_argument("--out", default="dirty_inpaint.png", help="Output path")
    ap.add_argument("--steps", type=int, default=28)
    ap.add_argument("--cfg", type=float, default=5.5, help="Guidance scale")
    ap.add_argument("--strength", type=float, default=0.7, help="Img2img strength for the masked region")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--scheduler", default="dpmpp_2m", help="dpmpp_2m|euler_a|euler|k_dpm2_a|k_dpm2|ddim")
    ap.add_argument("--sdxl", action="store_true", help="Hint: treat as SDXL model")

    # size control
    ap.add_argument("--keep-size", action="store_true", help="Use the original image size (default)")
    ap.add_argument("--size", type=str, default=None, help="Override size WxH, e.g. 768x768")

    # mask controls
    ap.add_argument("--invert-mask", action="store_true", help="Invert mask polarity")
    ap.add_argument("--feather", type=int, default=4, help="Gaussian blur (px) applied to mask edges")

    # init fill controls
    ap.add_argument("--fill", default="noise", choices=["noise", "blur"], help="How to fill masked region before img2img")
    ap.add_argument("--noise-sigma", type=float, default=30.0, help="Stddev for noise fill (if --fill=noise)")
    ap.add_argument("--blur-radius", type=float, default=8.0, help="Blur radius for blur fill (if --fill=blur)")

    args = ap.parse_args()

    # device & dtype
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    if device == "cpu":
        print("WARNING: CUDA not available; running on CPU will be slow.")

    # seed
    if args.seed is None:
        args.seed = random.randrange(0, 2**31 - 1)
    gen = torch.Generator(device=device).manual_seed(args.seed)

    # load images
    base = load_rgb(args.image)
    mask = load_mask_L(args.mask)

    # size handling
    target_size = None
    if args.size:
        try:
            w, h = map(int, args.size.lower().split("x"))
            target_size = to_multiple_of_8((w, h))
        except Exception:
            raise ValueError("--size must be like 768x768")
    elif args.keep_size:
        target_size = to_multiple_of_8(base.size)
    else:
        # default: keep original size (rounded down to multiple of 8)
        target_size = to_multiple_of_8(base.size)

    base = resize_if_needed(base, target_size)
    mask = normalize_mask(mask, invert=args.invert_mask, feather_px=args.feather, target_size=target_size)
    # Force strict black/white interpretation: only pure white stays white
    mask = binarize_mask(mask)

    # prepare init image (mask region replaced by noise/blur; rest untouched)
    init_img = fill_mask_with_noise_or_blur(base, mask, args.fill, args.noise_sigma, args.blur_radius)

    # load pipeline
    pipe = choose_img2img_pipe(args.model, args.sdxl, dtype, device)

    # Prefer SDPA; fall back to attention slicing if needed (no xFormers)
    try:
        pipe.enable_attention_slicing("max")
    except Exception:
        pass

    # optional scheduler
    set_scheduler(pipe, args.scheduler)

    # some SDXL community models ship with watermarker; disable if present
    if hasattr(pipe, "watermarker"):
        try:
            pipe.watermarker = None
        except Exception:
            pass

    # run img2img over whole image
    with torch.inference_mode():
        result = pipe(
            prompt=args.prompt,
            negative_prompt=args.neg,
            image=init_img,
            guidance_scale=args.cfg,
            strength=args.strength,
            num_inference_steps=args.steps,
            generator=gen,
        ).images[0]

    # composite back: keep original outside mask, use generated inside mask
    # PIL compositing: Image.composite(foreground, background, mask) uses mask as alpha of 'foreground'
    final = Image.composite(result, base, mask)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    final.save(args.out)
    print(f"Saved {args.out} (seed={args.seed}, size={final.size[0]}x{final.size[1]})")

if __name__ == "__main__":
    main()
