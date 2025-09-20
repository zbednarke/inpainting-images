import argparse, torch, os
from diffusers import StableDiffusionXLPipeline, StableDiffusionPipeline, AutoPipelineForText2Image

def choose_pipe(model_path: str, sdxl: bool, device: str):
    dtype = torch.float16 if device == "cuda" else torch.float32

    # Prefer a specific pipeline if you know it:
    if sdxl:
        pipe = StableDiffusionXLPipeline.from_single_file(model_path, torch_dtype=dtype)
    else:
        try:
            pipe = StableDiffusionPipeline.from_single_file(model_path, torch_dtype=dtype)
        except Exception:
            # Fallback: auto-detect pipeline type
            pipe = AutoPipelineForText2Image.from_single_file(model_path, torch_dtype=dtype)
    pipe = pipe.to(device)

    # Memory-friendly attention
    try:
        pipe.enable_xformers_memory_efficient_attention()
    except Exception:
        pipe.enable_attention_slicing("max")
        if hasattr(pipe, "set_progress_bar_config"):
            pipe.set_progress_bar_config(disable=False)

    # Mixed precision / TF32 hints
    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    return pipe

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Path to .safetensors/.ckpt")
    ap.add_argument("--out", default="out.png")
    ap.add_argument("--prompt", default="cute puppy catching a ball in a green grassy field")
    ap.add_argument("--neg", default="low quality, blurry, artifacts")
    ap.add_argument("--steps", type=int, default=28)
    ap.add_argument("--cfg", type=float, default=5.5)
    ap.add_argument("--w", type=int, default=768)
    ap.add_argument("--h", type=int, default=768)
    ap.add_argument("--sdxl", action="store_true", help="Hint: treat as SDXL")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    pipe = choose_pipe(args.model, args.sdxl, device)

    # Some SDXL community models expect "add_watermark=False"
    if hasattr(pipe, "watermarker"):
        try:
            pipe.watermarker = None
        except Exception:
            pass

    # Generate
    with torch.inference_mode():
        img = pipe(
            prompt=args.prompt,
            negative_prompt=args.neg,
            num_inference_steps=args.steps,
            guidance_scale=args.cfg,
            width=args.w,
            height=args.h
        ).images[0]

    img.save(args.out)
    print(f"Saved {args.out}")

if __name__ == "__main__":
    main()
