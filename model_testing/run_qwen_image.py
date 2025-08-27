#!/usr/bin/env python3
"""
Apply ONE fixed prompt to every image in image_inputs/,
save to image_outputs/, and log to logs/qwen-image_test_logs.csv.
Uses diffusers.QwenImageEditPipeline with local Qwen-Image module utilities.
"""

import os
import csv
import sys
from datetime import datetime
from PIL import Image
import torch
from diffusers import QwenImageEditPipeline

# Removed prompt utils import to simplify loading

# -------------------------------
# CONFIG – edit these as needed
# -------------------------------
MODEL_ID = "Qwen/Qwen-Image-Edit"  # Full model - will use memory optimization
INPUT_DIR = r"image_inputs"
OUTDIR = r"image_outputs"
LOG_PATH = r"logs\qwen-image_test_logs.csv"

PROMPT = "Make this look like a matte black chair with product lighting"
NEG_PROMPT = " "  # optional negative prompt

STEPS = 20  # Further reduced for maximum memory efficiency
TRUE_CFG = 4.0  # Qwen-specific cfg scale
DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32  # Use float16 for lower memory usage
SEED = 123  # None for random

# Disabled prompt enhancement to simplify loading
USE_PROMPT_ENHANCEMENT = False  # Set to False to avoid prompt utils dependency
# -------------------------------

def _safe(name, n=100):
    return "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in name.strip())[:n] or "out"

def enhance_prompt(prompt, image):
    """
    Simplified prompt enhancement - just returns the original prompt.
    """
    return prompt

def main():
    os.makedirs(OUTDIR, exist_ok=True)
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🚀 Using device: {device}")
    print(f"📊 Data type: {DTYPE}")

    print(f"📥 Loading model: {MODEL_ID}")
    print("   Note: This will download the model from Hugging Face on first run")
    print("   Subsequent runs will use the cached model")
    
    try:
        # Load model with maximum memory optimization for 12GB VRAM
        pipeline = QwenImageEditPipeline.from_pretrained(
            MODEL_ID, 
            torch_dtype=DTYPE,
            low_cpu_mem_usage=True
        )
        
        # Enable maximum memory optimization
        pipeline.enable_attention_slicing(slice_size="max")
        pipeline.enable_model_cpu_offload()
        pipeline.enable_sequential_cpu_offload()
        
        # Move to device after optimizations
        pipeline.to(device)
        
        pipeline.set_progress_bar_config(disable=None)
        print("✅ Model loaded successfully with maximum memory optimization")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        print("   Make sure you have the latest diffusers installed:")
        print("   pip install git+https://github.com/huggingface/diffusers")
        return

    with open(LOG_PATH, "a", encoding="utf-8", newline="") as lf:
        writer = csv.writer(lf)
        if lf.tell() == 0:
            writer.writerow([
                "timestamp", "model", "device",
                "prompt", "input_path", "output_path",
                "steps", "true_cfg", "seed"
            ])

        gen = torch.Generator(device=device).manual_seed(SEED) if SEED is not None else None

        image_files = [f for f in sorted(os.listdir(INPUT_DIR)) 
                      if os.path.isfile(os.path.join(INPUT_DIR, f))]
        
        print(f"🖼️  Found {len(image_files)} images to process")
        
        for i, fname in enumerate(image_files, 1):
            in_path = os.path.join(INPUT_DIR, fname)
            print(f"\n📸 Processing {i}/{len(image_files)}: {fname}")
            
            try:
                image = Image.open(in_path).convert("RGB")
            except Exception as e:
                print(f"❌ Skipping {fname}: {e}")
                continue  # skip non-images

            # Optional prompt enhancement using local module
            current_prompt = PROMPT
            if USE_PROMPT_ENHANCEMENT:
                current_prompt = enhance_prompt(PROMPT, image)

            inputs = {
                "image": image,
                "prompt": current_prompt,
                "generator": gen,
                "true_cfg_scale": TRUE_CFG,
                "negative_prompt": NEG_PROMPT,
                "num_inference_steps": STEPS,
            }

            try:
                with torch.inference_mode():
                    output = pipeline(**inputs)
                    out_img = output.images[0]

                stem = _safe(os.path.basename(in_path) + "_" + current_prompt)
                out_path = os.path.join(OUTDIR, f"{stem}.png")
                out_img.save(out_path)

                writer.writerow([
                    datetime.now().isoformat(timespec="seconds"),
                    MODEL_ID, device,
                    current_prompt, in_path, out_path,
                    STEPS, TRUE_CFG, SEED or ""
                ])

                print(f"✅ {fname} → {out_path}")
                
            except Exception as e:
                print(f"❌ Failed to process {fname}: {e}")

    print(f"\n🎉 Done! Outputs → {OUTDIR}")
    print(f"📝 Log → {LOG_PATH}")

if __name__ == "__main__":
    main()
