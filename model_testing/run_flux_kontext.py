#!/usr/bin/env python3
"""
Apply ONE fixed prompt to every image in image_inputs/,
save to image_outputs/, and log to logs/flux-kontext_test_logs.csv.
Uses FLUX Kontext with FP8 quantization for efficient inference.
"""

import os
import csv
import sys
import time
from datetime import datetime
from PIL import Image
import torch
import numpy as np

# Add FLUX Kontext module to path
flux_kontext_path = os.path.join(os.path.dirname(__file__), '..', 'flux-kontext', 'src')
sys.path.append(flux_kontext_path)

from flux.cli import SamplingOptions
from flux.sampling import denoise, get_noise, get_schedule, prepare, unpack
from flux.util import (
    configs,
    embed_watermark,
    load_ae,
    load_clip,
    load_flow_model,
    load_t5,
    save_image,
)

# -------------------------------
# CONFIG – edit these as needed
# -------------------------------
MODEL_ID = "flux-dev-kontext"  # FLUX.1 Kontext model
INPUT_DIR = r"image_inputs"
OUTDIR = r"image_outputs"
LOG_PATH = r"logs\flux-kontext_test_logs.csv"

PROMPT = "Make this look like a matte black chair with product lighting"
NEG_PROMPT = " "  # optional negative prompt

# FLUX Kontext specific settings
WIDTH = 1024
HEIGHT = 1024
STEPS = 50
GUIDANCE = 4.0
SEED = 123  # None for random

# FP8 quantization settings
USE_FP8 = True
DTYPE = torch.float8_e4m3fn if USE_FP8 and hasattr(torch, 'float8_e4m3fn') else torch.bfloat16

# Device settings
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OFFLOAD = False  # Set to True if you have limited VRAM
# -------------------------------

def _safe(name, n=100):
    """Make filename safe for filesystem."""
    return "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in name.strip())[:n] or "out"

def load_flux_models(model_name, device, offload=False):
    """Load FLUX models with FP8 quantization support."""
    print(f"📥 Loading FLUX models: {model_name}")
    print(f"   Device: {device}, Offload: {offload}, FP8: {USE_FP8}")
    
    # Load models
    t5 = load_t5(device, max_length=512)
    clip = load_clip(device)
    model = load_flow_model(model_name, device="cpu" if offload else device)
    ae = load_ae(model_name, device="cpu" if offload else device)
    
    # Apply FP8 quantization if available
    if USE_FP8 and hasattr(torch, 'float8_e4m3fn'):
        print("🔧 Applying FP8 quantization to models")
        model = model.to(dtype=torch.float8_e4m3fn)
        ae = ae.to(dtype=torch.float8_e4m3fn)
    
    return model, ae, t5, clip

def process_image_with_flux(
    model, ae, t5, clip, 
    image_path, prompt, width, height, steps, guidance, seed, device, offload
):
    """Process a single image with FLUX Kontext."""
    
    # Load and prepare input image
    print(f"🖼️  Processing: {os.path.basename(image_path)}")
    init_image = Image.open(image_path).convert("RGB")
    
    # Convert PIL to tensor
    init_image_tensor = torch.from_numpy(np.array(init_image)).permute(2, 0, 1).float() / 255.0
    init_image_tensor = init_image_tensor.unsqueeze(0).to(device)
    
    # Resize to target dimensions
    init_image_tensor = torch.nn.functional.interpolate(init_image_tensor, (height, width))
    
    # Encode with autoencoder
    if offload:
        ae.encoder.to(device)
    init_latent = ae.encode(init_image_tensor)
    if offload:
        ae = ae.cpu()
        torch.cuda.empty_cache()
    
    # Set up sampling options
    opts = SamplingOptions(
        prompt=prompt,
        width=width,
        height=height,
        num_steps=steps,
        guidance=guidance,
        seed=seed,
    )
    
    if opts.seed is None:
        opts.seed = torch.Generator(device="cpu").seed()
    
    print(f"🎯 Generating '{opts.prompt}' with seed {opts.seed}")
    t0 = time.perf_counter()
    
    # Prepare noise
    x = get_noise(
        1, opts.height, opts.width, 
        device=device, 
        dtype=DTYPE
    )
    
    # Prepare conditioning
    conditioning = prepare(
        opts.prompt, t5, clip, 
        height=opts.height, width=opts.width,
        device=device, dtype=DTYPE
    )
    
    # Get sampling schedule
    schedule = get_schedule(opts.num_steps, device=device, dtype=DTYPE)
    
    # Run denoising
    x = denoise(
        x, conditioning, schedule, 
        model, opts.guidance, 
        init_latent=init_latent
    )
    
    # Decode with autoencoder
    if offload:
        ae.to(device)
    output = ae.decode(x)
    if offload:
        ae = ae.cpu()
        torch.cuda.empty_cache()
    
    # Convert to image
    output = unpack(output)
    output = torch.clamp(output, 0, 1)
    
    # Add watermark
    output = embed_watermark(output)
    
    generation_time = time.perf_counter() - t0
    print(f"✅ Generated in {generation_time:.2f}s")
    
    return output, generation_time

def main():
    """Main function to process all images with FLUX Kontext."""
    os.makedirs(OUTDIR, exist_ok=True)
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    
    print(f"🚀 FLUX Kontext with FP8 Quantization")
    print(f"📊 Device: {DEVICE}")
    print(f"📊 Data type: {DTYPE}")
    print(f"📊 FP8 enabled: {USE_FP8}")
    
    # Load models
    try:
        model, ae, t5, clip = load_flux_models(MODEL_ID, DEVICE, OFFLOAD)
        print("✅ Models loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load models: {e}")
        print("   Make sure FLUX Kontext is properly installed:")
        print("   pip install -e flux-kontext[all]")
        return
    
    # Set up logging
    with open(LOG_PATH, "a", encoding="utf-8", newline="") as lf:
        writer = csv.writer(lf)
        if lf.tell() == 0:
            writer.writerow([
                "timestamp", "model", "device", "fp8_enabled",
                "prompt", "input_path", "output_path",
                "width", "height", "steps", "guidance", "seed",
                "generation_time", "memory_used_mb"
            ])
        
        # Get image files
        image_files = [f for f in sorted(os.listdir(INPUT_DIR)) 
                      if os.path.isfile(os.path.join(INPUT_DIR, f))]
        
        print(f"🖼️  Found {len(image_files)} images to process")
        
        for i, fname in enumerate(image_files, 1):
            in_path = os.path.join(INPUT_DIR, fname)
            
            # Generate output filename
            base_name = os.path.splitext(fname)[0]
            safe_prompt = _safe(PROMPT, 50)
            out_filename = f"flux_kontext_{base_name}_{safe_prompt}.png"
            out_path = os.path.join(OUTDIR, out_filename)
            
            print(f"\n📸 [{i}/{len(image_files)}] Processing: {fname}")
            
            try:
                # Process image
                output, generation_time = process_image_with_flux(
                    model, ae, t5, clip,
                    in_path, PROMPT, WIDTH, HEIGHT, STEPS, GUIDANCE, SEED,
                    DEVICE, OFFLOAD
                )
                
                # Save output
                save_image(output, out_path)
                print(f"💾 Saved: {out_filename}")
                
                # Get memory usage
                memory_used = torch.cuda.memory_allocated() / 1024**2 if torch.cuda.is_available() else 0
                
                # Log results
                writer.writerow([
                    datetime.now().isoformat(),
                    MODEL_ID, DEVICE, USE_FP8,
                    PROMPT, in_path, out_path,
                    WIDTH, HEIGHT, STEPS, GUIDANCE, SEED,
                    f"{generation_time:.2f}", f"{memory_used:.1f}"
                ])
                
            except Exception as e:
                print(f"❌ Error processing {fname}: {e}")
                # Log error
                writer.writerow([
                    datetime.now().isoformat(),
                    MODEL_ID, DEVICE, USE_FP8,
                    PROMPT, in_path, f"ERROR: {str(e)}",
                    WIDTH, HEIGHT, STEPS, GUIDANCE, SEED,
                    "0.0", "0.0"
                ])
    
    print(f"\n🎉 Processing complete!")
    print(f"📊 Results logged to: {LOG_PATH}")
    print(f"🖼️  Output images saved to: {OUTDIR}")

if __name__ == "__main__":
    main()
