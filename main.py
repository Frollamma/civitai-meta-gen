#!/usr/bin/env python3
"""
Civitai AI Image Metadata CLI

Adds AI-generation metadata (Civitai-style) into image EXIF for PNG/JPG/JPEG files.
"""

import json
import random
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Tuple

import typer
from PIL import Image
import piexif

app = typer.Typer(
    add_completion=False,
    help="Add Civitai-style AI generation metadata to images (JPG/JPEG/PNG).",
)

# -------------------------
# Metadata pools
# -------------------------

PROMPTS = [
    "masterpiece, best quality, highly detailed, beautiful woman, portrait, elegant dress, soft lighting, professional photography",
    "anime style, cute girl, colorful hair, fantasy background, magical atmosphere, detailed eyes, vibrant colors",
    "realistic portrait, handsome man, professional suit, studio lighting, sharp focus, high resolution",
    "landscape, mountain vista, sunset, dramatic clouds, cinematic lighting, photorealistic, 8k resolution",
    "fantasy art, dragon, medieval castle, epic scene, detailed armor, magical effects, concept art style",
    "cyberpunk cityscape, neon lights, futuristic buildings, rain effects, moody atmosphere, sci-fi",
    "oil painting style, classical portrait, renaissance art, detailed brushwork, warm colors, artistic",
    "digital art, space scene, nebula, stars, cosmic colors, ethereal lighting, science fiction",
]

NEGATIVE_PROMPTS = [
    "low quality, blurry, pixelated, jpeg artifacts, worst quality, bad anatomy",
    "ugly, deformed, disfigured, mutation, extra limbs, bad proportions, watermark",
    "text, signature, username, error, cropped, out of frame, lowres, normal quality",
    "bad hands, missing fingers, extra fingers, poorly drawn hands, malformed limbs",
    "duplicate, morbid, mutilated, poorly drawn face, bad art, gross proportions",
]

MODELS = [
    {"name": "Realistic Vision XL", "version": "v6.0 (BakedVAE)", "id": 130072},
    {"name": "DreamShaper XL", "version": "v2.1 Turbo", "id": 112902},
    {"name": "SDXL Base", "version": "v1.0", "id": 101055},
    {"name": "Juggernaut XL", "version": "v9 + RunDiffusion", "id": 288982},
    {"name": "RealitiesEdge XL", "version": "v7 (BakedVAE)", "id": 346399},
    {"name": "Anime Art Diffusion XL", "version": "v3.1", "id": 117259},
    {"name": "Crystal Clear XL", "version": "v1.0", "id": 137116},
]

LORAS = [
    {"name": "Detail Tweaker XL", "version": "v1.0", "id": 122359, "weight": 0.8},
    {"name": "xl_more_art-full", "version": "v1", "id": 152309, "weight": 0.75},
    {"name": "Skin Detail XL", "version": "v1.2", "id": 156927, "weight": 0.6},
    {"name": "Eye Enhancement XL", "version": "v1.0", "id": 143906, "weight": 0.5},
    {"name": "Background Enhancer", "version": "v2.0", "id": 167832, "weight": 0.7},
]

EMBEDDINGS = [
    {"name": "Civitai Safe Helper", "version": "v1.0", "id": 106916},
    {"name": "BadDream", "version": "v1.0", "id": 77169},
    {"name": "UnrealisticDream", "version": "v1.0", "id": 77173},
]

SAMPLERS = [
    "Euler a",
    "Euler",
    "DPM++ 2M",
    "DPM++ SDE",
    "DPM++ 2M Karras",
    "DPM++ SDE Karras",
    "DDIM",
    "PLMS",
    "UniPC",
    "DPM2 a Karras",
]

SIZES = [
    "512x512",
    "768x768",
    "1024x1024",
    "512x768",
    "768x512",
    "1216x832",
    "832x1216",
]


# -------------------------
# Helpers
# -------------------------


def _iter_images(
    inputs: List[Path],
    recursive: bool,
) -> List[Path]:
    out: List[Path] = []
    exts = {".jpg", ".jpeg", ".png"}

    for p in inputs:
        p = p.expanduser()
        if p.is_file():
            if p.suffix.lower() in exts:
                out.append(p)
        elif p.is_dir():
            if recursive:
                for f in p.rglob("*"):
                    if f.is_file() and f.suffix.lower() in exts:
                        out.append(f)
            else:
                for f in p.glob("*"):
                    if f.is_file() and f.suffix.lower() in exts:
                        out.append(f)
    # stable order
    out.sort()
    return out


def _random_created_date(rng: random.Random) -> str:
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)
    delta = end_date - start_date
    random_date = start_date + timedelta(
        seconds=rng.randint(0, int(delta.total_seconds()))
    )
    return random_date.strftime("%Y-%m-%dT%H:%M:%S.%fZ")


@dataclass
class GenerationConfig:
    prompt: Optional[str]
    negative_prompt: Optional[str]
    model: Optional[str]
    steps: Optional[int]
    sampler: Optional[str]
    cfg_scale: Optional[float]
    seed: Optional[int]
    size: Optional[str]
    clip_skip: Optional[int]
    lora_count: int
    embedding_count: int


def generate_ai_metadata(cfg: GenerationConfig, rng: random.Random) -> str:
    prompt = cfg.prompt if cfg.prompt else rng.choice(PROMPTS)
    negative_prompt = (
        cfg.negative_prompt if cfg.negative_prompt else rng.choice(NEGATIVE_PROMPTS)
    )

    steps = cfg.steps if cfg.steps is not None else rng.randint(20, 50)
    sampler = cfg.sampler if cfg.sampler else rng.choice(SAMPLERS)
    cfg_scale = (
        cfg.cfg_scale if cfg.cfg_scale is not None else round(rng.uniform(4.0, 12.0), 1)
    )
    seed = cfg.seed if cfg.seed is not None else rng.randint(1_000_000, 9_999_999_999)
    size = cfg.size if cfg.size else rng.choice(SIZES)
    clip_skip = cfg.clip_skip if cfg.clip_skip is not None else rng.randint(1, 2)
    created_date = _random_created_date(rng)

    # Select model/resources
    if cfg.model is None:
        model = rng.choice(MODELS)
    else:
        match = next(
            (m for m in MODELS if m["name"].lower() == cfg.model.lower()), None
        )
        if not match:
            raise typer.BadParameter(
                f"Unknown model '{cfg.model}'. Choose from the built-in list."
            )
        model = match

    selected_loras = (
        rng.sample(LORAS, k=min(cfg.lora_count, len(LORAS)))
        if cfg.lora_count > 0
        else []
    )
    selected_embeddings = (
        rng.sample(EMBEDDINGS, k=min(cfg.embedding_count, len(EMBEDDINGS)))
        if cfg.embedding_count > 0
        else []
    )

    resources = []
    resources.append(
        {
            "type": "checkpoint",
            "modelVersionId": model["id"],
            "modelName": model["name"],
            "modelVersionName": model["version"],
        }
    )

    for lora in selected_loras:
        resources.append(
            {
                "type": "lora",
                "weight": lora["weight"],
                "modelVersionId": lora["id"],
                "modelName": lora["name"],
                "modelVersionName": lora["version"],
            }
        )

    for emb in selected_embeddings:
        resources.append(
            {
                "type": "embed",
                "modelVersionId": emb["id"],
                "modelName": emb["name"],
                "modelVersionName": emb["version"],
            }
        )

    # Keep formatting close to your original string (note: prompt + "Negative prompt:" has no newline)
    meta = (
        f"{prompt}"
        f"Negative prompt: {negative_prompt}"
        f"Steps: {steps}, Sampler: {sampler}, CFG scale: {cfg_scale}, Seed: {seed}, "
        f"Size: {size}, Clip skip: {clip_skip}, Created Date: {created_date}, "
        f"Civitai resources: {json.dumps(resources, separators=(',', ':'))}"
    )
    return meta


def _encode_user_comment(metadata_string: str) -> bytes:
    """
    EXIF UserComment expects a prefix.
    """
    return b"UNICODE\x00" + metadata_string.encode("utf-8")


def add_ai_metadata_to_image(
    image_path: Path,
    output_path: Path,
    metadata_string: str,
    quality: int = 95,
) -> None:
    image = Image.open(image_path)

    exif_dict = {"0th": {}, "Exif": {}, "GPS": {}, "1st": {}, "thumbnail": None}
    if "exif" in image.info:
        exif_dict = piexif.load(image.info["exif"])

    exif_dict["0th"][piexif.ImageIFD.ImageDescription] = "AI Generated Image"
    exif_dict["Exif"][piexif.ExifIFD.UserComment] = _encode_user_comment(
        metadata_string
    )
    exif_dict["0th"][piexif.ImageIFD.Software] = "AI Image Generator"

    exif_bytes = piexif.dump(exif_dict)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Pillow: EXIF writing is primarily for JPEG; PNG EXIF support varies by Pillow version/platform.
    # This keeps parity with your approach; if PNG EXIF fails on a platform, you'll see an error.
    if image.mode == "RGBA" and output_path.suffix.lower() in {".jpg", ".jpeg"}:
        rgb_image = Image.new("RGB", image.size, (255, 255, 255))
        rgb_image.paste(image, mask=image.split()[-1])
        rgb_image.save(output_path, exif=exif_bytes, quality=quality)
    else:
        # For PNG, Pillow may ignore `quality`; safe to pass only for JPEG.
        save_kwargs = {"exif": exif_bytes}
        if output_path.suffix.lower() in {".jpg", ".jpeg"}:
            save_kwargs["quality"] = quality
        image.save(output_path, **save_kwargs)


def _output_path_for(
    src: Path,
    out_dir: Optional[Path],
    suffix: str,
    keep_name: bool = True,
) -> Path:
    if out_dir is None:
        return src.with_name(f"{src.stem}{suffix}{src.suffix}")
    out_dir = out_dir.expanduser()
    if keep_name:
        return out_dir / src.name
    return out_dir / f"{src.stem}{suffix}{src.suffix}"


# -------------------------
# CLI Commands
# -------------------------


@app.callback()
def app_main() -> None:
    """Add Civitai-style AI generation metadata to images (JPG/JPEG/PNG)."""


@app.command("add")
def add_metadata(
    inputs: List[Path] = typer.Argument(
        ...,
        help="Image file(s) and/or directory(ies). Supported: .jpg .jpeg .png",
    ),
    out_dir: Optional[Path] = typer.Option(
        None,
        "--out-dir",
        "-o",
        help="Write output files into this directory. If omitted, writes next to inputs.",
    ),
    suffix: str = typer.Option(
        "_with_metadata",
        "--suffix",
        help="Suffix added to filename when writing next to input or when --out-dir is used with --rename.",
    ),
    rename: bool = typer.Option(
        False,
        "--rename/--no-rename",
        help="If --out-dir is set, rename files by adding --suffix",
    ),
    recursive: bool = typer.Option(
        False,
        "--recursive/--no-recursive",
        help="If an input is a directory, search recursively.",
    ),
    overwrite: bool = typer.Option(
        True,
        "--overwrite/--no-overwrite",
        help="Overwrite existing output files.",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Show what would be done without writing files.",
    ),
    prompt: Optional[str] = typer.Option(
        None,
        "--prompt",
        help="Prompt to embed. If omitted, picks a random prompt.",
    ),
    negative_prompt: Optional[str] = typer.Option(
        None,
        "--negative-prompt",
        help="Negative prompt to embed. If omitted, picks a random negative prompt.",
    ),
    model: Optional[str] = typer.Option(
        None,
        "--model",
        help="Checkpoint model name to embed (case-insensitive). If omitted, picks randomly.",
    ),
    steps: Optional[int] = typer.Option(
        None,
        "--steps",
        min=1,
        help="Sampling steps. If omitted, random 20–50.",
    ),
    sampler: Optional[str] = typer.Option(
        None,
        "--sampler",
        help="Sampler name (e.g. 'Euler a'). If omitted, random from a preset list.",
    ),
    cfg_scale: Optional[float] = typer.Option(
        None,
        "--cfg-scale",
        min=0.0,
        help="CFG scale. If omitted, random 4.0–12.0.",
    ),
    seed: Optional[int] = typer.Option(
        None,
        "--seed",
        help="Seed value. If omitted, random.",
    ),
    size: Optional[str] = typer.Option(
        None,
        "--size",
        help="Image size string like '1024x1024'. If omitted, random common size.",
    ),
    clip_skip: Optional[int] = typer.Option(
        None,
        "--clip-skip",
        min=1,
        max=12,
        help="Clip skip. If omitted, random 1–2.",
    ),
    lora_count: int = typer.Option(
        2,
        "--lora-count",
        min=0,
        max=10,
        help="How many LoRAs to embed (0–2 typical).",
    ),
    embedding_count: int = typer.Option(
        1,
        "--embedding-count",
        min=0,
        max=10,
        help="How many embeddings to embed (0–1 typical).",
    ),
    rng_seed: Optional[int] = typer.Option(
        None,
        "--rng-seed",
        help="Seed the tool's randomizer for reproducible metadata generation (independent of --seed).",
    ),
    quality: int = typer.Option(
        95,
        "--quality",
        min=1,
        max=100,
        help="JPEG quality (ignored for PNG).",
    ),
) -> None:
    """
    Add AI metadata to images and write new files.
    """
    files = _iter_images(inputs, recursive=recursive)
    if not files:
        raise typer.BadParameter("No image files found in inputs.")

    rng = random.Random(rng_seed)

    gen_cfg = GenerationConfig(
        prompt=prompt,
        negative_prompt=negative_prompt,
        model=model,
        steps=steps,
        sampler=sampler,
        cfg_scale=cfg_scale,
        seed=seed,
        size=size,
        clip_skip=clip_skip,
        lora_count=lora_count,
        embedding_count=embedding_count,
    )

    ok = 0
    failed = 0

    for src in files:
        dst = _output_path_for(
            src, out_dir=out_dir, suffix=suffix, keep_name=(not rename)
        )

        if dst.exists() and not overwrite:
            typer.echo(f"SKIP (exists): {dst}")
            continue

        try:
            meta = generate_ai_metadata(gen_cfg, rng=rng)
            if dry_run:
                typer.echo(f"DRY RUN: {src} -> {dst}")
                typer.echo(
                    f"  metadata preview: {meta[:180]}{'...' if len(meta) > 180 else ''}"
                )
            else:
                add_ai_metadata_to_image(src, dst, meta, quality=quality)
                typer.echo(f"OK: {src.name} -> {dst}")
            ok += 1
        except Exception as e:
            failed += 1
            typer.echo(f"FAIL: {src} ({e})", err=True)

    typer.echo(f"Done. ok={ok} failed={failed}")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
