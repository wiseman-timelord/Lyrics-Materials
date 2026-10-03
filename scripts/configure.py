"""
configure.py - Constants, paths, settings I/O for Lyrics-Materials.
Lyrics → per-line visual materials (images) for external AI video use.
Models:
  Encoder  : Qwen3-VL-4B Instruct / Uncensored (Q4)
  Thinking : Qwen3-VL-4B Thinking (Q5) — optional, richer prompts + song analysis
  Diffuser : FLUX.2-klein-4B (Q8)
  mmproj quarantined under models/mmproj/ (not used for pure text)
"""
from __future__ import annotations

import configparser
import json
import os
import re
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

def _get_project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def get_project_root() -> Path:
    return _get_project_root()


def get_data_dir() -> Path:
    return _get_project_root() / "data"


def get_models_dir() -> Path:
    return _get_project_root() / "models"


def get_output_dir() -> Path:
    return _get_project_root() / "output"


def get_llama_bin_dir() -> Path:
    return get_data_dir() / "llama_cpp_binaries"


def get_sd_bin_dir() -> Path:
    return get_data_dir() / "stable_diffusion_binaries"


def get_dictionaries_dir() -> Path:
    return get_data_dir() / "dictionaries"


def get_constants_path() -> Path:
    return get_data_dir() / "constants.ini"


def get_configuration_path() -> Path:
    return get_data_dir() / "configuration.json"


def get_preferences_path() -> Path:
    return get_data_dir() / "preferences.json"


def get_generation_path() -> Path:
    return get_data_dir() / "generation.json"


def get_prompting_path() -> Path:
    return get_data_dir() / "prompting.json"


def ensure_data_dirs() -> None:
    for d in (
        get_data_dir(),
        get_models_dir(),
        get_mmproj_quarantine_dir(),
        get_output_dir(),
        get_llama_bin_dir(),
        get_sd_bin_dir(),
        get_dictionaries_dir(),
        get_data_dir() / "temp_images",
        get_data_dir() / "temp_audio",
    ):
        d.mkdir(parents=True, exist_ok=True)
    # Keep vision projectors out of models/ root so pure-text loads stay clean
    quarantine_mmproj_files()


# ---------------------------------------------------------------------------
# Fixed model expectations
# ---------------------------------------------------------------------------
ENCODER_MODEL_HINT = (
    "Huihui-Qwen3-VL-4B-Instruct-abliterated*.gguf  OR  "
    "Qwen3-VL-4B-Instruct-Uncensored-abliterated*.gguf"
)
THINKING_MODEL_HINT = "Huihui-Qwen3-VL-4B-Thinking-abliterated*.gguf"
MMPROJ_HINT = "mmproj*.gguf  (quarantined under models/mmproj/ — not used for pure text)"
DIFFUSER_MODEL_HINT = "flux-2-klein-4b-Q8_0.gguf  OR  diffusion_pytorch_model_4b.safetensors"
VAE_HINT = "flux2_ae.safetensors (Flux.2 VAE from black-forest-labs/FLUX.2-dev — NOT Flux.1 ae.safetensors)"

# Qwen3-VL-4B backbone (Instruct / Thinking / Uncensored share architecture)
ENCODER_MAX_LAYERS = 36
ENCODER_MAX_CONTEXT = 40960
ENCODER_DIM = 2560

# Flux.2-klein-4B
DIFFUSER_MAX_LAYERS = 30  # informational only; sd.cpp is whole-module

# Hard-coded CPU threads for heavy work (user request)
HEAVY_THREADS = 10
# Single shared worker-thread count for all heavy work (llama / sd).
# Logical CPUs 0 and 1 are reserved for the OS/UI — affinity starts at core 2.
WORKER_THREADS_MIN = 2
WORKER_THREADS_DEFAULT = 8
AFFINITY_CORE_OFFSET = 2  # skip cores 0 and 1

# Model weight load mode (per model column: encoder / thinking / diffuser)
# One-Shot  → mmap / default load (lazy page-in; lower peak RAM)
# M-Lock    → force model resident in RAM (-lm mlock) so weights are not swapped
LOAD_MODE_ONE_SHOT = "One-Shot"
LOAD_MODE_MLOCK = "M-Lock"
LOAD_MODE_CHOICES = [LOAD_MODE_ONE_SHOT, LOAD_MODE_MLOCK]
DEFAULT_LOAD_MODE = LOAD_MODE_ONE_SHOT

# GPU layer offload: how many transformer layers go to VRAM; remainder on system RAM.
# -1 = all layers (llama.cpp convention).
DEFAULT_GPU_LAYERS = -1

# Diffuser placement when image backend is a GPU (Vulkan/CUDA/…), not CPU.
# Gpu_Only → diffusion + te + vae all on the selected GPU device
# Split    → diffusion on GPU; text-encoder + VAE stay on CPU (lower VRAM)
PLACEMENT_GPU_ONLY = "Gpu_Only"
PLACEMENT_SPLIT = "Split"
PLACEMENT_CHOICES = [PLACEMENT_GPU_ONLY, PLACEMENT_SPLIT]
DEFAULT_PLACEMENT = PLACEMENT_SPLIT

# ---------------------------------------------------------------------------
# Approximate weight + runtime VRAM budgets (MiB).
# Disk sizes from HF GGUF listings; runtime adds headroom for short context.
# mmproj is NOT loaded for pure-text prompt work (quarantined).
# ---------------------------------------------------------------------------
MODEL_VRAM_PROFILES = {
    "q4_k_m": {"disk_mb": 2500, "full_gpu_mb": 3200, "layers": ENCODER_MAX_LAYERS},
    "q5_k_m": {"disk_mb": 2890, "full_gpu_mb": 3800, "layers": ENCODER_MAX_LAYERS},
    "q8_0": {"disk_mb": 4280, "full_gpu_mb": 5200, "layers": ENCODER_MAX_LAYERS},
    "flux_q8": {
        "disk_mb": 4300,
        "full_gpu_mb": 5600,   # diffusion on GPU, Split (TE/VAE on CPU)
        "gpu_only_mb": 8200,   # diffusion + TE + VAE co-resident
        "layers": DIFFUSER_MAX_LAYERS,
    },
}


def _match_vram_profile(filename: str) -> Dict[str, int]:
    n = (filename or "").lower()
    if "flux" in n or "klein" in n:
        return dict(MODEL_VRAM_PROFILES["flux_q8"])
    if "q5_k" in n or "q5k" in n:
        return dict(MODEL_VRAM_PROFILES["q5_k_m"])
    if "q8_0" in n or "q8." in n:
        return dict(MODEL_VRAM_PROFILES["q8_0"])
    return dict(MODEL_VRAM_PROFILES["q4_k_m"])


def free_vram_floor_mb(vram_free_mb: int) -> int:
    """Round free VRAM down to the next whole GB (2731 → 2000, 7367 → 7000)."""
    if vram_free_mb <= 0:
        return 0
    return (int(vram_free_mb) // 1000) * 1000


def device_safe_vram_mb(device: Dict[str, Any]) -> int:
    """Safe budget for a probed GPU: free MiB floored to whole GB."""
    return free_vram_floor_mb(int(device.get("vram_free_mb") or 0))


def gguf_block_count(model_path: str) -> int:
    """
    Read transformer block/layer count from a GGUF file header metadata.
    Looks for keys ending in '.block_count' (e.g. qwen3vl.block_count).
    Pure-Python reader — no external gguf package required.
    Returns 0 if unreadable; callers fall back to ENCODER_MAX_LAYERS.
    """
    p = Path(model_path or "")
    if not p.is_file():
        return 0
    try:
        import struct
        with open(p, "rb") as f:
            magic = f.read(4)
            if magic != b"GGUF":
                return 0
            version = struct.unpack("<I", f.read(4))[0]
            if version not in (1, 2, 3):
                return 0
            _tensor_count = struct.unpack("<Q", f.read(8))[0]
            kv_count = struct.unpack("<Q", f.read(8))[0]

            def _read_str() -> str:
                n = struct.unpack("<Q", f.read(8))[0]
                if n > 10_000_000:
                    raise ValueError("string too large")
                return f.read(n).decode("utf-8", errors="replace")

            def _skip_value(vtype: int) -> None:
                # GGUF value types: 0=uint8 … 12=array
                sizes = {
                    0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 8, 8: 8,
                    9: None,  # string
                    10: 1, 11: 8,
                }
                if vtype == 9:  # string
                    _read_str()
                    return
                if vtype == 12:  # array
                    atype = struct.unpack("<I", f.read(4))[0]
                    alen = struct.unpack("<Q", f.read(8))[0]
                    if atype == 9:
                        for _ in range(min(alen, 100000)):
                            _read_str()
                    else:
                        es = sizes.get(atype, 0) or 0
                        f.seek(es * alen, 1)
                    return
                es = sizes.get(vtype)
                if es:
                    f.seek(es, 1)

            for _ in range(min(kv_count, 5000)):
                key = _read_str()
                vtype = struct.unpack("<I", f.read(4))[0]
                if key.endswith(".block_count") or key.endswith(".n_layer") or key == "block_count":
                    if vtype == 4:  # uint32
                        return int(struct.unpack("<I", f.read(4))[0])
                    if vtype == 5:  # int32
                        return int(struct.unpack("<i", f.read(4))[0])
                    if vtype == 6:  # float32 — unlikely
                        f.seek(4, 1)
                        continue
                    if vtype == 11:  # uint64
                        return int(struct.unpack("<Q", f.read(8))[0])
                    _skip_value(vtype)
                else:
                    _skip_value(vtype)
    except Exception:
        return 0
    return 0


def model_layer_count(model_path: str) -> int:
    """GGUF block_count if available, else architecture default."""
    n = gguf_block_count(model_path)
    if n > 0:
        return n
    return ENCODER_MAX_LAYERS


def suggest_gpu_layers(
    model_path: str,
    vram_free_mb: int,
    is_diffuser: bool = False,
    placement: str = DEFAULT_PLACEMENT,
) -> int:
    """
    Concrete layer count that fits in the safe VRAM floor.
    Never returns -1 — callers that store -1 mean "auto"; this resolves auto.
    """
    floor = free_vram_floor_mb(int(vram_free_mb or 0))
    if floor < 1500:
        return 0
    prof = _match_vram_profile(Path(model_path or "").name)
    if is_diffuser:
        need = prof.get(
            "gpu_only_mb" if normalize_placement(placement) == PLACEMENT_GPU_ONLY
            else "full_gpu_mb",
            5600,
        )
        # sd.cpp is whole-module; 1 = try GPU, 0 = force CPU-side params
        return 1 if floor >= need else 0
    need = max(1, int(prof.get("full_gpu_mb", 3200)))
    layers = model_layer_count(model_path)
    if floor >= need:
        return layers  # all layers fit
    # Proportional: how many layers fit in the safe floor
    ratio = max(0.0, min(1.0, floor / float(need)))
    n = int(layers * ratio)
    # Leave a little headroom — never claim more than 95% of proportional
    n = int(n * 0.95) if n > 2 else n
    return max(0, min(layers, n))


def resolve_text_gpu_layers(
    model_path: str,
    requested: int,
    backend: str,
    vram_free_mb: int = 0,
) -> int:
    """
    Map UI GPU-layers value to concrete -ngl.
    requested == -1 → auto from safe VRAM floor + GGUF layer count.
    CPU backend → 0.
    """
    if not str(backend or "").upper().startswith(("VULKAN", "CUDA")):
        return 0
    try:
        req = int(requested)
    except (TypeError, ValueError):
        req = DEFAULT_GPU_LAYERS
    if req == -1:
        # If caller did not pass free VRAM, use largest probed device floor
        free = int(vram_free_mb or 0)
        if free <= 0:
            vk = get_vulkan_info()
            frees = [int(d.get("vram_free_mb") or 0) for d in vk.get("devices") or []]
            free = max(frees) if frees else 0
        return suggest_gpu_layers(model_path, free, is_diffuser=False)
    return max(0, req)


def vram_free_for_backend(backend: str) -> int:
    """Look up install-time free MiB for the selected VulkanN / CUDA N label."""
    backend = str(backend or "")
    m = __import__("re").search(r"(\d+)", backend)
    idx = int(m.group(1)) if m else 0
    for d in get_vulkan_info().get("devices") or []:
        if int(d.get("index", -1)) == idx:
            return int(d.get("vram_free_mb") or 0)
    return 0


def identify_encoder_variant(path_str: str) -> str:
    """Label Qwen3-VL-4B family GGUF: Instruct / Thinking / Uncensored-Instruct."""
    n = Path(path_str or "").name.lower()
    if "thinking" in n:
        return "Thinking"
    if "uncensored" in n:
        return "Uncensored-Instruct"
    if "instruct" in n or "huihui" in n or "qwen3-vl-4b" in n:
        return "Instruct"
    return "Encoder" if path_str else ""


def get_mmproj_quarantine_dir() -> Path:
    return get_models_dir() / "mmproj"


def quarantine_mmproj_files() -> List[str]:
    """
    Move mmproj*.gguf from models/ root into models/mmproj/ so llama.cpp
    does not auto-attach vision projector during pure-text loads.
    """
    models = get_models_dir()
    dest = get_mmproj_quarantine_dir()
    dest.mkdir(parents=True, exist_ok=True)
    moved: List[str] = []
    if not models.is_dir():
        return moved
    for p in list(models.iterdir()):
        if not p.is_file():
            continue
        name = p.name
        low = name.lower()
        if not low.startswith("mmproj") or not low.endswith(".gguf"):
            continue
        target = dest / name
        try:
            if target.exists():
                p.unlink(missing_ok=True)
                moved.append(f"{name} (duplicate removed)")
            else:
                shutil.move(str(p), str(target))
                moved.append(name)
        except OSError:
            pass
    return moved

# Styles
STYLE_LIGHT = "light and bright"
STYLE_DARK = "dark and gloomy"
STYLE_COLORFUL = "colorful and wild"
STYLE_CHOICES = [STYLE_LIGHT, STYLE_DARK, STYLE_COLORFUL]
# Default visual style for new projects / fresh installs
STYLE_DEFAULT = STYLE_DARK

# ---------------------------------------------------------------------------
# Hair style (subject token — optional via None)
# Injected into character-bearing prompts when not None.
# ---------------------------------------------------------------------------
HAIR_STYLE_SHORT = "Short"
HAIR_STYLE_NATURAL = "Natural"
HAIR_STYLE_BOB = "Bob"
HAIR_STYLE_PONY = "Pony"
HAIR_STYLE_PIGTAIL = "PigTail"
HAIR_STYLE_DUDE = "Dude"
HAIR_STYLE_NONE = "None"
ALL_HAIR_STYLES = "All Styles"
HAIR_STYLE_CHOICES = [
    HAIR_STYLE_SHORT, HAIR_STYLE_NATURAL, HAIR_STYLE_DUDE, HAIR_STYLE_BOB,
    HAIR_STYLE_PONY, HAIR_STYLE_PIGTAIL, HAIR_STYLE_NONE,
    ALL_HAIR_STYLES,
]
HAIR_STYLE_CONCRETE = [
    HAIR_STYLE_SHORT, HAIR_STYLE_NATURAL, HAIR_STYLE_DUDE, HAIR_STYLE_BOB,
    HAIR_STYLE_PONY, HAIR_STYLE_PIGTAIL, HAIR_STYLE_NONE,
]
HAIR_STYLE_DEFAULT = HAIR_STYLE_NONE
HAIR_STYLE_TOKEN = "<hair_style>"
HAIR_STYLE_WORDS = {
    HAIR_STYLE_SHORT: "cut short and even, three inches length",
    HAIR_STYLE_NATURAL: "worn naturally, 3 inches length all over",
    HAIR_STYLE_DUDE: "center parting, shaggy and shoulder length",
    HAIR_STYLE_BOB: "styled in a short bob with a fringe",
    HAIR_STYLE_PONY: "tied-back in a shoulder-length high-ponytail",
    HAIR_STYLE_PIGTAIL: "tied-back in 2 shoulder-length high-pigtails",
    HAIR_STYLE_NONE: "",
}


def normalize_hair_style(value: str) -> str:
    v = (value or "").strip()
    if v in HAIR_STYLE_CHOICES:
        return v
    for c in HAIR_STYLE_CONCRETE:
        if c.lower() == v.lower():
            return c
    return HAIR_STYLE_DEFAULT


def hair_style_phrase(value: str) -> str:
    """Concrete hair description, or empty when None / All Styles."""
    key = normalize_hair_style(value)
    if key in (HAIR_STYLE_NONE, ALL_HAIR_STYLES):
        return ""
    return (HAIR_STYLE_WORDS.get(key) or "").strip()


# ---------------------------------------------------------------------------
# Outfit worn (subject token — optional via None)
# ---------------------------------------------------------------------------
OUTFIT_SMART_SUIT = "Smart Suit"
OUTFIT_SMART_CASUAL_MALE = "Casual_Male"
OUTFIT_SMART_CASUAL_FEMALE = "Casual_Female"
OUTFIT_JOGGERS = "Joggers"
OUTFIT_ROCKER = "Rocker"
OUTFIT_SKIMPY = "Skimpy"
OUTFIT_UNDIES = "Undies"
OUTFIT_NONE = "None"
ALL_OUTFITS = "All Outfits"
OUTFIT_CHOICES = [
    OUTFIT_SMART_SUIT, OUTFIT_SMART_CASUAL_MALE, OUTFIT_SMART_CASUAL_FEMALE,
    OUTFIT_JOGGERS, OUTFIT_ROCKER, OUTFIT_SKIMPY, OUTFIT_UNDIES, OUTFIT_NONE,
    ALL_OUTFITS,
]
OUTFIT_CONCRETE = [
    OUTFIT_SMART_SUIT, OUTFIT_SMART_CASUAL_MALE, OUTFIT_SMART_CASUAL_FEMALE,
    OUTFIT_JOGGERS, OUTFIT_ROCKER, OUTFIT_SKIMPY, OUTFIT_UNDIES, OUTFIT_NONE,
]
OUTFIT_DEFAULT = OUTFIT_NONE
OUTFIT_TOKEN = "<outfit_worn>"
OUTFIT_WORDS = {
    OUTFIT_SMART_SUIT: "smart-suit with unbuttoned-shirt outfit",
    OUTFIT_SMART_CASUAL_MALE: "black-tshirt with grey-smart-jeans outfit",
    OUTFIT_SMART_CASUAL_FEMALE: "black-tshirt with grey-short-skirt outfit",
    OUTFIT_JOGGERS: "black-crop-top with grey-jogging-shorts outfit",
    OUTFIT_ROCKER: "long-black-leather-coat with black shirt and grey-jeans outfit",
    OUTFIT_SKIMPY: "skimpy-revealing version of same outfit",
    OUTFIT_UNDIES: "underwear only",
    OUTFIT_NONE: "",
}


def normalize_outfit(value: str) -> str:
    v = (value or "").strip()
    if v in OUTFIT_CHOICES:
        return v
    for c in OUTFIT_CONCRETE:
        if c.lower() == v.lower():
            return c
    return OUTFIT_DEFAULT


def outfit_phrase(value: str) -> str:
    """Concrete outfit noun-phrase, or empty when None / All Outfits."""
    key = normalize_outfit(value)
    if key in (OUTFIT_NONE, ALL_OUTFITS):
        return ""
    return (OUTFIT_WORDS.get(key) or "").strip()


def subject_appearance_clause(hair: str = "", outfit: str = "") -> str:
    """
    Build an optional appearance clause for character-bearing prompts.
    Empty strings are omitted so None leaves the prompt unchanged.
    """
    bits: List[str] = []
    hp = hair_style_phrase(hair)
    op = outfit_phrase(outfit)
    if hp:
        bits.append(f"hair {hp}")
    if op:
        bits.append(f"wearing a {op}")
    if not bits:
        return ""
    return "Subject appearance: " + "; ".join(bits) + "."

# Fade colours (RGB 0-255) used for intro/outro and lyric gaps
STYLE_FADE_RGB = {
    STYLE_LIGHT: (255, 255, 255),      # white
    STYLE_DARK: (0, 0, 0),             # black
    STYLE_COLORFUL: (128, 0, 128),     # purple proxy for rainbow (ffmpeg limitation)
}

# Default prompt templates per visual style (seed for prompting.json)
STYLE_PROMPT_TEMPLATES = {
    STYLE_LIGHT: (
        "Create a bright, airy, high-key visual description for a music-video still. "
        "Emphasize soft light, clean composition, and hopeful mood. "
        "Subject of the image: {line}"
    ),
    STYLE_DARK: (
        "Create a dark, moody, low-key visual description for a music-video still. "
        "Emphasize shadows, contrast, and a sombre atmosphere. "
        "Subject of the image: {line}"
    ),
    STYLE_COLORFUL: (
        "Create a vivid, colourful, high-energy visual description for a music-video still. "
        "Emphasize saturated colours, dynamic shapes, and playful wildness. "
        "Subject of the image: {line}"
    ),
}


def prompt_template_for_style(style: str) -> str:
    """Return the editable prompt template for a style (from prompting.json)."""
    data = load_prompting()
    return data.get(style) or STYLE_PROMPT_TEMPLATES.get(style, STYLE_PROMPT_TEMPLATES[STYLE_LIGHT])


# Section timing is set by the user via editable markers on the audio
# timeline (Intro Start/End, Chorus N, Outro Start/End). No Whisper.
MARKER_GAP_SECONDS = 0.5  # minimum gap between consecutive markers

# Video containers
VIDEO_MP4 = "mp4"
VIDEO_MKV = "mkv"
VIDEO_CHOICES = [VIDEO_MP4, VIDEO_MKV]

# Output resolution presets (final video frame size)
RESOLUTION_720P = "720p"
RESOLUTION_1080P = "1080p"
RESOLUTION_CHOICES = [RESOLUTION_720P, RESOLUTION_1080P]
RESOLUTION_PIXELS = {
    RESOLUTION_720P: (1280, 720),
    RESOLUTION_1080P: (1920, 1080),
}

# Default generation values for Flux.2-klein-4B distilled
DEFAULT_WIDTH = 768
DEFAULT_HEIGHT = 512
DEFAULT_STEPS = 8  # 8 improves eyes / fine detail vs 4 on Flux.2-klein
DEFAULT_CFG = 1.0
DEFAULT_NEGATIVE_PROMPT = (
    "cartoon, pixelated, graphical overlay, text overlay, watermark, logo, blurry, low quality, deformed, extra limbs"
)
DEFAULT_SAMPLER = "euler_a"
DEFAULT_SEED = -1

# Still output sizes (width × height) — user-selectable in Generation tab
IMAGE_SIZE_768x512 = "768 × 512"
IMAGE_SIZE_1024x512 = "1024 × 512"
IMAGE_SIZE_1024x768 = "1024 × 768"
IMAGE_SIZE_1280x768 = "1280 × 768"
IMAGE_SIZE_CHOICES = [
    IMAGE_SIZE_768x512,
    IMAGE_SIZE_1024x512,
    IMAGE_SIZE_1024x768,
    IMAGE_SIZE_1280x768,
]
IMAGE_SIZE_PIXELS = {
    IMAGE_SIZE_768x512: (768, 512),
    IMAGE_SIZE_1024x512: (1024, 512),
    IMAGE_SIZE_1024x768: (1024, 768),
    IMAGE_SIZE_1280x768: (1280, 768),
}
DEFAULT_IMAGE_SIZE = IMAGE_SIZE_768x512


def normalize_image_size(value: str) -> str:
    """Map a label or 'WxH' string to a canonical IMAGE_SIZE_* choice."""
    v = (value or "").strip()
    if v in IMAGE_SIZE_PIXELS:
        return v
    # Accept "768x512", "768 × 512", "768*512", etc.
    compact = re.sub(r"\s+", "", v.lower().replace("×", "x").replace("*", "x"))
    for label, (w, h) in IMAGE_SIZE_PIXELS.items():
        if compact == f"{w}x{h}":
            return label
    return DEFAULT_IMAGE_SIZE


def image_size_pixels(value: str) -> Tuple[int, int]:
    """Return (width, height) for a size label; falls back to defaults."""
    label = normalize_image_size(value)
    return IMAGE_SIZE_PIXELS.get(label, (DEFAULT_WIDTH, DEFAULT_HEIGHT))


def image_size_label_from_wh(width: int, height: int) -> str:
    """Best matching dropdown label for stored width/height."""
    for label, (w, h) in IMAGE_SIZE_PIXELS.items():
        if int(width) == w and int(height) == h:
            return label
    return DEFAULT_IMAGE_SIZE



# Image frequency presets: Cover / Theme / Lyrics-per-line
# Labels shown in the Generation dropdown.
#   C = cover stills · T = theme (ambient) stills · L = stills per lyric line
IMAGE_FREQUENCY_CHOICES = [
    "C1/T2/L1", "C2/T4/L1", "C3/T6/L1",
    "C1/T2/L2", "C2/T4/L2", "C3/T6/L2",
    "C1/T2/L3", "C2/T4/L3", "C3/T6/L3",
]
DEFAULT_IMAGE_FREQUENCY = "C1/T2/L1"

IMAGE_FREQUENCY_MAP = {
    "C1/T2/L1": {"cover": 1, "theme": 2, "lyrics": 1},
    "C2/T4/L1": {"cover": 2, "theme": 4, "lyrics": 1},
    "C3/T6/L1": {"cover": 3, "theme": 6, "lyrics": 1},
    "C1/T2/L2": {"cover": 1, "theme": 2, "lyrics": 2},
    "C2/T4/L2": {"cover": 2, "theme": 4, "lyrics": 2},
    "C3/T6/L2": {"cover": 3, "theme": 6, "lyrics": 2},
    "C1/T2/L3": {"cover": 1, "theme": 2, "lyrics": 3},
    "C2/T4/L3": {"cover": 2, "theme": 4, "lyrics": 3},
    "C3/T6/L3": {"cover": 3, "theme": 6, "lyrics": 3},
    # Legacy aliases from v1
    "C1/T2/LX": {"cover": 1, "theme": 2, "lyrics": 1},
    "C2/T4/LX": {"cover": 2, "theme": 4, "lyrics": 1},
    "C3/T6/LX": {"cover": 3, "theme": 6, "lyrics": 1},
}

# Framing hints when generating multiple stills for one lyric line
IMAGE_FREQUENCY_SEQUENCE = {
    1: [""],
    2: [
        "opening beat of this moment — establish the scene",
        "closing beat of this moment — resolve the beat",
    ],
    3: [
        "beginning of this moment",
        "middle / peak of this moment",
        "end of this moment",
    ],
}


def normalize_image_frequency(value) -> str:
    """Return a canonical frequency preset label (C1/T2/L1 …). Accepts legacy ints/LX."""
    v = str(value or "").strip().upper().replace(" ", "")
    # Normalise LX → L1 for lookup display
    if v.endswith("/LX"):
        v = v[:-2] + "L1"
    if v in IMAGE_FREQUENCY_MAP and not v.endswith("/LX"):
        # Prefer non-legacy key when both exist
        if v in IMAGE_FREQUENCY_CHOICES:
            return v
    if v in IMAGE_FREQUENCY_MAP:
        # Map legacy LX keys to L1 display labels
        legacy = {
            "C1/T2/LX": "C1/T2/L1",
            "C2/T4/LX": "C2/T4/L1",
            "C3/T6/LX": "C3/T6/L1",
        }
        return legacy.get(v, v if v in IMAGE_FREQUENCY_CHOICES else DEFAULT_IMAGE_FREQUENCY)
    try:
        n = int(value)
        if n <= 1:
            return "C1/T2/L1"
        if n == 2:
            return "C2/T4/L2"
        return "C3/T6/L3"
    except (TypeError, ValueError):
        pass
    return DEFAULT_IMAGE_FREQUENCY


def frequency_counts(value) -> Dict[str, int]:
    """Return {'cover': N, 'theme': M, 'lyrics': K} for a preset or legacy value."""
    key = normalize_image_frequency(value)
    # Also try raw string for legacy map keys
    raw = str(value or "").strip().upper().replace(" ", "")
    src = IMAGE_FREQUENCY_MAP.get(key) or IMAGE_FREQUENCY_MAP.get(raw) or IMAGE_FREQUENCY_MAP[DEFAULT_IMAGE_FREQUENCY]
    return dict(src)


def frequency_cover_count(value) -> int:
    return int(frequency_counts(value).get("cover", 1))


def frequency_theme_count(value) -> int:
    return int(frequency_counts(value).get("theme", 2))


def frequency_lyrics_per_line(value) -> int:
    return int(frequency_counts(value).get("lyrics", 1))


def image_frequency_hints(freq) -> list:
    """Ordered sequence framing strings for multi-variant lyric stills (length == lyrics-per-line)."""
    n = frequency_lyrics_per_line(freq)
    hints = IMAGE_FREQUENCY_SEQUENCE.get(n) or IMAGE_FREQUENCY_SEQUENCE[1]
    if len(hints) < n:
        hints = list(hints) + [""] * (n - len(hints))
    return list(hints[:n])




# Window geometry defaults
WINDOW_DEFAULT_WIDTH = 1280
WINDOW_DEFAULT_HEIGHT = 900
WINDOW_GEOMETRY_UNSET = -1

SPELLCHECK_LANGUAGE = "en-US"

# Status bar element id (display.py)
STATUS_BAR_KEY = "status-bar"

# Greyed-out placeholders for Configuration path fields
MODEL_PATH_PLACEHOLDERS = {
    "encoder": "Huihui-…Instruct…Q4_K_M.gguf or Uncensored…Q4_K_M.gguf",
    "thinking": "Huihui-…Thinking-abliterated…Q5_K_M.gguf",
    "mmproj": "mmproj*.gguf (auto-quarantined; not used for pure text)",
    "diffuser": "flux-2-klein-4b-Q8_0.gguf",
    "llm": "path/to/Qwen3-4B-*.gguf / qwen_3_4b.safetensors (Flux.2 text encoder)",
    "vae": "flux2_ae.safetensors OR BFL vae/diffusion_pytorch_model.safetensors (~350MB)",
}


# Preview / gallery sizes
PREVIEW_IMAGE_HEIGHT = 420
THUMBNAIL_GALLERY_HEIGHT = 140
INPUT_GALLERY_PADDING = 16
DEFAULT_MAX_THUMBNAILS = 50
MAX_THUMBNAIL_CHOICES = [25, 50, 100]
THUMBNAIL_COUNT_CHOICES = [25, 50, 100]
DEFAULT_INPUT_THUMBNAIL = 96
INPUT_THUMBNAIL_CHOICES = [64, 96, 128, 160]

# ---------------------------------------------------------------------------
# Session state (in-memory)
# ---------------------------------------------------------------------------
APP_STATE: Dict[str, Any] = {
    "active_processes": [],
    "generation_output_paths": [],
    "cancel_requested": False,
    "last_image_browse_dir": "",
    "models_unloaded": False,
    "last_image_path": "",
    "current_project_folder": "",
    # Session management (Generation left column)
    "active_session_id": "",       # folder name under output/, or "" for blank new
    "sessions_sidebar_expanded": True,
    "session_status": "idle",      # idle | running | stopped | done
    # Per-still regenerate queue (0-based line indices); drained when idle
    "regen_queue": [],
    "generating": False,
    # 1-based lyric line numbers listed for generation but not started yet
    # (drives thumbnails_qued_for_generation.jpg in the Materials grid)
    "thumb_queued_lines": [],
}


def init_session_state() -> None:
    cfg = load_configuration()
    APP_STATE["last_image_browse_dir"] = cfg.get("last_image_browse_dir", "")
    APP_STATE["cancel_requested"] = False
    APP_STATE["active_processes"] = []
    APP_STATE["generation_output_paths"] = []


# ---------------------------------------------------------------------------
# constants.ini
# ---------------------------------------------------------------------------

def get_cpu_info() -> Dict[str, Any]:
    path = get_constants_path()
    defaults = {
        "brand": "unknown",
        "vendor": "unknown",
        "cores_logical": 8,
        "cores_physical": 4,
        "default_threads": HEAVY_THREADS,
        "build_jobs": HEAVY_THREADS,
        "arch": "x86_64",
        "has_avx2": False,
        "has_avx512": False,
    }
    if not path.exists():
        return defaults
    cp = configparser.ConfigParser()
    try:
        cp.read(path, encoding="utf-8")
        sec = cp["cpu"] if "cpu" in cp else {}
        return {
            "brand": sec.get("brand", defaults["brand"]),
            "vendor": sec.get("vendor", defaults["vendor"]),
            "cores_logical": sec.getint("cores_logical", defaults["cores_logical"]),
            "cores_physical": sec.getint("cores_physical", defaults["cores_physical"]),
            "default_threads": sec.getint("default_threads", HEAVY_THREADS),
            "build_jobs": sec.getint("build_jobs", HEAVY_THREADS),
            "arch": sec.get("arch", defaults["arch"]),
            "has_avx2": sec.getboolean("has_avx2", False) if hasattr(sec, "getboolean") else False,
            "has_avx512": sec.getboolean("has_avx512", False) if hasattr(sec, "getboolean") else False,
        }
    except Exception:
        return defaults


def get_vulkan_info() -> Dict[str, Any]:
    path = get_constants_path()
    result: Dict[str, Any] = {
        "available": False,
        "version": "unknown",
        "sdk": "",
        "devices": [],
        "enumerated_by": "none",
        "install_type": "cpu",
    }
    if not path.exists():
        return result
    cp = configparser.ConfigParser()
    try:
        cp.read(path, encoding="utf-8")
        if "vulkan" in cp:
            sec = cp["vulkan"]
            result["available"] = sec.getboolean("available", False)
            result["version"] = sec.get("version", "unknown")
            result["sdk"] = sec.get("sdk", "")
            result["enumerated_by"] = sec.get("enumerated_by", "none")
        if "install" in cp:
            result["install_type"] = cp["install"].get("install_type", "cpu")
        devices = []
        for section in cp.sections():
            if section.startswith("device_"):
                d = cp[section]
                devices.append({
                    "backend": d.get("backend", "Vulkan"),
                    "index": d.getint("index", 0),
                    "name": d.get("name", "GPU"),
                    "vram_total_mb": d.getint("vram_total_mb", 0),
                    "vram_free_mb": d.getint("vram_free_mb", 0),
                    "fp16": d.getboolean("fp16", False),
                })
        result["devices"] = sorted(devices, key=lambda x: x["index"])
    except Exception:
        pass
    return result


def get_install_info() -> Dict[str, Any]:
    """Install-time constants from constants.ini [install] section."""
    defaults = {
        "install_type": "cpu",
        "backend_method": "download",
        "llama_ref": "",
        "sd_ref": "",
        "heavy_threads": HEAVY_THREADS,
    }
    path = get_constants_path()
    if not path.exists():
        return defaults
    cp = configparser.ConfigParser()
    try:
        cp.read(path, encoding="utf-8")
        if "install" not in cp:
            return defaults
        sec = cp["install"]
        return {
            "install_type": sec.get("install_type", defaults["install_type"]),
            "backend_method": sec.get("backend_method", defaults["backend_method"]),
            "llama_ref": sec.get("llama_ref", ""),
            "sd_ref": sec.get("sd_ref", ""),
            "heavy_threads": sec.getint("heavy_threads", HEAVY_THREADS),
        }
    except Exception:
        return defaults


def get_default_threads() -> int:
    return get_cpu_info().get("default_threads", HEAVY_THREADS)


def get_backend_choices() -> Dict[str, Any]:
    """Populate backend dropdown from constants.ini devices + CPU."""
    vk = get_vulkan_info()
    choices = ["CPU"]
    for d in vk.get("devices", []):
        label = f"Vulkan{d['index']} — {d['name']}"
        choices.append(label)
    return {"all_choices": choices, "cpu_label": "CPU"}


def get_thread_choices() -> List[int]:
    logical = get_cpu_info().get("cores_logical", 8)
    # Always offer 10 as primary, plus a few sensible values
    base = [4, 6, 8, 10, 12, 16]
    return sorted(set(b for b in base if b <= logical) | {HEAVY_THREADS})



def get_worker_threads() -> int:
    """Single shared thread count for all intensive backends."""
    cfg = load_configuration()
    try:
        n = int(cfg.get("worker_threads", WORKER_THREADS_DEFAULT))
    except (TypeError, ValueError):
        n = WORKER_THREADS_DEFAULT
    logical = get_cpu_info().get("cores_logical", 8) or 8
    # Never use fewer than MIN; never more than (logical - offset) usable cores
    max_usable = max(WORKER_THREADS_MIN, logical - AFFINITY_CORE_OFFSET)
    return max(WORKER_THREADS_MIN, min(int(n), max_usable))


def affinity_core_list(n_threads: Optional[int] = None) -> List[int]:
    """
    Logical CPU indices for worker threads, always skipping 0 and 1.
    Example: n=8 → [2, 3, 4, 5, 6, 7, 8, 9]
    """
    n = int(n_threads if n_threads is not None else get_worker_threads())
    logical = get_cpu_info().get("cores_logical", 16) or 16
    cores = []
    i = AFFINITY_CORE_OFFSET
    while len(cores) < n and i < logical:
        cores.append(i)
        i += 1
    return cores


def affinity_mask(n_threads: Optional[int] = None) -> int:
    """Windows process affinity bitmask for affinity_core_list()."""
    mask = 0
    for c in affinity_core_list(n_threads):
        mask |= (1 << c)
    return mask



# ---------------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------------

def _load_json_with_defaults(path: Path, defaults: Dict[str, Any]) -> Dict[str, Any]:
    data = dict(defaults)
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                loaded = json.load(f)
            if isinstance(loaded, dict):
                data.update(loaded)
        except (json.JSONDecodeError, OSError):
            pass
    return data


def _save_json_atomic(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    tmp.replace(path)


# ---------------------------------------------------------------------------
# configuration.json  (models, backends, threads, window)
# ---------------------------------------------------------------------------

CONFIGURATION_KEYS = [
    "encoder_model_path",
    "mmproj_path",
    "thinking_model_path",
    "imagegen_model_path",
    "vae_model_path",
    "llm_model_path",
    "encoder_backend",
    "thinking_backend",
    "imagegen_backend",
    "worker_threads",
    "encoder_threads",
    "thinking_threads",
    "imagegen_threads",
    "imagegen_vulkan_device",
    "imagegen_placement",
    "model_load_mode",
    "text_load_mode",
    "encoder_load_mode",
    "thinking_load_mode",
    "imagegen_load_mode",
    "text_gpu_layers",
    "encoder_gpu_layers",
    "thinking_gpu_layers",
    "imagegen_gpu_layers",
    "last_model_browse_dir",
    "last_image_browse_dir",
    "last_audio_browse_dir",
    "window_x",
    "window_y",
    "window_width",
    "window_height",
    "window_maximized",
]


def _default_configuration() -> Dict[str, Any]:
    return {
        "encoder_model_path": "",
        "mmproj_path": "",
        "thinking_model_path": "",
        "imagegen_model_path": "",
        "vae_model_path": "",
        "llm_model_path": "",
        "encoder_backend": "CPU",
        "thinking_backend": "CPU",
        "imagegen_backend": "CPU",
        "worker_threads": WORKER_THREADS_DEFAULT,
        "encoder_threads": HEAVY_THREADS,
        "thinking_threads": HEAVY_THREADS,
        "imagegen_threads": HEAVY_THREADS,
        "imagegen_vulkan_device": -1,
        "imagegen_placement": DEFAULT_PLACEMENT,
        "model_load_mode": DEFAULT_LOAD_MODE,  # legacy shared default
        "text_load_mode": DEFAULT_LOAD_MODE,
        "encoder_load_mode": DEFAULT_LOAD_MODE,
        "thinking_load_mode": DEFAULT_LOAD_MODE,
        "imagegen_load_mode": DEFAULT_LOAD_MODE,
        "text_gpu_layers": DEFAULT_GPU_LAYERS,
        "encoder_gpu_layers": DEFAULT_GPU_LAYERS,
        "thinking_gpu_layers": DEFAULT_GPU_LAYERS,
        "imagegen_gpu_layers": DEFAULT_GPU_LAYERS,
        "last_model_browse_dir": "",
        "last_image_browse_dir": "",
        "last_audio_browse_dir": "",
        "window_x": WINDOW_GEOMETRY_UNSET,
        "window_y": WINDOW_GEOMETRY_UNSET,
        "window_width": WINDOW_DEFAULT_WIDTH,
        "window_height": WINDOW_DEFAULT_HEIGHT,
        "window_maximized": False,
    }


def normalize_placement(value: str) -> str:
    """Map legacy placement labels to current Gpu_Only / Split values."""
    v = (value or "").strip()
    if v in PLACEMENT_CHOICES:
        return v
    # Legacy labels from earlier builds
    low = v.lower().replace(" ", "").replace("_", "")
    if low in ("fullgpu", "gpuonly", "gpu"):
        return PLACEMENT_GPU_ONLY
    if low in ("split", "partial", "cpuoffload"):
        return PLACEMENT_SPLIT
    return DEFAULT_PLACEMENT


def normalize_load_mode(value: str) -> str:
    v = (value or "").strip()
    if v in LOAD_MODE_CHOICES:
        return v
    low = v.lower().replace(" ", "").replace("_", "-")
    if low in ("mlock", "m-lock", "lock"):
        return LOAD_MODE_MLOCK
    return LOAD_MODE_ONE_SHOT


def load_configuration() -> Dict[str, Any]:
    return _load_json_with_defaults(get_configuration_path(), _default_configuration())


def save_configuration(data: Dict[str, Any]) -> None:
    filtered = {k: v for k, v in data.items() if k in CONFIGURATION_KEYS}
    _save_json_atomic(get_configuration_path(), filtered)


def update_configuration(updates: Dict[str, Any]) -> Dict[str, Any]:
    data = load_configuration()
    data.update({k: v for k, v in updates.items() if k in CONFIGURATION_KEYS})
    save_configuration(data)
    return data


# ---------------------------------------------------------------------------
# preferences.json
# ---------------------------------------------------------------------------

PREFERENCES_KEYS = [
    "style",
    "video_format",
    "max_thumbnails",
    "input_thumbnail_size",
    "bleep_section_completion",
    "bleep_video_completion",
]


def _default_preferences() -> Dict[str, Any]:
    return {
        "style": STYLE_DEFAULT,
        "video_format": VIDEO_MP4,
        "max_thumbnails": DEFAULT_MAX_THUMBNAILS,
        "input_thumbnail_size": DEFAULT_INPUT_THUMBNAIL,
        "bleep_section_completion": False,
        "bleep_video_completion": False,
    }


def load_preferences() -> Dict[str, Any]:
    return _load_json_with_defaults(get_preferences_path(), _default_preferences())


def save_preferences(data: Dict[str, Any]) -> None:
    filtered = {k: v for k, v in data.items() if k in PREFERENCES_KEYS}
    _save_json_atomic(get_preferences_path(), filtered)


def update_preferences(updates: Dict[str, Any]) -> Dict[str, Any]:
    data = load_preferences()
    data.update({k: v for k, v in updates.items() if k in PREFERENCES_KEYS})
    save_preferences(data)
    return data


# ---------------------------------------------------------------------------
# prompting.json  (editable visual-style prompt templates)
# ---------------------------------------------------------------------------

PROMPTING_KEYS = [
    STYLE_LIGHT,
    STYLE_DARK,
    STYLE_COLORFUL,
]


def _default_prompting() -> Dict[str, Any]:
    return {
        STYLE_LIGHT: STYLE_PROMPT_TEMPLATES[STYLE_LIGHT],
        STYLE_DARK: STYLE_PROMPT_TEMPLATES[STYLE_DARK],
        STYLE_COLORFUL: STYLE_PROMPT_TEMPLATES[STYLE_COLORFUL],
    }


def load_prompting() -> Dict[str, Any]:
    return _load_json_with_defaults(get_prompting_path(), _default_prompting())


def save_prompting(data: Dict[str, Any]) -> None:
    filtered = {k: str(v) for k, v in data.items() if k in PROMPTING_KEYS}
    # Ensure all three styles are always present
    defaults = _default_prompting()
    for k in PROMPTING_KEYS:
        if k not in filtered or not str(filtered[k]).strip():
            filtered[k] = defaults[k]
    _save_json_atomic(get_prompting_path(), filtered)


def update_prompting(updates: Dict[str, Any]) -> Dict[str, Any]:
    data = load_prompting()
    data.update({k: v for k, v in updates.items() if k in PROMPTING_KEYS})
    save_prompting(data)
    return data


# ---------------------------------------------------------------------------
# generation.json  (per-run / last-used generation params)
# ---------------------------------------------------------------------------

GENERATION_KEYS = [
    "imagegen_width",
    "imagegen_height",
    "imagegen_size",
    "imagegen_frequency",
    "imagegen_steps",
    "imagegen_cfg_scale",
    "imagegen_seed",
    "imagegen_sampling",
    "song_length_seconds",
    "last_lyrics",
    "last_audio_path",
    "output_resolution",
    "last_project_folder",
    "last_markers",
    "reference_image_path",
    "hair_style",
    "outfit_worn",
    "project_label",
    "last_image_gen_seconds",
]


def _default_generation() -> Dict[str, Any]:
    return {
        "imagegen_width": DEFAULT_WIDTH,
        "imagegen_height": DEFAULT_HEIGHT,
        "imagegen_size": DEFAULT_IMAGE_SIZE,
        "imagegen_frequency": DEFAULT_IMAGE_FREQUENCY,
        "imagegen_steps": DEFAULT_STEPS,
        "imagegen_cfg_scale": DEFAULT_CFG,
        "imagegen_seed": DEFAULT_SEED,
        "imagegen_sampling": DEFAULT_SAMPLER,
        "song_length_seconds": 180,
        "last_lyrics": "",
        "last_audio_path": "",
        "output_resolution": "720p",
        "last_project_folder": "",
        "last_markers": [],
        "reference_image_path": "",
        "hair_style": HAIR_STYLE_DEFAULT,
        "outfit_worn": OUTFIT_DEFAULT,
        "project_label": "",
        "last_image_gen_seconds": 0.0,
    }


def load_generation() -> Dict[str, Any]:
    return _load_json_with_defaults(get_generation_path(), _default_generation())


def save_generation(data: Dict[str, Any]) -> None:
    filtered = {k: v for k, v in data.items() if k in GENERATION_KEYS}
    _save_json_atomic(get_generation_path(), filtered)


def update_generation(updates: Dict[str, Any]) -> Dict[str, Any]:
    data = load_generation()
    data.update({k: v for k, v in updates.items() if k in GENERATION_KEYS})
    save_generation(data)
    return data


def generation_config() -> Dict[str, Any]:
    """Merged view used by the pipeline."""
    merged = dict(load_configuration())
    merged.update(load_preferences())
    merged.update(load_generation())
    return merged


# ---------------------------------------------------------------------------
# Window geometry
# ---------------------------------------------------------------------------

def get_window_geometry() -> Dict[str, Any]:
    cfg = load_configuration()

    def _int(key: str, default: int) -> int:
        try:
            return int(cfg.get(key, default))
        except (TypeError, ValueError):
            return default

    return {
        "x": _int("window_x", WINDOW_GEOMETRY_UNSET),
        "y": _int("window_y", WINDOW_GEOMETRY_UNSET),
        "width": max(640, _int("window_width", WINDOW_DEFAULT_WIDTH)),
        "height": max(480, _int("window_height", WINDOW_DEFAULT_HEIGHT)),
        "maximized": bool(cfg.get("window_maximized", False)),
    }


def save_window_geometry(x: int, y: int, width: int, height: int, maximized: bool) -> None:
    update_configuration({
        "window_x": int(x),
        "window_y": int(y),
        "window_width": int(width),
        "window_height": int(height),
        "window_maximized": bool(maximized),
    })


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def resolve_model_path(path_str: str, fallback_dir: Optional[Path] = None) -> Optional[Path]:
    if not path_str:
        return None
    p = Path(path_str).expanduser()
    if p.is_absolute() and p.exists():
        return p
    fb = fallback_dir or get_models_dir()
    if (fb / p).exists():
        return fb / p
    root_rel = _get_project_root() / p
    if root_rel.exists():
        return root_rel
    return None


def get_last_image_dir() -> str:
    d = APP_STATE.get("last_image_browse_dir") or load_configuration().get("last_image_browse_dir", "")
    if d and Path(d).is_dir():
        return d
    pictures = Path.home() / "Pictures"
    return str(pictures) if pictures.is_dir() else str(get_output_dir())


def set_last_image_dir(path: str) -> None:
    APP_STATE["last_image_browse_dir"] = path
    update_configuration({"last_image_browse_dir": path})


def get_images_dir() -> Path:
    """Optional icons / banner folder."""
    return _get_project_root() / "images"


# ---------------------------------------------------------------------------
# Session management (project folders under output/)
# ---------------------------------------------------------------------------

SESSION_META_NAME = "session.json"

# session.json schema (written by pipeline / UI):
# {
#   "song_name": str,
#   "lyrics": str,
#   "style": str,
#   "steps": int,
#   "cfg_scale": float,
#   "reference_image": str,   # original path or project-relative
#   "phase": "none"|"analysis"|"prompts"|"images"|"done"|"stopped",
#   "line_count": int,
#   "images_done": int,
#   "created": float,         # unix time
#   "updated": float,
# }


def _session_meta_path(project_dir: Path) -> Path:
    return project_dir / SESSION_META_NAME


def load_session_meta(project_dir: Path) -> Dict[str, Any]:
    path = _session_meta_path(project_dir)
    defaults: Dict[str, Any] = {
        "song_name": project_dir.name,
        "lyrics": "",
        "style": STYLE_LIGHT,
        "steps": DEFAULT_STEPS,
        "cfg_scale": DEFAULT_CFG,
        "reference_image": "",
        "negative_prompt": DEFAULT_NEGATIVE_PROMPT,
        "phase": "none",
        "line_count": 0,
        "images_done": 0,
        "created": 0.0,
        "updated": 0.0,
    }
    if not path.exists():
        return defaults
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            defaults.update(data)
    except (json.JSONDecodeError, OSError):
        pass
    return defaults


def save_session_meta(project_dir: Path, updates: Dict[str, Any]) -> Dict[str, Any]:
    data = load_session_meta(project_dir)
    data.update(updates or {})
    import time as _time
    data["updated"] = _time.time()
    if not data.get("created"):
        data["created"] = data["updated"]
    path = _session_meta_path(project_dir)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        tmp.replace(path)
    except OSError as e:
        print(f"[session] could not write meta: {e}", flush=True)
    return data


def list_session_images(project_dir: Path) -> List[str]:
    """Sorted list of numbered stills only (001-… / 001 - …). Never reference.*."""
    if not project_dir or not Path(project_dir).is_dir():
        return []
    imgs: List[Tuple[int, Path]] = []
    for p in Path(project_dir).iterdir():
        if not p.is_file():
            continue
        if p.suffix.lower() not in (".png", ".jpg", ".jpeg", ".webp"):
            continue
        if p.name.lower().startswith("reference"):
            continue
        # Accept "001-slug.png", "001 - lyric.png", "001.png"
        m = re.match(r"^(\d{3})(?:\s*[-–—]|[-.]|$)", p.name)
        if not m:
            continue
        imgs.append((int(m.group(1)), p))
    imgs.sort(key=lambda t: (t[0], t[1].name))
    return [str(p) for _, p in imgs]


def list_sessions() -> List[Dict[str, Any]]:
    """
    Enumerate project folders under output/.
    Each entry: id (folder name), path, label, phase, images_done, line_count, mtime.
    Sorted newest-first by updated/mtime.
    """
    root = get_output_dir()
    if not root.is_dir():
        return []
    sessions: List[Dict[str, Any]] = []
    for p in root.iterdir():
        if not p.is_dir():
            continue
        # Skip hidden / temp
        if p.name.startswith("."):
            continue
        meta = load_session_meta(p)
        images = list_session_images(p)
        images_done = len(images)
        # Prefer meta; fall back to counting files / lyrics
        line_count = int(meta.get("line_count") or 0)
        if line_count <= 0:
            lyrics_file = p / "lyrics.txt"
            if lyrics_file.exists():
                try:
                    # rough: non-empty non-header lines
                    text = lyrics_file.read_text(encoding="utf-8", errors="replace")
                    n = 0
                    for raw in text.splitlines():
                        t = raw.strip()
                        if not t:
                            continue
                        if t.startswith("[") and t.endswith("]"):
                            continue
                        n += 1
                    line_count = n
                except OSError:
                    pass
        phase = str(meta.get("phase") or "none")
        if phase in ("none", "") and images_done > 0:
            if line_count > 0 and images_done >= line_count:
                phase = "done"
            else:
                phase = "stopped"
        elif phase == "images" and line_count > 0 and images_done >= line_count:
            phase = "done"
        mtime = float(meta.get("updated") or 0) or p.stat().st_mtime
        song = (meta.get("song_name") or p.name).strip() or p.name
        sessions.append({
            "id": p.name,
            "path": str(p),
            "label": p.name,  # folder name already encodes song + serial
            "song_name": song,
            "phase": phase,
            "images_done": images_done,
            "line_count": line_count,
            "mtime": mtime,
            "image_paths": images,
            "lyrics": meta.get("lyrics") or "",
            "style": meta.get("style") or STYLE_LIGHT,
            "steps": meta.get("steps", DEFAULT_STEPS),
            "cfg_scale": meta.get("cfg_scale", DEFAULT_CFG),
            "reference_image": meta.get("reference_image") or "",
            "negative_prompt": meta.get("negative_prompt") if meta.get("negative_prompt") is not None else DEFAULT_NEGATIVE_PROMPT,
        })
    sessions.sort(key=lambda s: s["mtime"], reverse=True)
    return sessions


def get_session_by_id(session_id: str) -> Optional[Dict[str, Any]]:
    if not session_id:
        return None
    for s in list_sessions():
        if s["id"] == session_id:
            return s
    # Direct path check
    p = get_output_dir() / session_id
    if p.is_dir():
        meta = load_session_meta(p)
        images = list_session_images(p)
        return {
            "id": p.name,
            "path": str(p),
            "label": p.name,
            "song_name": meta.get("song_name") or p.name,
            "phase": meta.get("phase") or "none",
            "images_done": len(images),
            "line_count": int(meta.get("line_count") or 0),
            "mtime": float(meta.get("updated") or p.stat().st_mtime),
            "image_paths": images,
            "lyrics": meta.get("lyrics") or "",
            "style": meta.get("style") or STYLE_LIGHT,
            "steps": meta.get("steps", DEFAULT_STEPS),
            "cfg_scale": meta.get("cfg_scale", DEFAULT_CFG),
            "reference_image": meta.get("reference_image") or "",
            "negative_prompt": meta.get("negative_prompt") if meta.get("negative_prompt") is not None else DEFAULT_NEGATIVE_PROMPT,
        }
    return None


def delete_session(session_id: str) -> bool:
    """Remove one project folder under output/. Returns True on success."""
    if not session_id:
        return False
    p = get_output_dir() / session_id
    if not p.is_dir():
        return False
    # Safety: only delete under output/
    try:
        p.resolve().relative_to(get_output_dir().resolve())
    except ValueError:
        return False
    try:
        shutil.rmtree(p)
        if APP_STATE.get("current_project_folder") == str(p):
            APP_STATE["current_project_folder"] = ""
        if APP_STATE.get("active_session_id") == session_id:
            APP_STATE["active_session_id"] = ""
            APP_STATE["session_status"] = "idle"
        return True
    except OSError:
        return False


def delete_all_sessions() -> int:
    """Delete every project folder under output/. Returns count removed."""
    root = get_output_dir()
    if not root.is_dir():
        return 0
    n = 0
    for p in list(root.iterdir()):
        if p.is_dir() and not p.name.startswith("."):
            try:
                shutil.rmtree(p)
                n += 1
            except OSError:
                pass
    APP_STATE["current_project_folder"] = ""
    APP_STATE["active_session_id"] = ""
    APP_STATE["session_status"] = "idle"
    APP_STATE["generation_output_paths"] = []
    return n


def session_status_label(phase: str, images_done: int, line_count: int) -> str:
    """Short UI badge for a session row."""
    phase = (phase or "none").lower()
    if phase == "done" or (line_count > 0 and images_done >= line_count):
        return "done"
    if phase in ("stopped", "images", "prompts", "analysis") and images_done > 0:
        return f"{images_done}/{line_count or '?'}"
    if phase in ("stopped", "images", "prompts", "analysis"):
        return phase
    if images_done > 0:
        return f"{images_done}/{line_count or '?'}"
    return "new"


# end session helpers
