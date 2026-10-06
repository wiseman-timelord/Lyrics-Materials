"""
inference.py - Lyrics → song/section analysis → visual prompts → images (Flux.2-klein-4B).
Produces numbered stills for external AI video generation (Lyrics-Materials).
"""
from __future__ import annotations

import gc
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import traceback
import unicodedata
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import scripts.configure as configure
import scripts.utilities as utilities

# Serialize every sd-cli invocation — never load two Diffusers into VRAM at once
_SD_CLI_LOCK = threading.Lock()


# ---------------------------------------------------------------------------
# Completion bleeps
# ---------------------------------------------------------------------------

def _play_bleep(count: int = 1) -> None:
    try:
        if sys.platform == "win32":
            import winsound
            for i in range(max(1, int(count))):
                winsound.MessageBeep(winsound.MB_OK)
                if i + 1 < count:
                    time.sleep(0.18)
        else:
            for i in range(max(1, int(count))):
                print("\a", end="", flush=True)
                if i + 1 < count:
                    time.sleep(0.18)
    except Exception:
        pass



def record_last_image_gen_seconds(elapsed: float) -> None:
    """Persist the time taken for the most recent successful still generation."""
    try:
        sec = round(float(elapsed), 1)
        if sec <= 0:
            return
        configure.update_generation({"last_image_gen_seconds": sec})
        configure.APP_STATE["last_image_gen_seconds"] = sec
    except Exception:
        pass


def last_image_gen_seconds() -> float:
    try:
        v = configure.APP_STATE.get("last_image_gen_seconds")
        if v is not None:
            return float(v)
    except (TypeError, ValueError):
        pass
    try:
        return float(configure.load_generation().get("last_image_gen_seconds") or 0)
    except (TypeError, ValueError):
        return 0.0


def maybe_bleep_section() -> None:
    prefs = configure.load_preferences()
    if prefs.get("bleep_section_completion"):
        _play_bleep(1)


def maybe_bleep_done() -> None:
    prefs = configure.load_preferences()
    section_on = bool(prefs.get("bleep_section_completion"))
    video_on = bool(prefs.get("bleep_video_completion"))  # reused as "batch done"
    if video_on and section_on:
        _play_bleep(2)
    elif video_on or section_on:
        _play_bleep(1)


# ---------------------------------------------------------------------------
# Process registry / Emergency Stop
# ---------------------------------------------------------------------------

def register_process(proc) -> None:
    if proc is None:
        return
    procs = configure.APP_STATE.setdefault("active_processes", [])
    if proc not in procs:
        procs.append(proc)
    _apply_worker_affinity(proc)


def unregister_process(proc) -> None:
    procs = configure.APP_STATE.get("active_processes") or []
    try:
        while proc in procs:
            procs.remove(proc)
    except Exception:
        pass
    configure.APP_STATE["active_processes"] = procs


def is_cancel_requested() -> bool:
    return bool(configure.APP_STATE.get("cancel_requested"))


def request_cancel() -> None:
    configure.APP_STATE["cancel_requested"] = True


def clear_cancel_state() -> None:
    configure.APP_STATE["cancel_requested"] = False
    configure.APP_STATE["active_processes"] = []
    configure.APP_STATE["generation_output_paths"] = []


def clear_project_image_assets(project_dir: Path) -> int:
    """
    Delete Cover / Theme / Lyrics stills from a project folder.
    Keeps analysis.txt, lyrics.txt, prompts.txt, character_map.txt, session.json, reference.*.
    Returns number of files removed.
    """
    if not project_dir or not Path(project_dir).is_dir():
        return 0
    removed = 0
    root = Path(project_dir)
    for p in list(root.iterdir()):
        if not p.is_file():
            continue
        low = p.name.lower()
        suf = p.suffix.lower()
        if suf not in (".png", ".jpg", ".jpeg", ".webp"):
            continue
        if low.startswith("reference"):
            continue
        # cover-*, theme-*, numbered stills (001-… / 001 - …)
        if (
            low.startswith("cover")
            or low.startswith("theme")
            or re.match(r"^\d{3}", p.name)
        ):
            try:
                p.unlink()
                removed += 1
            except OSError:
                pass
    # Also clear theme_prompts so theme phase rewrites them
    for name in ("theme_prompts.txt", "prompts.txt"):
        tp = root / name
        if tp.is_file():
            try:
                tp.unlink()
            except OSError:
                pass
    configure.APP_STATE["generation_output_paths"] = []
    configure.APP_STATE["thumb_expected_count"] = 0
    configure.APP_STATE["thumb_generating_line"] = None
    configure.APP_STATE["thumb_queued_lines"] = []
    return removed


def kill_active_processes() -> int:
    procs = list(configure.APP_STATE.get("active_processes") or [])
    killed = 0
    for proc in procs:
        try:
            if proc.poll() is None:
                proc.terminate()
                killed += 1
        except Exception:
            pass
    if procs:
        time.sleep(0.15)
    for proc in procs:
        try:
            if proc.poll() is None:
                proc.kill()
        except Exception:
            pass
        unregister_process(proc)
    return killed


def emergency_stop() -> str:
    request_cancel()
    n = kill_active_processes()
    msg = "Pipeline stopped by user."
    if n:
        msg += f" Terminated {n} process(es)."
    # Persist partial progress so the session can be resumed
    try:
        pf = configure.APP_STATE.get("current_project_folder") or ""
        if pf and Path(pf).is_dir():
            imgs = configure.list_session_images(Path(pf))
            configure.save_session_meta(Path(pf), {
                "phase": "stopped",
                "images_done": len(imgs),
            })
            configure.APP_STATE["session_status"] = "stopped"
            configure.APP_STATE["generation_output_paths"] = imgs
    except Exception as e:
        print(f"[stop] session meta update failed: {e}", flush=True)
    print(f"[stop] {msg}", flush=True)
    return msg


# ---------------------------------------------------------------------------
# Affinity / threads / binary finders
# ---------------------------------------------------------------------------

def _apply_worker_affinity(proc) -> None:
    if proc is None or sys.platform != "win32":
        return
    try:
        import ctypes
        mask = configure.affinity_mask()
        if mask <= 0:
            return
        handle = ctypes.windll.kernel32.OpenProcess(0x0200 | 0x0400, False, proc.pid)
        if handle:
            ctypes.windll.kernel32.SetProcessAffinityMask(handle, mask)
            ctypes.windll.kernel32.CloseHandle(handle)
    except Exception:
        pass


def _worker_thread_count(cfg: Optional[Dict[str, Any]] = None) -> int:
    if cfg:
        try:
            n = int(cfg.get("worker_threads") or cfg.get("encoder_threads") or 0)
            if n >= configure.WORKER_THREADS_MIN:
                return n
        except (TypeError, ValueError):
            pass
    return configure.get_worker_threads()


def find_llama_completion() -> Optional[Path]:
    bin_dir = configure.get_llama_bin_dir()
    for name in ("llama-completion.exe", "llama-completion"):
        p = bin_dir / name
        if p.exists():
            return p
    found = shutil.which("llama-completion")
    return Path(found) if found else None


def find_sd_cpp() -> Optional[Path]:
    bin_dir = configure.get_sd_bin_dir()
    for name in ("sd-cli.exe", "sd-cli", "sd.exe", "sd"):
        p = bin_dir / name
        if p.exists():
            return p
    for name in ("sd-cli", "sd"):
        found = shutil.which(name)
        if found:
            return Path(found)
    return None


# ---------------------------------------------------------------------------
# Backend / load-mode helpers
# ---------------------------------------------------------------------------

def _load_mode_for_role(cfg: Dict[str, Any], role: str) -> str:
    """
    Resolve per-role load mode:
      thinking / prompts → thinking_load_mode (Prompts Load Mode)
      encoder            → encoder_load_mode
      imagegen           → imagegen_load_mode
    Falls back to legacy text_load_mode / model_load_mode only if the
    role-specific key is missing.
    """
    role = (role or "encoder").lower()
    if role in ("thinking", "prompts", "prompt"):
        raw = (
            cfg.get("thinking_load_mode")
            or cfg.get("text_load_mode")
            or cfg.get("model_load_mode")
            or configure.DEFAULT_LOAD_MODE
        )
    elif role == "imagegen":
        raw = (
            cfg.get("imagegen_load_mode")
            or cfg.get("text_load_mode")
            or configure.DEFAULT_LOAD_MODE
        )
    else:  # encoder
        raw = (
            cfg.get("encoder_load_mode")
            or cfg.get("text_load_mode")
            or cfg.get("model_load_mode")
            or configure.DEFAULT_LOAD_MODE
        )
    return configure.normalize_load_mode(str(raw))


def _load_mode_args_for(cfg: Dict[str, Any], role: str = "encoder") -> List[str]:
    """
    Map per-role load mode → llama.cpp CLI flags.

    Modern llama.cpp (2026+) removed the bare ``--mlock`` / ``--mmap`` flags
    (PR #28334). The replacement is a single option:

        -lm, --load-mode MODE
          auto        — mmap when the device supports it (default; One-Shot)
          none        — no special mode (old --no-mmap)
          mmap        — memory-map model
          mlock       — force system to keep model in RAM (no swap)
          mmap+mlock  — mmap and pin in RAM
          dio         — DirectIO when available

    Our UI:
      One-Shot  →  omit flag (binary default = auto / lazy page-in)
      M-Lock    →  -lm mlock   (pin weights in RAM; never silently degrade)

    sd-cli has no equivalent — imagegen returns [].
    """
    if role in ("encoder", "thinking"):
        if _load_mode_for_role(cfg, role) == configure.LOAD_MODE_MLOCK:
            # Correct modern flag — bare --mlock is rejected as "invalid argument"
            return ["-lm", "mlock"]
        # One-Shot: leave default load-mode (auto)
        return []
    return []


def _gpu_layers_for(cfg: Dict[str, Any], backend: str, role: str = "encoder",
                    model_path: str = "") -> int:
    if not backend.upper().startswith("VULKAN") and not backend.upper().startswith("CUDA"):
        return 0
    if role in ("encoder", "thinking"):
        try:
            if role == "thinking":
                requested = int(
                    cfg.get("thinking_gpu_layers",
                            cfg.get("text_gpu_layers", configure.DEFAULT_GPU_LAYERS))
                )
            else:
                requested = int(
                    cfg.get("encoder_gpu_layers",
                            cfg.get("text_gpu_layers", configure.DEFAULT_GPU_LAYERS))
                )
        except (TypeError, ValueError):
            requested = configure.DEFAULT_GPU_LAYERS
        path = model_path or (
            cfg.get("thinking_model_path") if role == "thinking" else cfg.get("encoder_model_path")
        ) or ""
        free = configure.vram_free_for_backend(backend)
        ngl = configure.resolve_text_gpu_layers(str(path), requested, backend, free)
        if requested == -1:
            floor = configure.free_vram_floor_mb(free)
            blocks = configure.model_layer_count(str(path))
            print(
                f"[ngl] auto {role}: {ngl}/{blocks} layers "
                f"(safe VRAM floor {floor} MiB on {backend})",
                flush=True,
            )
        return ngl
    try:
        return int(cfg.get("imagegen_gpu_layers", configure.DEFAULT_GPU_LAYERS))
    except (TypeError, ValueError):
        return configure.DEFAULT_GPU_LAYERS


def _llama_backend_args(cfg: Dict[str, Any], role: str = "encoder",
                        model_path: str = "") -> List[str]:
    backend_key = "thinking_backend" if role == "thinking" else "encoder_backend"
    backend = str(cfg.get(backend_key) or cfg.get("encoder_backend") or "CPU")
    path = model_path or (
        cfg.get("thinking_model_path") if role == "thinking" else cfg.get("encoder_model_path")
    ) or ""
    extra: List[str] = []
    if backend.upper().startswith("VULKAN"):
        m = re.search(r"(\d+)", backend)
        idx = int(m.group(1)) if m else 0
        ngl = _gpu_layers_for(cfg, backend, role, model_path=str(path))
        extra.extend(["-dev", f"Vulkan{idx}", "-ngl", str(ngl)])
    elif backend.upper().startswith("CUDA"):
        m = re.search(r"(\d+)", backend)
        idx = int(m.group(1)) if m else 0
        ngl = _gpu_layers_for(cfg, backend, role, model_path=str(path))
        extra.extend(["-dev", f"CUDA{idx}", "-ngl", str(ngl)])
    else:
        extra.extend(["-ngl", "0"])
    extra.extend(_load_mode_args_for(cfg, role))
    return extra


def _resolve_prompt_model(cfg: Dict[str, Any]) -> Tuple[str, str]:
    th = (cfg.get("thinking_model_path") or "").strip()
    if th and Path(th).exists():
        return th, "thinking"
    enc = (cfg.get("encoder_model_path") or "").strip()
    return enc, "encoder"


def _backend_device_key(backend: str) -> str:
    """
    Canonical device key for conflict checks, e.g. 'vulkan0', 'cuda1', 'cpu'.
    Same key ⇒ same physical accelerator (must not hold two mlocked models).
    """
    b = (backend or "CPU").strip()
    low = b.lower()
    if low == "cpu" or not low:
        return "cpu"
    m = re.search(r"(vulkan|cuda)\s*(\d+)", low, re.I)
    if m:
        return f"{m.group(1).lower()}{m.group(2)}"
    m = re.search(r"(\d+)", low)
    if "vulkan" in low:
        return f"vulkan{m.group(1) if m else 0}"
    if "cuda" in low:
        return f"cuda{m.group(1) if m else 0}"
    return low


def _text_role_backend(cfg: Dict[str, Any], role: str) -> str:
    if role == "thinking":
        return str(cfg.get("thinking_backend") or cfg.get("encoder_backend") or "CPU")
    return str(cfg.get("encoder_backend") or "CPU")


def _imagegen_backend(cfg: Dict[str, Any]) -> str:
    return str(cfg.get("imagegen_backend") or "CPU")


def _is_mlock_enabled(cfg: Dict[str, Any], role: str = "text") -> bool:
    """
    Per-role M-Lock check. role may be 'text' (→ thinking/prompts for phase plan),
    'thinking', 'encoder', or 'imagegen'.
    """
    if role in ("text", "prompts", "prompt"):
        # Phase-plan default: the model used for prompts (thinking if set)
        path, resolved = _resolve_prompt_model(cfg)
        role = resolved  # 'thinking' or 'encoder'
    return _load_mode_for_role(cfg, role) == configure.LOAD_MODE_MLOCK


def _same_device_conflict(cfg: Dict[str, Any], text_role: str) -> bool:
    """True when text model and Flux share one accelerator."""
    text_key = _backend_device_key(_text_role_backend(cfg, text_role))
    img_key = _backend_device_key(_imagegen_backend(cfg))
    if text_key == "cpu" or img_key == "cpu":
        return False
    return text_key == img_key


def phase_barrier(label: str = "phase boundary") -> None:
    """
    Ensure every registered worker process has fully exited before the next
    phase starts. Critical when M-Lock + same-device: the previous model's
    VRAM must be released before the next model loads.
    """
    print(f"[phase] ── {label}: waiting for worker processes to exit…", flush=True)
    procs = list(configure.APP_STATE.get("active_processes") or [])
    still = []
    for proc in procs:
        try:
            if proc.poll() is None:
                still.append(proc)
        except Exception:
            pass
    if still:
        print(f"[phase] {len(still)} process(es) still live — terminating…", flush=True)
        for proc in still:
            try:
                proc.terminate()
            except Exception:
                pass
        time.sleep(0.4)
        for proc in still:
            try:
                if proc.poll() is None:
                    proc.kill()
            except Exception:
                pass
            unregister_process(proc)
    # Drop any stale handles
    configure.APP_STATE["active_processes"] = []
    # Brief pause so the driver can reclaim VRAM after mlock release
    time.sleep(0.6)
    print(f"[phase] ── {label}: clear — safe to load next model", flush=True)


def log_phase_plan(cfg: Dict[str, Any], text_role: str) -> None:
    """Print the two-phase device plan so the user can see unload intent."""
    text_backend = _text_role_backend(cfg, text_role)
    img_backend = _imagegen_backend(cfg)
    text_mode = _load_mode_for_role(cfg, text_role)
    enc_mode = _load_mode_for_role(cfg, "encoder")
    img_mode = _load_mode_for_role(cfg, "imagegen")
    text_mlock = text_mode == configure.LOAD_MODE_MLOCK
    img_mlock = img_mode == configure.LOAD_MODE_MLOCK
    conflict = _same_device_conflict(cfg, text_role)
    print("[plan] ══════════════════════════════════════════════", flush=True)
    print(f"[plan] Load modes — Prompts/Thinking: {text_mode if text_role == 'thinking' else enc_mode}"
          f"  |  Encoder: {enc_mode}  |  ImageGen: {img_mode}", flush=True)
    print(f"[plan] Phase 1 — assessment + prompts", flush=True)
    print(f"[plan]   model role : {text_role}", flush=True)
    print(f"[plan]   backend    : {text_backend}  (device key {_backend_device_key(text_backend)})", flush=True)
    print(f"[plan]   load mode  : {text_mode}"
          f"{' (-lm mlock)' if text_mlock else ''}", flush=True)
    print(f"[plan] Phase 2 — image generation (Flux)", flush=True)
    print(f"[plan]   backend    : {img_backend}  (device key {_backend_device_key(img_backend)})", flush=True)
    print(f"[plan]   load mode  : {img_mode} "
          f"(sd-cli has no --mlock; process exit releases VRAM)", flush=True)
    if conflict and (text_mlock or img_mlock):
        print(
            f"[plan] SAME DEVICE + M-Lock → Thinking will fully unload before "
            f"Flux loads on {_backend_device_key(text_backend)}. "
            f"They will NOT be resident together.",
            flush=True,
        )
    elif conflict:
        print(
            f"[plan] Same device ({_backend_device_key(text_backend)}) but One-Shot — "
            f"still sequential: text phase finishes before Flux.",
            flush=True,
        )
    else:
        print(
            f"[plan] Different devices — text on {_backend_device_key(text_backend)}, "
            f"Flux on {_backend_device_key(img_backend)}; sequential phases still apply.",
            flush=True,
        )
    print("[plan] ══════════════════════════════════════════════", flush=True)



def _sd_placement_args(cfg: Dict[str, Any], use_gpu: bool, vk_idx: int = 0) -> List[str]:
    args: List[str] = []
    if not use_gpu:
        args.extend(["--backend", "cpu", "--params-backend", "cpu"])
    else:
        dev = f"vulkan{vk_idx}"
        placement = configure.normalize_placement(
            str(cfg.get("imagegen_placement", configure.DEFAULT_PLACEMENT))
        )
        if placement == configure.PLACEMENT_GPU_ONLY:
            spec = f"diffusion={dev},te={dev},vae={dev}"
        else:
            spec = f"diffusion={dev},te=cpu,vae=cpu"
        args.extend(["--backend", spec, "--params-backend", spec])
    args.extend(_load_mode_args_for(cfg, "imagegen"))
    return args


# ---------------------------------------------------------------------------
# Lyrics parsing (section-aware; skip blank + pure marker lines for images)
# ---------------------------------------------------------------------------

_SECTION_RE = re.compile(
    r"^\[?\s*(intro|outro|fade[_\s\-]*out|chorus|verse|bridge|pre[- ]?chorus|hook|refrain)"
    r"[\s_\-]*(\d*)\s*\]?$",
    re.IGNORECASE,
)


def _normalize_section_label(kind: str, number: str) -> str:
    k = (kind or "").strip().lower().replace("-", " ").replace("_", " ")
    k = re.sub(r"\s+", " ", k)
    if k.startswith("fade"):
        return "Fade Out"
    if k == "intro":
        return "Intro"
    if k == "outro":
        return "Outro"
    if k.startswith("pre"):
        return "Pre-Chorus" + (f" {number}" if number else "")
    title = k.title()
    if number:
        return f"{title} {number}"
    return title


def parse_lyrics(raw: str) -> List[Dict[str, Any]]:
    """
    Split pasted lyrics into ordered entries.
    Section headers become context; only non-empty non-header lines become image targets.
    """
    entries: List[Dict[str, Any]] = []
    idx = 0
    current_section = "Body"
    for raw_line in (raw or "").splitlines():
        text = raw_line.strip()
        if not text:
            continue
        bare = text.strip("[]").strip()
        m = _SECTION_RE.match(text) or _SECTION_RE.match(bare)
        if m or (text.startswith("[") and text.endswith("]")):
            if m:
                label = _normalize_section_label(m.group(1), m.group(2) or "")
            else:
                label = bare or text.strip("[]")
            current_section = label
            entries.append({
                "type": "section",
                "text": text.strip("[]"),
                "label": label,
                "index": -1,
            })
        else:
            entries.append({
                "type": "line",
                "text": text,
                "index": idx,
                "section": current_section,
            })
            idx += 1
    return entries


def lyric_lines_only(parsed: List[Dict[str, Any]]) -> List[str]:
    return [e["text"] for e in parsed if e["type"] == "line"]


def sections_with_lines(parsed: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    order: List[str] = []
    buckets: Dict[str, Dict[str, Any]] = {}
    for e in parsed:
        if e["type"] != "line":
            continue
        label = e.get("section") or "Body"
        if label not in buckets:
            buckets[label] = {"label": label, "lines": [], "line_indices": []}
            order.append(label)
        buckets[label]["lines"].append(e["text"])
        buckets[label]["line_indices"].append(e["index"])
    return [buckets[k] for k in order]


# ---------------------------------------------------------------------------
# Text generation helpers
# ---------------------------------------------------------------------------

_NAME_CTX = 8192
_NAME_PREDICT = 32
_ANALYSIS_CTX = 16384
# Full OVERALL + CHARACTER + MAP + multi-paragraph SECTIONS needs room;
# 512 truncated mid-Intro and left later sections blank.
_ANALYSIS_PREDICT = 2048
_PROMPT_CTX = 4096
_PROMPT_PREDICT = 180

# Generation timeouts (seconds) — analysis can run long with large n_predict
_TIMEOUT_NAME = 120.0
_TIMEOUT_ANALYSIS = 720.0
_TIMEOUT_PROMPT = 240.0
_TIMEOUT_SD = 600.0

_THINK_TAG_RE = re.compile(r"<think>.*?</think>", re.IGNORECASE | re.DOTALL)
_SLUG_RE = re.compile(r"[^a-z0-9]+")
_SAFE_FILENAME_RE = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
_WINDOWS_RESERVED_NAMES = {
    "con", "prn", "aux", "nul",
    *(f"com{i}" for i in range(1, 10)),
    *(f"lpt{i}" for i in range(1, 10)),
}


def _strip_think_tags(text: str) -> str:
    return _THINK_TAG_RE.sub("", text or "").strip()


def _wrap_qwen_chat(user_prompt: str, system: str = "") -> str:
    """
    Wrap a plain instruction in Qwen2/Qwen3 ChatML so the model continues as
    the assistant instead of treating the long prompt as already-complete text
    and emitting EOS immediately (the blank-assessment failure mode).
    """
    sys = (system or (
        "You are a precise music-video art director. "
        "Follow the user's structure exactly. Write real sentences, not keyword lists."
    )).strip()
    user = (user_prompt or "").strip()
    # End with the assistant header so generation starts in the right role
    return (
        f"<|im_start|>system\n{sys}<|im_end|>\n"
        f"<|im_start|>user\n{user}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


def _is_llama_log_line(line: str) -> bool:
    """True for llama.cpp verbose / ggml noise that is not model text."""
    s = (line or "").strip()
    if not s:
        return True
    low = s.lower()
    # Common log / telemetry prefixes
    if any(
        low.startswith(p) for p in (
            "llama_", "ggml_", "gguf_", "print_info", "load_", "llm_",
            "slot ", "sampler", "sampling:", "prompt eval", "eval time",
            "total time", "system info", "build:", "log_", "common_",
            "init:", "loading", "offloaded", "graph splits", "kv cache",
            "llama_model", "llama_context", "llama_new", "clip_",
        )
    ):
        return True
    if re.match(r"^[\d./\s%|]+$", s):  # progress bars / pure numbers
        return True
    if "tokens/s" in low or "ms/token" in low or "t/s" in low:
        return True
    if low in ("[end of text]", "end of text", "<|endoftext|>", "<|im_end|>"):
        return True
    return False


def _extract_completion_text(raw_out: str, prompt: str) -> str:
    """
    Pull the model answer out of interleaved verbose logs + optional prompt echo.
    Prefers text from the first OVERALL: / CHARACTER: heading onward.
    """
    text = (raw_out or "").replace("\r\n", "\n").replace("\r", "\n")
    # Drop pure log lines
    kept: List[str] = []
    for line in text.split("\n"):
        if _is_llama_log_line(line):
            continue
        kept.append(line)
    text = "\n".join(kept).strip()

    # Strip prompt echo (full or ChatML-wrapped)
    for candidate in (prompt, prompt.strip()):
        if candidate and text.startswith(candidate):
            text = text[len(candidate):].strip()
            break
    # Strip leading ChatML debris
    for marker in ("<|im_start|>assistant", "<|im_start|>assistant\n"):
        if marker in text:
            text = text.split(marker, 1)[-1].strip()
    for end in ("<|im_end|>", "<|endoftext|>", "[end of text]"):
        if end in text:
            text = text.split(end)[0].strip()

    # Prefer structured assessment body
    upper = text.upper()
    for key in ("OVERALL:", "OVERALL\n", "CHARACTER:", "SECTIONS:"):
        idx = upper.find(key)
        if idx >= 0:
            text = text[idx:].strip()
            break

    return _strip_think_tags(text)


def _run_llama_completion(
    prompt: str,
    cfg: Dict[str, Any],
    role: str,
    model_path: str,
    n_predict: int,
    ctx_size: int,
    temperature: float = 0.7,
    timeout: float = 360.0,
) -> str:
    if is_cancel_requested():
        return ""
    exe = find_llama_completion()
    if not exe:
        raise RuntimeError("llama-completion binary not found. Run Installation.")
    if not model_path or not Path(model_path).exists():
        raise RuntimeError(f"{role.title()} model not set/found: {model_path or '(empty)'}")

    backend_args = _llama_backend_args(cfg, role, model_path=str(model_path))
    threads = _worker_thread_count(cfg)
    # Qwen Instruct/Thinking need ChatML; plain -p often yields immediate EOS
    chat_prompt = _wrap_qwen_chat(prompt)
    cmd = [
        str(exe),
        "-m", str(model_path),
        "-p", chat_prompt,
        "-n", str(int(n_predict)),
        "-c", str(int(ctx_size)),
        "-t", str(threads),
        "--temp", str(temperature),
        # Verbose logging so the console shows load / offload / generate progress
        "-lv", "1",
        "-no-cnv",  # one-shot; template already applied in -p
        # Discourage instant stop after a long prompt
        "--repeat-penalty", "1.1",
    ]
    cmd.extend(backend_args)

    model_name = Path(model_path).name
    print(f"[llama] role={role}  model={model_name}", flush=True)
    print(f"[llama] threads={threads}  ctx={ctx_size}  n_predict={n_predict}  "
          f"timeout={timeout:.0f}s", flush=True)
    print(f"[llama] backend args: {' '.join(backend_args) or '(CPU / defaults)'}", flush=True)
    print(f"[llama] starting process (verbose)…", flush=True)
    t0 = time.time()

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,  # merge so we stream everything live
        text=True,
        encoding="utf-8",
        errors="replace",
        cwd=str(Path(exe).parent),
        bufsize=1,
    )
    register_process(proc)

    out_lines: List[str] = []
    model_loaded_announced = False
    generating_announced = False

    def _maybe_announce(line: str) -> None:
        nonlocal model_loaded_announced, generating_announced
        low = line.lower()
        # Common llama.cpp load milestones
        if not model_loaded_announced and any(
            k in low for k in (
                "llama_model_load", "model loaded", "loaded model",
                "load_tensors", "llm_load_tensors", "print_info: model size",
                "llama_context", "graph splits", "offloaded",
            )
        ):
            # Prefer a clear single banner when weights are in
            if any(k in low for k in ("model size", "offloaded", "model loaded", "loaded model")):
                model_loaded_announced = True
                elapsed = time.time() - t0
                print(f"[llama] *** MODEL LOADED ({elapsed:.1f}s) ***  {line.strip()[:160]}", flush=True)
        if model_loaded_announced and not generating_announced:
            if any(k in low for k in ("sampling", "eval", "prompt eval", "generating", "n_tokens")):
                generating_announced = True
                print(f"[llama] generating tokens…", flush=True)

    try:
        deadline = time.time() + timeout
        assert proc.stdout is not None
        while True:
            if is_cancel_requested():
                proc.kill()
                break
            if time.time() > deadline:
                proc.kill()
                raise subprocess.TimeoutExpired(cmd, timeout)
            line = proc.stdout.readline()
            if line == "" and proc.poll() is not None:
                break
            if not line:
                time.sleep(0.02)
                continue
            text_line = line.rstrip("\n\r")
            out_lines.append(text_line)
            # Echo load / device / error lines only — not the full prompt dump
            low = text_line.lower()
            interesting = any(
                k in low for k in (
                    "load", "layer", "vram", "vulkan", "cuda", "ggml", "offload",
                    "device", "error", "fail", "warn", "tensor", "mmap",
                    "mlock", "backend", "rpc", "sampling", "eval time",
                    "tokens/s", "system info", "build:", "print_info",
                    "model size", "model loaded", "offloaded",
                )
            )
            if interesting:
                print(f"[llama] {text_line[:200]}", flush=True)
            _maybe_announce(text_line)
        # Drain remainder
        rest = proc.stdout.read() if proc.stdout else ""
        if rest:
            for text_line in rest.splitlines():
                out_lines.append(text_line)
                print(f"[llama] {text_line}", flush=True)
                _maybe_announce(text_line)
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        try:
            proc.kill()
        except Exception:
            pass
        unregister_process(proc)
        elapsed = time.time() - t0
        print(f"[llama] TIMEOUT after {elapsed:.1f}s (limit {timeout:.0f}s)", flush=True)
        raise RuntimeError(
            f"{role.title()} completion timed out after {timeout:.0f}s "
            f"(model load + generate)."
        )
    unregister_process(proc)
    elapsed = time.time() - t0
    if not model_loaded_announced:
        print(f"[llama] (no explicit 'model loaded' marker seen — process finished in {elapsed:.1f}s)", flush=True)
    else:
        print(f"[llama] done in {elapsed:.1f}s  exit={proc.returncode}", flush=True)

    raw_out = "\n".join(out_lines)
    if proc.returncode not in (0, None) and not raw_out.strip():
        raise RuntimeError(
            f"{role.title()} completion failed (exit {proc.returncode}) with empty output."
        )
    if proc.returncode not in (0, None):
        # Still try to use any generated text; only hard-fail if nothing usable
        tail = raw_out.strip()[-500:]
        if len(raw_out.strip()) < 8:
            raise RuntimeError(
                f"{role.title()} completion failed (exit {proc.returncode}): {tail}"
            )
        print(f"[llama] warning: exit={proc.returncode} but output present — continuing", flush=True)

    cleaned = _extract_completion_text(raw_out, chat_prompt)
    if len(cleaned) < 40:
        # Fallback: try extracting against the unwrapped prompt too
        cleaned2 = _extract_completion_text(raw_out, prompt)
        if len(cleaned2) > len(cleaned):
            cleaned = cleaned2
    if len(cleaned) < 40:
        print(
            f"[llama] WARNING: completion very short ({len(cleaned)} chars). "
            f"raw stdout was {len(raw_out)} chars.",
            flush=True,
        )
        # Last resort: keep any non-log tail from raw_out
        tail = _extract_completion_text(raw_out, "")
        if len(tail) > len(cleaned):
            cleaned = tail
    return cleaned



def _slugify_folder_name(raw: str, fallback: str = "lyric_materials") -> str:
    text = (raw or "").strip()
    text = text.splitlines()[0] if text else ""
    text = text.strip().strip('"').strip("'").strip(".")
    slug = _SLUG_RE.sub("_", text.lower()).strip("_")
    slug = re.sub(r"_+", "_", slug)
    if not slug:
        slug = fallback
    if slug in _WINDOWS_RESERVED_NAMES:
        slug = f"{slug}_project"
    return slug[:60]


def _ascii_line_slug(line: str, max_len: int = 48) -> str:
    """Filesystem-safe ASCII slug for still filenames (no spaces, no unicode)."""
    t = (line or "").strip()
    for src, dst in (
        ("\u2018", "'"), ("\u2019", "'"), ("\u201c", '"'), ("\u201d", '"'),
        ("\u2013", "-"), ("\u2014", "-"), ("\u2026", "..."),
    ):
        t = t.replace(src, dst)
    t = unicodedata.normalize("NFKD", t).encode("ascii", "ignore").decode("ascii")
    t = re.sub(r"[^A-Za-z0-9]+", "-", t).strip("-").lower()
    if not t:
        t = "line"
    if len(t) > max_len:
        t = t[:max_len].rstrip("-")
    return t or "line"


def _still_filename(line_no: int, line: str, variant: int = 1, frequency: int = 1) -> str:
    """
    001-lyric-slug.png (freq 1) or 001-1-lyric-slug.png … 001-N-… (freq > 1).
    ASCII only so sd-cli and Python agree on the path.
    """
    slug = _ascii_line_slug(line)
    n = int(line_no)
    freq = max(1, int(frequency or 1))
    var = max(1, int(variant or 1))
    if freq <= 1:
        return f"{n:03d}-{slug}.png"
    return f"{n:03d}-{var}-{slug}.png"


def _find_still_for_line(
    out_dir: Path, line_no: int, variant: Optional[int] = None,
) -> Optional[Path]:
    """Find PNG/JPG for this 1-based line number (optional sequence variant)."""
    if not out_dir.is_dir():
        return None
    n = int(line_no)
    hits: List[Path] = []
    if variant is not None:
        v = int(variant)
        pats = (
            f"{n:03d}-{v}-*.png",
            f"{n:03d}-{v}-*.jpg",
            f"{n:03d}-{v}.*",
        )
    else:
        pats = (
            f"{n:03d}-*.png",
            f"{n:03d}-*.jpg",
            f"{n:03d} - *",
            f"{n:03d}.*",
        )
    for pat in pats:
        for m in out_dir.glob(pat):
            if not m.is_file() or m.suffix.lower() not in (".png", ".jpg", ".jpeg", ".webp"):
                continue
            if ".__partial__" in m.name.lower():
                continue
            hits.append(m)
    if not hits:
        return None
    hits.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return hits[0]


def _existing_variants_for_line(out_dir: Path, line_no: int) -> set:
    """Set of variant indices (1-based) already on disk for a lyric line."""
    found: set = set()
    if not out_dir.is_dir():
        return found
    n = int(line_no)
    prefix = f"{n:03d}-"
    for m in out_dir.iterdir():
        if not m.is_file() or m.suffix.lower() not in (".png", ".jpg", ".jpeg", ".webp"):
            continue
        if ".__partial__" in m.name.lower():
            continue
        if not m.name.startswith(prefix):
            continue
        rest = m.stem[len(prefix):]
        vm = re.match(r"^(\d+)-", rest)
        if vm:
            found.add(int(vm.group(1)))
        else:
            found.add(1)
    return found


def _safe_line_filename(line: str, max_len: int = 80) -> str:
    """Sanitize a lyric line for use inside an image filename (ASCII slug)."""
    return _ascii_line_slug(line, max_len=max_len)


def next_project_number(output_root: Path) -> int:
    """Next ### sequence number under output/ based on existing 'NNN - *' folders."""
    highest = 0
    if not output_root.is_dir():
        return 1
    for p in output_root.iterdir():
        if not p.is_dir():
            continue
        m = re.match(r"^(\d+)\s*[-–—]", p.name)
        if m:
            highest = max(highest, int(m.group(1)))
    return highest + 1


def generate_project_folder_name(
    lyrics: str,
    cfg: Dict[str, Any],
    progress_callback: Optional[Callable] = None,
) -> str:
    if progress_callback:
        progress_callback("Naming project…", 0.03, {"phase": "project"})

    model_path = (cfg.get("encoder_model_path") or "").strip()
    excerpt = "\n".join((lyrics or "").splitlines()[:12])[:1200]
    prompt = (
        "You are naming a folder for a lyric image-materials project. "
        "Read the song lyrics below and reply with ONLY a short, "
        "filesystem-safe project name: 2 to 5 words, lowercase, "
        "words separated by underscores, no punctuation, no quotes, "
        "no explanation.\n\n"
        f"Lyrics excerpt:\n{excerpt}\n\nProject name:"
    )
    try:
        raw = _run_llama_completion(
            prompt, cfg, role="encoder", model_path=model_path,
            n_predict=_NAME_PREDICT, ctx_size=_NAME_CTX,
            temperature=0.6, timeout=_TIMEOUT_NAME,
        )
    except Exception as e:
        print(f"[project] naming failed, using fallback: {e}", flush=True)
        raw = ""
    return _slugify_folder_name(raw)


def ensure_project_dir(
    folder_label: str,
    sequential: bool = True,
    resume_path: Optional[str] = None,
) -> Path:
    """
    Create output/### - label/ (or label alone if sequential=False).
    If resume_path points at an existing project folder, reuse it.
    Records path in APP_STATE.
    """
    if resume_path:
        existing = Path(resume_path)
        if existing.is_dir():
            configure.APP_STATE["current_project_folder"] = str(existing)
            configure.APP_STATE["active_session_id"] = existing.name
            return existing

    base = configure.get_output_dir()
    base.mkdir(parents=True, exist_ok=True)
    label = folder_label or "lyric_materials"
    if sequential:
        num = next_project_number(base)
        name = f"{num:03d} - {label}"
    else:
        name = label
    candidate = base / name
    n = 2
    while candidate.exists():
        candidate = base / f"{name}_{n}"
        n += 1
    candidate.mkdir(parents=True, exist_ok=True)
    configure.APP_STATE["current_project_folder"] = str(candidate)
    configure.APP_STATE["active_session_id"] = candidate.name
    return candidate


def _load_analysis_from_disk(project_dir: Path) -> Optional[Dict[str, Any]]:
    """Reconstruct analysis dict from analysis.txt + character_map.txt if present."""
    analysis_path = project_dir / "analysis.txt"
    if not analysis_path.exists():
        return None
    try:
        text = analysis_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    overall = ""
    character_guidance = ""
    character_presence: Dict[str, str] = {}
    sections: Dict[str, str] = {}
    mode = None
    buf: List[str] = []

    def _flush():
        nonlocal overall, character_guidance
        body = "\n".join(buf).strip()
        if mode == "OVERALL":
            overall = body
        elif mode == "CHARACTER":
            character_guidance = body
        elif mode == "CHARACTER_MAP":
            for line in body.splitlines():
                if ":" in line:
                    k, v = line.split(":", 1)
                    character_presence[k.strip()] = v.strip().lower()
        elif mode == "SECTIONS":
            for line in body.splitlines():
                if ":" in line:
                    k, v = line.split(":", 1)
                    sections[k.strip()] = v.strip()

    for raw in text.splitlines():
        head = raw.strip().upper()
        if head in ("OVERALL", "CHARACTER", "CHARACTER_MAP", "SECTIONS"):
            _flush()
            buf = []
            mode = head
        else:
            buf.append(raw)
    _flush()

    cmap = project_dir / "character_map.txt"
    if cmap.exists():
        try:
            for raw in cmap.read_text(encoding="utf-8", errors="replace").splitlines():
                if ":" in raw and not raw.strip().lower().startswith("section"):
                    k, v = raw.split(":", 1)
                    k, v = k.strip(), v.strip().lower()
                    if k and v in ("none", "silhouette", "partial", "full"):
                        character_presence[k] = v
        except OSError:
            pass

    return {
        "overall": overall,
        "sections": sections,
        "character_guidance": character_guidance,
        "character_presence": character_presence,
    }


def _load_prompts_from_disk(
    project_dir: Path, expected_lines: int,
) -> Tuple[Optional[List[str]], Optional[List[str]]]:
    """
    Parse prompts.txt written by the pipeline.
    Returns (prompts, presence_per_line) or (None, None) if incomplete/missing.
    """
    path = project_dir / "prompts.txt"
    if not path.exists():
        return None, None
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None, None

    prompts: List[str] = []
    presence: List[str] = []
    current_prompt: List[str] = []
    current_pres = "none"
    in_prompt = False

    def _commit():
        nonlocal current_prompt, current_pres
        if in_prompt or current_prompt:
            prompts.append("\n".join(current_prompt).strip())
            presence.append(current_pres)
        current_prompt = []
        current_pres = "none"

    for raw in text.splitlines():
        if raw.startswith("=== line "):
            if prompts or current_prompt or in_prompt:
                _commit()
            in_prompt = False
            continue
        if raw.lower().startswith("character:"):
            current_pres = raw.split(":", 1)[1].strip().lower() or "none"
            continue
        if raw.strip() == "--- prompt ---":
            in_prompt = True
            current_prompt = []
            continue
        if in_prompt:
            current_prompt.append(raw)
    if current_prompt or in_prompt:
        _commit()

    if expected_lines > 0 and len(prompts) < expected_lines:
        return None, None
    if not prompts:
        return None, None
    return prompts, presence


def _existing_image_indices(project_dir: Path) -> set:
    """Set of 1-based line numbers that already have a still."""
    found = set()
    if not project_dir.is_dir():
        return found
    for p in project_dir.iterdir():
        if not p.is_file():
            continue
        if p.suffix.lower() not in (".png", ".jpg", ".jpeg", ".webp"):
            continue
        if ".__partial__" in p.name.lower():
            continue
        m = re.match(r"^(\d+)(?:\s*[-–—]|[-.])", p.name)
        if m:
            found.add(int(m.group(1)))
        else:
            m2 = re.match(r"^(\d+)\.", p.name)
            if m2:
                found.add(int(m2.group(1)))
    return found



def _snapshot_session(
    project_dir: Path,
    *,
    lyrics: str = "",
    song_name: str = "",
    cfg: Optional[Dict[str, Any]] = None,
    reference_image: str = "",
    phase: str = "",
    line_count: int = 0,
    images_done: Optional[int] = None,
) -> None:
    """
    Write a complete session.json so the UI can resume after stop/restart.
    Always stores style, steps, cfg, reference path, song name, lyrics.
    """
    cfg = cfg or {}
    if images_done is None:
        images_done = len(_existing_image_indices(project_dir))
    updates: Dict[str, Any] = {
        "song_name": (song_name or cfg.get("project_label") or project_dir.name).strip()
        or project_dir.name,
        "lyrics": lyrics if lyrics is not None else "",
        "style": cfg.get("style") or configure.STYLE_LIGHT,
        "steps": int(cfg.get("imagegen_steps") or configure.DEFAULT_STEPS),
        "cfg_scale": float(cfg.get("imagegen_cfg_scale") or configure.DEFAULT_CFG),
        "reference_image": reference_image or "",
        "negative_prompt": (cfg.get("negative_prompt") if cfg.get("negative_prompt") is not None
                            else configure.DEFAULT_NEGATIVE_PROMPT) or "",
        "line_count": int(line_count or 0),
        "images_done": int(images_done),
    }
    if phase:
        updates["phase"] = phase
    configure.save_session_meta(project_dir, updates)
    # Side file for easy inspection / external tools
    try:
        neg = updates.get("negative_prompt") or ""
        payload = (neg + "\n") if neg else ""
        (project_dir / "negative_prompt.txt").write_text(payload, encoding="utf-8")
    except OSError:
        pass



# ---------------------------------------------------------------------------
# Song + section analysis (overall idea → section notes)
# ---------------------------------------------------------------------------



def _parse_labeled_section_notes(sec_blob: str, labels: List[str]) -> Dict[str, str]:
    """
    Line-based parse of SECTIONS block. More reliable than a single regex when
    labels share prefixes (Chorus 1 / Chorus 4) or blank lines separate paragraphs.
    """
    notes: Dict[str, str] = {lab: "" for lab in labels}
    if not sec_blob or not labels:
        return notes
    # Longest labels first so "Chorus 11" wins over "Chorus 1" if both exist
    labels_sorted = sorted(labels, key=lambda s: len(s), reverse=True)
    current: Optional[str] = None
    buf: List[str] = []

    def _flush():
        nonlocal buf, current
        if current is not None:
            body = " ".join(x for x in buf if x).strip()
            body = re.sub(r"\[end of text\]", "", body, flags=re.I).strip()
            notes[current] = body
        buf = []

    for raw_line in sec_blob.splitlines():
        line = raw_line.strip()
        if not line:
            if current is not None:
                buf.append("")  # keep paragraph breaks as spaces later
            continue
        matched: Optional[str] = None
        rest = ""
        for lab in labels_sorted:
            # "Chorus 4:" or "Chorus 4 -" at line start
            m = re.match(
                rf"^{re.escape(lab)}\s*[:.\-–—]\s*(.*)$",
                line,
                re.I,
            )
            if m:
                matched = lab
                rest = (m.group(1) or "").strip()
                break
            m2 = re.match(rf"^{re.escape(lab)}\s*$", line, re.I)
            if m2:
                matched = lab
                rest = ""
                break
        if matched is not None:
            _flush()
            current = matched
            if rest:
                buf.append(rest)
        elif current is not None:
            buf.append(line)
    _flush()
    return notes


def _looks_like_keyword_spam(text: str) -> bool:
    """Detect slash-separated adjective loops / non-sentence model dumps."""
    t = (text or "").strip()
    if not t:
        return False
    if t.count("/") >= 8:
        return True
    # Same short token repeated many times
    words = re.findall(r"[A-Za-z][A-Za-z\-']{2,}", t.lower())
    if len(words) >= 20:
        from collections import Counter
        common = Counter(words).most_common(1)
        if common and common[0][1] >= max(8, len(words) // 5):
            return True
    # Almost no sentence punctuation in a long blob
    if len(t) > 200 and t.count(".") + t.count("!") + t.count("?") < 1:
        return True
    return False


def analyze_song_and_sections(
    lyrics: str,
    parsed: List[Dict[str, Any]],
    cfg: Dict[str, Any],
    has_character_ref: bool = False,
    progress_callback: Optional[Callable] = None,
) -> Dict[str, Any]:
    """
    One LLM call: overall narrative + per-section notes + character presence map.

    Returns:
      overall: str
      sections: {label: notes}
      character_guidance: str
      character_presence: {section_label: "none"|"silhouette"|"partial"|"full"}
        — used programmatically to include/exclude the reference image per line
    """
    if progress_callback:
        progress_callback("Loading model & analysing song/sections…", 0.06, {"phase": "analysis"})
    print("[analysis] === starting song + section analysis ===", flush=True)
    print("[analysis] (watch for [llama] MODEL LOADED, then generation)", flush=True)

    model_path, role = _resolve_prompt_model(cfg)
    empty = {
        "overall": "",
        "sections": {},
        "character_guidance": "",
        "character_presence": {},
    }
    if not model_path:
        return empty

    sections = sections_with_lines(parsed)
    section_labels = [s["label"] for s in sections]
    section_summary = "\n".join(
        f"- {s['label']}: {len(s['lines'])} lines — "
        + "; ".join(s["lines"][:3])
        + ("…" if len(s["lines"]) > 3 else "")
        for s in sections
    ) or "(no section headers; treat as continuous Body)"
    labels_list = ", ".join(section_labels) if section_labels else "Body"

    if has_character_ref:
        char_instr = (
            "CHARACTER: 1-2 sentences on the central character's face, body type, age range, "
            "expression, and attitude/energy only.\n"
            "Do NOT mention clothing, outfit, wardrobe, gear, uniform, or hair style — "
            "those are chosen later in the GUI and must not appear here.\n"
            "A reference photo will supply likeness on some stills.\n"
            "CHARACTER_MAP: one line per section as  SectionName: none|silhouette|partial|full\n"
            "  none=no character, silhouette=outline only, partial=partly visible, full=clearly present.\n"
            "  Vary presence across the song when the lyrics support it.\n"
        )
    else:
        char_instr = (
            "CHARACTER: 1-2 sentences inventing a central figure from the lyrics "
            "(face, body type, attitude only), or 'none specified'.\n"
            "Do NOT mention clothing, outfit, wardrobe, gear, or hair style.\n"
            "CHARACTER_MAP: one line per section as  SectionName: none|silhouette|partial|full\n"
        )

    # Rigid, example-led prompt — no trailing "Notes:" (models often free-associate after it)
    example_labels = section_labels[:2] if section_labels else ["Intro", "Chorus 1"]
    ex0 = example_labels[0]
    ex1 = example_labels[1] if len(example_labels) > 1 else "Chorus 1"
    prompt = (
        "You are a music-video art director. Write structured production notes.\n"
        "Reply with ONLY the four blocks below. Use real sentences, not keyword lists.\n"
        "Do NOT output slash-separated adjectives. Do NOT repeat the same phrase.\n"
        "Never describe the character's clothing or hair style in any block — "
        "wardrobe and hair are configured separately in the UI.\n\n"
        "OVERALL:\n"
        "Write 3-5 complete sentences: narrative arc, mood, setting, colour palette, visual tone.\n\n"
        f"{char_instr}\n"
        "SECTIONS:\n"
        f"For EACH of these section names write one non-empty paragraph (2-3 sentences): {labels_list}.\n"
        "Label each paragraph with the exact section name followed by a colon.\n"
        "Focus on camera, setting, action, and mood — not wardrobe.\n\n"
        "Example format (replace with content about THIS song):\n"
        "OVERALL:\n"
        "A lone figure moves through cold neon streets at night. The palette is steel blue and amber. "
        "The tone is defiant and intimate, not chaotic.\n\n"
        "CHARACTER:\n"
        "A sharp-eyed nonconformist with a calm, deliberate presence; mid-adult, lean build, unreadable expression.\n\n"
        "CHARACTER_MAP:\n"
        f"{ex0}: silhouette\n"
        f"{ex1}: full\n\n"
        "SECTIONS:\n"
        f"{ex0}: Wide establishing shots of an anonymous crowd. The camera glides; the figure is barely present.\n"
        f"{ex1}: Closer framing on the figure at the edge of a queue. Gestures are subtle, lighting harder.\n\n"
        f"Sections detected:\n{section_summary}\n\n"
        f"Full lyrics:\n{lyrics[:5000]}\n\n"
        "Start your reply with the word OVERALL: and follow the structure exactly."
    )

    try:
        raw = _run_llama_completion(
            prompt, cfg, role=role, model_path=model_path,
            n_predict=_ANALYSIS_PREDICT, ctx_size=_ANALYSIS_CTX,
            temperature=0.35, timeout=_TIMEOUT_ANALYSIS,
        )
    except Exception as e:
        print(f"[analysis] failed: {e}", flush=True)
        raw = ""

    if raw:
        print(f"[analysis] raw reply length = {len(raw)} chars "
              f"(n_predict budget {_ANALYSIS_PREDICT})", flush=True)

    # Retry once with a shorter, stricter prompt if the first pass was empty/stub
    if len((raw or "").strip()) < 80 or not re.search(r"(?i)\bOVERALL\b", raw or ""):
        print("[analysis] first pass empty/stub — retrying with compact prompt…", flush=True)
        retry_prompt = (
            "Write production notes for this song. Start with OVERALL:\n\n"
            f"Sections: {labels_list}\n\n"
            f"Lyrics:\n{lyrics[:3500]}\n\n"
            "Required format:\n"
            "OVERALL:\n"
            "<3-5 sentences on narrative, mood, setting, colour>\n\n"
            "CHARACTER:\n"
            "<1-2 sentences on face, body type, attitude only — "
            "NO clothing, outfit, wardrobe, or hair style>\n\n"
            "CHARACTER_MAP:\n"
            + "\n".join(f"{lab}: partial" for lab in (section_labels or ["Body"]))
            + "\n\nSECTIONS:\n"
            + "\n".join(
                f"{lab}: <2 sentences of visual notes for this section>"
                for lab in (section_labels or ["Body"])
            )
            + "\n\nFill every section with real content about these lyrics. "
            "Do not leave blanks. Never describe clothing or hair."
        )
        try:
            raw2 = _run_llama_completion(
                retry_prompt, cfg, role=role, model_path=model_path,
                n_predict=_ANALYSIS_PREDICT, ctx_size=_ANALYSIS_CTX,
                temperature=0.5, timeout=_TIMEOUT_ANALYSIS,
            )
            if len((raw2 or "").strip()) > len((raw or "").strip()):
                raw = raw2
                print(f"[analysis] retry reply length = {len(raw)} chars", flush=True)
        except Exception as e:
            print(f"[analysis] retry failed: {e}", flush=True)

    overall = ""
    character = ""
    section_notes: Dict[str, str] = {}
    presence: Dict[str, str] = {}

    if raw:
        # Prefer structured split; fall back to whole text only for overall if needed
        parts = re.split(r"(?i)\b(OVERALL|CHARACTER_MAP|CHARACTER|SECTIONS)\s*:", raw)
        current = None
        buf: List[str] = []
        blocks: Dict[str, str] = {}
        for part in parts:
            key = part.strip().upper().replace(" ", "_")
            if key in ("OVERALL", "CHARACTER", "CHARACTER_MAP", "SECTIONS"):
                if current and buf:
                    blocks[current] = "\n".join(buf).strip()
                current = key
                buf = []
            else:
                buf.append(part)
        if current and buf:
            blocks[current] = "\n".join(buf).strip()

        overall = blocks.get("OVERALL", "").strip()
        character = blocks.get("CHARACTER", "").strip()
        sec_blob = blocks.get("SECTIONS", "")
        map_blob = blocks.get("CHARACTER_MAP", "")

        if not overall:
            # First paragraph before CHARACTER_MAP if model omitted heading
            overall = (raw[:500] or "").strip()

        # Reject slash-spam / looping keyword lists
        if _looks_like_keyword_spam(overall):
            print("[analysis] OVERALL looks like keyword spam — discarding", flush=True)
            overall = ""
        if _looks_like_keyword_spam(character):
            character = ""

        parsed_notes = _parse_labeled_section_notes(
            sec_blob, [s["label"] for s in sections],
        )
        for s in sections:
            lab = s["label"]
            note = (parsed_notes.get(lab) or "").strip()
            if note and not _looks_like_keyword_spam(note):
                section_notes[lab] = note
            else:
                section_notes[lab] = ""
                if not note:
                    print(f"[analysis] missing section notes for: {lab}", flush=True)

        for s in sections:
            lab = s["label"]
            found = "partial"
            for line in map_blob.splitlines():
                if re.match(
                    rf"^{re.escape(lab)}\s*[:.\-–—]",
                    line.strip(),
                    re.I,
                ):
                    for kw in ("none", "silhouette", "partial", "full"):
                        if re.search(rf"\b{kw}\b", line, re.I):
                            found = kw
                            break
                    break
            presence[lab] = found

        if "SECTIONS" not in (raw or "").upper():
            print("[analysis] WARNING: no SECTIONS heading in reply", flush=True)
        filled = sum(1 for v in section_notes.values() if (v or "").strip())
        print(f"[analysis] section notes filled: {filled}/{len(section_labels)}", flush=True)

    # Defaults for any section missing a map entry
    for s in sections:
        lab = s["label"]
        section_notes.setdefault(lab, "")
        if lab not in presence:
            low = lab.lower()
            if any(k in low for k in ("intro", "outro", "fade")):
                presence[lab] = "silhouette" if has_character_ref else "none"
            else:
                presence[lab] = "full" if has_character_ref else "partial"

    print("[analysis] character presence map:", flush=True)
    for lab, val in presence.items():
        print(f"[analysis]   {lab}: {val}", flush=True)

    # Do NOT invent a default OVERALL when the model failed — empty signals unusable
    return {
        "overall": overall,
        "sections": section_notes,
        "character_guidance": character,
        "character_presence": presence,
    }



def generate_visual_prompts(
    lines: List[str],
    parsed: List[Dict[str, Any]],
    analysis: Dict[str, Any],
    cfg: Dict[str, Any],
    has_character_ref: bool = False,
    progress_callback: Optional[Callable] = None,
) -> Tuple[List[str], List[str]]:
    """
    One visual-description prompt per lyric line.
    Returns (prompts, presence_per_line) where presence_per_line[i] is
    none|silhouette|partial|full for programmatic ref-image include/exclude.
    """
    model_path, role = _resolve_prompt_model(cfg)
    if not model_path:
        raise RuntimeError("No Encoder/Thinking model configured for visual prompts.")

    style = cfg.get("style") or configure.STYLE_LIGHT
    template = (
        cfg.get("prompt_template")
        or configure.prompt_template_for_style(style)
        or configure.STYLE_PROMPT_TEMPLATES[configure.STYLE_LIGHT]
    )

    line_section: Dict[int, str] = {}
    for e in parsed:
        if e["type"] == "line":
            line_section[e["index"]] = e.get("section") or "Body"

    overall = (analysis or {}).get("overall") or ""
    section_notes = (analysis or {}).get("sections") or {}
    char_guide = (analysis or {}).get("character_guidance") or ""
    presence_map = (analysis or {}).get("character_presence") or {}

    prompts: List[str] = []
    presence_per_line: List[str] = []
    total = len(lines)
    print(f"[prompts] === Phase 1 prompts: {total} lyric line(s) via {role} ===", flush=True)
    for i, line in enumerate(lines):
        if is_cancel_requested():
            break
        line_no = i + 1
        sec = line_section.get(i, "Body")
        pres = (presence_map.get(sec) or ("full" if has_character_ref else "none")).lower()
        if pres not in ("none", "silhouette", "partial", "full"):
            pres = "full" if has_character_ref else "none"
        presence_per_line.append(pres)

        print(f"[prompts] ────────────────────────────────────────", flush=True)
        print(f"[prompts] Lyrics Line {line_no}/{total}: {line}", flush=True)
        print(f"[prompts] section={sec}  character={pres}"
              f"{' (ref image will be attached)' if pres != 'none' and has_character_ref else ' (no ref image)'}",
              flush=True)
        if progress_callback:
            progress_callback(
                f"Prompt {line_no}/{total}: {line[:60]}",
                0.10 + 0.25 * (i / max(total, 1)),
                {"phase": "prompts", "line": line_no, "total": total},
            )
        sec_note = section_notes.get(sec, "")
        style_part = template.replace("{line}", line) if "{line}" in template else f"{template}\n{line}"

        if has_character_ref and pres != "none":
            how = {
                "silhouette": "show the reference character only as a silhouette or shadow outline",
                "partial": "show the reference character partially (cropped, turned, or distant)",
                "full": "show the reference character clearly, matching the reference likeness",
            }.get(pres, "show the reference character")
            char_clause = (
                f"CHARACTER: {how}. "
                f"Character notes: {char_guide or 'match the reference subject.'} "
            )
        elif has_character_ref and pres == "none":
            char_clause = (
                "CHARACTER: Do NOT depict the reference person. "
                "No recognisable central character from the reference photo — "
                "environment, abstract, or anonymous figures only. "
            )
        else:
            char_clause = (
                "No reference photo is provided — invent figures purely from the lyrics "
                "and notes when needed. "
            )

        instruction = (
            f"{style_part}\n\n"
            f"Song context: {overall[:350]}\n"
            f"Section ({sec}): {sec_note[:250]}\n"
            f"{char_clause}"
            "Reply with ONLY the visual description for this single still, "
            "one paragraph, no preamble, no quotes, no section labels."
        )
        try:
            text = _run_llama_completion(
                instruction, cfg, role=role, model_path=model_path,
                n_predict=_PROMPT_PREDICT, ctx_size=_PROMPT_CTX,
                temperature=0.8, timeout=_TIMEOUT_PROMPT,
            )
            preview = (text or "").strip().replace("\n", " ")[:120]
            print(f"[prompts] Lyrics Line {line_no} OK — prompt preview: {preview}…", flush=True)
        except Exception as e:
            print(f"[prompts] Lyrics Line {line_no} FAILED — fallback to raw lyric: {e}", flush=True)
            text = ""
        prompts.append(text.strip() or line)

    print(f"[prompts] === finished {len(prompts)}/{total} prompts ===", flush=True)
    # Pad presence if cancelled mid-loop
    while len(presence_per_line) < len(prompts):
        presence_per_line.append("none")
    return prompts, presence_per_line



def _run_sd_cli_once(cmd: List[str], exe: Path, timeout: float = 600.0) -> Tuple[str, int]:
    """
    One-shot sd-cli. Returns (captured_tail, returncode).

    Serialized via _SD_CLI_LOCK so two Gradio handlers cannot load Flux twice.
    Keeps only a tail of stdout so verbose Flux logs cannot grow unbounded
    across a 40-line batch. Always closes pipes and collects after the
    process exits so Vulkan/CPU buffers from the child are released.
    """
    print("[sd-cli] waiting for exclusive image-gen slot…", flush=True)
    with _SD_CLI_LOCK:
        print("[sd-cli] acquired image-gen slot", flush=True)
        return _run_sd_cli_once_unlocked(cmd, exe, timeout=timeout)


def _run_sd_cli_once_unlocked(cmd: List[str], exe: Path, timeout: float = 600.0) -> Tuple[str, int]:
    if "-v" not in cmd and "--verbose" not in cmd:
        cmd = list(cmd) + ["-v"]
    print("[sd-cli] " + " ".join(str(c) for c in cmd), flush=True)
    print(f"[sd-cli] starting (timeout {timeout:.0f}s) — streaming output…", flush=True)
    t0 = time.time()
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding="utf-8", errors="replace",
        cwd=str(exe.parent),
        bufsize=1,
    )
    register_process(proc)
    tail: List[str] = []
    saved_ok = False
    model_loaded = False
    rc = -1
    flags: List[str] = []
    try:
        deadline = time.time() + timeout
        assert proc.stdout is not None
        while True:
            if is_cancel_requested():
                proc.kill()
                flags.append("[[CANCELLED]] sd-cli was stopped by a cancel request.")
                break
            if time.time() > deadline:
                proc.kill()
                print(f"[sd-cli] TIMEOUT after {time.time() - t0:.1f}s", flush=True)
                flags.append(
                    f"[[TIMEOUT]] sd-cli was killed after {time.time() - t0:.0f}s "
                    f"(limit {timeout:.0f}s) before saving an image."
                )
                break
            line = proc.stdout.readline()
            if line == "" and proc.poll() is not None:
                break
            if not line:
                time.sleep(0.02)
                continue
            text_line = line.rstrip("\n\r")
            tail.append(text_line)
            if len(tail) > 250:
                tail = tail[-200:]
            print(f"[sd-cli] {text_line}", flush=True)
            low = text_line.lower()
            if "images saved" in low or "save result image" in low and "success" in low:
                saved_ok = True
            if not model_loaded and any(
                k in low for k in ("loaded", "load model", "diffusion", "vae", "tensor")
            ):
                if "load" in low or "loaded" in low:
                    model_loaded = True
                    print(f"[sd-cli] *** MODEL ACTIVITY ({time.time() - t0:.1f}s) ***", flush=True)
        try:
            proc.wait(timeout=8)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass
            try:
                proc.wait(timeout=3)
            except Exception:
                pass
        rc = proc.returncode if proc.returncode is not None else -1
    except Exception as e:
        print(f"[sd-cli] stream error: {e}", flush=True)
        try:
            proc.kill()
        except Exception:
            pass
        rc = proc.returncode if proc.returncode is not None else -1
    finally:
        try:
            if proc.stdout:
                proc.stdout.close()
        except Exception:
            pass
        unregister_process(proc)
        # Drop child handles so the OS can reclaim Vulkan allocations
        try:
            del proc
        except Exception:
            pass
        gc.collect()
        time.sleep(0.25)

    elapsed = time.time() - t0
    print(f"[sd-cli] finished in {elapsed:.1f}s  exit={rc}", flush=True)
    # Marker so callers can treat "saved" even if the exact Path lookup fails
    if saved_ok and "[[SAVED_OK]]" not in "\n".join(tail[-5:]):
        tail.append("[[SAVED_OK]]")
    tail.extend(flags)
    return "\n".join(tail), int(rc if rc is not None else -1)



def _norm_path_key(p: str) -> str:
    return (p or "").replace("\\", "/").lower()


def _path_has_flux2_marker(p: str) -> bool:
    n = _norm_path_key(p)
    return any(k in n for k in ("flux2", "flux.2", "flux-2", "flux_2"))


def _path_has_flux1_marker(p: str) -> bool:
    n = _norm_path_key(p)
    if _path_has_flux2_marker(p):
        return False
    return any(k in n for k in (
        "flux1", "flux.1", "flux-1", "flux_1", "schnell", "flux-schnell",
    ))


def _vae_is_known_flux1(vae_path: str) -> bool:
    """Hard reject: path is clearly Flux.1 / schnell ae (16-ch), not Flux.2."""
    if not vae_path:
        return False
    if _path_has_flux2_marker(vae_path):
        return False
    if _path_has_flux1_marker(vae_path):
        return True
    name = Path(vae_path).name.lower()
    # Bare ae.safetensors with no flux2 marker in the full path → Flux.1-style
    if name in ("ae.safetensors", "ae.sft", "flux_ae.safetensors"):
        return True
    return False


def _file_size_mb(path: str) -> float:
    try:
        return Path(path).stat().st_size / (1024 * 1024)
    except OSError:
        return -1.0


def _vae_is_dit_weights(vae_path: str) -> bool:
    """
    True when the configured 'VAE' is almost certainly the DiT / diffusion
    weights, not an autoencoder.

    BFL ships the Flux.2 VAE as diffusion_pytorch_model.safetensors (~300–400 MB)
    from the vae/ folder — that name is valid and must be allowed (same rule as
    the companion glamour program: empty/unknown VAE name → allow through).

    The DiT next to flux-2-klein-*.gguf is also often named
    diffusion_pytorch_model*.safetensors but is multi-GB. Size is the reliable
    discriminator when the filename alone cannot tell.
    """
    if not vae_path or not Path(vae_path).is_file():
        return False
    name = Path(vae_path).name.lower()
    # Explicit VAE names are never DiT
    if any(k in name for k in ("flux2_ae", "flux2-ae", "flux2_vae", "flux2-vae",
                               "flux.2_ae", "autoencoder")):
        return False
    if name.endswith("_vae.safetensors") or name.endswith("-vae.safetensors"):
        return False
    # GGUF is never a VAE for Flux.2 (user sometimes swaps slots)
    if name.endswith(".gguf"):
        return True
    size = _file_size_mb(vae_path)
    # Flux.2 AE is typically ~320–450 MB; DiT 4B Q8 / bf16 is multi-GB
    if size > 900:
        return True
    if name.startswith("diffusion_pytorch_model") or name.startswith("diffusion_model"):
        # Small/medium = likely BFL VAE export; huge = DiT
        if size < 0:
            return False  # unknown size → allow (other program: cannot tell)
        return size > 900
    return False


def _flux2_vae_candidates_near(diffusion_path: str) -> List[Path]:
    """Search beside the diffusion model for a plausible Flux.2 VAE file."""
    found: List[Path] = []
    try:
        root = Path(diffusion_path).expanduser().resolve().parent
    except OSError:
        return found
    preferred = (
        "flux2_ae.safetensors",
        "flux2-ae.safetensors",
        "flux2_vae.safetensors",
        "flux2-vae.safetensors",
        "flux.2_ae.safetensors",
        "ae.safetensors",  # only accepted when under a flux2-marked folder
    )
    search_dirs = [
        root, root / "vae", root / "VAE",
        root.parent, root.parent / "vae", root.parent / "VAE",
    ]
    seen = set()
    for d in search_dirs:
        if not d.is_dir():
            continue
        for name in preferred:
            if name == "ae.safetensors" and not _path_has_flux2_marker(str(d)):
                continue
            p = d / name
            try:
                key = str(p.resolve())
            except OSError:
                key = str(p)
            if key in seen or not p.is_file():
                continue
            if _vae_is_dit_weights(str(p)) or _vae_is_known_flux1(str(p)):
                continue
            seen.add(key)
            found.append(p)
        # BFL generic VAE name — only if size looks like an autoencoder
        for p in d.glob("diffusion_pytorch_model*.safetensors"):
            try:
                key = str(p.resolve())
            except OSError:
                key = str(p)
            if key in seen:
                continue
            if _vae_is_dit_weights(str(p)):
                continue
            if _file_size_mb(str(p)) > 900:
                continue
            seen.add(key)
            found.append(p)
    return found


def _resolve_flux2_vae(cfg: Dict[str, Any], diffusion_path: str) -> str:
    """
    Pick a Flux.2-compatible VAE path.

    Matches the companion program's rule: a filename that identifies nothing
    (e.g. BFL's diffusion_pytorch_model.safetensors VAE) is allowed when the
    user points at it. Only POSITIVE mismatches are rejected (Flux.1 ae,
    multi-GB DiT weights, GGUF used as VAE).
    """
    configured = (cfg.get("vae_model_path") or "").strip()
    if configured and Path(configured).exists():
        if _vae_is_known_flux1(configured):
            print(
                f"[images] configured VAE is Flux.1/schnell (16-ch) — not valid for Flux.2:\n"
                f"  {configured}",
                flush=True,
            )
        elif _vae_is_dit_weights(configured):
            mb = _file_size_mb(configured)
            print(
                f"[images] configured VAE looks like DiT/diffusion weights "
                f"({mb:.0f} MB), not an autoencoder — ignoring:\n"
                f"  {configured}",
                flush=True,
            )
        else:
            # Explicit flux2_ae*, small diffusion_pytorch_model*, or unknown name
            return configured

    for cand in _flux2_vae_candidates_near(diffusion_path):
        print(f"[images] auto-selected Flux.2 VAE: {cand}", flush=True)
        return str(cand)

    return ""


def _flux2_vae_help(diffusion_path: str = "", tried: str = "") -> str:
    nearby = _flux2_vae_candidates_near(diffusion_path) if diffusion_path else []
    nearby_txt = (
        "\nFound nearby candidates:\n  " + "\n  ".join(str(p) for p in nearby)
        if nearby else
        "\nNo Flux.2 VAE found next to the diffusion model."
    )
    return (
        "Flux.2-klein needs its own autoencoder (32 latent channels).\n\n"
        "Valid VAE files:\n"
        "  • flux2_ae.safetensors  (preferred name)\n"
        "  • diffusion_pytorch_model.safetensors from the BFL FLUX.2 **vae/** folder\n"
        "    (~300–450 MB — NOT the multi-GB DiT file next to the GGUF)\n\n"
        "Download: https://huggingface.co/black-forest-labs/FLUX.2-dev\n"
        "  (vae/diffusion_pytorch_model.safetensors or flux2_ae.safetensors)\n\n"
        "Do NOT use:\n"
        "  • Flux.1 / schnell ae.safetensors (16 channels)\n"
        "  • The DiT GGUF or multi-GB diffusion_pytorch_model as --vae\n"
        "  • Swapping slots: diffusion = GGUF/DiT, VAE = autoencoder only\n\n"
        f"Configured VAE: {tried or '(none)'}\n"
        f"Diffusion model: {diffusion_path or '(none)'}"
        f"{nearby_txt}\n\n"
        "Configuration → ImageGen → VAE → select the Flux.2 autoencoder → Save → resume."
    )


_SD_DUMP_LINE_RE = re.compile(r"^\s*[\w\-]+:\s.*,\s*$")
_SD_ERROR_HINTS = (
    "error", "fail", "assert", "abort", "exception", "out of memory", "oom",
    "cannot", "can't", "unable", "invalid", "not found", "device lost",
    "vk::", "vulkan", "ggml_", "[[timeout]]", "[[cancelled]]", "access violation",
)


def _format_sd_failure(out: str, vae_path: str = "", diffusion_path: str = "", rc: Optional[int] = None) -> str:
    """Turn sd-cli output into an actionable message (cause first, params dump removed)."""
    text = out or ""
    low = text.lower()
    if (
        "wrong shape" in low
        or "model metadata validation failed" in low
        or "not in model metadata" in low
        or "new_sd_ctx_t failed" in low
    ):
        return _flux2_vae_help(diffusion_path, vae_path)
    if not text.strip():
        return f"(no output captured){f'  exit code {rc}' if rc is not None else ''}"

    lines = text.splitlines()
    flags = [l for l in lines if l.startswith("[[TIMEOUT]]") or l.startswith("[[CANCELLED]]")]
    # Drop the long parameter dump (``key: value,`` lines) and blanks
    useful = [l for l in lines if l.strip() and not _SD_DUMP_LINE_RE.match(l)
              and not l.startswith("[[")]
    errs = []
    for l in useful:
        ll = l.lower()
        if any(h in ll for h in _SD_ERROR_HINTS):
            errs.append(l.strip())
    # de-duplicate, keep order, cap
    seen, key_errs = set(), []
    for l in errs:
        if l not in seen:
            seen.add(l)
            key_errs.append(l[:300])
    parts: List[str] = []
    if rc is not None:
        parts.append(f"sd-cli exit code: {rc}")
    parts.extend(flags)
    if key_errs:
        parts.append("Key messages:\n  " + "\n  ".join(key_errs[-10:]))
    parts.append("Last output:\n  " + "\n  ".join(l.rstrip()[:300] for l in useful[-18:]))
    return "\n".join(parts)


def _write_sd_failure_log(out_dir: Path, line_no: int, attempt: str, cmd: List[str], out: str, rc: int) -> Optional[Path]:
    """Save the full sd-cli output for a failed line next to the stills."""
    try:
        path = Path(out_dir) / f"sd_failure_line_{int(line_no):03d}.log"
        with open(path, "a", encoding="utf-8") as f:
            f.write(f"===== {time.strftime('%Y-%m-%d %H:%M:%S')}  attempt: {attempt}  exit={rc} =====\n")
            f.write("COMMAND: " + " ".join(str(c) for c in cmd) + "\n\n")
            f.write(out or "(no output)")
            f.write("\n\n")
        return path
    except OSError:
        return None


_PROMPT_LEAK_MARKERS = (
    "reply with only the visual description",
    "character: do not show the reference",
    "character: show the reference character",
    "no preamble, no quotes",
    "no section labels",
    "--- prompt ---",
    "character notes:",
)


def _sanitize_visual_prompt(prompt: str, lyric: str, style_hint: str = "") -> str:
    """
    Drop llama instruction-echo leaks (the model sometimes dumps remaining
    section templates instead of a still description). Truncate huge prompts.
    """
    p = (prompt or "").strip()
    low = p.lower()
    leaked = (
        not p
        or any(m in low for m in _PROMPT_LEAK_MARKERS)
        or low.lstrip().startswith("section (")
    )
    if leaked:
        print("[images] prompt looks like instruction leak — using lyric fallback", flush=True)
        bits = []
        if style_hint:
            bits.append(style_hint.strip().split("\n")[0][:240])
        bits.append(f"Music-video still of: {lyric.strip()}")
        bits.append("Cinematic composition, coherent lighting, no on-image text.")
        p = " ".join(bits)
    # Soft cap — prefer ending on a sentence boundary so Flux is not fed a stump
    if len(p) > 1400:
        cut = p[:1400]
        for sep in (". ", "! ", "? ", "; "):
            pos = cut.rfind(sep)
            if pos > 600:
                cut = cut[: pos + 1]
                break
        else:
            cut = cut.rsplit(" ", 1)[0]
        p = cut
    return p


def _appearance_bit(cfg: Dict[str, Any]) -> str:
    """Identity + hair/outfit clause from cfg (character-bearing stills only)."""
    return configure.character_identity_clause(
        hair=cfg.get("hair_style") or "",
        outfit=cfg.get("outfit_worn") or "",
        gender=cfg.get("ref_gender") or "",
        bodyshape=cfg.get("ref_bodyshape") or "",
    )


def _sd_attempt_succeeded(out: str, rc: int, found: Optional[Path]) -> bool:
    """True when sd-cli actually produced a still — never retry in that case."""
    if found is not None and found.is_file() and found.stat().st_size > 64:
        return True
    low = (out or "").lower()
    if rc == 0 and ("images saved" in low or "[[saved_ok]]" in low or "save result image" in low):
        return True
    return False


def generate_images_from_prompts(
    prompts: List[str],
    lines: List[str],
    cfg: Dict[str, Any],
    progress_callback: Optional[Callable] = None,
    project_dir: Optional[Path] = None,
    reference_image: str = "",
    presence_per_line: Optional[List[str]] = None,
    force_line_numbers: Optional[List[int]] = None,
) -> List[Path]:
    """
    One still per prompt → project_dir / "001-lyric-slug.png".
    Reference image is attached ONLY when presence_per_line[i] is not "none".
    force_line_numbers: optional 1-based indices (used by regenerate).
    """
    exe = find_sd_cpp()
    if not exe:
        raise RuntimeError("sd-cli binary not found. Run Installation.")

    model_path = (cfg.get("imagegen_model_path") or "").strip()
    if not model_path or not Path(model_path).exists():
        raise RuntimeError(f"Diffuser model not set/found: {model_path or '(empty)'}")

    # Flux.2 --llm text conditioning: reuse Encoder (Qwen3-VL Instruct) path
    llm_path = (cfg.get("encoder_model_path") or "").strip()
    if not llm_path or not Path(llm_path).exists():
        raise RuntimeError(
            "Encoder (Qwen3-VL) not set/found — required for Flux.2 --llm.\n"
            "Set Configuration → Encoder → Model location.\n"
            f"Current path: {llm_path or '(empty)'}"
        )

    # Resolve VAE: configured path must be Flux.2 autoencoder, not Flux.1 ae
    # and not the DiT safetensors that often sits beside the GGUF.
    vae_path = _resolve_flux2_vae(cfg, model_path)
    if not vae_path:
        msg = _flux2_vae_help(model_path, (cfg.get("vae_model_path") or "").strip())
        print(f"[images] ERROR: {msg}", flush=True)
        raise RuntimeError(msg)

    ref_path = (reference_image or "").strip()
    has_ref = bool(ref_path and Path(ref_path).is_file())
    print(f"[images] diffusion = {Path(model_path).name}", flush=True)
    print(f"[images] encoder/--llm = {Path(llm_path).name}", flush=True)
    print(f"[images] vae       = {Path(vae_path).name if vae_path else '(none)'}", flush=True)

    out_dir = project_dir if project_dir is not None else configure.get_output_dir()
    out_dir.mkdir(parents=True, exist_ok=True)

    backend = str(cfg.get("imagegen_backend") or "CPU")
    use_gpu = backend.upper().startswith(("VULKAN", "CUDA"))
    vk_idx = 0
    if use_gpu:
        m = re.search(r"(\d+)", backend)
        vk_idx = int(m.group(1)) if m else 0

    # Output size from Generation tab / generation.json
    try:
        width = int(cfg.get("imagegen_width") or configure.DEFAULT_WIDTH)
        height = int(cfg.get("imagegen_height") or configure.DEFAULT_HEIGHT)
    except (TypeError, ValueError):
        width, height = configure.DEFAULT_WIDTH, configure.DEFAULT_HEIGHT
    if width < 64 or height < 64:
        width, height = configure.DEFAULT_WIDTH, configure.DEFAULT_HEIGHT
    steps = int(cfg.get("imagegen_steps") or configure.DEFAULT_STEPS)
    freq_label = configure.normalize_image_frequency(
        cfg.get("imagegen_frequency") or configure.DEFAULT_IMAGE_FREQUENCY
    )
    freq = configure.frequency_lyrics_per_line(freq_label)
    seq_hints = configure.image_frequency_hints(freq_label)
    print(f"[images] frequency = {freq_label} → {freq} still(s) per lyric line", flush=True)
    cfg_scale = float(cfg.get("imagegen_cfg_scale") or configure.DEFAULT_CFG)
    sampler = str(cfg.get("imagegen_sampling") or configure.DEFAULT_SAMPLER)
    seed_val = cfg.get("imagegen_seed")
    seed = int(seed_val) if seed_val not in (None, "") else configure.DEFAULT_SEED
    threads = _worker_thread_count(cfg)

    images: List[Path] = []
    total = len(prompts)
    already = _existing_image_indices(out_dir)
    # Collect any already-present stills so gallery/resume stay consistent
    for idx in sorted(already):
        found = _find_still_for_line(out_dir, idx)
        if found is not None:
            images.append(found)
            configure.APP_STATE.setdefault("generation_output_paths", []).append(str(found))
    skip_n = len(already)
    print(
        f"[images] === Phase 2: generating {total} still(s)"
        f"{f' (skipping {skip_n} already on disk)' if skip_n else ''} ===",
        flush=True,
    )

    style_hint = str(cfg.get("prompt_template") or cfg.get("style") or "")

    def _build_sd_cmd(
        prompt_text: str,
        dest: Path,
        attach_ref: bool,
        ref_override: str = "",
    ) -> List[str]:
        c = [
            str(exe),
            "--diffusion-model", str(model_path),
            "--llm", str(llm_path),
            "-p", prompt_text,
            "-o", str(dest),
            "-W", str(width), "-H", str(height),
            "--steps", str(steps),
            "--cfg-scale", str(cfg_scale),
            "--sampling-method", sampler,
            "-s", str(seed if seed >= 0 else seed),
            "-t", str(threads),
        ]
        if vae_path and Path(vae_path).exists():
            c.extend(["--vae", str(vae_path)])
        c.append("--vae-tiling")
        try:
            vk = configure.get_vulkan_info()
            for d in vk.get("devices") or []:
                if int(d.get("index", -1)) == int(vk_idx) and d.get("fp16"):
                    c.append("--diffusion-fa")
                    break
        except Exception:
            pass
        # Progressive L2/L3: prior still as -r; else character reference photo
        r_use = (ref_override or "").strip()
        if not r_use and attach_ref and has_ref:
            r_use = str(ref_path)
        if r_use and Path(r_use).is_file():
            c.extend(["-r", str(r_use)])
        neg = configure.merge_negative_prompt(
            (cfg.get("negative_prompt") or "").strip(),
            cover=bool(cfg.get("cover_mode")),
        )
        if neg:
            c.extend(["-n", neg])
        c.extend(_sd_placement_args(cfg, use_gpu, vk_idx))
        return c

    work_total = max(1, total * freq)
    work_done = 0
    for i, prompt in enumerate(prompts):
        if is_cancel_requested():
            break
        line_text = lines[i] if i < len(lines) else f"line_{i + 1}"
        line_no = (
            int(force_line_numbers[i])
            if force_line_numbers and i < len(force_line_numbers)
            else i + 1
        )
        existing_vars = _existing_variants_for_line(out_dir, line_no)

        pres = "none"
        if presence_per_line and i < len(presence_per_line):
            pres = (presence_per_line[i] or "none").lower()
        use_char_ref = bool(has_ref and pres != "none")
        clean_prompt = _sanitize_visual_prompt(prompt, line_text, style_hint)
        prev_still: Optional[Path] = None  # progressive chain within this line

        for var in range(1, freq + 1):
            if is_cancel_requested():
                break
            img_path = out_dir / _still_filename(
                line_no, line_text, variant=var, frequency=freq,
            )
            already_here = var in existing_vars or (
                freq <= 1 and _find_still_for_line(out_dir, line_no) is not None
            )
            if already_here:
                dest = _find_still_for_line(
                    out_dir, line_no, variant=var if freq > 1 else None,
                ) or img_path
                if dest.exists() and dest not in images:
                    images.append(dest)
                    configure.APP_STATE.setdefault("generation_output_paths", []).append(str(dest))
                prev_still = dest if dest.exists() else prev_still
                print(
                    f"[images] Lyrics Line {line_no}/{total} var {var}/{freq}: SKIP (exists)",
                    flush=True,
                )
                work_done += 1
                if progress_callback:
                    progress_callback(
                        f"Image {line_no}/{total} [{var}/{freq}]: skipped (exists)",
                        0.35 + 0.55 * (work_done / work_total),
                        {"phase": "images", "line": line_no, "total": total},
                    )
                continue

            print("[images] ────────────────────────────────────────", flush=True)
            print(
                f"[images] Lyrics Line {line_no}/{total} [{var}/{freq}]: {line_text}",
                flush=True,
            )
            if progress_callback:
                progress_callback(
                    f"Image {line_no}/{total} [{var}/{freq}]: {line_text[:50]}",
                    0.35 + 0.55 * (work_done / work_total),
                    {"phase": "images", "line": line_no, "total": total},
                )

            hint = seq_hints[var - 1] if var - 1 < len(seq_hints) else ""
            seq_bit = f" [Sequence {var}/{freq}: {hint}.]" if hint else ""

            # Progressive chain: var>1 uses previous still as visual reference
            progressive = bool(freq > 1 and var > 1 and prev_still and prev_still.is_file())
            ref_override = ""
            attach_ref = False
            if progressive:
                ref_override = str(prev_still)
                attach_ref = True
                appear = _appearance_bit(cfg) if use_char_ref else ""
                appear_bit = f" {appear}" if appear else ""
                final_prompt = (
                    f"{clean_prompt}{seq_bit}{appear_bit} "
                    f"This is beat {var} of {freq} in a continuous moment. "
                    "Evolve naturally from the provided reference still "
                    "(same scene, characters, hair, outfit, and lighting language) — "
                    "show the next beat of the action, not a new unrelated shot."
                )
                print(
                    f"[images] progressive ref ← {prev_still.name} (var {var}/{freq})",
                    flush=True,
                )
            elif use_char_ref:
                attach_ref = True
                appear = _appearance_bit(cfg)
                appear_bit = f" {appear}" if appear else ""
                final_prompt = (
                    f"{clean_prompt}{seq_bit}{appear_bit} "
                    f"[Reference character presence: {pres}. "
                    "Match the provided reference image likeness accordingly.]"
                )
                print(f"[images] character ref ATTACHED ({pres})"
                      f"{' +appearance' if appear else ''}", flush=True)
            else:
                if has_ref:
                    print("[images] character ref OMITTED (presence=none)", flush=True)
                final_prompt = (
                    f"{clean_prompt}{seq_bit} "
                    "[Do not depict any specific real person from a reference photo.]"
                ) if has_ref else f"{clean_prompt}{seq_bit}"

            cmd = _build_sd_cmd(
                final_prompt, img_path, attach_ref=attach_ref, ref_override=ref_override,
            )
            t_img = time.time()
            configure.APP_STATE["image_gen_t0"] = t_img
            out, rc = _run_sd_cli_once(cmd, exe)
            found = _find_still_for_line(
                out_dir, line_no, variant=var if freq > 1 else None,
            )
            log_path = None
            if not _sd_attempt_succeeded(out, rc, found):
                log_path = _write_sd_failure_log(
                    out_dir, line_no,
                    f"var{var} with reference" if attach_ref else f"var{var} no reference",
                    cmd, out, rc,
                )

            if not _sd_attempt_succeeded(out, rc, found) and attach_ref:
                print(
                    f"[images] line {line_no} var {var} failed with ref — "
                    "retrying without reference…",
                    flush=True,
                )
                cmd2 = _build_sd_cmd(final_prompt, img_path, attach_ref=False, ref_override="")
                out, rc = _run_sd_cli_once(cmd2, exe)
                found = _find_still_for_line(
                    out_dir, line_no, variant=var if freq > 1 else None,
                )
                if not _sd_attempt_succeeded(out, rc, found):
                    cmd = cmd2
                    log_path = _write_sd_failure_log(
                        out_dir, line_no, f"var{var} retry, no reference", cmd2, out, rc,
                    )

            if not _sd_attempt_succeeded(out, rc, found):
                detail = _format_sd_failure(out, vae_path, model_path, rc)
                where = f"\nFull log: {log_path}" if log_path else ""
                raise RuntimeError(
                    f"Image generation failed for line {line_no}/{total} "
                    f"variant {var}/{freq}"
                    f" ({'reference attached' if attach_ref else 'no reference'}).\n"
                    f"{detail}{where}"
                )

            dest = found if found is not None else img_path
            images.append(dest)
            prev_still = dest  # feed next variant in this line
            configure.APP_STATE.setdefault("generation_output_paths", []).append(str(dest))
            already.add(line_no)
            existing_vars.add(var)
            try:
                sz = dest.stat().st_size
            except OSError:
                sz = 0
            img_elapsed = time.time() - t_img
            # Always record the *actual* duration so the next still's estimate
            # reflects recent speed — never pad to match a previous slow run.
            record_last_image_gen_seconds(img_elapsed)
            configure.APP_STATE["image_gen_t0"] = None
            print(
                f"[images] saved {dest.name} ({sz} bytes) in {img_elapsed:.1f}s",
                flush=True,
            )
            work_done += 1
            if progress_callback:
                progress_callback(
                    f"Image {line_no}/{total} [{var}/{freq}] done ({img_elapsed:.0f}s)",
                    0.35 + 0.55 * (work_done / work_total),
                    {"phase": "images", "line": line_no, "total": total},
                )
            try:
                configure.save_session_meta(out_dir, {
                    "phase": "images",
                    "images_done": len(images),
                    "line_count": total,
                })
            except Exception:
                pass
            gc.collect()


    return images



def regenerate_named_still(
    kind: str,
    path: str,
    cfg: Dict[str, Any],
    *,
    song_name: str = "",
    lyrics: str = "",
    reference_image: str = "",
    theme_index: int = 0,
) -> Path:
    """
    Re-generate one cover or theme still in place (same filename).
    kind: "cover" | "theme"
    """
    dest = Path(path)
    if not dest.parent.is_dir():
        raise RuntimeError(f"Project folder missing: {dest.parent}")
    cfg = _build_cfg_for_imagegen(cfg)
    has_ref = bool((reference_image or "").strip() and Path(reference_image).is_file())
    project_dir = dest.parent
    analysis = _load_analysis_from_disk(project_dir) or {}
    style = str(cfg.get("style") or configure.STYLE_LIGHT)
    style_hint = configure.prompt_template_for_style(style)
    label = (song_name or project_dir.name or "song").strip()
    title = label.replace("_", " ").strip() or label

    if kind == "cover":
        prompt, strategy, use_char_ref = _build_cover_prompt(
            title, analysis, cfg, has_ref=has_ref,
        )
        # use_char_ref applied below via has_ref override
        if not use_char_ref:
            has_ref = False
    else:
        # Theme: extract the Nth prompt from theme_prompts.txt structured format:
        #   OVERALL\n...\n=== theme 1 ===\nprompt\n\n=== theme 2 ===\n...
        # Fall back to OVERALL-only ambient prompt. Never use character / reference.
        prompt = ""
        prompts_file = project_dir / "theme_prompts.txt"
        if prompts_file.is_file():
            try:
                raw_tp = prompts_file.read_text(encoding="utf-8", errors="replace")
                blocks = re.split(
                    r"(?im)^\s*===\s*theme\s*(\d+)\s*===\s*$",
                    raw_tp,
                )
                by_num: Dict[int, str] = {}
                i = 1
                while i + 1 < len(blocks):
                    try:
                        num = int(blocks[i])
                    except ValueError:
                        i += 2
                        continue
                    body = (blocks[i + 1] or "").strip()
                    body = re.split(r"(?im)^\s*OVERALL\b", body)[0].strip()
                    if body:
                        by_num[num] = body
                    i += 2
                want = int(theme_index) + 1
                if want in by_num:
                    prompt = by_num[want]
                elif by_num:
                    ordered = [by_num[k] for k in sorted(by_num)]
                    if 0 <= theme_index < len(ordered):
                        prompt = ordered[theme_index]
            except OSError:
                pass
        if not prompt:
            overall = (analysis.get("overall") or "").strip()
            prompt = (
                f"{style_hint.replace('{line}', (overall[:160] or title))} "
                f"Ambient theme still from the song's OVERALL world only. {overall[:520]} "
                "Environment and atmosphere only — no person, no character, no face, "
                "no silhouette. No readable text, logos, or watermarks."
            )
        prompt = (
            f"{prompt.strip()} "
            "Atmospheric worldspace still with no central character portrait, "
            "no specific person, no face close-up — environment and mood only."
        ).strip()

    cfg_run = dict(cfg)
    if kind == "cover":
        cfg_run["cover_mode"] = True
        return _generate_named_still(
            dest, prompt, cfg_run,
            reference_image=reference_image if has_ref else "",
            attach_ref=has_ref,
        )
    # Theme: never attach character reference
    return _generate_named_still(
        dest, prompt, cfg_run,
        reference_image="",
        attach_ref=False,
    )


def regenerate_single_still(
    project_dir: Path,
    line_index: int,
    cfg: Dict[str, Any],
    reference_image: str = "",
    progress_callback: Optional[Callable] = None,
    variant: int = 1,
) -> Path:
    """
    Force-regenerate ONE still (0-based line_index) only.

    variant: 1-based sequence index when Image Frequency is L2/L3
    (filename 001-2-slug.png for variant 2). Defaults to 1 (L1 naming).

    Does not call the batch generator. Runs one sd-cli invocation for that
    line+variant, leaves every other still untouched.
    """
    project_dir = Path(project_dir)
    if not project_dir.is_dir():
        raise RuntimeError(f"Project folder not found: {project_dir}")

    raw = (
        (project_dir / "lyrics.txt").read_text(encoding="utf-8")
        if (project_dir / "lyrics.txt").exists()
        else ""
    )
    lines = lyric_lines_only(parse_lyrics(raw))
    if not lines:
        raise RuntimeError("No lyrics lines in project — cannot regenerate.")

    if line_index < 0 or line_index >= len(lines):
        raise RuntimeError(
            f"Line index {line_index + 1} out of range (1..{len(lines)})."
        )

    # Allow partial prompts.txt (resume mid-song): only need this line's prompt
    prompts, presence = _load_prompts_from_disk(project_dir, expected_lines=0)
    if not prompts or line_index >= len(prompts):
        prompts, presence = _load_prompts_from_disk(
            project_dir, expected_lines=len(lines),
        )
    if not prompts or line_index >= len(prompts):
        raise RuntimeError(
            "prompts.txt missing or incomplete for this line — run full Generate first."
        )

    line_no = int(line_index) + 1
    line_text = lines[line_index]
    prompt = prompts[line_index]
    pres = "none"
    if presence and line_index < len(presence):
        pres = (presence[line_index] or "none").lower()

    # Keep the existing still on disk until the new one is written (sd-cli -o
    # overwrites). Pre-deleting made the gallery show "No Image" and shifted
    # neighbouring slots while regeneration was in flight.
    removed = 0

    ref = (reference_image or "").strip()
    if not ref:
        for cand in project_dir.glob("reference.*"):
            if cand.is_file():
                ref = str(cand)
                break
    has_ref = bool(ref and Path(ref).is_file())
    use_ref = bool(has_ref and pres != "none")

    exe = find_sd_cpp()
    if not exe:
        raise RuntimeError("sd-cli binary not found. Run Installation.")
    model_path = (cfg.get("imagegen_model_path") or "").strip()
    if not model_path or not Path(model_path).exists():
        raise RuntimeError(f"Diffuser model not set/found: {model_path or '(empty)'}")
    llm_path = (cfg.get("encoder_model_path") or "").strip()
    if not llm_path or not Path(llm_path).exists():
        raise RuntimeError(f"Encoder not set/found: {llm_path or '(empty)'}")
    vae_path = _resolve_flux2_vae(cfg, model_path)
    if not vae_path:
        raise RuntimeError(_flux2_vae_help(model_path, (cfg.get("vae_model_path") or "").strip()))

    backend = str(cfg.get("imagegen_backend") or "CPU")
    use_gpu = backend.upper().startswith(("VULKAN", "CUDA"))
    vk_idx = 0
    if use_gpu:
        m = re.search(r"(\d+)", backend)
        vk_idx = int(m.group(1)) if m else 0
    try:
        width = int(cfg.get("imagegen_width") or configure.DEFAULT_WIDTH)
        height = int(cfg.get("imagegen_height") or configure.DEFAULT_HEIGHT)
    except (TypeError, ValueError):
        width, height = configure.DEFAULT_WIDTH, configure.DEFAULT_HEIGHT
    if width < 64 or height < 64:
        width, height = configure.DEFAULT_WIDTH, configure.DEFAULT_HEIGHT
    steps = int(cfg.get("imagegen_steps") or configure.DEFAULT_STEPS)
    cfg_scale = float(cfg.get("imagegen_cfg_scale") or configure.DEFAULT_CFG)
    sampler = str(cfg.get("imagegen_sampling") or configure.DEFAULT_SAMPLER)
    seed_val = cfg.get("imagegen_seed")
    seed = int(seed_val) if seed_val not in (None, "") else configure.DEFAULT_SEED
    threads = _worker_thread_count(cfg)

    style_hint = str(cfg.get("prompt_template") or cfg.get("style") or "")
    clean_prompt = _sanitize_visual_prompt(prompt, line_text, style_hint)
    if use_ref:
        appear = _appearance_bit(cfg)
        appear_bit = f" {appear}" if appear else ""
        final_prompt = (
            f"{clean_prompt}{appear_bit} "
            f"[Reference character presence: {pres}. "
            "Match the provided reference image likeness accordingly.]"
        )
    else:
        final_prompt = (
            f"{clean_prompt} "
            "[Do not depict any specific real person from a reference photo.]"
        ) if has_ref else clean_prompt

    freq_label = configure.normalize_image_frequency(
        cfg.get("imagegen_frequency") or configure.DEFAULT_IMAGE_FREQUENCY
    )
    freq = max(1, int(configure.frequency_lyrics_per_line(freq_label)))
    var = max(1, min(int(variant or 1), freq))
    # Apply sequence framing hint when regenerating a multi-variant still
    if freq > 1:
        hints = configure.image_frequency_hints(freq_label)
        hint = hints[var - 1] if var - 1 < len(hints) else ""
        if hint:
            final_prompt = f"{final_prompt} {hint}".strip()
    img_path = project_dir / _still_filename(
        line_no, line_text, variant=var, frequency=freq,
    )
    # Write to a sibling temp path first so the existing still stays on disk
    # until success (avoids "No Image" flash and slot-list shifts mid-regen).
    partial = img_path.with_name(img_path.stem + ".__partial__" + img_path.suffix)
    try:
        if partial.exists():
            partial.unlink()
    except OSError:
        pass

    print(
        f"[regen] ONLY regenerating still {line_no}/{len(lines)}: {line_text[:60]}",
        flush=True,
    )
    if progress_callback:
        progress_callback(
            f"Regenerating still {line_no} only…",
            0.5,
            {"phase": "regen", "line": line_no, "total": 1},
        )

    def _one_cmd(attach_ref: bool) -> list:
        c = [
            str(exe),
            "--diffusion-model", str(model_path),
            "--llm", str(llm_path),
            "-p", final_prompt,
            "-o", str(partial),
            "-W", str(width), "-H", str(height),
            "--steps", str(steps),
            "--cfg-scale", str(cfg_scale),
            "--sampling-method", sampler,
            "-s", str(seed if seed >= 0 else seed),
            "-t", str(threads),
        ]
        if vae_path and Path(vae_path).exists():
            c.extend(["--vae", str(vae_path)])
        c.append("--vae-tiling")
        try:
            vk = configure.get_vulkan_info()
            for d in vk.get("devices") or []:
                if int(d.get("index", -1)) == int(vk_idx) and d.get("fp16"):
                    c.append("--diffusion-fa")
                    break
        except Exception:
            pass
        if attach_ref and has_ref:
            c.extend(["-r", str(ref)])
        neg = configure.merge_negative_prompt(
            (cfg.get("negative_prompt") or "").strip(),
            cover=bool(cfg.get("cover_mode")),
        )
        if neg:
            c.extend(["-n", neg])
        c.extend(_sd_placement_args(cfg, use_gpu, vk_idx))
        return c

    if use_ref:
        print(f"[regen] ref image ATTACHED ({pres})", flush=True)
    else:
        print("[regen] ref image OMITTED", flush=True)

    t_img = time.time()
    out, rc = _run_sd_cli_once(_one_cmd(attach_ref=use_ref), exe)
    found = partial if partial.is_file() else None
    if found is None:
        found = _find_still_for_line(
            project_dir, line_no, variant=var if freq > 1 else None,
        )

    if not _sd_attempt_succeeded(out, rc, found) and use_ref:
        print(f"[regen] line {line_no} var {var} failed with ref — one retry without reference…", flush=True)
        try:
            if partial.exists():
                partial.unlink()
        except OSError:
            pass
        out, rc = _run_sd_cli_once(_one_cmd(attach_ref=False), exe)
        found = partial if partial.is_file() else None
        if found is None:
            found = _find_still_for_line(
                project_dir, line_no, variant=var if freq > 1 else None,
            )

    if not _sd_attempt_succeeded(out, rc, found):
        try:
            if partial.exists():
                partial.unlink()
        except OSError:
            pass
        detail = _format_sd_failure(out, vae_path, model_path, rc)
        raise RuntimeError(
            f"Regenerate failed for line {line_no} variant {var} only.\n{detail}"
        )

    # Atomic-ish replace: only now remove the previous still for this variant
    dest = img_path
    try:
        if found is not None and found.resolve() != img_path.resolve():
            if img_path.exists():
                img_path.unlink()
            found.replace(img_path)
        elif found is not None and found.resolve() == img_path.resolve():
            pass  # already at dest
        dest = img_path
    except OSError:
        dest = found if found is not None else img_path

    img_elapsed = time.time() - t_img
    record_last_image_gen_seconds(img_elapsed)
    kept = []
    for p in (configure.APP_STATE.get("generation_output_paths") or []):
        try:
            n = Path(p).name
            # Drop only the same line+variant path we just replaced
            if freq > 1:
                if n.startswith(f"{line_no:03d}-{var}-") or n.startswith(f"{line_no:03d}-{var}."):
                    continue
            else:
                m = re.match(r"^(\d+)", n)
                if m and int(m.group(1)) == line_no:
                    continue
        except Exception:
            pass
        kept.append(p)
    kept.append(str(dest))
    configure.APP_STATE["generation_output_paths"] = kept
    print(f"[regen] done — only {dest.name} ({dest.stat().st_size} bytes) in {img_elapsed:.1f}s", flush=True)
    try:
        configure.save_session_meta(project_dir, {
            "phase": "images",
            "images_done": len(_existing_image_indices(project_dir)),
        })
    except Exception:
        pass
    gc.collect()
    return dest


# ---------------------------------------------------------------------------
# Top-level pipeline
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Cover image + Theme images (assessment-driven)
# ---------------------------------------------------------------------------


def _analysis_is_usable(analysis: Optional[Dict[str, Any]]) -> bool:
    if not analysis or not isinstance(analysis, dict):
        return False
    overall = (analysis.get("overall") or "").strip()
    if overall and not _looks_like_keyword_spam(overall):
        return True
    secs = analysis.get("sections") or {}
    if isinstance(secs, dict):
        good = [
            (v or "").strip()
            for v in secs.values()
            if (v or "").strip() and not _looks_like_keyword_spam(v or "")
        ]
        if len(good) >= 1:
            return True
    return False


def _write_analysis_file(project_dir: Path, analysis: Dict[str, Any]) -> None:
    try:
        (project_dir / "analysis.txt").write_text(
            "OVERALL\n"
            + (analysis.get("overall") or "")
            + "\n\nCHARACTER\n"
            + (analysis.get("character_guidance") or "")
            + "\n\nCHARACTER_MAP\n"
            + "\n".join(
                f"{k}: {v}"
                for k, v in (analysis.get("character_presence") or {}).items()
            )
            + "\n\nSECTIONS\n"
            + "\n".join(
                f"{k}: {v}" for k, v in (analysis.get("sections") or {}).items()
            ),
            encoding="utf-8",
        )
    except OSError as e:
        print(f"[analysis] could not write analysis.txt: {e}", flush=True)


def ensure_project_assessment(
    project_dir: Path,
    lyrics: str,
    cfg: Dict[str, Any],
    *,
    has_character_ref: bool = False,
    progress_callback: Optional[Callable] = None,
    force: bool = False,
) -> Dict[str, Any]:
    """
    Load analysis.txt if usable; otherwise run song/section assessment and save it.
    All three generation modes (cover / theme / lyrics) call this before acting.
    """
    project_dir = Path(project_dir)
    project_dir.mkdir(parents=True, exist_ok=True)
    try:
        (project_dir / "lyrics.txt").write_text(lyrics or "", encoding="utf-8")
    except OSError:
        pass

    if not force:
        disk = _load_analysis_from_disk(project_dir)
        if _analysis_is_usable(disk):
            print("[analysis] RESUME — usable analysis.txt on disk", flush=True)
            if progress_callback:
                progress_callback("Using existing song assessment", 0.08, {"phase": "analysis"})
            return disk
        if disk:
            print("[analysis] disk analysis unusable (spam or empty) — re-running", flush=True)

    if not (lyrics or "").strip():
        print("[analysis] no lyrics — cannot run full assessment", flush=True)
        return {
            "overall": "",
            "sections": {},
            "character_guidance": "",
            "character_presence": {},
        }

    # Drop stale/bad analysis before a forced or recovery run
    try:
        stale = project_dir / "analysis.txt"
        if force and stale.is_file():
            stale.unlink()
            print("[analysis] removed prior analysis.txt for re-run", flush=True)
    except OSError:
        pass

    parsed = parse_lyrics(lyrics)
    if progress_callback:
        progress_callback("Assessing song & sections…", 0.06, {"phase": "analysis"})
    analysis = analyze_song_and_sections(
        lyrics, parsed, cfg,
        has_character_ref=has_character_ref,
        progress_callback=progress_callback,
    )
    if _analysis_is_usable(analysis):
        _write_analysis_file(project_dir, analysis)
        maybe_bleep_section()
    else:
        print("[analysis] model reply unusable — not saving analysis.txt", flush=True)
        # Still write a short stub so the Assessment panel shows what went wrong
        try:
            (project_dir / "analysis.txt").write_text(
                "OVERALL\n"
                "(Assessment failed — model returned unstructured or repetitive text. "
                "Click Re-run Assessment.)\n\n"
                "CHARACTER\n\nCHARACTER_MAP\n\nSECTIONS\n",
                encoding="utf-8",
            )
        except OSError:
            pass
    return analysis


def generate_theme_prompts_from_assessment(
    analysis: Dict[str, Any],
    cfg: Dict[str, Any],
    n_prompts: int,
    *,
    song_name: str = "",
    progress_callback: Optional[Callable] = None,
) -> List[str]:
    """
    Thinking/Encoder writes `n_prompts` distinct ambient worldspace prompts
    from the OVERALL assessment only. Theme stills are fillers for non-lyric
    stretches — no central character, no reference photo.
    """
    n = max(1, min(8, int(n_prompts or 1)))
    overall = (analysis.get("overall") or "").strip()
    if not overall:
        return []

    model_path, role = _resolve_prompt_model(cfg)
    if not model_path:
        return [overall] * n

    style = str(cfg.get("style") or configure.STYLE_DEFAULT)
    title = (song_name or "").replace("_", " ").strip()
    prompt = (
        "You are writing visual prompts for music-video THEME stills "
        "(ambient fillers between lyric lines — NOT character action, NOT cover art).\n"
        f"Song title (context only, not the subject): {title or '(untitled)'}\n"
        f"Visual style preference: {style}\n\n"
        "ASSESSMENT — OVERALL (the ONLY source — ignore any character or section notes):\n"
        f"{overall}\n\n"
        f"Write exactly {n} distinct image prompts, numbered 1..{n}.\n"
        "Each prompt must:\n"
        "- Be drawn only from the OVERALL world, mood, setting, and colour language\n"
        "- Be a single dense paragraph for one ambient environmental still\n"
        "- Explore a different facet of that OVERALL space "
        "(architecture, atmosphere, light, weather, scale, time of day)\n"
        "- Contain NO person, NO character, NO face, NO silhouette of a person, NO figure\n"
        "- Contain NO reference to a protagonist or central subject\n"
        "- Work as a Flux image prompt (concrete environment, lighting, composition)\n"
        "- No quotes, no 'Prompt:' labels, no lyrics, no readable text in the image\n\n"
        "Output format ONLY:\n"
        "1. <prompt>\n"
        "2. <prompt>\n"
        f"…\n{n}. <prompt>\n"
    )
    if progress_callback:
        progress_callback(
            f"Writing {n} theme prompt(s) from assessment…",
            0.18,
            {"phase": "theme_prompts"},
        )
    print(f"[theme] generating {n} theme prompt(s) via {role}", flush=True)
    try:
        raw = _run_llama_completion(
            prompt, cfg, role=role, model_path=model_path,
            n_predict=max(_PROMPT_PREDICT, 160 * n, 400),
            ctx_size=_PROMPT_CTX + 2048,
            temperature=0.7,
            timeout=max(_TIMEOUT_PROMPT, 120.0 * n),
        )
    except Exception as e:
        print(f"[theme] prompt generation failed: {e}", flush=True)
        raw = ""

    prompts: List[str] = []
    if raw:
        for m in re.finditer(
            r"(?m)^\s*(?:\*\*)?(\d+)(?:\*\*)?[.)\:\-]\s*(.+?)(?=^\s*(?:\*\*)?\d+(?:\*\*)?[.)\:\-]|\Z)",
            raw,
            re.S,
        ):
            body = re.sub(r"\s+", " ", m.group(2)).strip().strip('"').strip()
            if len(body) > 20:
                prompts.append(body)
        if not prompts:
            # Fallback: split non-empty lines
            for line in raw.splitlines():
                t = re.sub(r"^\s*\d+[.)\:\-]\s*", "", line).strip()
                if len(t) > 40:
                    prompts.append(t)

    # Pad / trim to exact n
    if not prompts and overall:
        prompts = [overall]
    while len(prompts) < n and prompts:
        prompts.append(prompts[-1])
    prompts = prompts[:n]
    print(f"[theme] got {len(prompts)} theme prompt(s)", flush=True)
    return prompts


def _theme_topics_from_analysis(analysis: Dict[str, Any]) -> List[Dict[str, str]]:
    """
    Flatten analysis notes into ordered theme topics for still generation.
    Each entry: {id, title, notes} — one still per non-empty paragraph.
    """
    topics: List[Dict[str, str]] = []
    overall = (analysis.get("overall") or "").strip()
    if overall:
        topics.append({"id": "overall", "title": "Overall", "notes": overall})
    char = (analysis.get("character_guidance") or "").strip()
    if char and char.lower() not in ("none specified", "none", "n/a"):
        topics.append({"id": "character", "title": "Character", "notes": char})
    sections = analysis.get("sections") or {}
    if isinstance(sections, dict):
        for label, notes in sections.items():
            n = (notes or "").strip()
            if not n:
                continue
            slug = _ascii_line_slug(str(label), max_len=40) or "section"
            topics.append({
                "id": f"section-{slug}",
                "title": str(label),
                "notes": n,
            })
    return topics


def _build_cfg_for_imagegen(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Ensure width/height/steps are concrete ints on a shallow copy."""
    out = dict(cfg or {})
    try:
        w = int(out.get("imagegen_width") or configure.DEFAULT_WIDTH)
        h = int(out.get("imagegen_height") or configure.DEFAULT_HEIGHT)
    except (TypeError, ValueError):
        w, h = configure.DEFAULT_WIDTH, configure.DEFAULT_HEIGHT
    if w < 64 or h < 64:
        w, h = configure.DEFAULT_WIDTH, configure.DEFAULT_HEIGHT
    out["imagegen_width"] = w
    out["imagegen_height"] = h
    try:
        out["imagegen_steps"] = int(out.get("imagegen_steps") or configure.DEFAULT_STEPS)
    except (TypeError, ValueError):
        out["imagegen_steps"] = configure.DEFAULT_STEPS
    try:
        out["imagegen_cfg_scale"] = float(out.get("imagegen_cfg_scale") or configure.DEFAULT_CFG)
    except (TypeError, ValueError):
        out["imagegen_cfg_scale"] = configure.DEFAULT_CFG
    return out


def _generate_named_still(
    dest: Path,
    prompt: str,
    cfg: Dict[str, Any],
    reference_image: str = "",
    attach_ref: bool = False,
) -> Path:
    """
    Run sd-cli once for a named output path (cover / theme / custom).
    Does not use the lyric-line numbering scheme.
    """
    cfg = _build_cfg_for_imagegen(cfg)
    exe = find_sd_cpp()
    if not exe:
        raise RuntimeError("sd-cli binary not found. Run Installation.")
    model_path = (cfg.get("imagegen_model_path") or "").strip()
    if not model_path or not Path(model_path).exists():
        raise RuntimeError(f"Diffuser model not set/found: {model_path or '(empty)'}")
    llm_path = (cfg.get("encoder_model_path") or "").strip()
    if not llm_path or not Path(llm_path).exists():
        raise RuntimeError(f"Encoder not set/found: {llm_path or '(empty)'}")
    vae_path = _resolve_flux2_vae(cfg, model_path)
    if not vae_path:
        raise RuntimeError(_flux2_vae_help(model_path, (cfg.get("vae_model_path") or "").strip()))

    backend = str(cfg.get("imagegen_backend") or "CPU")
    use_gpu = backend.upper().startswith(("VULKAN", "CUDA"))
    vk_idx = 0
    if use_gpu:
        m = re.search(r"(\d+)", backend)
        vk_idx = int(m.group(1)) if m else 0
    width = int(cfg["imagegen_width"])
    height = int(cfg["imagegen_height"])
    steps = int(cfg["imagegen_steps"])
    cfg_scale = float(cfg["imagegen_cfg_scale"])
    sampler = str(cfg.get("imagegen_sampling") or configure.DEFAULT_SAMPLER)
    seed_val = cfg.get("imagegen_seed")
    seed = int(seed_val) if seed_val not in (None, "") else configure.DEFAULT_SEED
    threads = _worker_thread_count(cfg)
    ref_path = (reference_image or "").strip()
    has_ref = bool(ref_path and Path(ref_path).is_file() and attach_ref)

    dest.parent.mkdir(parents=True, exist_ok=True)
    # Write to a sibling temp path first so the existing still stays on disk
    # until success. Pre-deleting dest shrank sorted cover/theme lists mid-run
    # and made other slots show "No Image" / wrong queue indices.
    partial = dest.with_name(dest.stem + ".__partial__" + dest.suffix)
    try:
        if partial.exists():
            partial.unlink()
    except OSError:
        pass

    configure.APP_STATE["image_gen_t0"] = time.time()
    clean = _sanitize_visual_prompt(prompt, "", str(cfg.get("style") or ""))
    cmd = [
        str(exe),
        "--diffusion-model", str(model_path),
        "--llm", str(llm_path),
        "-p", clean,
        "-o", str(partial),
        "-W", str(width), "-H", str(height),
        "--steps", str(steps),
        "--cfg-scale", str(cfg_scale),
        "--sampling-method", sampler,
        "-s", str(seed if seed >= 0 else seed),
        "-t", str(threads),
    ]
    if vae_path and Path(vae_path).exists():
        cmd.extend(["--vae", str(vae_path)])
    cmd.append("--vae-tiling")
    try:
        vk = configure.get_vulkan_info()
        for d in vk.get("devices") or []:
            if int(d.get("index", -1)) == int(vk_idx) and d.get("fp16"):
                cmd.append("--diffusion-fa")
                break
    except Exception:
        pass
    if has_ref:
        cmd.extend(["-r", str(ref_path)])
    neg = configure.merge_negative_prompt(
        (cfg.get("negative_prompt") or "").strip(),
        cover=bool(cfg.get("cover_mode")),
    )
    if neg:
        cmd.extend(["-n", neg])
    cmd.extend(_sd_placement_args(cfg, use_gpu, vk_idx))

    t_img = time.time()
    out, rc = _run_sd_cli_once(cmd, exe)
    # Prefer the partial we asked for; fall back to newest matching partial stem
    found = partial if partial.is_file() else None
    if found is None and dest.parent.is_dir():
        stem = partial.stem  # e.g. cover-01-foo.__partial__
        cands = sorted(
            [p for p in dest.parent.glob(stem + ".*")
             if p.is_file() and p.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp")],
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if cands:
            found = cands[0]
    if not _sd_attempt_succeeded(out, rc, found):
        # Leave original dest intact; clean up failed partial
        try:
            if partial.exists():
                partial.unlink()
        except OSError:
            pass
        detail = _format_sd_failure(out, vae_path, model_path, rc)
        raise RuntimeError(f"Still generation failed for {dest.name}.\n{detail}")
    # Atomic-ish replace: only now remove the previous still
    dest_final = dest
    try:
        if dest.exists():
            dest.unlink()
        if found is not None and found.resolve() != dest.resolve():
            found.replace(dest)
        elif found is not None:
            dest_final = found
    except OSError as e:
        # If replace failed but partial exists, keep it as dest_final for return
        if found is not None and found.is_file():
            dest_final = found
        print(f"[named] replace warning: {e}", flush=True)
    img_elapsed = time.time() - t_img
    record_last_image_gen_seconds(img_elapsed)
    configure.APP_STATE["image_gen_t0"] = None
    try:
        sz = dest_final.stat().st_size
    except OSError:
        sz = 0
    print(f"[named] saved {dest_final.name} ({sz} bytes) in {img_elapsed:.1f}s", flush=True)
    gc.collect()
    return dest_final



def _build_cover_prompt(
    title: str,
    analysis: Dict[str, Any],
    cfg: Dict[str, Any],
    *,
    has_ref: bool = False,
    strategy: str = "",
) -> Tuple[str, str, bool]:
    """
    Title-first album cover prompt.

    The song name is the visual subject. OVERALL assessment is only a short
    interpretation aid (what the title likely means in this song) — not a
    license to paint the same ambient worldspace as Theme stills.

    Returns (prompt, strategy, attach_ref).
    """
    title = (title or "").strip() or "Untitled"
    overall = (analysis.get("overall") or "").strip()
    character = (analysis.get("character_guidance") or "").strip()
    style = str(cfg.get("style") or configure.STYLE_DEFAULT)
    style_hint = configure.prompt_template_for_style(style)
    # Prefer first 1-2 sentences of OVERALL as title interpretation only
    interpret = ""
    if overall:
        parts = re.split(r"(?<=[.!?])\s+", overall)
        interpret = " ".join(parts[:2]).strip()
        if len(interpret) > 280:
            interpret = interpret[:280].rsplit(" ", 1)[0] + "…"

    if not strategy:
        strategy = classify_cover_strategy(title, analysis, cfg, has_ref)
    use_char_ref = strategy == "character" and has_ref

    # Optional short LLM concept locked to the title words
    concept = ""
    model_path, role = _resolve_prompt_model(cfg)
    if model_path and Path(model_path).is_file():
        try:
            concept_prompt = (
                "You write ONE album-cover concept sentence for an AI image model.\n"
                f'Song title (must be the visual subject): "{title}"\n'
            )
            if interpret:
                concept_prompt += (
                    f"Assessment hint (interpret the title only — do not describe generic scenery): "
                    f"{interpret}\n"
                )
            concept_prompt += (
                "Rules:\n"
                "- The image idea must make a viewer think of the song title itself.\n"
                "- Literal or metaphorical is fine, but the title's meaning must be obvious.\n"
                "- Do NOT describe a generic ambient corridor, crowd, or mood board.\n"
                "- No text, letters, logos, or watermarks in the image.\n"
                "- Describe a single focused subject (close-up or strong symbol), not a busy scene.\n"
                "- Reply with ONE sentence only. No preamble.\n"
            )
            raw = _run_llama_completion(
                concept_prompt, cfg, role=role, model_path=model_path,
                n_predict=120, ctx_size=min(_PROMPT_CTX, 4096),
                temperature=0.4, timeout=120.0,
            )
            concept = (raw or "").strip().splitlines()[0].strip()
            # Drop quotes / length runaway
            concept = concept.strip(" \"'`")
            if len(concept) > 320:
                concept = concept[:320].rsplit(" ", 1)[0]
            if len(concept) < 20:
                concept = ""
            else:
                print(f"[cover] title concept: {concept[:120]}", flush=True)
        except Exception as e:
            print(f"[cover] concept LLM skipped: {e}", flush=True)
            concept = ""

    appear = _appearance_bit(cfg) if use_char_ref else ""
    # Do NOT put the title in quotes as on-image text — Flux will try to write it.
    title_lead = (
        f"Album cover still inspired by the song name: {title}. "
        "One significant, isolated subject fills the frame — a close-up or strong "
        "figurative symbol the eye locks onto immediately. Clean composition, "
        "shallow depth of field or solid negative space, high visual impact. "
    )
    if concept:
        title_lead += f"Subject concept: {concept} "
    elif interpret:
        title_lead += (
            f"Use this only to interpret what the song name means (not as a scenery brief): "
            f"{interpret} "
        )

    no_text = (
        "Absolutely no text, letters, words, writing, typography, captions, subtitles, "
        "logos, watermarks, signatures, UI, or graphical overlays anywhere in the image. "
    )
    if use_char_ref:
        char_bit = f" Central figure: {character[:220]}." if character else ""
        cover_prompt = (
            f"{style_hint.replace('{line}', title)} "
            f"{title_lead}"
            f"Feature the central character as the single focused subject embodying the song name.{char_bit} "
            f"{appear + ' ' if appear else ''}"
            "Iconic cover portrait or figure study — not a busy scene. "
            f"{no_text}"
        )
    else:
        cover_prompt = (
            f"{style_hint.replace('{line}', title)} "
            f"{title_lead}"
            "Bold symbolic or literal object/figure for the song name — not a wide landscape, "
            "corridor, crowd, or generic mood board. One clear focal subject. "
            f"{no_text}"
        )
    return cover_prompt, strategy, use_char_ref


def classify_cover_strategy(
    title: str,
    analysis: Dict[str, Any],
    cfg: Dict[str, Any],
    has_ref: bool,
) -> str:
    """
    Decide cover composition strategy:
      'character' — song title is about the central subject/character;
                    use CHARACTER assessment + reference image when available.
      'world'     — title is atmospheric / setting / abstract;
                    use OVERALL worldspace only, no character / no reference image.

    Uses the Thinking (or Encoder) model when available; falls back to a
    lightweight heuristic from assessment text.
    """
    title = (title or "").strip()
    overall = (analysis.get("overall") or "").strip()
    character = (analysis.get("character_guidance") or "").strip()
    if not has_ref or not character or character.lower() in ("none specified", "none", "n/a", ""):
        return "world"

    model_path, role = _resolve_prompt_model(cfg)
    if model_path and Path(model_path).is_file():
        prompt = (
            "Classify this song title for album cover art strategy.\n\n"
            f"Song title: {title}\n\n"
            f"OVERALL assessment (world/mood):\n{overall[:600] or '(none)'}\n\n"
            f"CHARACTER guidance:\n{character[:400] or '(none)'}\n\n"
            "Answer with EXACTLY one word:\n"
            "  CHARACTER — if the title primarily names, describes, or is about "
            "the central character/subject (a person, named figure, or the story's protagonist).\n"
            "  WORLD — if the title is mainly about setting, mood, abstract idea, "
            "place, time, or atmosphere, not a specific character.\n\n"
            "Reply with only CHARACTER or WORLD."
        )
        try:
            raw = _run_llama_completion(
                prompt, cfg, role, model_path,
                n_predict=16,
                ctx_size=2048,
                temperature=0.1,
                timeout=90.0,
            )
            ans = (raw or "").strip().upper()
            first = ans.replace(".", " ").split()[0] if ans.split() else ""
            if first.startswith("CHAR"):
                print(f"[cover] strategy=CHARACTER (LLM: {ans[:40]!r})", flush=True)
                return "character"
            if first.startswith("WORLD") or "ATMOSPHER" in ans or "SETTING" in ans:
                print(f"[cover] strategy=WORLD (LLM: {ans[:40]!r})", flush=True)
                return "world"
            print(f"[cover] strategy=WORLD (LLM default from {ans[:40]!r})", flush=True)
            return "world"
        except Exception as e:
            print(f"[cover] LLM classify failed ({e}); using heuristic", flush=True)

    # Heuristic: title words overlapping character notes → character
    title_tokens = set(re.findall(r"[a-z0-9]+", title.lower()))
    char_tokens = set(re.findall(r"[a-z0-9]+", character.lower()[:500]))
    stop = {"the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "with", "my", "your", "is", "it"}
    overlap = (title_tokens - stop) & (char_tokens - stop)
    if overlap:
        print(f"[cover] strategy=CHARACTER (heuristic overlap={overlap})", flush=True)
        return "character"
    print("[cover] strategy=WORLD (heuristic, no title/character overlap)", flush=True)
    return "world"


def generate_cover_image(
    song_name: str,
    cfg: Dict[str, Any],
    reference_image: str = "",
    progress_callback: Optional[Callable] = None,
    resume_folder: str = "",
    lyrics: str = "",
) -> Dict[str, Any]:
    """
    Cover stills from song title + assessment strategy:
      CHARACTER → title + character guidance + reference image
      WORLD     → title + OVERALL worldspace only (no character, no ref)
    """
    clear_cancel_state()
    configure.APP_STATE["session_status"] = "running"
    t0 = time.time()
    result: Dict[str, Any] = {
        "success": False,
        "message": "",
        "project_folder": "",
        "image_count": 0,
        "image_paths": [],
        "elapsed_seconds": 0.0,
        "session_id": "",
    }
    try:
        cfg = _build_cfg_for_imagegen(cfg)
        label = _slugify_folder_name(
            (song_name or cfg.get("project_label") or "").strip(),
            fallback="",
        )
        if not label and not resume_folder:
            result["message"] = "Song name is required for a cover image."
            configure.APP_STATE["session_status"] = "idle"
            return result
        if not label and resume_folder:
            label = Path(resume_folder).name
        project_dir = ensure_project_dir(
            label, sequential=False, resume_path=resume_folder or None,
        )
        result["project_folder"] = str(project_dir)
        result["session_id"] = project_dir.name
        configure.APP_STATE["active_session_id"] = project_dir.name

        _diff = (cfg.get("imagegen_model_path") or "").strip()
        _enc = (cfg.get("encoder_model_path") or "").strip()
        if not _diff or not Path(_diff).is_file() or not _enc or not Path(_enc).is_file():
            result["message"] = "Encoder + Diffuser models must be configured before generating a cover."
            configure.APP_STATE["session_status"] = "idle"
            return result
        vae = _resolve_flux2_vae(cfg, _diff)
        if not vae:
            result["message"] = _flux2_vae_help(_diff, (cfg.get("vae_model_path") or "").strip())
            configure.APP_STATE["session_status"] = "idle"
            return result
        cfg["vae_model_path"] = vae

        has_ref = bool(reference_image and Path(reference_image).is_file())
        if has_ref:
            local_ref = configure.ensure_project_reference(project_dir, reference_image)
            if local_ref:
                reference_image = local_ref

        analysis: Dict[str, Any] = {}
        if (lyrics or "").strip():
            analysis = ensure_project_assessment(
                project_dir,
                lyrics,
                cfg,
                has_character_ref=has_ref,
                progress_callback=progress_callback,
            )
            if is_cancel_requested():
                configure.APP_STATE["session_status"] = "stopped"
                result["message"] = "Cancelled during analysis."
                return result
        else:
            disk = _load_analysis_from_disk(project_dir) or {}
            if _analysis_is_usable(disk):
                analysis = disk
            else:
                print("[cover] no lyrics / assessment; title-only cover prompt", flush=True)

        title = (song_name or label).replace("_", " ").strip() or label
        cover_prompt, strategy, use_char_ref = _build_cover_prompt(
            title, analysis, cfg, has_ref=has_ref,
        )
        print(
            f"[cover] strategy={strategy}  attach_ref={use_char_ref}  "
            f"title={title!r}",
            flush=True,
        )
        if progress_callback:
            progress_callback(f"Cover for '{title}' ({strategy})", 0.2, {"phase": "cover"})

        phase_barrier("cover image")
        slug = _ascii_line_slug(title, max_len=40) or "song"
        target_n = configure.frequency_cover_count(
            cfg.get("imagegen_frequency") or configure.DEFAULT_IMAGE_FREQUENCY
        )
        # Count existing covers so "Complete" only fills the remainder;
        # when already at target, wipe and fully re-generate.
        existing_covers = sorted(project_dir.glob("cover-*.png"))
        have_n = len(existing_covers)
        if have_n >= target_n:
            for old in existing_covers:
                try:
                    old.unlink()
                except OSError:
                    pass
            have_n = 0
            cover_n = target_n
            start_idx = 1
            print(f"[cover] complete set removed — re-generating {cover_n}", flush=True)
        elif have_n > 0:
            cover_n = max(0, target_n - have_n)
            start_idx = 1
            while (project_dir / f"cover-{start_idx:02d}-{slug}.png").exists():
                start_idx += 1
                if start_idx > 999:
                    break
        else:
            cover_n = target_n
            start_idx = 1
        images: List[Path] = []
        print(f"[cover] target={target_n} have={have_n} generating={cover_n}", flush=True)
        configure.APP_STATE["cover_slot_expected"] = cover_n
        configure.APP_STATE["cover_slot_queued"] = list(range(cover_n))
        configure.APP_STATE["cover_slot_generating"] = None
        for i in range(cover_n):
            if is_cancel_requested():
                break
            cq = list(configure.APP_STATE.get("cover_slot_queued") or [])
            if i in cq:
                cq = [x for x in cq if x != i]
            configure.APP_STATE["cover_slot_queued"] = cq
            configure.APP_STATE["cover_slot_generating"] = i
            idx = start_idx + i
            dest = project_dir / f"cover-{idx:02d}-{slug}.png"
            if progress_callback:
                progress_callback(
                    f"Generating cover still {i + 1}/{cover_n}…",
                    0.35 + 0.55 * (i / max(cover_n, 1)),
                    {"phase": "cover", "line": i + 1, "total": cover_n},
                )
            print(f"[cover] {i + 1}/{cover_n} — {dest.name}", flush=True)
            var_bit = (
                f" Variation {i + 1} of {cover_n}: alternate angle or crop emphasis."
                if cover_n > 1 else ""
            )
            cfg_cover = dict(cfg)
            cfg_cover["cover_mode"] = True
            out = _generate_named_still(
                dest, cover_prompt + var_bit, cfg_cover,
                reference_image=reference_image if use_char_ref else "",
                attach_ref=use_char_ref,
            )
            images.append(out)
        elapsed = time.time() - t0
        configure.APP_STATE["session_status"] = "stopped"
        configure.APP_STATE["cover_slot_generating"] = None
        configure.APP_STATE["cover_slot_queued"] = []
        result.update(
            success=bool(images),
            image_count=len(images),
            image_paths=[str(p) for p in images],
            message=(
                f"Cover image(s) ready: {len(images)} still(s) in {project_dir.name}/ "
                f"({int(elapsed)}s)"
                if images else "No cover images were generated."
            ),
            elapsed_seconds=round(elapsed, 1),
            session_id=project_dir.name,
            project_folder=str(project_dir),
        )
        if progress_callback:
            progress_callback(result["message"], 1.0, {"phase": "done"})
        if images:
            maybe_bleep_done()
    except Exception as e:
        traceback.print_exc()
        result["message"] = f"Cover generation error: {e}"
        result["elapsed_seconds"] = round(time.time() - t0, 1)
        configure.APP_STATE["session_status"] = "stopped"
    return result



def generate_theme_images(
    lyrics: str,
    cfg: Dict[str, Any],
    song_name: str = "",
    reference_image: str = "",
    progress_callback: Optional[Callable] = None,
    resume_folder: str = "",
) -> Dict[str, Any]:
    """
    Theme stills from song assessment:
      1. ensure_project_assessment (OVERALL / CHARACTER / SECTIONS)
      2. Thinking/Encoder writes Image-Frequency distinct visual prompts from OVERALL
      3. Each prompt → one Flux still (theme-01-….png …)
    """
    clear_cancel_state()
    configure.APP_STATE["session_status"] = "running"
    t0 = time.time()
    result: Dict[str, Any] = {
        "success": False,
        "message": "",
        "project_folder": "",
        "image_count": 0,
        "image_paths": [],
        "elapsed_seconds": 0.0,
        "session_id": "",
    }
    try:
        cfg = _build_cfg_for_imagegen(cfg)
        parsed = parse_lyrics(lyrics)
        lines = lyric_lines_only(parsed)
        if not lines and not (lyrics or "").strip():
            result["message"] = "Lyrics are required so the assessment can produce theme notes."
            configure.APP_STATE["session_status"] = "idle"
            return result

        label = _slugify_folder_name(
            (song_name or cfg.get("project_label") or "").strip(),
            fallback="",
        )
        if not label and not resume_folder:
            result["message"] = "Song name is required for theme images."
            configure.APP_STATE["session_status"] = "idle"
            return result
        if not label and resume_folder:
            label = Path(resume_folder).name
        project_dir = ensure_project_dir(
            label, sequential=False, resume_path=resume_folder or None,
        )
        result["project_folder"] = str(project_dir)
        result["session_id"] = project_dir.name
        configure.APP_STATE["active_session_id"] = project_dir.name

        try:
            (project_dir / "lyrics.txt").write_text(lyrics or "", encoding="utf-8")
        except OSError:
            pass

        # Model preflight
        _diff = (cfg.get("imagegen_model_path") or "").strip()
        _enc = (cfg.get("encoder_model_path") or "").strip()
        if not _diff or not Path(_diff).is_file() or not _enc or not Path(_enc).is_file():
            result["message"] = "Encoder + Diffuser models must be configured before theme images."
            configure.APP_STATE["session_status"] = "idle"
            return result
        vae = _resolve_flux2_vae(cfg, _diff)
        if not vae:
            result["message"] = _flux2_vae_help(_diff, (cfg.get("vae_model_path") or "").strip())
            configure.APP_STATE["session_status"] = "idle"
            return result
        cfg["vae_model_path"] = vae

        has_ref = bool(reference_image and Path(reference_image).is_file())
        if has_ref:
            local_ref = configure.ensure_project_reference(project_dir, reference_image)
            if local_ref:
                reference_image = local_ref

        _text_path, _text_role = _resolve_prompt_model(cfg)
        log_phase_plan(cfg, _text_role)


        # Assessment must already be saved (Run Assessment + optional edit/Save)
        analysis = ensure_project_assessment(
            project_dir,
            lyrics or "",
            cfg,
            has_character_ref=has_ref,
            progress_callback=progress_callback,
            force=False,
        )
        if is_cancel_requested():
            configure.APP_STATE["session_status"] = "stopped"
            result["message"] = "Cancelled during analysis."
            return result

        if not _analysis_is_usable(analysis):
            result["message"] = (
                "No usable assessment on disk — click Run Assessment first, "
                "edit if needed, then Save Assessment."
            )
            configure.APP_STATE["session_status"] = "stopped"
            return result

        freq_label = configure.normalize_image_frequency(
            cfg.get("imagegen_frequency") or configure.DEFAULT_IMAGE_FREQUENCY
        )
        target_theme = configure.frequency_theme_count(freq_label)
        existing_themes = sorted(project_dir.glob("theme-*.png"))
        existing_theme_n = len(existing_themes)
        # Complete: fill remainder; already full: wipe and re-generate all
        if existing_theme_n >= target_theme:
            for old in existing_themes:
                try:
                    old.unlink()
                except OSError:
                    pass
            existing_theme_n = 0
            theme_n = target_theme
            print(f"[theme] complete set removed — re-generating {theme_n}", flush=True)
        elif existing_theme_n > 0:
            theme_n = target_theme - existing_theme_n
        else:
            theme_n = target_theme
        style = str(cfg.get("style") or configure.STYLE_DEFAULT)
        print(
            f"[theme] frequency = {freq_label} → target={target_theme} "
            f"have={existing_theme_n} generating={theme_n}",
            flush=True,
        )

        # Phase 1b: thinking model writes frequency theme prompts from OVERALL
        theme_prompts = generate_theme_prompts_from_assessment(
            analysis, cfg, theme_n,
            song_name=song_name or label,
            progress_callback=progress_callback,
        )
        if not theme_prompts:
            result["message"] = "Could not derive theme prompts from the assessment."
            configure.APP_STATE["session_status"] = "stopped"
            return result

        try:
            with open(project_dir / "theme_prompts.txt", "w", encoding="utf-8") as f:
                f.write("OVERALL\n" + (analysis.get("overall") or "") + "\n\n")
                for i, pr in enumerate(theme_prompts, 1):
                    f.write(f"=== theme {i} ===\n{pr}\n\n")
        except OSError:
            pass

        if is_cancel_requested():
            configure.APP_STATE["session_status"] = "stopped"
            result["message"] = "Cancelled during theme prompts."
            return result

        phase_barrier("text → theme images")
        if progress_callback:
            progress_callback(
                f"Generating {len(theme_prompts)} theme still(s)…",
                0.35,
                {"phase": "theme"},
            )

        images: List[Path] = []
        total = len(theme_prompts)
        # Start numbering after any existing theme-NN files so re-runs accumulate
        base_idx = 0
        for p in project_dir.glob("theme-*.png"):
            m = re.match(r"theme-(\d+)", p.name, re.I)
            if m:
                base_idx = max(base_idx, int(m.group(1)))
        # Thumbnail queue: all slots qued, then each becomes generating
        configure.APP_STATE["theme_slot_expected"] = total
        configure.APP_STATE["theme_slot_queued"] = list(range(total))
        configure.APP_STATE["theme_slot_generating"] = None
        for i, tprompt in enumerate(theme_prompts):
            if is_cancel_requested():
                break
            q = list(configure.APP_STATE.get("theme_slot_queued") or [])
            if i in q:
                q = [x for x in q if x != i]
            configure.APP_STATE["theme_slot_queued"] = q
            configure.APP_STATE["theme_slot_generating"] = i
            style_tpl = configure.prompt_template_for_style(style)
            # Theme = ambient worldspace fillers (no character, no reference image)
            filled = style_tpl.replace(
                "{line}",
                f"ambient theme {i + 1}: {(analysis.get('overall') or '')[:120]}",
            )
            final = (
                f"{filled} {tprompt} "
                "Atmospheric worldspace still with no central character portrait, "
                "no specific person, no face close-up — environment and mood only."
            ).strip()
            theme_no = base_idx + i + 1
            fname = (
                f"theme-{theme_no:02d}-"
                f"{_ascii_line_slug(song_name or label, max_len=32) or 'theme'}.png"
            )
            dest = project_dir / fname
            if progress_callback:
                progress_callback(
                    f"encoding & generating still",
                    0.35 + 0.55 * (i / max(total, 1)),
                    {"phase": "theme", "line": i + 1, "total": total},
                )
            print(f"[theme] {i + 1}/{total} — {fname} (no character / no ref)", flush=True)
            try:
                out = _generate_named_still(
                    dest, final, cfg,
                    reference_image="",
                    attach_ref=False,
                )
                images.append(out)
            except Exception as e:
                print(f"[theme] failed theme {i + 1}: {e}", flush=True)
                raise

        elapsed = time.time() - t0
        configure.APP_STATE["session_status"] = "stopped"
        configure.APP_STATE["theme_slot_generating"] = None
        configure.APP_STATE["theme_slot_queued"] = []
        result.update(
            success=bool(images),
            image_count=len(images),
            image_paths=[str(p) for p in images],
            message=(
                f"Theme images ready: {len(images)} still(s) in {project_dir.name}/ "
                f"({int(elapsed)}s)"
                if images else "No theme images were generated."
            ),
            elapsed_seconds=round(elapsed, 1),
            session_id=project_dir.name,
            project_folder=str(project_dir),
        )
        if progress_callback:
            progress_callback(result["message"], 1.0, {"phase": "done"})
        if images:
            maybe_bleep_done()
    except Exception as e:
        traceback.print_exc()
        result["message"] = f"Theme generation error: {e}"
        result["elapsed_seconds"] = round(time.time() - t0, 1)
        configure.APP_STATE["session_status"] = "stopped"
    return result




def run_materials_pipeline(
    lyrics: str,
    cfg: Dict[str, Any],
    song_name: str = "",
    reference_image: str = "",
    progress_callback: Optional[Callable] = None,
    resume_folder: str = "",
) -> Dict[str, Any]:
    """
    Full Lyrics-Materials pipeline:
      1. Parse lyrics (skip blanks / pure section markers for image targets)
      2. Project folder from user song name → output/<song_name_with_underscores>/
         (or resume an existing folder when resume_folder is set)
      3. Song + section analysis notes (skipped if analysis.txt present on resume)
      4. Per-line visual prompts (skipped if prompts.txt complete on resume)
      5. Generate stills (user-selected size, default 768×512) named "NNN - lyric line.png"
         (skips indices that already exist on disk)
    """
    clear_cancel_state()
    configure.APP_STATE["session_status"] = "running"
    t0 = time.time()
    result: Dict[str, Any] = {
        "success": False,
        "message": "",
        "project_folder": "",
        "image_count": 0,
        "image_paths": [],
        "elapsed_seconds": 0.0,
        "session_id": "",
    }

    try:
        # Resolve image size from cfg / generation defaults
        cfg = dict(cfg)
        try:
            w = int(cfg.get("imagegen_width") or configure.DEFAULT_WIDTH)
            h = int(cfg.get("imagegen_height") or configure.DEFAULT_HEIGHT)
        except (TypeError, ValueError):
            w, h = configure.DEFAULT_WIDTH, configure.DEFAULT_HEIGHT
        if w < 64 or h < 64:
            w, h = configure.DEFAULT_WIDTH, configure.DEFAULT_HEIGHT
        cfg["imagegen_width"] = w
        cfg["imagegen_height"] = h

        parsed = parse_lyrics(lyrics)
        lines = lyric_lines_only(parsed)
        if not lines:
            result["message"] = "No lyric lines found (section headers and blanks are skipped)."
            configure.APP_STATE["session_status"] = "idle"
            return result

        if progress_callback:
            progress_callback(f"Parsed {len(lines)} lyric lines", 0.02, {"phase": "parse"})

        # Folder from user-entered song name (required) — or resume existing
        label = _slugify_folder_name(
            (song_name or cfg.get("project_label") or "").strip(),
            fallback="",
        )
        if not label and not resume_folder:
            result["message"] = "Song name is required — enter a name above the lyrics."
            configure.APP_STATE["session_status"] = "idle"
            return result
        if not label and resume_folder:
            label = Path(resume_folder).name
        project_dir = ensure_project_dir(
            label, sequential=False, resume_path=resume_folder or None,
        )
        result["project_folder"] = str(project_dir)
        result["session_id"] = project_dir.name
        configure.APP_STATE["active_session_id"] = project_dir.name

        # Persist full session meta early so emergency-stop / restart can resume
        _snapshot_session(
            project_dir,
            lyrics=lyrics,
            song_name=(song_name or label).strip() or label,
            cfg=cfg,
            reference_image=reference_image or "",
            phase="analysis",
            line_count=len(lines),
        )

        try:
            (project_dir / "lyrics.txt").write_text(lyrics, encoding="utf-8")
        except OSError:
            pass
        if progress_callback:
            progress_callback(f"Project folder: output/{project_dir.name}/", 0.05, {"phase": "project"})
        print(f"[project] writing into {project_dir}", flush=True)
        maybe_bleep_section()

        has_ref = bool(reference_image and Path(reference_image).is_file())

        # ── Fail FAST on missing Flux.2 components (before 40+ prompt calls) ──
        # Text conditioning for Flux.2 uses the Encoder column (Qwen3-VL Instruct).
        # VAE must be Flux.2 autoencoder — not Flux.1 ae, not the DiT safetensors.
        _diff = (cfg.get("imagegen_model_path") or "").strip()
        _enc = (cfg.get("encoder_model_path") or "").strip()
        _vae_cfg = (cfg.get("vae_model_path") or "").strip()
        _vae = _resolve_flux2_vae(cfg, _diff) if _diff else ""
        missing = []
        if not _diff or not Path(_diff).is_file():
            missing.append(
                f"  • Diffusion model  (Configuration → ImageGen → Diffusion model)\n"
                f"    current: {_diff or '(empty)'}"
            )
        if not _enc or not Path(_enc).is_file():
            missing.append(
                f"  • Encoder (Qwen3-VL) (Configuration → Encoder → Model location)\n"
                f"    Used for prompts AND as Flux.2 --llm text conditioning\n"
                f"    current: {_enc or '(empty)'}"
            )
        if not _vae:
            missing.append(
                "  • VAE  (Configuration → ImageGen → VAE)\n"
                "    Flux.2 autoencoder (32-ch), e.g. flux2_ae.safetensors\n"
                "    or BFL vae/diffusion_pytorch_model.safetensors (~350 MB)\n"
                "    https://huggingface.co/black-forest-labs/FLUX.2-dev\n"
                "    NOT Flux.1 ae, NOT the multi-GB DiT next to the GGUF\n"
                f"    configured: {_vae_cfg or '(empty)'}"
            )
        if missing:
            msg = (
                "Image generation cannot start — model paths incomplete:\n\n"
                + "\n\n".join(missing)
                + "\n\nSave Configuration after setting paths, then run again."
            )
            print(f"[config] {msg}", flush=True)
            result["message"] = msg
            return result

        # Keep resolved VAE on cfg so Phase 2 uses the same path
        cfg["vae_model_path"] = _vae
        print(f"[config] diffusion = {Path(_diff).name}", flush=True)
        print(f"[config] encoder   = {Path(_enc).name}  (also used as Flux --llm)", flush=True)
        print(f"[config] vae       = {Path(_vae).name}", flush=True)

        if has_ref:
            local_ref = configure.ensure_project_reference(project_dir, reference_image)
            if local_ref:
                reference_image = local_ref
                print(f"[project] using reference {Path(local_ref).name}", flush=True)
            else:
                print("[project] WARNING: could not place reference in project folder", flush=True)

        # ── Phase plan (device conflict + M-Lock awareness) ──────────────
        _text_path, _text_role = _resolve_prompt_model(cfg)
        log_phase_plan(cfg, _text_role)
        if progress_callback:
            progress_callback(
                f"Phase 1: {_text_role} on {_text_role_backend(cfg, _text_role)} "
                f"({'M-Lock' if _is_mlock_enabled(cfg, 'text') else 'One-Shot'})",
                0.05,
                {"phase": "plan"},
            )

                # ── Phase 1: assessment + prompts (text model only) ──────────────
        # Shared assessment gate (Cover / Theme / Lyrics)
        analysis = ensure_project_assessment(
            project_dir,
            lyrics,
            cfg,
            has_character_ref=has_ref,
            progress_callback=progress_callback,
        )

        _snapshot_session(
            project_dir, lyrics=lyrics, song_name=song_name or label, cfg=cfg,
            reference_image=reference_image or "", phase="prompts", line_count=len(lines),
        )

        if is_cancel_requested():
            _snapshot_session(
                project_dir, lyrics=lyrics, song_name=song_name or label, cfg=cfg,
                reference_image=reference_image or "", phase="stopped", line_count=len(lines),
            )
            configure.APP_STATE["session_status"] = "stopped"
            result["message"] = "Cancelled during analysis."
            return result

        disk_prompts, disk_presence = _load_prompts_from_disk(project_dir, len(lines))
        if disk_prompts is not None:
            print(f"[prompts] RESUME — loaded {len(disk_prompts)} prompts from disk", flush=True)
            prompts, presence_per_line = disk_prompts, disk_presence or []
            if progress_callback:
                progress_callback(
                    f"Resumed {len(prompts)} prompts from disk", 0.30, {"phase": "prompts"},
                )
        else:
            # Prompts (same text model / same phase)
            prompts, presence_per_line = generate_visual_prompts(
                lines, parsed, analysis, cfg,
                has_character_ref=has_ref,
                progress_callback=progress_callback,
            )
            if is_cancel_requested():
                _snapshot_session(
                    project_dir, lyrics=lyrics, song_name=song_name or label, cfg=cfg,
                    reference_image=reference_image or "", phase="stopped", line_count=len(lines),
                )
                configure.APP_STATE["session_status"] = "stopped"
                result["message"] = "Cancelled during prompt generation."
                return result
            if not prompts:
                _snapshot_session(
                    project_dir, lyrics=lyrics, song_name=song_name or label, cfg=cfg,
                    reference_image=reference_image or "", phase="stopped", line_count=len(lines),
                )
                result["message"] = "Prompt generation produced nothing."
                configure.APP_STATE["session_status"] = "stopped"
                return result

            try:
                with open(project_dir / "prompts.txt", "w", encoding="utf-8") as f:
                    for i, (line, pr) in enumerate(zip(lines, prompts), 1):
                        pres = presence_per_line[i - 1] if i - 1 < len(presence_per_line) else "?"
                        f.write(
                            f"=== line {i} ===\n{line}\n"
                            f"character: {pres}\n"
                            f"--- prompt ---\n{pr}\n\n"
                        )
                with open(project_dir / "character_map.txt", "w", encoding="utf-8") as f:
                    f.write("section → presence (none|silhouette|partial|full)\n")
                    for lab, val in (analysis.get("character_presence") or {}).items():
                        f.write(f"{lab}: {val}\n")
                    f.write("\nper-line:\n")
                    for i, (line, pres) in enumerate(zip(lines, presence_per_line), 1):
                        f.write(f"{i:03d} [{pres}] {line}\n")
            except OSError as e:
                print(f"[project] could not write prompts/character_map: {e}", flush=True)
            maybe_bleep_section()

        _snapshot_session(
            project_dir, lyrics=lyrics, song_name=song_name or label, cfg=cfg,
            reference_image=reference_image or "", phase="images", line_count=len(lines),
        )

        # ── Barrier: unload text model before Flux ───────────────────────
        # Required when Thinking and Flux share a device under M-Lock so
        # both are never resident together. Also runs on different devices
        # for a clean hand-off.
        if progress_callback:
            progress_callback(
                "Phase 1 complete — unloading text model before Flux…",
                0.34,
                {"phase": "barrier"},
            )
        phase_barrier("text → image (unload Thinking/Encoder before Flux)")

        # ── Phase 2: image generation (Flux only) ────────────────────────
        img_backend = _imagegen_backend(cfg)
        if progress_callback:
            progress_callback(
                f"Phase 2: Flux on {img_backend}",
                0.35,
                {"phase": "images"},
            )
        print(
            f"[phase] Phase 2 start — Flux on {img_backend} "
            f"(device {_backend_device_key(img_backend)})",
            flush=True,
        )
        images = generate_images_from_prompts(
            prompts, lines, cfg,
            progress_callback=progress_callback,
            project_dir=project_dir,
            reference_image=reference_image if has_ref else "",
            presence_per_line=presence_per_line,
        )
        if is_cancel_requested():
            _snapshot_session(
                project_dir, lyrics=lyrics, song_name=song_name or label, cfg=cfg,
                reference_image=reference_image or "", phase="stopped",
                line_count=len(lines), images_done=len(images),
            )
            configure.APP_STATE["session_status"] = "stopped"
            result["message"] = "Cancelled during image generation."
            result["image_count"] = len(images)
            result["image_paths"] = [str(p) for p in images]
            result["session_id"] = project_dir.name
            return result
        if not images:
            _snapshot_session(
                project_dir, lyrics=lyrics, song_name=song_name or label, cfg=cfg,
                reference_image=reference_image or "", phase="stopped", line_count=len(lines),
            )
            configure.APP_STATE["session_status"] = "stopped"
            result["message"] = "No images were generated."
            return result

        elapsed = time.time() - t0
        phase = "done" if len(images) >= len(lines) else "stopped"
        _snapshot_session(
            project_dir, lyrics=lyrics, song_name=song_name or label, cfg=cfg,
            reference_image=reference_image or "", phase=phase,
            line_count=len(lines), images_done=len(images),
        )
        configure.APP_STATE["session_status"] = phase if phase == "done" else "stopped"
        result.update(
            success=True,
            project_folder=str(project_dir),
            image_count=len(images),
            image_paths=[str(p) for p in images],
            session_id=project_dir.name,
            message=(
                f"Materials ready: {len(images)} images in {project_dir.name}/ "
                f"({int(elapsed)}s)"
            ),
            elapsed_seconds=round(elapsed, 1),
        )
        if progress_callback:
            progress_callback(result["message"], 1.0, {"phase": "done"})
        maybe_bleep_done()

    except Exception as e:
        traceback.print_exc()
        result["message"] = f"Pipeline error: {e}"
        result["elapsed_seconds"] = round(time.time() - t0, 1)
        try:
            pf = result.get("project_folder") or configure.APP_STATE.get("current_project_folder")
            if pf:
                configure.save_session_meta(Path(pf), {"phase": "stopped"})
            configure.APP_STATE["session_status"] = "stopped"
        except Exception:
            pass

    return result
