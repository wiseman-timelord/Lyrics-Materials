"""
inference.py - Lyrics → visual prompts (Qwen3-VL) → images (Flux.2-klein-4B)
→ timed slideshow video (ffmpeg + user-placed section markers on audio).
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import scripts.configure as configure
import scripts.utilities as utilities


# ---------------------------------------------------------------------------
# Completion bleeps (Windows MessageBeep / winsound)
# ---------------------------------------------------------------------------

def _play_bleep(count: int = 1) -> None:
    """Play a short system beep (count times). Best-effort; never raises."""
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


def maybe_bleep_section() -> None:
    """Bleep if 'Bleep upon section completion' is enabled in preferences."""
    prefs = configure.load_preferences()
    if prefs.get("bleep_section_completion"):
        _play_bleep(1)


def maybe_bleep_video_done() -> None:
    """
    Bleep when the final video is ready.
    If both section and video bleeps are on → double bleep; else single when video is on.
    """
    prefs = configure.load_preferences()
    section_on = bool(prefs.get("bleep_section_completion"))
    video_on = bool(prefs.get("bleep_video_completion"))
    if video_on and section_on:
        _play_bleep(2)
    elif video_on:
        _play_bleep(1)
    elif section_on:
        # Video stage is also a "section"; section-only users still get one beep
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
    print(f"[stop] {msg}", flush=True)
    return msg


# ---------------------------------------------------------------------------
# Binary finders
# ---------------------------------------------------------------------------


def _apply_worker_affinity(proc) -> None:
    """Pin a worker process to cores starting at 2 (skip 0/1 for OS/UI)."""
    if proc is None or sys.platform != "win32":
        return
    try:
        import ctypes
        mask = configure.affinity_mask()
        if mask <= 0:
            return
        handle = ctypes.windll.kernel32.OpenProcess(0x0200 | 0x0400, False, proc.pid)  # QUERY|SET
        if handle:
            ctypes.windll.kernel32.SetProcessAffinityMask(handle, mask)
            ctypes.windll.kernel32.CloseHandle(handle)
    except Exception:
        pass


def _worker_thread_count(cfg: Optional[Dict[str, Any]] = None) -> int:
    """Shared Threads Used value for -t / cpu_threads."""
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
# Backend / load-mode helpers (One-Shot vs M-Lock, GPU layers, placement)
# ---------------------------------------------------------------------------

def _load_mode_args_for(cfg: Dict[str, Any], role: str = "encoder") -> List[str]:
    """
    Map load mode → llama.cpp flags.
    Text models (encoder/thinking) share text_load_mode.
    One-Shot → default mmap; M-Lock → --mlock.

    IMPORTANT: --mlock is a llama.cpp-only flag (llama-completion). sd-cli
    (stable-diffusion.cpp) does NOT have a --mlock option at all — passing it
    makes sd-cli reject the whole command line and dump its --help usage,
    which is what a failed-line error tends to show. sd-cli's related flag is
    --mmap (opt-in memory-mapping; opposite polarity — omitting it already
    gives a fully resident load, which is what "M-Lock" is asking for), so for
    the imagegen role we deliberately never emit --mlock.
    """
    if role in ("encoder", "thinking"):
        mode = configure.normalize_load_mode(
            str(
                cfg.get("text_load_mode")
                or cfg.get("encoder_load_mode")
                or cfg.get("model_load_mode")
                or configure.DEFAULT_LOAD_MODE
            )
        )
        if mode == configure.LOAD_MODE_MLOCK:
            return ["--mlock"]
        return []
    # role == "imagegen": sd-cli has no --mlock equivalent; nothing to add
    # regardless of the saved Load Mode value.
    return []


def _gpu_layers_for(cfg: Dict[str, Any], backend: str, role: str = "encoder",
                    model_path: str = "") -> int:
    """
    Return concrete -ngl for role. CPU → 0.
    Text models share text_gpu_layers; -1 resolves via GGUF layer count
    and the selected device's safe VRAM floor (install free MiB → whole GB).
    """
    if not backend.upper().startswith("VULKAN") and not backend.upper().startswith("CUDA"):
        return 0
    if role in ("encoder", "thinking"):
        try:
            requested = int(
                cfg.get("text_gpu_layers",
                        cfg.get("encoder_gpu_layers", configure.DEFAULT_GPU_LAYERS))
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
    # role == "imagegen": sd-cli has no -ngl / --gpu-layers equivalent — Flux
    # offload is per-component (diffusion/te/vae), controlled entirely by
    # _sd_placement_args()'s --backend spec (the Placement setting). This
    # branch is not called by the image pipeline; it's kept only so an old
    # imagegen_gpu_layers config value still normalizes to an int cleanly.
    try:
        return int(cfg.get("imagegen_gpu_layers", configure.DEFAULT_GPU_LAYERS))
    except (TypeError, ValueError):
        return configure.DEFAULT_GPU_LAYERS


def _llama_backend_args(cfg: Dict[str, Any], role: str = "encoder",
                        model_path: str = "") -> List[str]:
    """
    Device + n-gpu-layers + load-mode for llama-completion.
    role=encoder → project naming; role=thinking → visual prompts (falls back).
    """
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
    """
    Prefer Thinking model for visual prompts when configured; else Encoder.
    Returns (model_path, role) where role is 'thinking' or 'encoder'.
    """
    th = (cfg.get("thinking_model_path") or "").strip()
    if th and Path(th).exists():
        return th, "thinking"
    enc = (cfg.get("encoder_model_path") or "").strip()
    return enc, "encoder"


def _sd_placement_args(cfg: Dict[str, Any], use_gpu: bool, vk_idx: int = 0) -> List[str]:
    """
    --backend / --params-backend for sd-cli.
    Gpu_Only → diffusion+te+vae on GPU; Split → diffusion GPU, te+vae CPU.
    (Load Mode's --mlock never applies here — see _load_mode_args_for.)
    """
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
# Lyrics parsing (section-aware)
# ---------------------------------------------------------------------------

_SECTION_RE = re.compile(
    r"^\[?\s*(intro|outro|fade[_\s\-]*out|chorus|verse|bridge|pre[- ]?chorus|hook|refrain)"
    r"[\s_\-]*(\d*)\s*\]?$",
    re.IGNORECASE,
)


def _normalize_section_label(kind: str, number: str) -> str:
    """Map matched header to a canonical section name."""
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
    Split pasted lyrics into ordered entries, grouping lines under section headers.

    Returns list of:
      {"type": "section", "text": str, "label": str, "index": -1}
      {"type": "line", "text": str, "index": int, "section": str}

    Recognised headers (brackets optional): Intro, Outro, Fade_Out / Fade Out,
    Chorus / Chorus_1 / Chorus 2, Verse, Bridge, Pre-Chorus, Hook, Refrain.
    Lines before any header belong to an implicit "Body" section.
    Empty / pure-whitespace lines are dropped.
    """
    entries: List[Dict[str, Any]] = []
    idx = 0
    current_section = "Body"
    for raw_line in (raw or "").splitlines():
        text = raw_line.strip()
        if not text:
            continue
        # Strip surrounding brackets for matching, but keep original for display
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
    """
    Group lyric lines under their section labels, in order of appearance.
    Returns [{"label": str, "lines": [str, ...], "line_indices": [int, ...]}, ...]
    Sections that have no lyric lines are omitted (headers alone don't count).
    """
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


def marker_plan_from_sections(sections: List[Dict[str, Any]]) -> List[str]:
    """
    Build the ordered list of marker labels for the audio timeline.

    Every section that has lyric lines gets a Start + End pair so the user
    can define exactly where that section's images play:

      Intro          → Intro Start, Intro End
      Body           → Body Start, Body End   (lines before any header)
      Chorus 1       → Chorus 1 Start, Chorus 1 End
      Verse 2        → Verse 2 Start, Verse 2 End
      Outro / Fade Out → Outro Start, Outro End  (Fade Out folds into Outro)

    Order follows the order of sections in the lyrics. Outro Start/End is
    always present so the user can close the track.
    """
    markers: List[str] = []
    seen_outro = False
    for s in sections:
        lab = s["label"]
        if lab in ("Outro", "Fade Out"):
            if not seen_outro:
                markers.extend(["Outro Start", "Outro End"])
                seen_outro = True
            continue
        # Intro, Body, Chorus N, Verse N, Bridge, Pre-Chorus, …
        markers.extend([f"{lab} Start", f"{lab} End"])
    if not seen_outro:
        markers.extend(["Outro Start", "Outro End"])
    return markers


def default_marker_times(
    marker_labels: List[str],
    song_length: float,
    gap: float = 0.5,
) -> List[Dict[str, Any]]:
    """
    Evenly space markers across [0, song_length], leaving a small gap between
    consecutive markers so sections never collapse to zero duration.
    Returns [{"label": str, "time": float}, ...].
    First * Start is forced to 0; last * End is forced to song length.
    """
    n = len(marker_labels)
    if n == 0:
        return []
    length = max(float(song_length or 1.0), 1.0)
    # Reserve a little headroom so the last marker isn't past the end
    usable = max(0.0, length - gap)
    if n == 1:
        times = [0.0]
    else:
        step = usable / (n - 1)
        times = [i * step for i in range(n)]
    # Clamp & enforce monotonic + gap
    out: List[Dict[str, Any]] = []
    prev = -gap
    for lab, t in zip(marker_labels, times):
        t = max(prev + gap, min(length, float(t)))
        out.append({"label": lab, "time": round(t, 3)})
        prev = t
    # Pin first Start to 0 and last End to song end
    if out and str(out[0]["label"]).endswith(" Start"):
        out[0]["time"] = 0.0
    if out and str(out[-1]["label"]).endswith(" End"):
        out[-1]["time"] = round(length, 3)
    return out


def markers_to_show_intervals(
    markers: List[Dict[str, Any]],
    sections: List[Dict[str, Any]],
    song_length: float,
) -> List[Tuple[float, float, List[int]]]:
    """
    Convert user markers into show intervals for the slideshow.

    Each interval is (start, end, line_indices) — images for those lyric lines
    are spread evenly across [start, end].

    Mapping (Start/End pairs for every section):
      "X Start" + "X End" → lines belonging to section X
      Outro Start/End also covers "Fade Out" lines
      Body lines map to Body Start/End (or fall back to Intro if only Intro exists)
    """
    if not markers:
        all_idx = [i for s in sections for i in s["line_indices"]]
        return [(0.0, float(song_length), all_idx)]

    by_label = {m["label"]: float(m["time"]) for m in markers}
    length = max(float(song_length or 1.0), 1.0)

    intervals: List[Tuple[float, float, List[int]]] = []

    for s in sections:
        lab = s["label"]
        if lab == "Fade Out":
            start_key, end_key = "Outro Start", "Outro End"
        else:
            start_key, end_key = f"{lab} Start", f"{lab} End"
        # Legacy fallback: Body lines can use Intro markers when no Body pair
        if lab == "Body" and (start_key not in by_label or end_key not in by_label):
            start_key, end_key = "Intro Start", "Intro End"
        if start_key not in by_label or end_key not in by_label:
            continue
        s0 = by_label[start_key]
        e0 = by_label[end_key]
        if e0 > s0 and s["line_indices"]:
            intervals.append((s0, e0, list(s["line_indices"])))

    intervals.sort(key=lambda x: x[0])
    if not intervals:
        all_idx = [i for s in sections for i in s["line_indices"]]
        intervals.append((0.0, length, all_idx))
    return intervals


# ---------------------------------------------------------------------------
# Video assembly (ffmpeg)
# ---------------------------------------------------------------------------

def _fade_color_hex(style: str) -> str:
    rgb = configure.STYLE_FADE_RGB.get(style, (0, 0, 0))
    return f"0x{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"


def assemble_slideshow(
    images: List[Path],
    song_length: float,
    show_intervals: List[Tuple[float, float, List[int]]],
    style: str,
    video_format: str,
    audio_path: Optional[str] = None,
    progress_callback: Optional[Callable] = None,
    project_dir: Optional[Path] = None,
    resolution: str = "720p",
) -> Path:
    """
    Build a timed slideshow from section Start/End markers:

    - Each interval (start, end, line_indices) receives that section's images,
      spaced evenly across [start, end].
    - Per section: fade-in from the style colour at the beginning of the
      interval, fade-out to the style colour at the end.
    - Gaps between an End and the next Start stay solid style colour
      (black / white / saturated purple). Fade durations adapt to available
      gap size and section length so short gaps stay snappy and long gaps
      can hold colour in the middle.
    - Requires an audio track for final mux (markers are relative to it).
    """
    ffmpeg = utilities.find_ffmpeg()
    if not ffmpeg:
        raise RuntimeError("ffmpeg not found. Place ffmpeg.exe in data/ffmpeg/ or PATH.")

    if not images:
        raise RuntimeError("No images to assemble.")

    out_dir = project_dir if project_dir is not None else configure.get_output_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"lyric_video_{ts}.{video_format}"
    _rw, _rh = configure.RESOLUTION_PIXELS.get(
        resolution, configure.RESOLUTION_PIXELS[configure.RESOLUTION_720P]
    )
    fade_col = _fade_color_hex(style)
    length = max(float(song_length or 1.0), 1.0)

    # Work directory for per-section clips and concat lists
    work = (project_dir if project_dir is not None else configure.get_data_dir() / "temp_images")
    work.mkdir(parents=True, exist_ok=True)
    section_clips: List[Tuple[Path, float, float, float, float]] = []
    # each: (clip_path, start, end, fade_in, fade_out)

    # Sort intervals by start time
    intervals = sorted(
        [(float(s), float(e), list(idxs)) for s, e, idxs in (show_intervals or [])
         if float(e) > float(s)],
        key=lambda x: x[0],
    )
    if not intervals:
        # Fallback: all images across the whole song
        all_idx = list(range(len(images)))
        intervals = [(0.0, length, all_idx)]

    used_indices: set = set()
    for i, (start, end, line_indices) in enumerate(intervals):
        idxs = [j for j in line_indices if 0 <= j < len(images)]
        if not idxs:
            continue
        for j in idxs:
            used_indices.add(j)

        sec_dur = max(0.05, end - start)
        # Neighbouring gaps drive how long we can fade
        prev_end = intervals[i - 1][1] if i > 0 else 0.0
        next_start = intervals[i + 1][0] if i + 1 < len(intervals) else length
        pre_gap = max(0.0, start - prev_end)
        post_gap = max(0.0, next_start - end)
        # Adaptive fade: use up to 60% of the adjacent gap, capped by section
        # length and a hard 1.5s ceiling; floor at 0.2s so there is always a
        # visible transition. When gap is zero the fade sits inside the section.
        fi = min(1.5, max(0.2, (pre_gap * 0.6) if pre_gap > 0.05 else min(0.5, sec_dur * 0.2)))
        fo = min(1.5, max(0.2, (post_gap * 0.6) if post_gap > 0.05 else min(0.5, sec_dur * 0.2)))
        # Never let fades consume the whole section
        if fi + fo > sec_dur * 0.85:
            scale = (sec_dur * 0.85) / max(fi + fo, 0.01)
            fi = max(0.15, fi * scale)
            fo = max(0.15, fo * scale)

        per = sec_dur / len(idxs)
        list_file = work / f"section_{i}_concat.txt"
        with open(list_file, "w", encoding="utf-8") as f:
            for j in idxs:
                img = images[j]
                f.write(f"file '{img.resolve().as_posix()}'\n")
                f.write(f"duration {per:.4f}\n")
            # concat demuxer requires a final entry without duration
            f.write(f"file '{images[idxs[-1]].resolve().as_posix()}'\n")

        clip_path = work / f"section_{i}.mp4"
        scale_vf = (
            f"scale={_rw}:{_rh}:force_original_aspect_ratio=decrease,"
            f"pad={_rw}:{_rh}:(ow-iw)/2:(oh-ih)/2,format=yuv420p"
        )
        # Build section clip (images only, full opacity); fades applied later
        cmd = [
            str(ffmpeg), "-y",
            "-f", "concat", "-safe", "0",
            "-i", str(list_file),
            "-vf", scale_vf,
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-r", "24",
            "-t", f"{sec_dur:.4f}",
            str(clip_path),
        ]
        if progress_callback:
            progress_callback(
                f"Encoding section {i + 1}/{len(intervals)}…",
                0.82 + 0.08 * (i / max(len(intervals), 1)),
                {"phase": "encode", "section": i + 1},
            )
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace",
        )
        register_process(proc)
        proc.communicate(timeout=180)
        unregister_process(proc)
        try:
            list_file.unlink(missing_ok=True)
        except OSError:
            pass
        if not clip_path.exists():
            print(f"[assemble] section {i} clip failed — skipping", flush=True)
            continue
        section_clips.append((clip_path, start, end, fi, fo))

    # Orphan images (never assigned to a section) → append to last section
    # if we still have one; otherwise create a short tail interval.
    orphans = [i for i in range(len(images)) if i not in used_indices]
    if orphans and section_clips:
        # Re-encode last section to include orphans at the tail (best-effort)
        # For simplicity leave orphans out of the timed show; they were already
        # generated and live in the project folder.
        print(f"[assemble] {len(orphans)} orphan image(s) not mapped to a marker interval", flush=True)

    if not section_clips:
        raise RuntimeError("No section clips could be produced.")

    # Final composite: solid style-colour background + each section overlaid
    # with internal fade-in / fade-out. Gaps between sections remain pure colour.
    if progress_callback:
        progress_callback("Compositing sections + fades…", 0.92, {"phase": "encode"})

    inputs: List[str] = []
    filter_parts: List[str] = [
        f"color=c={fade_col}:s={_rw}x{_rh}:d={length:.4f}[bg]"
    ]
    # Chain overlays: [bg][v0]overlay → [tmp0]; [tmp0][v1]overlay → …
    prev_label = "bg"
    for k, (clip, start, end, fi, fo) in enumerate(section_clips):
        inputs.extend(["-i", str(clip)])
        inp_idx = k  # 0-based input after the implicit color is filter-only
        # Input index in the command: first -i is section 0 → index 0
        sec_dur = max(0.05, end - start)
        # Fade filters run on the section's own timeline (0 … sec_dur)
        fade_f = (
            f"[{inp_idx}:v]"
            f"fade=t=in:st=0:d={fi:.3f}:color={fade_col},"
            f"fade=t=out:st={max(0.0, sec_dur - fo):.3f}:d={fo:.3f}:color={fade_col},"
            f"setpts=PTS-STARTPTS+{start:.4f}/TB[v{k}]"
        )
        filter_parts.append(fade_f)
        out_label = "v" if k == len(section_clips) - 1 else f"tmp{k}"
        ov = (
            f"[{prev_label}][v{k}]overlay=0:0:"
            f"enable='between(t\\,{start:.4f}\\,{end:.4f})'[{out_label}]"
        )
        filter_parts.append(ov)
        prev_label = out_label

    fc = ";".join(filter_parts)

    cmd2 = [str(ffmpeg), "-y"]
    cmd2.extend(inputs)
    map_args = ["-map", "[v]"]
    if audio_path and Path(audio_path).exists():
        cmd2.extend(["-i", str(audio_path)])
        # audio is the last input
        audio_idx = len(section_clips)
        map_args.extend([
            "-map", f"{audio_idx}:a",
            "-c:a", "aac", "-b:a", "192k", "-shortest",
        ])
    else:
        map_args.append("-an")

    cmd2.extend([
        "-filter_complex", fc,
        *map_args,
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-r", "24",
        "-t", f"{length:.4f}",
        str(out_path),
    ])

    if progress_callback:
        progress_callback("Final encode…", 0.95, {"phase": "encode"})

    proc = subprocess.Popen(
        cmd2, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding="utf-8", errors="replace",
    )
    register_process(proc)
    out, _ = proc.communicate(timeout=600)
    unregister_process(proc)

    if not out_path.exists():
        print((out or "")[-1200:], flush=True)
        raise RuntimeError("Final ffmpeg encode failed.")

    # Cleanup section clips
    for clip, *_ in section_clips:
        try:
            clip.unlink(missing_ok=True)
        except OSError:
            pass

    return out_path


# ---------------------------------------------------------------------------
# Text generation (llama-completion): project naming + visual prompts
# ---------------------------------------------------------------------------

_NAME_CTX = 8192          # "8k ctx" per pipeline docstring
_NAME_PREDICT = 32        # "32 tokens" per pipeline docstring
_PROMPT_CTX = 4096
_PROMPT_PREDICT = 160

_THINK_TAG_RE = re.compile(r"<think>.*?</think>", re.IGNORECASE | re.DOTALL)
_SLUG_RE = re.compile(r"[^a-z0-9]+")
_WINDOWS_RESERVED_NAMES = {
    "con", "prn", "aux", "nul",
    *(f"com{i}" for i in range(1, 10)),
    *(f"lpt{i}" for i in range(1, 10)),
}


def _strip_think_tags(text: str) -> str:
    """Drop <think>...</think> reasoning blocks the Thinking model may emit."""
    return _THINK_TAG_RE.sub("", text or "").strip()


def _run_llama_completion(
    prompt: str,
    cfg: Dict[str, Any],
    role: str,
    model_path: str,
    n_predict: int,
    ctx_size: int,
    temperature: float = 0.7,
    timeout: float = 120.0,
) -> str:
    """
    Run llama-completion once with `prompt`, return the generated continuation
    (prompt echo and <think> blocks stripped). role is 'encoder' or 'thinking'
    (selects backend/device/-ngl/load-mode via _llama_backend_args).
    """
    if is_cancel_requested():
        return ""
    exe = find_llama_completion()
    if not exe:
        raise RuntimeError("llama-completion binary not found. Run Installation.")
    if not model_path or not Path(model_path).exists():
        raise RuntimeError(f"{role.title()} model not set/found: {model_path or '(empty)'}")

    cmd = [
        str(exe),
        "-m", str(model_path),
        "-p", prompt,
        "-n", str(int(n_predict)),
        "-c", str(int(ctx_size)),
        "-t", str(_worker_thread_count(cfg)),
        "--temp", str(temperature),
    ]
    cmd.extend(_llama_backend_args(cfg, role, model_path=str(model_path)))

    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, encoding="utf-8", errors="replace",
        cwd=str(Path(exe).parent),
    )
    register_process(proc)
    try:
        out, err = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        out, err = proc.communicate()
        unregister_process(proc)
        raise RuntimeError(f"{role.title()} completion timed out after {timeout:.0f}s.")
    unregister_process(proc)

    text = (out or "").strip()
    if not text and proc.returncode not in (0, None):
        tail = (err or "").strip()[-500:]
        raise RuntimeError(f"{role.title()} completion failed (exit {proc.returncode}): {tail}")

    # Raw completion echoes the prompt before continuing — drop that prefix.
    stripped_prompt = prompt.strip()
    if stripped_prompt and text.startswith(stripped_prompt):
        text = text[len(stripped_prompt):]

    return _strip_think_tags(text)


def _slugify_folder_name(raw: str, fallback: str = "lyric_video") -> str:
    """Sanitize free-form model output into a short, filesystem-safe folder name."""
    text = (raw or "").strip()
    text = text.splitlines()[0] if text else ""  # models sometimes ramble past 32 tokens
    text = text.strip().strip('"').strip("'").strip(".")
    slug = _SLUG_RE.sub("_", text.lower()).strip("_")
    slug = re.sub(r"_+", "_", slug)
    if not slug:
        slug = fallback
    if slug in _WINDOWS_RESERVED_NAMES:
        slug = f"{slug}_project"
    return slug[:60]


def generate_project_folder_name(
    lyrics: str,
    cfg: Dict[str, Any],
    progress_callback: Optional[Callable] = None,
) -> str:
    """
    Ask the Encoder (Qwen3-VL) for a short, filesystem-safe project name
    based on the pasted lyrics. 8k context / 32-token budget — this is a
    quick naming call, not a creative-writing one. Falls back to a generic
    name on any failure so a naming hiccup never aborts the whole pipeline.
    """
    if progress_callback:
        progress_callback("Naming project…", 0.03, {"phase": "project"})

    model_path = (cfg.get("encoder_model_path") or "").strip()
    excerpt = "\n".join((lyrics or "").splitlines()[:12])[:1200]
    prompt = (
        "You are naming a folder for a music video project. "
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
            temperature=0.6, timeout=60.0,
        )
    except Exception as e:
        print(f"[project] naming failed, using fallback: {e}", flush=True)
        raw = ""

    return _slugify_folder_name(raw)


def ensure_project_dir(folder_name: str) -> Path:
    """
    Create (and return) a unique project directory under output/ for this run.
    Appends _2, _3, … on collision. Records it in APP_STATE so the UI can show
    the live gallery / "open project folder" while the pipeline is running.
    """
    base = configure.get_output_dir()
    base.mkdir(parents=True, exist_ok=True)
    name = folder_name or "lyric_video"
    candidate = base / name
    n = 2
    while candidate.exists():
        candidate = base / f"{name}_{n}"
        n += 1
    candidate.mkdir(parents=True, exist_ok=True)
    configure.APP_STATE["current_project_folder"] = str(candidate)
    return candidate


def generate_visual_prompts(
    lines: List[str],
    cfg: Dict[str, Any],
    progress_callback: Optional[Callable] = None,
) -> List[str]:
    """
    One visual-description prompt per lyric line, via the Thinking model when
    configured (richer prompts) else the Encoder — see _resolve_prompt_model().
    Uses the active style's editable template (cfg["prompt_template"], from
    prompting.json) with {line} substitution. A line that fails to generate
    falls back to the raw lyric text so every line still gets an image prompt.
    """
    model_path, role = _resolve_prompt_model(cfg)
    if not model_path:
        raise RuntimeError("No Encoder/Thinking model configured for visual prompts.")

    template = (
        cfg.get("prompt_template")
        or configure.STYLE_PROMPT_TEMPLATES[configure.STYLE_LIGHT]
    )

    prompts: List[str] = []
    total = len(lines)
    for i, line in enumerate(lines):
        if is_cancel_requested():
            break
        if progress_callback:
            progress_callback(
                f"Writing visual prompt {i + 1}/{total}…",
                0.05 + 0.25 * (i / max(total, 1)),
                {"phase": "prompts", "line": i + 1, "total": total},
            )
        instruction = template.replace("{line}", line) if "{line}" in template else f"{template}\n{line}"
        full_prompt = (
            f"{instruction}\n\n"
            "Reply with ONLY the visual description, one paragraph, no preamble, no quotes."
        )
        try:
            text = _run_llama_completion(
                full_prompt, cfg, role=role, model_path=model_path,
                n_predict=_PROMPT_PREDICT, ctx_size=_PROMPT_CTX,
                temperature=0.8, timeout=120.0,
            )
        except Exception as e:
            print(f"[prompts] line {i + 1} failed, using lyric line as fallback: {e}", flush=True)
            text = ""
        prompts.append(text.strip() or line)

    return prompts


def _run_sd_cli_once(cmd: List[str], exe: Path, timeout: float = 300.0) -> str:
    print("[sd-cli] " + " ".join(str(c) for c in cmd), flush=True)
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding="utf-8", errors="replace",
        cwd=str(exe.parent),
    )
    register_process(proc)
    try:
        out, _ = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        out, _ = proc.communicate()
    unregister_process(proc)
    return out or ""


def generate_images_from_prompts(
    prompts: List[str],
    cfg: Dict[str, Any],
    progress_callback: Optional[Callable] = None,
    project_dir: Optional[Path] = None,
) -> List[Path]:
    """
    Generate one still per visual prompt with FLUX.2-klein-4B (sd-cli), saved
    into project_dir as slide_001.png, slide_002.png, … in prompt order (the
    order assemble_slideshow / markers_to_show_intervals rely on). A failed
    line is retried once; if it still fails the pipeline aborts rather than
    silently shifting every later slide out of alignment with its lyric line.
    """
    exe = find_sd_cpp()
    if not exe:
        raise RuntimeError("sd-cli binary not found. Run Installation.")

    model_path = (cfg.get("imagegen_model_path") or "").strip()
    if not model_path or not Path(model_path).exists():
        raise RuntimeError(f"Diffuser model not set/found: {model_path or '(empty)'}")
    vae_path = (cfg.get("vae_model_path") or "").strip()

    out_dir = project_dir if project_dir is not None else configure.get_output_dir()
    out_dir.mkdir(parents=True, exist_ok=True)

    backend = str(cfg.get("imagegen_backend") or "CPU")
    use_gpu = backend.upper().startswith(("VULKAN", "CUDA"))
    vk_idx = 0
    if use_gpu:
        m = re.search(r"(\d+)", backend)
        vk_idx = int(m.group(1)) if m else 0

    width = int(cfg.get("imagegen_width") or configure.DEFAULT_WIDTH)
    height = int(cfg.get("imagegen_height") or configure.DEFAULT_HEIGHT)
    steps = int(cfg.get("imagegen_steps") or configure.DEFAULT_STEPS)
    cfg_scale = float(cfg.get("imagegen_cfg_scale") or configure.DEFAULT_CFG)
    sampler = str(cfg.get("imagegen_sampling") or configure.DEFAULT_SAMPLER)
    seed_val = cfg.get("imagegen_seed")
    seed = int(seed_val) if seed_val not in (None, "") else configure.DEFAULT_SEED
    threads = _worker_thread_count(cfg)

    images: List[Path] = []
    total = len(prompts)
    for i, prompt in enumerate(prompts):
        if is_cancel_requested():
            break
        if progress_callback:
            progress_callback(
                f"Generating image {i + 1}/{total}…",
                0.30 + 0.50 * (i / max(total, 1)),
                {"phase": "images", "line": i + 1, "total": total},
            )
        img_path = out_dir / f"slide_{i + 1:03d}.png"
        cmd = [
            str(exe),
            "--diffusion-model", str(model_path),
            "-p", prompt,
            "-o", str(img_path),
            "-W", str(width), "-H", str(height),
            "--steps", str(steps),
            "--cfg-scale", str(cfg_scale),
            "--sampling-method", sampler,
            "-s", str(seed),
            "-t", str(threads),
        ]
        if vae_path and Path(vae_path).exists():
            cmd.extend(["--vae", str(vae_path)])
        cmd.extend(_sd_placement_args(cfg, use_gpu, vk_idx))

        out = _run_sd_cli_once(cmd, exe)
        if not img_path.exists():
            print(f"[images] line {i + 1} failed — retrying once…", flush=True)
            out = _run_sd_cli_once(cmd, exe)
        if not img_path.exists():
            cmd_str = " ".join(str(c) for c in cmd)
            text = out or "(no output captured)"
            # sd-cli prints its real error first, then (on a bad flag) a long
            # --help dump. Showing only a small tail can land mid-way through
            # that dump and hide the actual error line, so show the head
            # (where the real error is) and the tail (final status) together.
            if len(text) > 2000:
                detail = text[:1200] + "\n…[truncated]…\n" + text[-800:]
            else:
                detail = text
            raise RuntimeError(
                f"Image generation failed for line {i + 1}/{total}.\n"
                f"Command: {cmd_str}\n"
                f"sd-cli output:\n{detail}"
            )

        images.append(img_path)

    return images


# ---------------------------------------------------------------------------
# Top-level pipeline
# ---------------------------------------------------------------------------

def run_lyric_video_pipeline(
    lyrics: str,
    song_length_seconds: float,
    audio_path: str,
    cfg: Dict[str, Any],
    progress_callback: Optional[Callable] = None,
) -> Dict[str, Any]:
    """
    Full pipeline:
      1. Name project folder via Qwen (8k ctx / 32 tokens)
      2. Line-by-line visual prompts → save prompts.txt in project folder
      3. Generate images into project folder
      4. Apply user section markers (Intro / Chorus / Outro) on the audio
      5. Assemble video into project folder at chosen resolution
    """
    clear_cancel_state()
    t0 = time.time()
    result: Dict[str, Any] = {
        "success": False,
        "message": "",
        "video_path": "",
        "project_folder": "",
        "image_count": 0,
        "elapsed_seconds": 0.0,
    }

    try:
        parsed = parse_lyrics(lyrics)
        lines = lyric_lines_only(parsed)
        if not lines:
            result["message"] = "No lyric lines found."
            return result

        if progress_callback:
            progress_callback(f"Parsed {len(lines)} lyric lines", 0.02, {"phase": "parse"})

        # 0. Project folder
        folder_name = generate_project_folder_name(lyrics, cfg, progress_callback)
        project_dir = ensure_project_dir(folder_name)
        result["project_folder"] = str(project_dir)
        # Save original lyrics + markers snapshot
        try:
            (project_dir / "lyrics.txt").write_text(lyrics, encoding="utf-8")
        except OSError:
            pass
        try:
            markers_snap = cfg.get("markers") or []
            if markers_snap:
                import json as _json
                (project_dir / "markers.json").write_text(
                    _json.dumps(markers_snap, indent=2), encoding="utf-8"
                )
        except OSError:
            pass
        if progress_callback:
            progress_callback(
                f"Project folder: {folder_name}",
                0.05,
                {"phase": "project"},
            )
        maybe_bleep_section()  # section: project named

        # 1. Visual prompts
        prompts = generate_visual_prompts(lines, cfg, progress_callback)
        if is_cancel_requested():
            result["message"] = "Cancelled during prompt generation."
            return result
        if not prompts:
            result["message"] = "Prompt generation produced nothing."
            return result

        # Save prompts alongside lyrics
        try:
            with open(project_dir / "prompts.txt", "w", encoding="utf-8") as f:
                for i, (line, pr) in enumerate(zip(lines, prompts), 1):
                    f.write(f"=== line {i} ===\n{line}\n--- prompt ---\n{pr}\n\n")
        except OSError as e:
            print(f"[project] could not write prompts.txt: {e}", flush=True)
        maybe_bleep_section()  # section: prompts generated

        # 2. Images into project folder
        images = generate_images_from_prompts(
            prompts, cfg, progress_callback, project_dir=project_dir
        )
        if is_cancel_requested():
            result["message"] = "Cancelled during image generation."
            return result
        if not images:
            result["message"] = "No images were generated."
            return result
        result["image_count"] = len(images)
        maybe_bleep_section()  # section: all images generated

        # 3. Timing from user-placed section markers (required audio)
        if not audio_path or not Path(audio_path).exists():
            result["message"] = "Audio file is required — attach a track and set section markers."
            return result
        markers = cfg.get("markers") or []
        if not markers:
            result["message"] = "No section markers set — place Intro / Chorus / Outro markers on the timeline."
            return result
        sections = sections_with_lines(parsed)
        intervals = markers_to_show_intervals(markers, sections, song_length_seconds)
        if progress_callback:
            progress_callback(
                f"Markers: {len(markers)} → {len(intervals)} section interval(s)",
                0.82,
                {"phase": "timing"},
            )

        # 4. Assemble video into project folder
        style = cfg.get("style", configure.STYLE_LIGHT)
        vfmt = cfg.get("video_format", configure.VIDEO_MP4)
        resolution = cfg.get("output_resolution", configure.RESOLUTION_720P)
        video_path = assemble_slideshow(
            images,
            song_length_seconds,
            intervals,
            style,
            vfmt,
            audio_path,
            progress_callback,
            project_dir=project_dir,
            resolution=resolution,
        )

        elapsed = time.time() - t0
        result.update(
            success=True,
            video_path=str(video_path),
            project_folder=str(project_dir),
            message=(
                f"Video saved: {video_path.name} in {folder_name}/ "
                f"({len(images)} images, {int(elapsed)}s)"
            ),
            elapsed_seconds=round(elapsed, 1),
        )
        if progress_callback:
            progress_callback(result["message"], 1.0, {"phase": "done"})
        maybe_bleep_video_done()  # video complete (double if both prefs on)

    except Exception as e:
        traceback.print_exc()
        result["message"] = f"Pipeline error: {e}"
        result["elapsed_seconds"] = round(time.time() - t0, 1)

    return result