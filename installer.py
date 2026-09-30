#!/usr/bin/env python3
"""
installer.py - Standalone setup for Lyrics-Materials.
Detects hardware, creates venv, installs Python deps, downloads (or compiles)
llama.cpp + stable-diffusion.cpp binaries for CPU or Vulkan, seeds config files,
and places ffmpeg.

Prebuilt GitHub release assets are the PRIMARY path (resumable, optimal across
CPUs via GGML_CPU_ALL_VARIANTS). Compile-from-source is available as fallback
when Git + CMake are present.
"""
from __future__ import annotations

import argparse
import configparser
import json
import math
import os
import platform
import re
import shutil
import stat
import subprocess
import sys
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Project layout
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).resolve().parent
_DATA_DIR = _ROOT / "data"
_VENV_DIR = _ROOT / "venv"
_CONST_PATH = _DATA_DIR / "constants.ini"
_CONFIG_PATH = _DATA_DIR / "configuration.json"
_PREFS_PATH = _DATA_DIR / "preferences.json"
_GENERATION_PATH = _DATA_DIR / "generation.json"
_PROMPTING_PATH = _DATA_DIR / "prompting.json"
_MODELS_DIR = _ROOT / "models"
_OUTPUT_DIR = _ROOT / "output"
_BUILD_TEMP = Path(os.environ.get("TEMP", "C:/build_temp")) / "lyrics_materials"

LLAMA_BIN_DIR = "data/llama_cpp_binaries"
SD_BIN_DIR = "data/stable_diffusion_binaries"
FFMPEG_DIR = "data/ffmpeg"
LLAMA_BIN_NAME = "llama-completion.exe"
HEAVY_THREADS = 10

REQUIREMENTS = [
    "gradio==6.19.0",
    "PyQt6==6.9.1",
    "PyQt6-WebEngine==6.9.0",
]

# Pinned upstream refs (bump together and retest)
# llama.cpp releases publish llama-{tag}-bin-win-{cpu|vulkan}-x64.zip
LLAMA_CPP_REPO = "ggml-org/llama.cpp"
LLAMA_CPP_REF = "b11100"  # recent stable-ish Windows asset set
LLAMA_CPP_SOURCE_URL = "https://github.com/ggml-org/llama.cpp.git"

# sd.cpp: tag is master-<n>-<sha>, asset embeds master-<sha>
SD_CPP_REPO = "leejet/stable-diffusion.cpp"
SD_CPP_RELEASE_TAG = "master-890-74988b2"
SD_CPP_ASSET_STEM = "master-74988b2"
SD_CPP_SOURCE_URL = "https://github.com/leejet/stable-diffusion.cpp.git"

# Downloads: no attempt limit, no socket timeouts — resume forever until complete.
RETRY_ATTEMPTS = None  # unused; kept for reference
BUILD_INACTIVITY_TIMEOUT = None  # compile builds also wait without a hard cutoff
CMAKE_CONFIGURE_TIMEOUT = None


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
def log(msg: str = "") -> None:
    print(f"  {msg}" if msg else "")


def header(title: str) -> None:
    os.system("cls" if platform.system() == "Windows" else "clear")
    print()
    print("  " + "=" * 78)
    print(f"      {title}")
    print("  " + "=" * 78)
    print()


def section(title: str) -> None:
    print()
    print(f"  {title}")
    print("  " + "-" * len(title))


class InstallAborted(Exception):
    def __init__(self, step: str, detail: str = ""):
        self.step = step
        self.detail = detail
        super().__init__(f"{step}: {detail}")


def fatal(step: str, detail: str = "") -> None:
    raise InstallAborted(step, detail)


def _safe_rmtree(path: Path) -> bool:
    def _on_error(func, fpath, exc_info):
        try:
            os.chmod(fpath, stat.S_IWRITE)
            func(fpath)
        except Exception:
            pass

    if not path.exists():
        return True
    shutil.rmtree(path, onerror=_on_error)
    return not path.exists()


def ensure_dirs() -> None:
    for d in (
        _DATA_DIR, _MODELS_DIR, _MODELS_DIR / "mmproj", _OUTPUT_DIR, _ROOT / "scripts",
        _DATA_DIR / "temp_images", _DATA_DIR / "temp_audio",
        _ROOT / LLAMA_BIN_DIR, _ROOT / SD_BIN_DIR, _ROOT / FFMPEG_DIR,
    ):
        d.mkdir(parents=True, exist_ok=True)
    init = _ROOT / "scripts" / "__init__.py"
    if not init.exists():
        init.write_text("# scripts package\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# CPU / Vulkan detection (Windows-focused, from original)
# ---------------------------------------------------------------------------
_PF_SSE3 = 13
_PF_SSE4_2 = 38
_PF_AVX = 39
_PF_AVX2 = 40
_PF_AVX512F = 41


def _has_cpu_feature(pf_const: int) -> bool:
    if platform.system() != "Windows":
        return False
    try:
        import ctypes
        fn = ctypes.windll.kernel32.IsProcessorFeaturePresent
        fn.argtypes = [ctypes.c_uint32]
        fn.restype = ctypes.c_int
        return bool(fn(ctypes.c_uint32(pf_const)))
    except Exception:
        return False


def _cpu_brand() -> str:
    if platform.system() == "Windows":
        try:
            import winreg
            with winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE,
                r"HARDWARE\DESCRIPTION\System\CentralProcessor\0",
            ) as k:
                val, _ = winreg.QueryValueEx(k, "ProcessorNameString")
                if val:
                    return str(val).strip()
        except Exception:
            pass
    return platform.processor() or "unknown"


def _physical_cores(logical: int) -> int:
    if platform.system() == "Windows":
        try:
            import ctypes
            RelationProcessorCore = 0
            k32 = ctypes.windll.kernel32
            size = ctypes.c_ulong(0)
            k32.GetLogicalProcessorInformationEx(RelationProcessorCore, None, ctypes.byref(size))
            if size.value == 0:
                return max(1, logical // 2)
            buf = (ctypes.c_byte * size.value)()
            if not k32.GetLogicalProcessorInformationEx(RelationProcessorCore, buf, ctypes.byref(size)):
                return max(1, logical // 2)
            count, off = 0, 0
            while off + 8 <= size.value:
                rel = ctypes.c_ulong.from_buffer(buf, off).value
                rec = ctypes.c_ulong.from_buffer(buf, off + 4).value
                if rec == 0:
                    break
                if rel == RelationProcessorCore:
                    count += 1
                off += rec
            return count if count > 0 else max(1, logical // 2)
        except Exception:
            pass
    return max(1, logical // 2)


def detect_cpu() -> Dict[str, Any]:
    logical = os.cpu_count() or 4
    physical = _physical_cores(logical)
    info: Dict[str, Any] = {
        "arch": "x86_64",
        "brand": _cpu_brand(),
        "vendor": "unknown",
        "cores_logical": logical,
        "cores_physical": physical,
        "default_threads": HEAVY_THREADS,
        "build_jobs": max(1, math.ceil(logical * 0.85)),
        "has_avx2": False,
        "has_avx512": False,
    }
    if platform.system() == "Windows":
        info["has_avx2"] = _has_cpu_feature(_PF_AVX2)
        info["has_avx512"] = _has_cpu_feature(_PF_AVX512F)
        n = _cpu_brand().lower()
        if any(k in n for k in ("amd", "ryzen", "epyc", "threadripper")):
            info["vendor"] = "AMD"
        elif any(k in n for k in ("intel", "xeon", "pentium", "celeron")):
            info["vendor"] = "Intel"
    return info


def detect_vulkan_presence() -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "available": False,
        "version": "unknown",
        "sdk": os.environ.get("VULKAN_SDK", ""),
        "devices": [],
        "enumerated_by": "none",
    }
    if platform.system() == "Windows":
        try:
            import ctypes
            ctypes.windll.LoadLibrary("vulkan-1.dll")
            result["available"] = True
            result["version"] = "1.x"
        except Exception:
            pass
    return result


def write_constants(
    cpu: Dict[str, Any],
    vk: Dict[str, Any],
    use_vulkan: bool = False,
    whisper_size: str = "",
    backend_method: str = "download",
) -> None:
    """
    Machine / install constants only. User-tunable settings live in the three
    JSON files (configuration.json, preferences.json, prompting.json, generation.json).
    """
    cp = configparser.ConfigParser()
    cp["cpu"] = {
        "brand": str(cpu.get("brand", "")),
        "vendor": str(cpu.get("vendor", "")),
        "cores_logical": str(cpu.get("cores_logical", 4)),
        "cores_physical": str(cpu.get("cores_physical", 2)),
        "default_threads": str(HEAVY_THREADS),
        "build_jobs": str(cpu.get("build_jobs", HEAVY_THREADS)),
        "arch": str(cpu.get("arch", "x86_64")),
        "has_avx2": str(cpu.get("has_avx2", False)).lower(),
        "has_avx512": str(cpu.get("has_avx512", False)).lower(),
    }
    cp["vulkan"] = {
        "available": str(vk.get("available", False)).lower(),
        "version": str(vk.get("version", "unknown")),
        "sdk": str(vk.get("sdk", "")),
        "enumerated_by": str(vk.get("enumerated_by", "none")),
        "device_count": str(len(vk.get("devices", []))),
    }
    # One section per ggml device — full name, VRAM, index used by -dev / --backend
    for i, d in enumerate(vk.get("devices", [])):
        sec = f"device_{i}"
        cp[sec] = {
            "backend": str(d.get("backend", "Vulkan")),
            "index": str(d.get("index", i)),
            "name": str(d.get("name", "GPU")),
            "vram_total_mb": str(d.get("vram_total_mb", 0)),
            "vram_free_mb": str(d.get("vram_free_mb", 0)),
            "fp16": str(d.get("fp16", False)).lower(),
        }
    # What this install actually provisioned (Debug tab reads these)
    cp["install"] = {
        "install_type": "vulkan" if use_vulkan else "cpu",
        "backend_method": str(backend_method),  # download | compile
        "llama_ref": LLAMA_CPP_REF,
        "sd_ref": SD_CPP_RELEASE_TAG,
        "sd_asset_stem": SD_CPP_ASSET_STEM,
        "heavy_threads": str(HEAVY_THREADS),
        "llama_bin": LLAMA_BIN_NAME,
        "sd_bin": "sd-cli.exe",
    }
    _CONST_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(_CONST_PATH, "w", encoding="utf-8") as f:
        cp.write(f)
    log(f"Wrote {_CONST_PATH}")


def write_default_configuration() -> None:
    if _CONFIG_PATH.exists():
        return
    data = {
        "encoder_model_path": "",
        "mmproj_path": "",
        "imagegen_model_path": "",
        "vae_model_path": "",
        "encoder_backend": "CPU",
        "imagegen_backend": "CPU",
        "worker_threads": 8,
        "encoder_threads": HEAVY_THREADS,
        "imagegen_threads": HEAVY_THREADS,
        "imagegen_vulkan_device": -1,
        "imagegen_placement": "Split",
        "model_load_mode": "One-Shot",
        "encoder_load_mode": "One-Shot",
        "thinking_load_mode": "One-Shot",
        "imagegen_load_mode": "One-Shot",
        "thinking_model_path": "",
        "thinking_backend": "CPU",
        "thinking_threads": HEAVY_THREADS,
        "encoder_gpu_layers": -1,
        "thinking_gpu_layers": -1,
        "imagegen_gpu_layers": -1,
        "last_model_browse_dir": "",
        "last_image_browse_dir": "",
        "last_audio_browse_dir": "",
        "window_x": -1, "window_y": -1,
        "window_width": 1280, "window_height": 900,
        "window_maximized": False,
    }
    with open(_CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    log(f"Wrote {_CONFIG_PATH}")


def write_default_preferences() -> None:
    if _PREFS_PATH.exists():
        return
    data = {
        "style": "light and bright",
        "video_format": "mp4",
        "max_thumbnails": 50,
        "input_thumbnail_size": 96,
        "bleep_section_completion": False,
        "bleep_video_completion": False,
    }
    with open(_PREFS_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    log(f"Wrote {_PREFS_PATH}")


def write_default_prompting() -> None:
    if _PROMPTING_PATH.exists():
        return
    data = {
        "light and bright": (
            "Create a bright, airy, high-key visual description for a music-video still. "
            "Emphasize soft light, clean composition, and hopeful mood. "
            "Subject of the image: {line}"
        ),
        "dark and gloomy": (
            "Create a dark, moody, low-key visual description for a music-video still. "
            "Emphasize shadows, contrast, and a sombre atmosphere. "
            "Subject of the image: {line}"
        ),
        "colorful and wild": (
            "Create a vivid, colourful, high-energy visual description for a music-video still. "
            "Emphasize saturated colours, dynamic shapes, and playful wildness. "
            "Subject of the image: {line}"
        ),
    }
    with open(_PROMPTING_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    log(f"Wrote {_PROMPTING_PATH}")


def write_default_generation() -> None:
    if _GENERATION_PATH.exists():
        return
    data = {
        "imagegen_width": 768,
        "imagegen_height": 512,
        "imagegen_steps": 4,
        "imagegen_cfg_scale": 1.0,
        "imagegen_seed": -1,
        "imagegen_sampling": "euler_a",
        "song_length_seconds": 180,
        "last_lyrics": "",
        "last_audio_path": "",
    }
    with open(_GENERATION_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    log(f"Wrote {_GENERATION_PATH}")


# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------
def _remote_content_length(url: str) -> Optional[int]:
    """Best-effort full object size via HEAD. No socket timeout — waits as long as needed."""
    try:
        req = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(req, timeout=None) as resp:
            cl = resp.headers.get("Content-Length")
            if cl and cl.isdigit():
                return int(cl)
    except Exception:
        pass
    return None


def _is_valid_zip(path: Path) -> bool:
    """True only if the file opens as a zip and its central directory is readable."""
    if not path.exists() or path.stat().st_size < 22:  # minimum empty zip size
        return False
    try:
        with zipfile.ZipFile(path, "r") as zf:
            # Force reading the central directory; BadZipFile if truncated.
            _ = zf.namelist()
            bad = zf.testzip()
            if bad is not None:
                return False
        return True
    except (zipfile.BadZipFile, OSError, EOFError):
        return False


def _download_with_retry(url: str, dest: Path, label: str) -> None:
    """
    Download to `dest` with infinite retries and no socket timeouts.
    Always RESUMES from the existing partial when present (Range requests).
    Never discards a valid complete download. Corrupt wholes are deleted and
    re-fetched; incomplete partials are kept and resumed.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    expected = _remote_content_length(url)
    if expected:
        log(f"  remote size: {expected // (1024 * 1024)} MB ({expected} bytes)")

    # Complete + valid → reuse, never re-download
    if dest.exists() and expected and dest.stat().st_size == expected:
        if dest.suffix.lower() != ".zip" or _is_valid_zip(dest):
            log(f"  cached OK -> {dest.name}")
            return
        log("  cached file is corrupt — deleting and re-downloading")
        dest.unlink(missing_ok=True)
    elif dest.exists() and dest.suffix.lower() == ".zip" and _is_valid_zip(dest):
        log(f"  cached OK (zip valid) -> {dest.name}")
        return

    attempt = 0
    while True:
        attempt += 1
        try:
            existing = dest.stat().st_size if dest.exists() else 0

            # Size matches but zip corrupt → only then discard
            if expected and existing >= expected:
                if dest.suffix.lower() == ".zip" and not _is_valid_zip(dest):
                    log("  on-disk size matches remote but zip is corrupt — restarting this file")
                    dest.unlink(missing_ok=True)
                    existing = 0
                elif dest.suffix.lower() != ".zip":
                    log(f"  cached OK -> {dest.name}")
                    return

            if expected and existing > expected:
                log("  partial larger than remote size — restarting this file")
                dest.unlink(missing_ok=True)
                existing = 0

            headers: Dict[str, str] = {
                "User-Agent": "Lyrics-Materials-Installer/1.0",
            }
            mode = "wb"
            if existing > 0:
                headers["Range"] = f"bytes={existing}-"
                mode = "ab"
                log(f"Downloading {label} (attempt {attempt}, "
                    f"resuming from {existing // (1024 * 1024)} MB)...")
            else:
                log(f"Downloading {label} (attempt {attempt})...")
            log(f"  {url}")

            req = urllib.request.Request(url, headers=headers)
            # timeout=None → no socket timeout; waits indefinitely
            with urllib.request.urlopen(req, timeout=None) as resp:
                status = getattr(resp, "status", None) or resp.getcode()
                if existing > 0 and status == 200:
                    # Server ignored Range — discard partial and retry THIS file only
                    log("  server ignored Range — restarting this file from byte 0")
                    dest.unlink(missing_ok=True)
                    raise RuntimeError("restart this file (server ignored Range)")
                if existing > 0 and status not in (206, 200):
                    raise RuntimeError(f"unexpected HTTP status {status} on resume")

                cl = resp.headers.get("Content-Length")
                body_len = int(cl) if cl and cl.isdigit() else None
                full_expected = expected
                if full_expected is None and body_len is not None:
                    full_expected = existing + body_len if status == 206 else body_len

                written = existing
                chunk = 1024 * 256
                with open(dest, mode) as out:
                    while True:
                        data = resp.read(chunk)
                        if not data:
                            break
                        out.write(data)
                        written += len(data)
                        if full_expected and full_expected > 0:
                            pct = min(100, int(100 * written / full_expected))
                            print(
                                f"\r  {pct}% ({written // (1024 * 1024)} MB / "
                                f"{full_expected // (1024 * 1024)} MB)",
                                end="", flush=True,
                            )
                        else:
                            print(
                                f"\r  {written // (1024 * 1024)} MB",
                                end="", flush=True,
                            )
                print()

            final_size = dest.stat().st_size if dest.exists() else 0
            if final_size < 1000:
                raise RuntimeError("download produced empty/tiny file — will resume/retry")

            if full_expected and final_size < full_expected:
                raise RuntimeError(
                    f"incomplete download: got {final_size} of {full_expected} bytes "
                    f"({100 * final_size // full_expected}%) — will resume"
                )
            if full_expected and final_size > full_expected:
                raise RuntimeError(
                    f"download larger than expected ({final_size} > {full_expected}) — restart this file"
                )

            if dest.suffix.lower() == ".zip":
                if not _is_valid_zip(dest):
                    raise RuntimeError(
                        "downloaded file is not a valid zip (truncated/corrupt) — will retry"
                    )

            log(f"  OK -> {dest.name} ({final_size // (1024 * 1024)} MB)")
            return

        except Exception as e:
            log(f"  failed: {e}")
            msg = str(e).lower()
            # Only delete partial when the WHOLE file is known-bad; keep incomplete for resume
            if any(k in msg for k in (
                "corrupt", "not a valid zip", "larger than expected", "tiny file", "restart this file"
            )):
                try:
                    dest.unlink(missing_ok=True)
                except OSError:
                    pass
            # Incomplete "will resume" keeps the partial on disk
            wait = min(60, 2 + attempt)
            log(f"  retrying in {wait}s (endless retries, resume when possible)...")
            time.sleep(wait)


def _extract_zip(zip_path: Path, dest_dir: Path, wanted_names: List[str]) -> List[Path]:
    """Extract .exe/.dll (and any basename in wanted_names). Aborts on bad zip."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    if not _is_valid_zip(zip_path):
        # Corrupt cache must not leave the install half-done — delete and abort.
        try:
            zip_path.unlink(missing_ok=True)
        except OSError:
            pass
        fatal(
            "Extract zip",
            f"File is not a valid zip (truncated or corrupt):\n  {zip_path}\n"
            "  The bad file was deleted. Re-run Installation to download again.",
        )
    found: List[Path] = []
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            for info in zf.infolist():
                if info.is_dir():
                    continue
                base = Path(info.filename).name
                take = base in wanted_names or base.lower().endswith((".dll", ".exe"))
                if not take:
                    continue
                target = dest_dir / base
                with zf.open(info) as src, open(target, "wb") as out:
                    shutil.copyfileobj(src, out)
                found.append(target)
                log(f"  extracted {base}")
    except zipfile.BadZipFile as e:
        try:
            zip_path.unlink(missing_ok=True)
        except OSError:
            pass
        fatal(
            "Extract zip",
            f"BadZipFile while extracting:\n  {zip_path}\n  {e}\n"
            "  The bad file was deleted. Re-run Installation to download again.",
        )
    return found


# ---------------------------------------------------------------------------
# llama.cpp prebuilt
# ---------------------------------------------------------------------------
def _llama_asset_name(use_vulkan: bool) -> str:
    kind = "vulkan" if use_vulkan else "cpu"
    return f"llama-{LLAMA_CPP_REF}-bin-win-{kind}-x64.zip"


def _llama_download_url(use_vulkan: bool) -> str:
    name = _llama_asset_name(use_vulkan)
    return f"https://github.com/{LLAMA_CPP_REPO}/releases/download/{LLAMA_CPP_REF}/{name}"


def install_llama_prebuilt(use_vulkan: bool) -> str:
    bin_dir = _ROOT / LLAMA_BIN_DIR
    bin_dir.mkdir(parents=True, exist_ok=True)
    existing = bin_dir / LLAMA_BIN_NAME
    if existing.exists():
        log(f"llama-completion already present: {existing}")
        return "already present"

    zip_name = _llama_asset_name(use_vulkan)
    cache = _DATA_DIR / "downloads" / zip_name
    url = _llama_download_url(use_vulkan)
    _download_with_retry(url, cache, f"llama.cpp {LLAMA_CPP_REF} ({'Vulkan' if use_vulkan else 'CPU'})")

    wanted = [LLAMA_BIN_NAME, "llama-cli.exe", "llama-server.exe"]
    extracted = _extract_zip(cache, bin_dir, wanted)
    # Official builds may still ship llama-cli; rename/copy if completion missing
    if not (bin_dir / LLAMA_BIN_NAME).exists():
        for alt in ("llama-cli.exe", "main.exe"):
            p = bin_dir / alt
            if p.exists():
                shutil.copy2(p, bin_dir / LLAMA_BIN_NAME)
                log(f"  note: using {alt} as llama-completion.exe fallback")
                break
    if not (bin_dir / LLAMA_BIN_NAME).exists():
        fatal("llama.cpp extract", f"{LLAMA_BIN_NAME} not found in {zip_name}")
    return f"downloaded ({'Vulkan' if use_vulkan else 'CPU'})"


# ---------------------------------------------------------------------------
# sd.cpp prebuilt
# ---------------------------------------------------------------------------
def _sd_asset_name(use_vulkan: bool) -> str:
    kind = "vulkan" if use_vulkan else "cpu"
    return f"sd-{SD_CPP_ASSET_STEM}-bin-win-{kind}-x64.zip"


def _sd_download_url(use_vulkan: bool) -> str:
    name = _sd_asset_name(use_vulkan)
    return (
        f"https://github.com/{SD_CPP_REPO}/releases/download/"
        f"{SD_CPP_RELEASE_TAG}/{name}"
    )


def install_sd_prebuilt(use_vulkan: bool) -> str:
    bin_dir = _ROOT / SD_BIN_DIR
    bin_dir.mkdir(parents=True, exist_ok=True)
    if (bin_dir / "sd-cli.exe").exists() or (bin_dir / "sd.exe").exists():
        log(f"sd-cli already present under {bin_dir}")
        return "already present"

    zip_name = _sd_asset_name(use_vulkan)
    cache = _DATA_DIR / "downloads" / zip_name
    url = _sd_download_url(use_vulkan)
    _download_with_retry(url, cache, f"sd.cpp {SD_CPP_RELEASE_TAG} ({'Vulkan' if use_vulkan else 'CPU'})")

    _extract_zip(cache, bin_dir, ["sd-cli.exe", "sd.exe", "sd-server.exe"])
    if not any((bin_dir / n).exists() for n in ("sd-cli.exe", "sd.exe")):
        fatal("sd.cpp extract", f"sd-cli.exe not found in {zip_name}")
    return f"downloaded ({'Vulkan' if use_vulkan else 'CPU'})"


# ---------------------------------------------------------------------------
# ffmpeg prebuilt (BtbN)
# Prefer the *static* gpl zip — single self-contained ffmpeg.exe / ffprobe.exe.
# The *shared* zip needs dozens of DLLs (avcodec-*.dll etc.); if we only pull the
# .exe files Windows shows "avcodec-XX.dll was not found" and probes hang.
# ---------------------------------------------------------------------------
def _ffmpeg_bin_ok(ff_dir: Path) -> bool:
    """True when ffmpeg.exe runs and prints a version line (DLLs resolved)."""
    exe = ff_dir / "ffmpeg.exe"
    if not exe.is_file():
        return False
    try:
        kwargs = dict(
            capture_output=True, text=True, encoding="utf-8",
            errors="replace", timeout=15,
        )
        if sys.platform == "win32":
            kwargs["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        r = subprocess.run([str(exe), "-hide_banner", "-version"], **kwargs)
        out = (r.stdout or "") + (r.stderr or "")
        return r.returncode == 0 and ("ffmpeg version" in out.lower() or "version" in out.lower())
    except Exception as e:
        log(f"  ffmpeg self-test failed: {e}")
        return False


def _extract_ffmpeg_zip(cache: Path, ff_dir: Path) -> int:
    """
    Extract ffmpeg.exe, ffprobe.exe, and every .dll from the archive into ff_dir.
    Returns number of files written.
    """
    n = 0
    with zipfile.ZipFile(cache, "r") as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            name = Path(info.filename).name
            low = name.lower()
            # Only top-level bin tools + their DLLs (skip include/doc/presets)
            if not (low.endswith(".exe") or low.endswith(".dll")):
                continue
            if low.endswith(".exe") and low not in ("ffmpeg.exe", "ffprobe.exe", "ffplay.exe"):
                continue
            target = ff_dir / name
            with zf.open(info) as src, open(target, "wb") as out:
                shutil.copyfileobj(src, out)
            n += 1
            log(f"  extracted {name}")
    return n


def install_ffmpeg() -> str:
    ff_dir = _ROOT / FFMPEG_DIR
    ff_dir.mkdir(parents=True, exist_ok=True)

    if _ffmpeg_bin_ok(ff_dir):
        log("ffmpeg already present and working.")
        return "already present"

    if (ff_dir / "ffmpeg.exe").exists():
        log("ffmpeg.exe present but broken (missing DLLs or won't start) — re-installing.")
        # Clear only the broken bin dir contents; keep downloads cache
        for p in list(ff_dir.iterdir()):
            try:
                if p.is_file():
                    p.unlink()
            except OSError:
                pass

    # Static first (no DLL deps), shared as fallback (we extract ALL dlls)
    candidates = [
        (
            "ffmpeg-master-latest-win64-gpl.zip",
            "https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-win64-gpl.zip",
        ),
        (
            "ffmpeg-master-latest-win64-gpl-shared.zip",
            "https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-win64-gpl-shared.zip",
        ),
    ]
    cache_dir = _DATA_DIR / "downloads"
    cache_dir.mkdir(parents=True, exist_ok=True)
    last_err = None
    for zip_name, url in candidates:
        try:
            cache = cache_dir / zip_name
            _download_with_retry(url, cache, f"ffmpeg ({zip_name})")
            written = _extract_ffmpeg_zip(cache, ff_dir)
            log(f"  extracted {written} file(s) from {zip_name}")
            if _ffmpeg_bin_ok(ff_dir):
                log("  ffmpeg self-test OK.")
                return "downloaded"
            log("  ffmpeg self-test FAILED after extract — trying next build…")
            last_err = RuntimeError(f"self-test failed for {zip_name}")
            # wipe partial extract before next candidate
            for p in list(ff_dir.iterdir()):
                try:
                    if p.is_file():
                        p.unlink()
                except OSError:
                    pass
        except InstallAborted as e:
            last_err = e
            log(f"  trying next ffmpeg mirror...")
            continue
        except Exception as e:
            last_err = e
            log(f"  ffmpeg candidate failed: {e}")
            continue
    fatal("ffmpeg download", str(last_err) if last_err else "all mirrors failed")
    return "failed"


# ---------------------------------------------------------------------------
# Tool finders (for compile path)
# ---------------------------------------------------------------------------
def find_git() -> Optional[Path]:
    g = shutil.which("git")
    return Path(g) if g else None


def find_cmake() -> Optional[Path]:
    c = shutil.which("cmake")
    if c:
        return Path(c)
    for p in (
        r"C:\Program Files\CMake\bin\cmake.exe",
        r"C:\Program Files (x86)\CMake\bin\cmake.exe",
    ):
        if Path(p).exists():
            return Path(p)
    return None


def _missing_build_tools() -> List[str]:
    missing = []
    if find_git() is None:
        missing.append("Git")
    if find_cmake() is None:
        missing.append("CMake")
    return missing


# ---------------------------------------------------------------------------
# Compile fallback (simplified from original)
# ---------------------------------------------------------------------------
def _git_clone(url: str, dest: Path, ref: str) -> None:
    git = find_git()
    if not git:
        fatal("git clone", "Git not found on PATH")
    if dest.exists():
        log(f"Source already cached: {dest}")
        # fetch + checkout
        subprocess.check_call([str(git), "-C", str(dest), "fetch", "--depth", "1", "origin", ref])
        subprocess.check_call([str(git), "-C", str(dest), "checkout", ref])
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    log(f"Cloning {url} @ {ref} ...")
    subprocess.check_call(
        [str(git), "clone", "--depth", "1", "--branch", ref, url, str(dest)],
        timeout=600,
    )


def _run_cmake_build(src: Path, build: Path, defs: List[str], jobs: int) -> bool:
    cmake = find_cmake()
    if not cmake:
        return False
    build.mkdir(parents=True, exist_ok=True)
    cfg_cmd = [str(cmake), "-S", str(src), "-B", str(build), "-DCMAKE_BUILD_TYPE=Release"] + defs
    log("cmake configure: " + " ".join(cfg_cmd))
    try:
        subprocess.check_call(cfg_cmd)
    except Exception as e:
        log(f"cmake configure failed: {e}")
        return False
    build_cmd = [str(cmake), "--build", str(build), "--config", "Release", "-j", str(jobs)]
    log("cmake build: " + " ".join(build_cmd))
    try:
        subprocess.check_call(build_cmd)
        return True
    except Exception as e:
        log(f"cmake build failed: {e}")
        return False


def compile_llama(cpu: Dict[str, Any], use_vulkan: bool) -> str:
    src = _BUILD_TEMP / "lc_src"
    build = _BUILD_TEMP / "lc_build"
    _git_clone(LLAMA_CPP_SOURCE_URL, src, LLAMA_CPP_REF)
    defs = ["-DGGML_NATIVE=ON", "-DLLAMA_BUILD_SERVER=OFF"]
    if use_vulkan:
        defs.append("-DGGML_VULKAN=ON")
    jobs = int(cpu.get("build_jobs", HEAVY_THREADS))
    if not _run_cmake_build(src, build, defs, jobs):
        fatal("Compile llama.cpp", "cmake build failed")
    bin_dir = _ROOT / LLAMA_BIN_DIR
    bin_dir.mkdir(parents=True, exist_ok=True)
    candidates = list(build.rglob("llama-completion.exe")) + list(build.rglob("llama-cli.exe"))
    if not candidates:
        fatal("Compile llama.cpp", "no llama-completion.exe produced")
    src_exe = candidates[0]
    shutil.copy2(src_exe, bin_dir / LLAMA_BIN_NAME)
    for dll in src_exe.parent.glob("*.dll"):
        shutil.copy2(dll, bin_dir / dll.name)
    return f"compiled ({'Vulkan' if use_vulkan else 'CPU'})"


def compile_sd(cpu: Dict[str, Any], use_vulkan: bool) -> str:
    src = _BUILD_TEMP / "sd_src"
    build = _BUILD_TEMP / "sd_build"
    # sd tags are not always valid git branch names; try tag then master
    try:
        _git_clone(SD_CPP_SOURCE_URL, src, SD_CPP_RELEASE_TAG)
    except Exception:
        _git_clone(SD_CPP_SOURCE_URL, src, "master")
    defs = ["-DGGML_NATIVE=ON", "-DSD_BUILD_EXAMPLES=ON"]
    if use_vulkan:
        defs.append("-DGGML_VULKAN=ON")
    jobs = int(cpu.get("build_jobs", HEAVY_THREADS))
    if not _run_cmake_build(src, build, defs, jobs):
        fatal("Compile sd.cpp", "cmake build failed")
    bin_dir = _ROOT / SD_BIN_DIR
    bin_dir.mkdir(parents=True, exist_ok=True)
    candidates = (
        list(build.rglob("sd-cli.exe")) + list(build.rglob("sd.exe"))
    )
    if not candidates:
        fatal("Compile sd.cpp", "no sd-cli.exe produced")
    src_exe = candidates[0]
    shutil.copy2(src_exe, bin_dir / src_exe.name)
    for dll in src_exe.parent.glob("*.dll"):
        shutil.copy2(dll, bin_dir / dll.name)
    return f"compiled ({'Vulkan' if use_vulkan else 'CPU'})"


# ---------------------------------------------------------------------------
# Post-build GPU enumeration (ask llama binary)
# ---------------------------------------------------------------------------
def probe_ggml_devices(exe: Path) -> List[Dict[str, Any]]:
    """
    Parse ggml device listing from llama-completion / sd-cli.

    Typical lines:
      Vulkan0: AMD Radeon RX 470 Series (8192 MiB, 7800 MiB free)
      ggml_vulkan: Found 2 Vulkan devices:
        Device 0: AMD Radeon RX 470 Series, compute unit 32, ...
    """
    devices: List[Dict[str, Any]] = []
    try:
        r = subprocess.run(
            [str(exe), "--list-devices"],
            capture_output=True, text=True, timeout=45,
            encoding="utf-8", errors="replace",
            cwd=str(exe.parent),  # DLLs next to the exe
        )
        raw = (r.stdout or "") + "\n" + (r.stderr or "")
        # Keep a copy for support / Debug
        try:
            probe_log = _DATA_DIR / "last_device_probe.txt"
            probe_log.write_text(
                f"exe: {exe}\nexit: {r.returncode}\n\n--- stdout+stderr ---\n{raw}",
                encoding="utf-8",
            )
        except OSError:
            pass

        for line in raw.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            if re.search(r"Vulkan|CUDA|Metal|Device\s+\d+", stripped, re.I):
                log(f"  [probe] {stripped}")

            backend = None
            idx = None
            name = ""
            vram_total = 0
            vram_free = 0

            # Form A: Vulkan0: Full Name (8192 MiB, 7800 MiB free)
            m = re.match(
                r"^(Vulkan|CUDA|Metal|CPU)\s*(\d+)\s*:\s*(.+)$",
                stripped, re.I,
            )
            if m:
                backend = m.group(1)
                idx = int(m.group(2))
                rest = m.group(3).strip()
                vm = re.search(
                    r"\((\d+)\s*MiB\s*(?:total\s*)?,?\s*(\d+)\s*MiB",
                    rest, re.I,
                )
                if vm:
                    vram_total = int(vm.group(1))
                    vram_free = int(vm.group(2))
                    name = rest[: vm.start()].strip().rstrip(",").strip()
                else:
                    name = rest
            else:
                # Form B: Device 0: AMD Radeon RX 470 Series, compute unit 32, ...
                m2 = re.match(
                    r"^Device\s+(\d+)\s*:\s*(.+)$",
                    stripped, re.I,
                )
                if m2:
                    idx = int(m2.group(1))
                    rest = m2.group(2).strip()
                    # Name is usually up to the first comma
                    name = rest.split(",")[0].strip()
                    backend = "Vulkan"
                    vm = re.search(r"(\d+)\s*MiB", rest, re.I)
                    if vm:
                        vram_total = int(vm.group(1))

            if backend is None or idx is None:
                continue
            if backend.upper() == "CPU":
                continue
            if not name:
                name = f"{backend}{idx}"
            devices.append({
                "backend": backend,
                "index": idx,
                "name": name,
                "vram_total_mb": vram_total,
                "vram_free_mb": vram_free,
                "fp16": False,
            })
    except Exception as e:
        log(f"  probe devices failed: {e}")

    seen = set()
    unique: List[Dict[str, Any]] = []
    for d in devices:
        key = (d["backend"].lower(), d["index"])
        if key in seen:
            continue
        seen.add(key)
        unique.append(d)
    return unique


def enumerate_devices_post_build(vk: Dict[str, Any], use_vulkan: bool = False) -> Dict[str, Any]:
    """
    Ask the on-disk binary for ggml's device list and fill vk["devices"].

    Always attempted when the binary exists (not only for Vulkan *installs*).
    CPU-only builds often report zero Vulkan devices — in that case we note it
    clearly so the user can re-run with Download Vulkan to get GPU labels in
    constants.ini / Configuration dropdowns.
    """
    vk = dict(vk)
    vk["devices"] = []
    exe = _ROOT / LLAMA_BIN_DIR / LLAMA_BIN_NAME
    sd_exe = next(
        (p for n in ("sd-cli.exe", "sd.exe")
         if (p := (_ROOT / SD_BIN_DIR / n)).exists()),
        None,
    )
    section("GPU enumeration (asking backend binary)...")

    if not exe.exists() and sd_exe is None:
        log("  No llama/sd binary found — cannot enumerate GPUs.")
        vk["enumerated_by"] = "none (binary missing)"
        return vk

    if not vk.get("available"):
        log("  Vulkan loader not detected (vulkan-1.dll) — no GPU list.")
        vk["enumerated_by"] = "none (no vulkan loader)"
        return vk

    devices: List[Dict[str, Any]] = []
    source = ""
    if exe.exists():
        devices = probe_ggml_devices(exe)
        source = f"{LLAMA_BIN_NAME} --list-devices"
    # If llama is a CPU build it often returns nothing; try sd-cli as well
    if not devices and sd_exe is not None:
        log("  llama reported no GPUs — trying sd-cli...")
        devices = probe_ggml_devices(sd_exe)
        if devices:
            source = f"{sd_exe.name} --list-devices"

    vk["devices"] = devices
    vk["enumerated_by"] = source or "none (no devices reported)"
    if devices:
        log(f"  ggml reports {len(devices)} device(s):")
        for d in devices:
            vram = ""
            if d.get("vram_total_mb"):
                vram = f"  ({d['vram_total_mb']} MiB total, {d.get('vram_free_mb', 0)} MiB free)"
            log(f"    {d['backend']}{d['index']}: {d['name']}{vram}")
    else:
        log("  ggml reports no GPU devices.")
        if not use_vulkan:
            log("  NOTE: You installed the CPU binary pack. CPU builds often")
            log("  cannot see Vulkan GPUs. Re-run Installation and choose")
            log("  'Download … for Vulkan' (or Compile Vulkan) so constants.ini")
            log("  gets full GPU names for the Configuration dropdowns.")
        else:
            log(f"  Vulkan install selected but no devices listed.")
            log(f"  SDK: {vk.get('sdk') or 'not set'}  — check GPU drivers.")
    return vk


# ---------------------------------------------------------------------------
# Venv + Python deps
# ---------------------------------------------------------------------------
def _venv_python() -> Path:
    if platform.system() == "Windows":
        return _VENV_DIR / "Scripts" / "python.exe"
    return _VENV_DIR / "bin" / "python"


def create_venv() -> None:
    if _VENV_DIR.exists() and _venv_python().exists():
        log("venv already present.")
        return
    log("Creating virtual environment...")
    subprocess.check_call([sys.executable, "-m", "venv", str(_VENV_DIR)])
    log("venv created.")


def install_deps() -> None:
    py = str(_venv_python())
    log("Upgrading pip...")
    subprocess.check_call([py, "-m", "pip", "install", "--upgrade", "pip", "wheel"])
    log("Installing Python packages...")
    for pkg in REQUIREMENTS:
        log(f"  {pkg}")
        # No timeout — large wheels / slow links must not abort
        subprocess.check_call([py, "-m", "pip", "install", pkg])
    log("Python packages OK.")


# ---------------------------------------------------------------------------
# Backend acquisition orchestration
# ---------------------------------------------------------------------------
def install_backends(cpu: Dict[str, Any], use_vulkan: bool, force_compile: bool) -> None:
    """
    Acquire llama.cpp, sd.cpp, and ffmpeg. Any failure raises InstallAborted
    and stops the install — no silent continue, no automatic compile fallback
    after a failed download (user must pick Compile explicitly if they want it).
    """
    mode = "Vulkan" if use_vulkan else "CPU"
    method = "compile" if force_compile else "download"
    section(f"Backend acquisition  ({mode}, {method})")

    log("llama.cpp...")
    if force_compile:
        status = compile_llama(cpu, use_vulkan)
    else:
        status = install_llama_prebuilt(use_vulkan)
    log(f"  llama.cpp -> {status}")

    log()
    log("stable-diffusion.cpp...")
    if force_compile:
        status = compile_sd(cpu, use_vulkan)
    else:
        status = install_sd_prebuilt(use_vulkan)
    log(f"  stable-diffusion.cpp -> {status}")

    log()
    log("ffmpeg...")
    status = install_ffmpeg()
    log(f"  ffmpeg -> {status}")


# ---------------------------------------------------------------------------
# Menus
# ---------------------------------------------------------------------------
def _print_install_banner(cpu: Dict[str, Any], vk: Dict[str, Any]) -> None:
    git, cmake = find_git(), find_cmake()
    header("Lyrics-Materials — Install Method")
    print()
    print()
    print("  System detections...")
    print(f"     Platform : Windows; Python {platform.python_version()}")
    print(f"     Build tools: Git {'OK' if git else 'NOT FOUND'}; CMake {'OK' if cmake else 'NOT FOUND'}")
    print(f"     CPU      : {cpu['brand']}")
    print(f"     Cores    : {cpu['cores_logical']} logical / {cpu['cores_physical']} physical")
    print(f"     Vulkan   : {vk['available']}  (SDK: {vk['sdk'] or 'not set'})")
    print()
    print()
    print("  " + "-" * 78)
    print()
    print()
    print("     1. Clean Install (purge venv + binaries, full setup)")
    print()
    print("     2. Check / Install (fill missing packages/backends; keeps models)")
    print()
    print("     3. Refresh Configs (re-ask CPU/Vulkan, re-probe GPUs)")
    print()
    print()
    print("  " + "=" * 78)



def _print_backend_banner(vk: Dict[str, Any], missing_tools: List[str]) -> None:
    header("Lyrics-Materials — Backend Selection")
    compile_suffix = f"   [BLOCKED - missing: {', '.join(missing_tools)}]" if missing_tools else ""
    if vk["available"]:
        print("     1. Download llama.cpp + sd.cpp for CPU")
        print("     2. Download llama.cpp + sd.cpp for Vulkan")
        print(f"     3. Compile llama.cpp + sd.cpp for CPU{compile_suffix}")
        print(f"     4. Compile llama.cpp + sd.cpp for Vulkan{compile_suffix}")
    else:
        print("     1. Download llama.cpp + sd.cpp for CPU")
        print(f"     2. Compile llama.cpp + sd.cpp for CPU{compile_suffix}")
    if missing_tools:
        print()
        print(f"  NOTE: Compile requires {', '.join(missing_tools)}.")
        print("        Use a Download option if you do not have them.")
    print()
    print("  " + "=" * 78)


def _choose_backend(vk: Dict[str, Any]) -> Tuple[bool, bool]:
    """Returns (use_vulkan, force_compile)."""
    has_vk = vk["available"]
    while True:
        missing = _missing_build_tools()
        _print_backend_banner(vk, missing)
        max_c = 4 if has_vk else 2
        choice = input(f"  Selection 1-{max_c}, Abandon = A: ").strip().upper()
        if choice == "A":
            raise SystemExit(0)
        compile_choice = choice in (("3", "4") if has_vk else ("2",))
        if compile_choice and missing:
            print(f"  Compile unavailable — missing: {', '.join(missing)}")
            input("  Press Enter...")
            continue
        if has_vk:
            if choice == "1":
                return False, False
            if choice == "2":
                return True, False
            if choice == "3":
                return False, True
            if choice == "4":
                return True, True
        else:
            if choice == "1":
                return False, False
            if choice == "2":
                return False, True
        print("  Invalid selection.")




def _purge_for_clean() -> None:
    """
    Clean Install only. Never deletes:
      - models/               (user model weights)
      - preferences / generation / prompting JSON
    """
    section("Purging previous installation...")
    targets = [
        (_VENV_DIR, "venv"),
        (_ROOT / LLAMA_BIN_DIR, "llama_cpp_binaries"),
        (_ROOT / SD_BIN_DIR, "stable_diffusion_binaries"),
        (_ROOT / FFMPEG_DIR, "ffmpeg"),
        (_BUILD_TEMP, "build_temp"),
    ]
    for target, label in targets:
        if target.exists():
            log(f"Removing {label}...")
            _safe_rmtree(target)
        else:
            log(f"{label} not present, skipping.")
    if _CONFIG_PATH.exists():
        _CONFIG_PATH.unlink()
        log("configuration.json removed.")
    if _CONST_PATH.exists():
        _CONST_PATH.unlink()
        log("constants.ini removed.")
    # preferences + generation kept (user taste / last run)


def _run_summary(t0: float) -> None:
    section("Installation summary")
    log(f"Time elapsed: {round(time.time() - t0, 1)}s")

    def _st(p: Path, what: str) -> str:
        return f"OK  {p}" if p.exists() else f"MISSING ({what})  {p}"

    log(f"constants.ini : {_st(_CONST_PATH, 'hardware')}")
    log(f"configuration : {_st(_CONFIG_PATH, 'models/backends')}")
    log(f"preferences   : {_st(_PREFS_PATH, 'style/detection')}")
    log(f"generation    : {_st(_GENERATION_PATH, 'last run params')}")
    log(f"venv python   : {_st(_venv_python(), 'python env')}")
    log(f"llama         : {_st(_ROOT / LLAMA_BIN_DIR / LLAMA_BIN_NAME, 'encoder binary')}")
    sd = _ROOT / SD_BIN_DIR
    sd_exe = next((sd / n for n in ("sd-cli.exe", "sd.exe") if (sd / n).exists()), sd / "sd-cli.exe")
    log(f"sd            : {_st(sd_exe, 'image binary')}")
    log(f"ffmpeg        : {_st(_ROOT / FFMPEG_DIR / 'ffmpeg.exe', 'video encoder')}")
    log()
    input("  Press Enter to return to the batch menu...")


def _report_abort(e: InstallAborted) -> None:
    print()
    print("=" * 78)
    print(f"  INSTALLATION FAILED — {e.step}")
    print("=" * 78)
    if e.detail:
        print(f"  {e.detail}")
    print()
    try:
        input("  Press Enter to return to the batch menu...")
    except EOFError:
        pass


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def run_detection() -> Tuple[Dict[str, Any], Dict[str, Any]]:
    section("Hardware detection...")
    cpu = detect_cpu()
    vk = detect_vulkan_presence()
    log(f"CPU  : {cpu['brand']}")
    log(f"Cores: {cpu['cores_logical']} logical / {cpu['cores_physical']} physical")
    log(f"       -> {HEAVY_THREADS} threads for heavy work")
    log(f"Vulkan: {vk['available']}  ver={vk['version']}")
    log(f"SDK   : {vk['sdk'] or 'not set'}")
    return cpu, vk


def main() -> None:
    parser = argparse.ArgumentParser(description="Lyrics-Materials Installer")
    parser.add_argument("--detect-only", action="store_true")
    args = parser.parse_args()

    ensure_dirs()
    header("Lyrics-Materials — Initialize Install")

    cpu, vk = run_detection()

    if args.detect_only:
        write_constants(cpu, vk)
        write_default_configuration()
        write_default_preferences()
        write_default_prompting()
        write_default_generation()
        log("Detection complete.")
        return

    while True:
        _print_install_banner(cpu, vk)
        choice = input("  Selection 1-3, Abandon = A: ").strip().upper()
        if choice == "A":
            return
        if choice == "1":
            t0 = time.time()
            use_vulkan, force_compile = _choose_backend(vk)
            whisper_size = ""
            method = "compile" if force_compile else "download"
            header("Lyrics-Materials — Installation")
            _purge_for_clean()
            write_constants(cpu, vk, use_vulkan=use_vulkan,
                            whisper_size=whisper_size, backend_method=method)
            section("Python virtual environment...")
            create_venv()
            section("Python dependencies...")
            install_deps()
            install_backends(cpu, use_vulkan, force_compile)
            vk = enumerate_devices_post_build(vk, use_vulkan)
            write_constants(cpu, vk, use_vulkan=use_vulkan,
                            whisper_size=whisper_size, backend_method=method)
            write_default_configuration()
            write_default_preferences()
            write_default_prompting()
            write_default_generation()
            _run_summary(t0)
            return
        if choice == "2":
            t0 = time.time()
            use_vulkan, force_compile = _choose_backend(vk)
            whisper_size = ""
            method = "compile" if force_compile else "download"
            header("Lyrics-Materials — Installation")
            write_constants(cpu, vk, use_vulkan=use_vulkan,
                            whisper_size=whisper_size, backend_method=method)
            section("Python virtual environment...")
            create_venv()
            section("Python dependencies...")
            install_deps()
            install_backends(cpu, use_vulkan, force_compile)
            vk = enumerate_devices_post_build(vk, use_vulkan)
            write_constants(cpu, vk, use_vulkan=use_vulkan,
                            whisper_size=whisper_size, backend_method=method)
            write_default_configuration()
            write_default_preferences()
            write_default_prompting()
            write_default_generation()
            _run_summary(t0)
            return
        if choice == "3":
            # Still ask backend flavour: constants.ini must reflect
            # the intended install type and GPU list, even when not re-downloading.
            t0 = time.time()
            use_vulkan, force_compile = _choose_backend(vk)
            whisper_size = ""
            method = "compile" if force_compile else "download"
            header("Lyrics-Materials — Refresh Configs")
            section("Re-detecting hardware and GPUs...")
            cpu, vk = run_detection()
            vk = enumerate_devices_post_build(vk, use_vulkan)
            write_constants(
                cpu, vk,
                use_vulkan=use_vulkan,
                whisper_size=whisper_size,
                backend_method=method,
            )
            # Only create missing JSON seeds; never wipe user settings on refresh
            write_default_configuration()
            write_default_preferences()
            write_default_prompting()
            write_default_generation()
            if not vk.get("devices") and use_vulkan:
                log("  WARNING: Vulkan selected but no GPUs were enumerated.")
                log("  Run option 2 (Check/Install) with Download Vulkan if binaries are CPU-only.")
            _run_summary(t0)
            return
        print("  Invalid selection.")


def _main_guarded() -> int:
    try:
        main()
        return 0
    except InstallAborted as e:
        _report_abort(e)
        return 1
    except KeyboardInterrupt:
        print("\n\n  Interrupted by user.")
        try:
            input("\n  Press Enter...")
        except EOFError:
            pass
        return 130
    except Exception:
        import traceback
        print()
        print("=" * 78)
        print("  INSTALLATION FAILED — unexpected error")
        print("=" * 78)
        traceback.print_exc()
        try:
            input("\n  Press Enter...")
        except EOFError:
            pass
        return 1


if __name__ == "__main__":
    sys.exit(_main_guarded())
