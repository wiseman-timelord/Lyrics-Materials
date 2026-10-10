"""
utilities.py - General helpers: build status, memory, formatting, env.
"""
from __future__ import annotations

import ctypes
import os
import platform
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

import scripts.configure as configure


def human_size(size_bytes: int) -> str:
    if size_bytes < 1024:
        return f"{size_bytes} B"
    if size_bytes < 1024 ** 2:
        return f"{size_bytes / 1024:.1f} KB"
    if size_bytes < 1024 ** 3:
        return f"{size_bytes / (1024 ** 2):.1f} MB"
    return f"{size_bytes / (1024 ** 3):.2f} GB"


def get_memory_info() -> Dict[str, Any]:
    info: Dict[str, Any] = {}
    if platform.system() != "Windows":
        return info
    try:
        class MS(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]
        mem = MS()
        mem.dwLength = ctypes.sizeof(MS)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(mem))
        total = mem.ullTotalPhys // (1024 * 1024)
        avail = mem.ullAvailPhys // (1024 * 1024)
        info = {
            "ram_total_mb": total,
            "ram_used_mb": total - avail,
            "ram_percent": mem.dwMemoryLoad,
        }
    except Exception:
        pass
    return info


def _find_exe_in_dir(directory: Path, names: List[str]) -> Optional[Path]:
    for name in names:
        p = directory / name
        if p.exists():
            return p
    return None


def get_build_status() -> Dict[str, Any]:
    """Check llama-completion and sd-cli binaries."""
    llama_bin_dir = configure.get_llama_bin_dir()
    sd_bin_dir = configure.get_sd_bin_dir()

    llama_exe = _find_exe_in_dir(
        llama_bin_dir, ["llama-completion.exe", "llama-completion"]
    )
    if not llama_exe:
        found = shutil.which("llama-completion")
        llama_exe = Path(found) if found else None

    sd_exe = _find_exe_in_dir(
        sd_bin_dir, ["sd-cli.exe", "sd-cli", "sd.exe", "sd"]
    )
    if not sd_exe:
        found = shutil.which("sd-cli") or shutil.which("sd")
        sd_exe = Path(found) if found else None

    return {
        "llama_built": llama_exe is not None,
        "llama_path": str(llama_exe) if llama_exe else "",
        "sd_built": sd_exe is not None,
        "sd_path": str(sd_exe) if sd_exe else "",
    }


def get_relevant_env() -> Dict[str, str]:
    keys = [
        "VULKAN_SDK", "VK_INSTANCE_LAYERS", "VK_LAYER_PATH",
        "GGML_VK_VISIBLE_DEVICES", "NUMBER_OF_PROCESSORS",
        "PROCESSOR_ARCHITECTURE", "PATH",
    ]
    result: Dict[str, str] = {}
    for k in keys:
        v = os.environ.get(k, "")
        if v:
            if k == "PATH" and len(v) > 200:
                v = v[:200] + "..."
            result[k] = v
    return result


def find_ffmpeg() -> Optional[Path]:
    """Locate ffmpeg.exe (bundled or PATH)."""
    candidates = [
        configure.get_project_root() / "data" / "ffmpeg" / "ffmpeg.exe",
        configure.get_project_root() / "ffmpeg" / "ffmpeg.exe",
    ]
    for c in candidates:
        if c.exists():
            return c
    found = shutil.which("ffmpeg")
    return Path(found) if found else None


def find_ffprobe() -> Optional[Path]:
    candidates = [
        configure.get_project_root() / "data" / "ffmpeg" / "ffprobe.exe",
        configure.get_project_root() / "ffmpeg" / "ffprobe.exe",
    ]
    for c in candidates:
        if c.exists():
            return c
    # Sibling of a found ffmpeg.exe
    ff = find_ffmpeg()
    if ff is not None:
        sib = ff.with_name("ffprobe.exe" if ff.suffix.lower() == ".exe" else "ffprobe")
        if sib.exists():
            return sib
    found = shutil.which("ffprobe")
    return Path(found) if found else None
