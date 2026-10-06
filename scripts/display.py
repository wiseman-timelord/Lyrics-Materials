"""
display.py - Gradio 6 UI for Lyrics-Materials.
Tabs: Generation | Prompting | Configuration | Preferences | Debug / Info
"""
from __future__ import annotations

import functools
import json
import os
import queue
import re
from pathlib import Path
import html
import time
import shutil
import subprocess
import sys
import threading
import time
import traceback
from typing import Any, Dict, List, Optional, Tuple

import gradio as gr

import scripts.configure as configure
import scripts.inference as inference
import scripts.utilities as utilities

_exit_handler = None


def set_exit_handler(handler) -> None:
    global _exit_handler
    _exit_handler = handler


def _handle_exit_click() -> None:
    if _exit_handler is not None:
        _exit_handler()
    else:
        os._exit(0)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cfg() -> Dict[str, Any]:
    return configure.load_configuration()


def _prefs() -> Dict[str, Any]:
    return configure.load_preferences()


def _gcfg() -> Dict[str, Any]:
    return configure.load_generation()


def _prompting() -> Dict[str, Any]:
    return configure.load_prompting()


_FILETYPES_MODEL = [
    ("Model files", "*.gguf *.safetensors"),
    ("GGUF", "*.gguf"),
    ("Safetensors", "*.safetensors"),
    ("All files", "*.*"),
]

_FILETYPES_IMAGE = [
    ("Images", "*.png *.jpg *.jpeg *.webp *.bmp"),
    ("All files", "*.*"),
]


def _browse_file(file_types=None, initial_key="last_model_browse_dir") -> str:
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        cfg = _cfg()
        last = cfg.get(initial_key, "")
        initial = str(configure.get_models_dir())
        if last:
            p = Path(last)
            if not p.is_absolute():
                p = configure.get_project_root() / p
            if p.is_dir():
                initial = str(p)
            elif p.parent.is_dir():
                initial = str(p.parent)
        path = filedialog.askopenfilename(
            initialdir=initial,
            filetypes=file_types or _FILETYPES_MODEL,
        )
        root.destroy()
        if path:
            configure.update_configuration({initial_key: str(Path(path).parent)})
            return path
        return ""
    except Exception:
        return ""


def _open_output_folder() -> str:
    out_dir = configure.get_output_dir()
    try:
        if sys.platform == "win32":
            subprocess.Popen(["explorer", str(out_dir)])
        else:
            subprocess.Popen(["xdg-open", str(out_dir)])
        return f"Opened: {out_dir}"
    except Exception as e:
        return f"ERROR: {e}"


def _open_project_folder() -> str:
    folder = (
        configure.APP_STATE.get("current_project_folder")
        or _gcfg().get("last_project_folder")
        or ""
    )
    if folder and Path(folder).is_dir():
        try:
            if sys.platform == "win32":
                subprocess.Popen(["explorer", str(folder)])
            else:
                subprocess.Popen(["xdg-open", str(folder)])
            return f"Opened project folder: {folder}"
        except Exception as e:
            return f"ERROR: {e}"
    return _open_output_folder()


def _models_configured() -> bool:
    cfg = _cfg()
    enc = (cfg.get("encoder_model_path") or "").strip()
    diff = (cfg.get("imagegen_model_path") or "").strip()
    # VAE is validated at generate time (Flux.2 resolver); not required to unlock Generate
    if not enc or not Path(enc).expanduser().exists():
        return False
    if not diff or not Path(diff).expanduser().exists():
        return False
    return True


def _assessment_path(project_dir: str = "") -> Optional[Path]:
    folder = (project_dir or configure.APP_STATE.get("current_project_folder") or "").strip()
    if not folder:
        return None
    return Path(folder) / "analysis.txt"


def _assessment_exists(project_dir: str = "") -> bool:
    """True when analysis.txt exists and has usable content."""
    path = _assessment_path(project_dir)
    if path is None or not path.is_file():
        return False
    try:
        raw = path.read_text(encoding="utf-8", errors="replace").strip()
    except OSError:
        return False
    if len(raw) < 40:
        return False
    # Prefer structured content; accept any substantial text the user saved
    upper = raw.upper()
    if "OVERALL" in upper or "SECTIONS" in upper or "CHARACTER" in upper:
        return True
    return len(raw) >= 80


def _can_run_assessment(lyrics: str = "", song_name: str = "") -> bool:
    if not _models_configured():
        return False
    if not (song_name or "").strip():
        return False
    if not (lyrics or "").strip():
        return False
    return True


def _can_create(lyrics: str, song_name: str = "") -> bool:
    """Full lyrics slideshow: needs assessment + song name + lyrics + models."""
    if not (lyrics or "").strip():
        return False
    if not (song_name or "").strip():
        return False
    if not _models_configured():
        return False
    return _assessment_exists()


def _can_cover(song_name: str = "") -> bool:
    """Cover image: needs assessment + song name + models."""
    if not (song_name or "").strip():
        return False
    if not _models_configured():
        return False
    return _assessment_exists()


def _can_theme(lyrics: str = "", song_name: str = "") -> bool:
    """Theme images: needs assessment + lyrics + song name + models."""
    if not (lyrics or "").strip():
        return False
    if not (song_name or "").strip():
        return False
    if not _models_configured():
        return False
    return _assessment_exists()


def _current_freq_label() -> str:
    try:
        g = configure.load_generation()
        return configure.normalize_image_frequency(
            g.get("imagegen_frequency") or configure.DEFAULT_IMAGE_FREQUENCY
        )
    except Exception:
        return configure.DEFAULT_IMAGE_FREQUENCY


def _lyrics_expected_stills(lyrics: str = "") -> int:
    """Expected numbered still count = lyric lines × L from frequency preset."""
    n_lines = 0
    try:
        if (lyrics or "").strip():
            from scripts.inference import parse_lyrics, lyric_lines_only
            n_lines = len(lyric_lines_only(parse_lyrics(lyrics)))
        if n_lines <= 0:
            n_lines = int(configure.APP_STATE.get("thumb_expected_count") or 0)
    except Exception:
        n_lines = int(configure.APP_STATE.get("thumb_expected_count") or 0)
    if n_lines <= 0:
        return 0
    per = configure.frequency_lyrics_per_line(_current_freq_label())
    return n_lines * max(1, per)


def _lyrics_have_stills(project_dir: str = "") -> int:
    """Count numbered lyric stills on disk (all variants)."""
    return len(_list_project_images(project_dir))


def _cover_counts() -> tuple:
    """(have, expected) for cover stills under current frequency.

    `have` counts only slots within the frequency expected range (cover-01 …
    cover-N) so raising frequency correctly reports incomplete when higher
    slots are missing.
    """
    folder = (configure.APP_STATE.get("current_project_folder") or "").strip()
    expected = max(1, configure.frequency_cover_count(_current_freq_label()))
    if not folder or not Path(folder).is_dir():
        return 0, expected
    smap = _named_slot_map("cover", folder)
    have = sum(1 for i in range(expected) if smap.get(i))
    return have, expected


def _theme_counts() -> tuple:
    """(have, expected) for theme stills under current frequency."""
    folder = (configure.APP_STATE.get("current_project_folder") or "").strip()
    expected = max(1, configure.frequency_theme_count(_current_freq_label()))
    if not folder or not Path(folder).is_dir():
        return 0, expected
    smap = _named_slot_map("theme", folder)
    have = sum(1 for i in range(expected) if smap.get(i))
    return have, expected


def _lyrics_counts(lyrics: str = "") -> tuple:
    """(have, expected) for lyric stills under current frequency."""
    folder = (configure.APP_STATE.get("current_project_folder") or "").strip()
    expected = _lyrics_expected_stills(lyrics)
    if expected <= 0:
        return 0, 0
    if not folder or not Path(folder).is_dir():
        return 0, expected
    return _lyrics_have_stills(folder), expected


def _partial_project_state(lyrics: str = "") -> bool:
    """True when the active project has SOME but not ALL lyric stills (incl. L2/L3)."""
    have, expected = _lyrics_counts(lyrics)
    return expected > 0 and 0 < have < expected


def _partial_cover_state() -> bool:
    have, expected = _cover_counts()
    return 0 < have < expected


def _partial_theme_state() -> bool:
    have, expected = _theme_counts()
    return 0 < have < expected


def _cover_complete() -> bool:
    have, expected = _cover_counts()
    return expected > 0 and have >= expected


def _theme_complete() -> bool:
    have, expected = _theme_counts()
    return expected > 0 and have >= expected


def _lyrics_complete(lyrics: str = "") -> bool:
    have, expected = _lyrics_counts(lyrics)
    return expected > 0 and have >= expected


def _all_assets_complete(lyrics: str = "") -> bool:
    """True only when Cover + Theme + Lyrics all meet the current Image Frequency."""
    return (
        _cover_complete()
        and _theme_complete()
        and _lyrics_complete(lyrics)
    )


def _lyrics_btn_label(lyrics: str = "") -> str:
    if _lyrics_complete(lyrics):
        return "Re-Generate All Lyrics Images"
    if _partial_project_state(lyrics):
        return "Complete Lyrics Images"
    return "Generate Lyrics Images"


def _cover_btn_label() -> str:
    if _cover_complete():
        return "Re-Generate All Cover Images"
    if _partial_cover_state():
        return "Complete Cover Images"
    return "Generate Cover Images"


def _theme_btn_label() -> str:
    if _theme_complete():
        return "Re-Generate All Theme Images"
    if _partial_theme_state():
        return "Complete Theme Images"
    return "Generate Theme Images"


def _all_assets_btn_label(lyrics: str = "") -> str:
    """Re-Generate only when every type is complete for the current frequency."""
    if _all_assets_complete(lyrics):
        return "Re-Generate All Assets"
    return "Generate All Assets"


def _project_has_any_images(project_dir: str = "") -> bool:
    """True when the active (or given) project has any Cover / Theme / Lyrics stills."""
    folder = (project_dir or configure.APP_STATE.get("current_project_folder") or "").strip()
    if not folder or not Path(folder).is_dir():
        return False
    if _list_cover_images(folder) or _list_theme_images(folder) or _list_project_images(folder):
        return True
    return False


def _action_btn_updates(lyrics: str = "", song_name: str = "", *, running: bool = False):
    """
    Button visibility:

      Models not set     → only assess_btn as "Configure the Pages First" (disabled)
      Models set, no assess → only "Run Assessment" (enabled when name+lyrics)
      Assessment saved   → "Run Assessment" + All Assets / Cover / Theme / Lyrics

    "Generate All Assets" when assessment exists and assets are incomplete.
    "Re-Generate All Assets" only when Cover + Theme + Lyrics all meet frequency.

    When running: hide action buttons, show Emergency Stop.

    Returns (assess_u, all_assets_u, cover_u, theme_u, lyrics_u, stop_u).
    """
    hide = gr.update(visible=False)
    if running:
        return hide, hide, hide, hide, hide, gr.update(visible=True)

    models_ok = _models_configured()
    if not models_ok:
        return (
            gr.update(
                value="Configure the Pages First",
                interactive=False,
                variant="secondary",
                visible=True,
            ),
            hide, hide, hide, hide,
            gr.update(visible=False),
        )

    has_assess = _assessment_exists()
    can_assess = _can_run_assessment(lyrics, song_name)
    assess_u = gr.update(
        value="Re-run Assessment" if has_assess else "Run Assessment",
        interactive=can_assess,
        variant="primary" if can_assess else "secondary",
        visible=True,
    )

    if not has_assess:
        return (
            assess_u,
            hide, hide, hide, hide,
            gr.update(visible=False),
        )

    cover_ok = _can_cover(song_name)
    theme_ok = _can_theme(lyrics, song_name)
    lyrics_ok = _can_create(lyrics, song_name)
    all_ok = cover_ok or theme_ok or lyrics_ok

    all_assets_u = gr.update(
        value=_all_assets_btn_label(lyrics),
        interactive=all_ok,
        variant="primary" if all_ok else "secondary",
        visible=True,
    )
    cover_u = gr.update(
        value=_cover_btn_label(),
        interactive=cover_ok,
        variant="primary" if cover_ok else "secondary",
        visible=True,
    )
    theme_u = gr.update(
        value=_theme_btn_label(),
        interactive=theme_ok,
        variant="primary" if theme_ok else "secondary",
        visible=True,
    )
    lyrics_u = gr.update(
        value=_lyrics_btn_label(lyrics),
        interactive=lyrics_ok,
        variant="primary" if lyrics_ok else "secondary",
        visible=True,
    )
    return assess_u, all_assets_u, cover_u, theme_u, lyrics_u, gr.update(visible=False)


def _run_btn_updates(lyrics: str = "", song_name: str = ""):
    """Back-compat: return only the lyrics-slideshow button update."""
    *_, lyrics_u, _ = _action_btn_updates(lyrics, song_name, running=False)
    return lyrics_u


def _run_btn_for_project(proj: str) -> Any:
    """Lyrics-slideshow button update for the active project."""
    lyrics = ""
    try:
        lp = Path(proj) / "lyrics.txt"
        if lp.is_file():
            lyrics = lp.read_text(encoding="utf-8", errors="replace")
    except OSError:
        pass
    return _run_btn_updates(lyrics, Path(proj).name if proj else "")


def _action_btns_for_project(proj: str):
    """Full action-button updates for a project folder."""
    lyrics = ""
    try:
        lp = Path(proj) / "lyrics.txt"
        if lp.is_file():
            lyrics = lp.read_text(encoding="utf-8", errors="replace")
    except OSError:
        pass
    return _action_btn_updates(lyrics, Path(proj).name if proj else "", running=False)


def _lyrics_line_count(text: str) -> int:
    """Visible lines for Song Lyrics while focused (full content)."""
    n = len((text or "").splitlines()) or 1
    # +2 breathing room; clamp so the UI cannot explode
    return max(12, min(120, n + 2))


def _last_still_seconds() -> float:
    try:
        return float(
            configure.APP_STATE.get("last_image_gen_seconds")
            or configure.load_generation().get("last_image_gen_seconds")
            or 0
        )
    except (TypeError, ValueError):
        return 0.0


def _current_still_elapsed() -> float:
    t0 = configure.APP_STATE.get("image_gen_t0")
    if not t0:
        return 0.0
    try:
        return max(0.0, time.time() - float(t0))
    except (TypeError, ValueError):
        return 0.0


def _format_progress_line(msg: str, frac: float, info: dict) -> str:
    """
    e.g. [ 45%] cover Image 1/1 Generating cover still…30s (est 90s)

    Previous still duration is a soft estimate only — never a wait gate.
    When the still finishes early, image_gen_t0 is cleared and the next
    pipeline frac advances immediately; the global last_* is updated to
    the actual shorter duration.
    """
    phase = str((info or {}).get("phase", "") or "")
    last = _last_still_seconds()
    elapsed = _current_still_elapsed()
    image_phases = ("images", "regen", "theme", "cover")
    f = float(frac or 0.0)
    # Soft ETA only while a still is actively generating (image_gen_t0 set).
    # Cap at 0.99 so we never look "done" before the worker reports completion.
    # Do NOT hold or sleep until elapsed reaches last — finish as soon as the binary exits.
    if phase in image_phases and last > 1.0 and elapsed > 0:
        f = min(0.99, max(f, elapsed / last))
    pct = int(round(max(0.0, min(1.0, f)) * 100))
    bits = [f"[{pct:3d}%]"]
    if phase and phase not in ("done",):
        bits.append(phase)
    line_n = (info or {}).get("line")
    total = (info or {}).get("total")
    msg_s = (msg or "").replace("\n", " ").strip()
    already_counted = bool(
        re.match(r"(?i)^(theme|image|cover|line)\s*\d+\s*/\s*\d+", msg_s)
        or re.search(r"(?i)\b(theme|image)\s+\d+\s*/\s*\d+", msg_s)
    )
    if line_n is not None and total is not None and not already_counted:
        bits.append(f"Image {int(line_n)}/{int(total)}")
    if line_n is not None and total is not None:
        msg_s = re.sub(
            rf"(?i)^(theme|image|cover|line)\s*{int(line_n)}\s*/\s*{int(total)}\s*[:.\-–—]?\s*",
            "",
            msg_s,
        ).strip()
    if msg_s:
        bits.append(msg_s)
    body = " ".join(bits).strip()
    if elapsed > 0:
        # Actively generating — show live seconds + soft estimate from previous still
        body = body.rstrip(".…")
        body = f"{body}…{int(elapsed)}s"
        if last > 0:
            body = f"{body} (est {int(last)}s)"
    elif phase in image_phases and last > 0 and phase not in ("done",):
        # Between stills — show last completed duration as reference only
        body = f"{body} (last {int(last)}s)"
    return body[:240]


def _status_from_progress(msg: str, frac: float, info: dict) -> str:
    """Plain status text with timers (no visual progress bar)."""
    return _format_progress_line(msg, frac, info or {})


def _status_plain(msg: str, frac: float = 0.0) -> str:
    return (msg or "Ready.").replace("\n", " ").strip()[:240]


def _load_assessment_text(project_dir: str = "") -> str:
    """Read analysis.txt for the active (or given) project for the Assessment panel."""
    folder = (project_dir or configure.APP_STATE.get("current_project_folder") or "").strip()
    if not folder:
        return "(No project folder — generate Cover / Theme / Lyrics first.)"
    path = Path(folder) / "analysis.txt"
    if not path.is_file():
        return "(No assessment yet — click Run Assessment with song name + lyrics.)"
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError as e:
        return f"(Could not read assessment: {e})"



# ---------------------------------------------------------------------------
# Generation tab
# ---------------------------------------------------------------------------

_gen: Dict[str, Any] = {}

# Max session rows rendered in the sidebar (Gradio needs fixed component count)
_MAX_SESSION_SLOTS = 24


def _session_choices_payload() -> List[Dict[str, Any]]:
    """Normalized session list for UI rendering."""
    return configure.list_sessions()


# Max still slots in the Generation thumb grid (4 columns).
THUMB_SLOTS = 80
THUMB_COLS = 8
# Cover / Theme gallery slot budgets (separate sections)
COVER_SLOTS = 8
THEME_SLOTS = 32


def _thumb_size_px() -> int:
    """Configured thumbnail height from Preferences (default 96)."""
    try:
        prefs = _prefs()
        return int(prefs.get("input_thumbnail_size", configure.DEFAULT_INPUT_THUMBNAIL) or 96)
    except Exception:
        return int(getattr(configure, "DEFAULT_INPUT_THUMBNAIL", 96) or 96)


def _gallery_height_for(n_images: int) -> int:
    """Dynamic materials panel height from thumb size + regen button row."""
    th = _thumb_size_px()
    cell_h = th + 40  # image + Regenerate button + gap
    if n_images <= 0:
        return cell_h + 16
    rows = max(1, (n_images + THUMB_COLS - 1) // THUMB_COLS)
    return min(1600, max(cell_h + 16, rows * cell_h + 24))


def _is_numbered_still(path: str | Path) -> bool:
    """True only for materials stills named like 001-… / 001 - …. Excludes reference.* and in-progress partials."""
    name = Path(path).name
    low = name.lower()
    if low.startswith("reference"):
        return False
    if ".__partial__" in low:
        return False
    return bool(re.match(r"^\d{3}(?:\s*[-–—]|[-.])", name) or re.match(r"^\d{3}\.", name) or re.match(r"^\d{3}$", Path(path).stem))


_PLACEHOLDER_WARNED: set = set()

# kind -> base stem under images/ (user-facing spelling "qued" is intentional).
# Actual files are thumbnails_{stem}_{regular|wide}.jpg
_PLACEHOLDER_STEMS = {
    "no_image": "thumbnails_no_image",
    "queued": "thumbnails_qued_for_generation",
    "generating": "thumbnails_generating",
}


def _current_thumb_aspect() -> str:
    """regular | wide from the Generation image size setting."""
    try:
        g = configure.load_generation()
        size = g.get("imagegen_size") or ""
        if not size:
            w = int(g.get("imagegen_width") or configure.DEFAULT_WIDTH)
            h = int(g.get("imagegen_height") or configure.DEFAULT_HEIGHT)
            size = configure.image_size_label_from_wh(w, h)
        return configure.image_size_aspect(str(size))
    except Exception:
        return configure.IMAGE_SIZE_ASPECT_REGULAR


def _placeholder_thumb(kind: str, aspect: str = "") -> Optional[str]:
    """Absolute path to images/thumbnails_*_{regular|wide}.jpg."""
    stem = _PLACEHOLDER_STEMS.get(kind, f"thumbnails_{kind}")
    asp = (aspect or _current_thumb_aspect() or "regular").strip().lower()
    if asp not in ("regular", "wide"):
        asp = "regular"
    names = [
        f"{stem}_{asp}.jpg",
        f"{stem}_{'wide' if asp == 'regular' else 'regular'}.jpg",
        f"{stem}.jpg",
    ]
    roots = []
    try:
        roots.append(configure.get_images_dir())
    except Exception:
        pass
    try:
        root = Path(configure._get_project_root())
        roots.append(root / "images")
        roots.append(root)
    except Exception:
        pass
    roots.append(Path("images"))
    candidates = []
    for r in roots:
        for n in names:
            candidates.append(r / n)
    for p in candidates:
        try:
            if p.is_file():
                return str(p.resolve())
        except Exception:
            continue
    warn_key = f"{stem}_{asp}"
    if warn_key not in _PLACEHOLDER_WARNED:
        _PLACEHOLDER_WARNED.add(warn_key)
        tried = ", ".join(str(c) for c in candidates[:4])
        print(f"  WARNING: placeholder not found: {warn_key} (tried {tried})", flush=True)
    return None


def _queued_lines() -> set:
    """1-based line numbers currently listed for generation (not started yet)."""
    out: set = set()
    for x in configure.APP_STATE.get("thumb_queued_lines") or []:
        try:
            out.add(int(x))
        except (TypeError, ValueError):
            pass
    return out


def _unqueue_line(line_no_1based: int) -> None:
    """Remove a line from the queued list (it is starting, or finished)."""
    try:
        n = int(line_no_1based)
    except (TypeError, ValueError):
        return
    configure.APP_STATE["thumb_queued_lines"] = sorted(_queued_lines() - {n})


def _list_project_images(project_dir: str = "") -> List[str]:
    """Sorted numbered still paths for the active project (never includes reference.*)."""
    paths: List[str] = []
    folder = (project_dir or configure.APP_STATE.get("current_project_folder") or "").strip()
    if folder and Path(folder).is_dir():
        for p in Path(folder).iterdir():
            if not p.is_file():
                continue
            if p.suffix.lower() not in (".png", ".jpg", ".jpeg", ".webp"):
                continue
            if _is_numbered_still(p):
                paths.append(str(p))
    if not paths:
        for p in configure.APP_STATE.get("generation_output_paths") or []:
            if p and Path(p).exists() and _is_numbered_still(p):
                paths.append(str(p))
    seen = set()
    out: List[str] = []
    for p in paths:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out



def _project_folder(project_dir: str = "") -> str:
    return (project_dir or configure.APP_STATE.get("current_project_folder") or "").strip()


def _list_cover_images(project_dir: str = "") -> List[str]:
    """Cover stills: cover*.png/jpg (not reference, not in-progress partials)."""
    folder = _project_folder(project_dir)
    if not folder or not Path(folder).is_dir():
        return []
    out: List[str] = []
    for p in sorted(Path(folder).iterdir(), key=lambda x: x.name.lower()):
        if not p.is_file():
            continue
        if p.suffix.lower() not in (".png", ".jpg", ".jpeg", ".webp"):
            continue
        low = p.name.lower()
        if ".__partial__" in low:
            continue
        if low.startswith("cover"):
            out.append(str(p.resolve()))
    return out


def _list_theme_images(project_dir: str = "") -> List[str]:
    """Theme stills: theme-*.png/jpg (not in-progress partials)."""
    folder = _project_folder(project_dir)
    if not folder or not Path(folder).is_dir():
        return []
    out: List[str] = []
    for p in sorted(Path(folder).iterdir(), key=lambda x: x.name.lower()):
        if not p.is_file():
            continue
        if p.suffix.lower() not in (".png", ".jpg", ".jpeg", ".webp"):
            continue
        low = p.name.lower()
        if ".__partial__" in low:
            continue
        if low.startswith("theme"):
            out.append(str(p.resolve()))
    return out


def _named_slot_map(kind: str, project_dir: str = "") -> Dict[int, str]:
    """0-based slot index → path, keyed by cover-NN / theme-NN in the filename.

    Using the number in the filename (not the live sorted-list index) keeps
    neighbouring slots stable when one file is briefly missing or replaced.
    """
    kind = (kind or "").strip().lower()
    files = _list_cover_images(project_dir) if kind == "cover" else _list_theme_images(project_dir)
    prefix = "cover" if kind == "cover" else "theme"
    out: Dict[int, str] = {}
    unmatched: List[str] = []
    for p in files:
        m = re.match(rf"^{prefix}-(\d+)", Path(p).name, re.I)
        if m:
            idx = int(m.group(1)) - 1
            if idx >= 0:
                out[idx] = p
        else:
            unmatched.append(p)
    used = set(out.keys())
    slot = 0
    for p in unmatched:
        while slot in used:
            slot += 1
        out[slot] = p
        used.add(slot)
        slot += 1
    return out


def _slot_aligned_named_paths(
    kind: str,
    project_dir: str = "",
    n_slots: int = 8,
    extra_slots: Optional[set] = None,
) -> List[str]:
    """Dense list index==slot, empty string when that numbered still is absent."""
    m = _named_slot_map(kind, project_dir)
    extra = extra_slots or set()
    hi = -1
    if m:
        hi = max(m.keys())
    if extra:
        hi = max(hi, max(extra))
    n_show = min(n_slots, max(hi + 1, 0))
    if n_show <= 0:
        return []
    arr = [""] * n_show
    for i, p in m.items():
        if 0 <= i < n_show:
            arr[i] = p
    return arr


def _lyrics_section_should_show(paths: Optional[List[str]] = None) -> bool:
    """True when the Lyrics Thumbnails section should be visible."""
    if configure.APP_STATE.get("generating") and configure.APP_STATE.get("thumb_queued_lines"):
        return True
    if configure.APP_STATE.get("thumb_generating_line") is not None:
        return True
    if int(configure.APP_STATE.get("thumb_expected_count") or 0) > 0:
        # Only show if we actually have lyric stills or an active lyrics run queued
        by_line = _line_path_map(paths)
        if by_line:
            return True
        if configure.APP_STATE.get("thumb_queued_lines"):
            return True
        return False
    by_line = _line_path_map(paths)
    return bool(by_line)



def _parse_still_line_variant(name: str) -> Tuple[int, int]:
    """
    Parse numbered still filename into (1-based line, 1-based variant).

    Patterns:
      001-lyric-slug.png      → (1, 1)   # L1 / legacy
      001-2-lyric-slug.png    → (1, 2)   # L2/L3 variant 2
      001 - lyric line.png    → (1, 1)   # space-dash legacy
    Returns (0, 0) when not a numbered still.
    """
    stem = Path(name or "").name
    m = re.match(r"^(\d{1,4})(?:\s*[-–—]\s*|[-.])(.*)$", stem)
    if not m:
        m2 = re.match(r"^(\d{1,4})\.", stem)
        if m2:
            return int(m2.group(1)), 1
        return 0, 0
    line_no = int(m.group(1))
    rest = (m.group(2) or "").lstrip()
    # Variant form: NNN-V-slug  (rest starts with digit then dash)
    vm = re.match(r"^(\d+)[-–—](.+)$", rest)
    if vm:
        return line_no, max(1, int(vm.group(1)))
    return line_no, 1


def _line_path_map(paths: Optional[List[str]] = None, variant: int = 0) -> Dict[int, str]:
    """
    Map 1-based line number → absolute still path.

    variant:
      0  → any/latest path per line (legacy single-map behaviour)
      N  → path for that sequence variant only (L2/L3 page N)
    """
    if paths is None:
        paths = _list_project_images()
    out: Dict[int, str] = {}
    # Prefer higher mtime when multiple match the same key
    mtimes: Dict[int, float] = {}
    want = int(variant or 0)
    for p in paths:
        if not p or not Path(p).is_file():
            continue
        line_no, var = _parse_still_line_variant(Path(p).name)
        if line_no <= 0:
            continue
        if want > 0 and var != want:
            continue
        try:
            mt = Path(p).stat().st_mtime
        except OSError:
            mt = 0.0
        if line_no not in out or mt >= mtimes.get(line_no, 0.0):
            out[line_no] = str(Path(p).resolve())
            mtimes[line_no] = mt
    return out


def _lyrics_per_line_count() -> int:
    """Current Image Frequency L value (stills per lyric line)."""
    try:
        return max(1, int(configure.frequency_lyrics_per_line(_current_freq_label())))
    except Exception:
        return 1


def _lyrics_page_current() -> int:
    """1-based variant page currently shown in Lyrics Thumbnails."""
    per = _lyrics_per_line_count()
    try:
        page = int(configure.APP_STATE.get("lyrics_page") or 1)
    except (TypeError, ValueError):
        page = 1
    return max(1, min(page, per))


def _set_lyrics_page(page: int) -> int:
    per = _lyrics_per_line_count()
    try:
        p = int(page or 1)
    except (TypeError, ValueError):
        p = 1
    p = max(1, min(p, per))
    configure.APP_STATE["lyrics_page"] = p
    return p


def _lyrics_page_choices() -> List[str]:
    """Labels for the Lyrics Image page switcher (Page 1 … Page N)."""
    per = _lyrics_per_line_count()
    return [f"Page {i}" for i in range(1, per + 1)]


def _lyrics_page_label(page: int = 0) -> str:
    p = int(page) if page else _lyrics_page_current()
    return f"Page {max(1, p)}"


def _thumb_expected_count() -> int:
    """How many lyric-line slots to show (line_count only — not × L), capped at THUMB_SLOTS.

    Multi-variant stills (L2/L3) are paged via the Lyrics Image page switcher;
    each page still shows one slot per lyric line.
    """
    n = int(configure.APP_STATE.get("thumb_expected_count") or 0)
    if n <= 0:
        # Fall back to highest existing still line number
        m = _line_path_map()
        if m:
            n = max(m.keys())
    return max(0, min(int(n), THUMB_SLOTS))


def _named_busy_set(kind: str = "") -> set:
    """Slot indices currently generating/regenerating for kind ('cover'|'theme')."""
    out: set = set()
    prefix = f"{kind}:" if kind else ""
    for x in (configure.APP_STATE.get("named_regen_busy") or []):
        s = str(x)
        if prefix and s.startswith(prefix):
            try:
                out.add(int(s.split(":", 1)[1]))
            except (TypeError, ValueError, IndexError):
                pass
        elif not prefix:
            out.add(s)
    # Batch generation current slot
    if kind in ("cover", "theme"):
        cur = configure.APP_STATE.get(f"{kind}_slot_generating")
        if cur is not None:
            try:
                out.add(int(cur))
            except (TypeError, ValueError):
                pass
    return out


def _named_queued_set(kind: str) -> set:
    """Slot indices waiting (batch queue or named_regen_queue) for this kind."""
    out: set = set()
    for job in (configure.APP_STATE.get("named_regen_queue") or []):
        try:
            if str(job.get("kind") or "") != kind:
                continue
            out.add(int(job.get("slot_idx", -1)))
        except (TypeError, ValueError):
            pass
    if kind in ("cover", "theme"):
        for x in (configure.APP_STATE.get(f"{kind}_slot_queued") or []):
            try:
                out.add(int(x))
            except (TypeError, ValueError):
                pass
    return out


def _simple_grid_updates(
    paths: List[str],
    n_slots: int,
    rows_key: str,
    cols_key: str,
    imgs_key: str,
    regen_key: str = "",
    remove_key: str = "",
    *,
    kind: str = "",
    expected: int = 0,
) -> List[Any]:
    """
    Cover/Theme grid updates with the same placeholder stages as Lyrics:
      real still → image path
      busy (regenerating) → thumbnails_generating.jpg
      queued → thumbnails_qued_for_generation.jpg
      empty expected slot → thumbnails_no_image.jpg
    """
    th = _thumb_size_px()
    n_rows = len(_gen.get(rows_key) or []) or max(1, (n_slots + THUMB_COLS - 1) // THUMB_COLS)
    # Frequency drives slot count: slim when expected drops, expand with no_image when it rises.
    # Caller passes expected from the current Image Frequency (busy/queued may raise it).
    exp = int(expected or 0)
    if exp > 0:
        n_show = min(exp, n_slots)
    else:
        n_show = min(len(paths), n_slots)
    has_btns = bool(regen_key and remove_key and (_gen.get(regen_key) or []))
    busy_idx = _named_busy_set(kind) if kind else set()
    queued_idx = _named_queued_set(kind) if kind else set()
    no_img = _placeholder_thumb("no_image")
    que_img = _placeholder_thumb("queued") or no_img
    gen_img = _placeholder_thumb("generating") or no_img

    updates: List[Any] = []
    for r in range(n_rows):
        row_start = r * THUMB_COLS
        updates.append(gr.update(visible=row_start < n_show))
    for i in range(n_slots):
        if i < n_show:
            real = paths[i] if i < len(paths) and paths[i] else None
            is_busy = i in busy_idx
            is_queued = (i in queued_idx) and not is_busy
            if is_busy:
                value = gen_img or real or no_img
            elif is_queued:
                value = que_img or no_img
            elif real:
                value = real
            else:
                value = no_img
            updates.append(gr.update(visible=True))
            updates.append(gr.update(value=value, height=th))
            if has_btns:
                updates.append(gr.update(
                    visible=True,
                    interactive=not is_busy and not is_queued,
                    value="…" if (is_busy or is_queued) else "Regenerate",
                ))
                updates.append(gr.update(
                    visible=True,
                    interactive=bool(real) and not is_busy,
                ))
        else:
            updates.append(gr.update(visible=False))
            updates.append(gr.update(value=None))
            if has_btns:
                updates.append(gr.update(visible=False, value="Regenerate"))
                updates.append(gr.update(visible=False))
    return updates


def _cover_panel_updates(project_dir: str = "") -> List[Any]:
    """Cover grid sized to the *current* Image Frequency (slim or expand).

    Frequency is the display authority: raise frequency → extra empty slots with
    no_image placeholders; lower frequency → hide higher slots (files stay on disk).
    During active generation/regen, busy/queued slots may temporarily expand the grid.
    """
    busy = _named_busy_set("cover")
    queued = _named_queued_set("cover")
    expected = 0
    try:
        expected = int(configure.APP_STATE.get("cover_slot_expected") or 0)
    except (TypeError, ValueError):
        expected = 0
    # Prefer live frequency unless a batch run explicitly pinned a higher target
    try:
        freq_n = int(configure.frequency_cover_count(_current_freq_label()))
    except Exception:
        freq_n = 0
    if expected <= 0:
        expected = freq_n
    else:
        # If user raised/lowered frequency after a run, follow the dropdown
        expected = max(freq_n, expected) if (busy or queued) else freq_n
    expected = max(0, min(int(expected), COVER_SLOTS))
    paths = _slot_aligned_named_paths(
        "cover", project_dir, COVER_SLOTS, extra_slots=busy | queued,
    )
    # Show when we have stills, pending work, or frequency asks for slots on an active project
    show = bool(any(paths)) or bool(busy) or bool(queued)
    if not show and expected > 0 and _assessment_exists():
        show = True
    # Slim to frequency expected; only expand for busy/queued beyond that
    n_expected = expected if show else 0
    if busy or queued:
        show = True
        n_expected = max(n_expected, max(busy | queued) + 1)
    n_expected = min(n_expected, COVER_SLOTS)
    return [gr.update(visible=show)] + _simple_grid_updates(
        paths, COVER_SLOTS, "cover_rows", "cover_cols", "cover_imgs",
        "cover_regen_btns", "cover_remove_btns",
        kind="cover", expected=n_expected if show else 0,
    )


def _theme_panel_updates(project_dir: str = "") -> List[Any]:
    """Theme grid sized to the *current* Image Frequency (slim or expand)."""
    busy = _named_busy_set("theme")
    queued = _named_queued_set("theme")
    expected = 0
    try:
        expected = int(configure.APP_STATE.get("theme_slot_expected") or 0)
    except (TypeError, ValueError):
        expected = 0
    try:
        freq_n = int(configure.frequency_theme_count(_current_freq_label()))
    except Exception:
        freq_n = 0
    if expected <= 0:
        expected = freq_n
    else:
        expected = max(freq_n, expected) if (busy or queued) else freq_n
    expected = max(0, min(int(expected), THEME_SLOTS))
    paths = _slot_aligned_named_paths(
        "theme", project_dir, THEME_SLOTS, extra_slots=busy | queued,
    )
    show = bool(any(paths)) or bool(busy) or bool(queued)
    if not show and expected > 0 and _assessment_exists():
        show = True
    n_expected = expected if show else 0
    if busy or queued:
        show = True
        n_expected = max(n_expected, max(busy | queued) + 1)
    n_expected = min(n_expected, THEME_SLOTS)
    return [gr.update(visible=show)] + _simple_grid_updates(
        paths, THEME_SLOTS, "theme_rows", "theme_cols", "theme_imgs",
        "theme_regen_btns", "theme_remove_btns",
        kind="theme", expected=n_expected if show else 0,
    )


def _thumb_panel_updates(paths: Optional[List[str]] = None) -> List[Any]:
    """
    Gradio updates for Lyrics Thumbnails grid (slot i = lyric line i+1):
      - has still on disk for the *current page variant* → that image
      - line is generating / regenerating now → thumbnails_generating.jpg
      - line is listed for generation, not started → thumbnails_qued_for_generation.jpg
      - expected but missing / removed, nothing pending → thumbnails_no_image.jpg

    When Image Frequency is L2 or L3, a page switcher (Page 1 / Page 2 / Page 3)
    sits under the Lyrics Thumbnails heading; each page shows that variant for
    every lyric line. L1 keeps the switcher hidden.
    Section is hidden when there are no lyrics stills and nothing queued.
    """
    page = _lyrics_page_current()
    per = _lyrics_per_line_count()
    # Clamp stored page if frequency dropped
    if page > per:
        page = _set_lyrics_page(per)
    by_line = _line_path_map(paths, variant=page if per > 1 else 0)
    # Any-variant map for section visibility / expected line count
    by_line_any = _line_path_map(paths, variant=0) if per > 1 else by_line
    expected = _thumb_expected_count()
    if by_line_any:
        expected = max(expected, min(max(by_line_any.keys()), THUMB_SLOTS))
    expected = min(expected, THUMB_SLOTS)
    show_section = _lyrics_section_should_show(paths)

    busy_raw = configure.APP_STATE.get("regen_busy_lines") or []
    busy: set = set()
    for x in busy_raw:
        try:
            busy.add(int(x))
        except (TypeError, ValueError):
            pass
    cur = configure.APP_STATE.get("thumb_generating_line")
    if cur is not None:
        try:
            busy.add(int(cur))
        except (TypeError, ValueError):
            pass

    queued = _queued_lines()
    no_img = _placeholder_thumb("no_image")
    que_img = _placeholder_thumb("queued") or no_img
    gen_img = _placeholder_thumb("generating") or no_img
    th = _thumb_size_px()
    n_rows = len(_gen.get("thumb_rows") or []) or max(1, (THUMB_SLOTS + THUMB_COLS - 1) // THUMB_COLS)

    # Panel + page switcher (visible only when L>1 and section shown)
    show_pages = bool(show_section and per > 1)
    choices = _lyrics_page_choices()
    updates: List[Any] = [
        gr.update(visible=show_section),
        gr.update(
            visible=show_pages,
            choices=choices,
            value=_lyrics_page_label(page) if show_pages else None,
        ),
    ]
    for r in range(n_rows):
        row_start = r * THUMB_COLS
        row_has = show_section and row_start < expected
        updates.append(gr.update(visible=row_has))

    for i in range(THUMB_SLOTS):
        line_no = i + 1
        if not show_section or line_no > expected:
            updates.append(gr.update(visible=False))
            updates.append(gr.update(value=None))
            updates.append(gr.update(visible=False, value="Regenerate"))
            updates.append(gr.update(visible=False))
            continue

        real = by_line.get(line_no)
        is_busy = line_no in busy
        is_queued = (line_no in queued) and not is_busy
        if is_busy:
            value = gen_img or no_img
        elif is_queued:
            value = que_img or no_img
        elif real:
            value = real
        else:
            value = no_img

        updates.append(gr.update(visible=True))
        updates.append(gr.update(value=value, height=th))
        can_regen = not is_busy and not is_queued
        can_remove = bool(real) and not is_busy
        updates.append(gr.update(
            visible=True,
            interactive=can_regen,
            value="…" if (is_busy or is_queued) else "Regenerate",
        ))
        updates.append(gr.update(visible=True, interactive=can_remove))
    return updates


def _all_gallery_updates(project_dir: str = "", lyric_paths: Optional[List[str]] = None) -> List[Any]:
    """Cover + Theme + Lyrics panel updates in a fixed order."""
    folder = _project_folder(project_dir)
    if lyric_paths is None:
        lyric_paths = _list_project_images(folder)
    return (
        _cover_panel_updates(folder)
        + _theme_panel_updates(folder)
        + _thumb_panel_updates(lyric_paths)
    )


def _thumb_panel_outputs() -> List[Any]:
    """Ordered list of Gradio components for Lyrics + Cover + Theme gallery updates.

    Order matches _all_gallery_updates:
      cover panel, cover rows/cols/imgs,
      theme panel, theme rows/cols/imgs,
      lyrics panel, lyrics rows + slots.
    """
    outs: List[Any] = []
    # Cover
    if _gen.get("cover_panel") is not None:
        outs.append(_gen["cover_panel"])
    for row in (_gen.get("cover_rows") or []):
        outs.append(row)
    for i in range(COVER_SLOTS):
        outs.append(_gen["cover_cols"][i])
        outs.append(_gen["cover_imgs"][i])
        if _gen.get("cover_regen_btns"):
            outs.append(_gen["cover_regen_btns"][i])
            outs.append(_gen["cover_remove_btns"][i])
    # Theme
    if _gen.get("theme_panel") is not None:
        outs.append(_gen["theme_panel"])
    for row in (_gen.get("theme_rows") or []):
        outs.append(row)
    for i in range(THEME_SLOTS):
        outs.append(_gen["theme_cols"][i])
        outs.append(_gen["theme_imgs"][i])
        if _gen.get("theme_regen_btns"):
            outs.append(_gen["theme_regen_btns"][i])
            outs.append(_gen["theme_remove_btns"][i])
    # Lyrics
    if _gen.get("thumbs_panel") is not None:
        outs.append(_gen["thumbs_panel"])
    if _gen.get("lyrics_page_radio") is not None:
        outs.append(_gen["lyrics_page_radio"])
    for row in (_gen.get("thumb_rows") or []):
        outs.append(row)
    for i in range(THUMB_SLOTS):
        outs.append(_gen["thumb_cols"][i])
        outs.append(_gen["thumb_imgs"][i])
        outs.append(_gen["thumb_btns"][i])
        outs.append(_gen["thumb_remove_btns"][i])
    return outs



def _session_row_label(s: Dict[str, Any], expanded: bool = True, index: int = 0) -> str:
    """Expanded: full folder label + badge. Compact: plain 1-based index."""
    if not expanded:
        return str(index + 1)
    badge = configure.session_status_label(
        s.get("phase", ""), int(s.get("images_done") or 0), int(s.get("line_count") or 0),
    )
    return f"{s['label']}  [{badge}]"


def _build_create_tab() -> None:
    # Fresh blank session on every program start — do not preload last lyrics/song
    prefs = _prefs()
    g0_init = _gcfg()
    initial_lyrics = ""
    initial_style = prefs.get("style") or configure.STYLE_DEFAULT
    if initial_style not in configure.STYLE_CHOICES:
        initial_style = configure.STYLE_DEFAULT
    initial_ref = ""
    initial_song = ""
    initial_hair = configure.normalize_hair_style(
        g0_init.get("hair_style") or configure.HAIR_STYLE_DEFAULT
    )
    initial_outfit = configure.normalize_outfit(
        g0_init.get("outfit_worn") or configure.OUTFIT_DEFAULT
    )
    initial_gender = configure.normalize_gender(
        g0_init.get("ref_gender") or configure.GENDER_DEFAULT
    )
    initial_bodyshape = configure.normalize_bodyshape(
        g0_init.get("ref_bodyshape") or configure.BODYSHAPE_DEFAULT
    )
    can = _can_create(initial_lyrics, initial_song)
    expanded = bool(configure.APP_STATE.get("sessions_sidebar_expanded", True))
    configure.APP_STATE["active_session_id"] = ""
    configure.APP_STATE["session_status"] = "idle"
    configure.APP_STATE["current_project_folder"] = ""
    configure.APP_STATE["generation_output_paths"] = []

    # Outer row: collapsible sessions column | main generation column
    with gr.Row(elem_id="gen-outer-row"):
        # ── Left: sessions sidebar ──────────────────────────────────────
        with gr.Column(
            scale=1 if expanded else 0,
            min_width=200 if expanded else 48,
            elem_id="sessions-sidebar",
            elem_classes=["sessions-sidebar-expanded"] if expanded else ["sessions-sidebar-compact"],
            visible=True,
        ) as _gen["sidebar_col"]:
            _gen["sidebar_toggle"] = gr.Button(
                "--><--" if expanded else "<-->",
                elem_id="sessions-toggle",
                size="sm",
            )
            # Heading always present when expanded; compact mode hides it via refresh
            _gen["sessions_heading"] = gr.Markdown(
                "**Session Slots**" if expanded else "",
                elem_id="sessions-heading",
            )
            # Start New / New — under the heading when expanded
            _gen["start_new_session"] = gr.Button(
                "Start New Session" if expanded else "New",
                variant="secondary",
                visible=False,  # shown when a stopped/done session is selected
                elem_id="start-new-session",
                size="sm",
            )
            # Fixed slot rows (select + delete) — visibility toggled by refresh
            _gen["session_select_btns"] = []
            _gen["session_delete_btns"] = []
            _gen["session_rows"] = []
            for i in range(_MAX_SESSION_SLOTS):
                with gr.Row(visible=False, elem_id=f"session-row-{i}") as row:
                    sel = gr.Button(
                        str(i + 1),
                        size="sm",
                        scale=5,
                        elem_classes=["session-select-btn"],
                    )
                    dele = gr.Button(
                        "Delete",
                        size="sm",
                        scale=1,
                        variant="stop",
                        elem_classes=["session-delete-btn"],
                        min_width=64,
                        visible=expanded,
                    )
                _gen["session_rows"].append(row)
                _gen["session_select_btns"].append(sel)
                _gen["session_delete_btns"].append(dele)
            _gen["delete_all_sessions"] = gr.Button(
                "Delete All Sessions",
                variant="stop",
                size="sm",
                elem_id="delete-all-sessions",
                visible=expanded,
            )
            # Hidden state carriers
            _gen["active_session_id"] = gr.State("")
            _gen["sidebar_expanded"] = gr.State(expanded)
            _gen["session_ids"] = gr.State([])  # parallel list of folder ids
            _gen["confirm_delete_all"] = gr.Checkbox(
                label="Confirm delete ALL sessions?",
                value=False,
                visible=False,
                elem_id="confirm-delete-all",
            )
            _gen["confirm_delete_all_btn"] = gr.Button(
                "Yes, delete all",
                variant="stop",
                visible=False,
                size="sm",
            )
            _gen["cancel_delete_all_btn"] = gr.Button(
                "Cancel",
                visible=False,
                size="sm",
            )

        # ── Right: main generation controls (single column + Details Mode) ──
        with gr.Column(scale=4, elem_id="gen-main-col"):
            _g0 = _gcfg()
            _init_size = configure.image_size_label_from_wh(
                int(_g0.get("imagegen_width") or configure.DEFAULT_WIDTH),
                int(_g0.get("imagegen_height") or configure.DEFAULT_HEIGHT),
            )
            if _g0.get("imagegen_size"):
                _init_size = configure.normalize_image_size(str(_g0.get("imagegen_size")))
            _init_freq = configure.normalize_image_frequency(
                _g0.get("imagegen_frequency") or configure.DEFAULT_IMAGE_FREQUENCY
            )

            _gen["details_mode"] = gr.Radio(
                label="Details Mode",
                choices=[
                    "Project Settings",
                    "Name and Lyrics",
                    "Song Assessment",
                    "Reference Character",
                ],
                value="Name and Lyrics",
                elem_id="details-mode-radio",
            )

            # 1 — Project Settings
            with gr.Column(visible=False, elem_id="details-project-settings") as _details_settings:
                gr.Markdown(
                    "### Project Settings\n"
                    "Visual style, still size, image frequency "
                    "(**C**over / **T**heme / **L**yrics-per-line), and sampling. "
                    "Default steps **8** (better eyes / detail on Flux.2)."
                )
                with gr.Row():
                    _gen["style"] = gr.Dropdown(
                        label="Visual Style",
                        choices=configure.STYLE_CHOICES,
                        value=initial_style,
                        info="Influences prompt templates for still generation.",
                    )
                with gr.Row():
                    _gen["image_size"] = gr.Dropdown(
                        label="Image size",
                        choices=configure.IMAGE_SIZE_CHOICES,
                        value=_init_size,
                        info="Output still dimensions (width × height).",
                    )
                    _gen["image_frequency"] = gr.Dropdown(
                        label="Image Frequency",
                        choices=configure.IMAGE_FREQUENCY_CHOICES,
                        value=_init_freq,
                        info="C = Cover count · T = Theme (ambient) count · L = stills per lyric line (L2/L3 progressive).",
                    )
                with gr.Row():
                    _gen["steps"] = gr.Slider(
                        label="Steps",
                        minimum=1,
                        maximum=20,
                        step=1,
                        value=int(_g0.get("imagegen_steps") or configure.DEFAULT_STEPS),
                    )
                    _gen["cfg"] = gr.Slider(
                        label="CFG",
                        minimum=0.5,
                        maximum=4.0,
                        step=0.1,
                        value=float(_g0.get("imagegen_cfg_scale") or configure.DEFAULT_CFG),
                    )
            _gen["details_project_settings"] = _details_settings

            # 2 — Name and Lyrics
            with gr.Column(visible=True, elem_id="details-name-lyrics") as _details_nl:
                _gen["song_name"] = gr.Textbox(
                    label="Song name (required — becomes output folder)",
                    value=initial_song,
                    placeholder="e.g. Midnight Drive",
                    elem_id="song-name-box",
                )
                _gen["lyrics"] = gr.Textbox(
                    label="Song Lyrics (one image per non-empty line)",
                    lines=12,
                    max_lines=12,
                    placeholder="Paste full lyrics here…\n[Intro]\nFirst line…\n…",
                    value=initial_lyrics,
                    elem_id="lyrics-box",
                )
                _gen["negative_prompt"] = gr.Textbox(
                    label="Negative prompt (saved with the project)",
                    lines=2,
                    max_lines=4,
                    value=getattr(configure, "DEFAULT_NEGATIVE_PROMPT", ""),
                    placeholder="Things to avoid in every still…",
                    elem_id="negative-prompt-box",
                )
            _gen["details_name_lyrics"] = _details_nl

            # 3 — Song Assessment
            with gr.Column(visible=False, elem_id="details-assessment") as _details_assess:
                _gen["assessment_view"] = gr.Textbox(
                    label="Song assessment (editable — Save before generating)",
                    lines=18,
                    max_lines=24,
                    value="",
                    interactive=True,
                    elem_id="assessment-view-box",
                )
                with gr.Row():
                    _gen["save_assessment_btn"] = gr.Button(
                        "Save Assessment",
                        variant="primary",
                        size="sm",
                        elem_id="save-assessment-btn",
                    )
                    _gen["reload_assessment_btn"] = gr.Button(
                        "Reload Assessment",
                        variant="secondary",
                        size="sm",
                        elem_id="reload-assessment-btn",
                    )
            _gen["details_assessment"] = _details_assess

            # 4 — Reference Character
            with gr.Column(visible=False, elem_id="details-reference") as _details_ref:
                gr.Markdown(
                    "### Reference Character (optional)\n"
                    "Central character likeness for stills that feature the subject. "
                    "Gender, bodyshape, hair, and outfit are injected only into "
                    "character-bearing prompts (when a reference image is attached)."
                )
                with gr.Row():
                    _gen["ref_image"] = gr.Textbox(
                        label="Reference image path",
                        value=initial_ref,
                        interactive=True,
                        scale=4,
                        placeholder="Enter full path to image or Browse",
                    )
                    with gr.Column(scale=1, min_width=100):
                        _gen["browse_ref"] = gr.Button("Browse", min_width=90)
                        _gen["remove_ref"] = gr.Button("Remove", min_width=90, variant="secondary")
                _ref_ok0 = bool(initial_ref and Path(initial_ref).is_file())
                _gen["ref_preview"] = gr.Image(
                    label="Reference preview",
                    value=initial_ref if _ref_ok0 else None,
                    type="filepath",
                    height=512,
                    interactive=False,
                    visible=_ref_ok0,
                    elem_id="ref-preview-image",
                )
                with gr.Row():
                    _gen["ref_gender"] = gr.Dropdown(
                        label="Reference Image Gender",
                        choices=configure.GENDER_CHOICES,
                        value=initial_gender,
                        info="Locks subject gender language in character stills.",
                    )
                    _gen["ref_bodyshape"] = gr.Dropdown(
                        label="Reference Image Bodyshape",
                        choices=configure.BODYSHAPE_CHOICES,
                        value=initial_bodyshape,
                        info="Counters Flux gym-fit prior; applied on character stills.",
                    )
                with gr.Row():
                    _gen["hair_style"] = gr.Dropdown(
                        label="Hair Style",
                        choices=configure.HAIR_STYLE_CHOICES,
                        value=initial_hair,
                        info="Locked hair description for character consistency. None = omit.",
                    )
                    _gen["outfit_worn"] = gr.Dropdown(
                        label="Outfit Worn",
                        choices=configure.OUTFIT_CHOICES,
                        value=initial_outfit,
                        info="Locked wardrobe for character consistency. None = omit.",
                    )
            _gen["details_reference"] = _details_ref

            with gr.Row(elem_id="gen-action-row"):
                _models_ok0 = _models_configured()
                _has_assess0 = _assessment_exists()
                _can_assess0 = _can_run_assessment(initial_lyrics, initial_song)
                _has_images0 = _project_has_any_images()
                _gen["assess_btn"] = gr.Button(
                    (
                        "Configure the Pages First"
                        if not _models_ok0
                        else ("Re-run Assessment" if _has_assess0 else "Run Assessment")
                    ),
                    variant="primary" if (_models_ok0 and _can_assess0) else "secondary",
                    interactive=bool(_models_ok0 and _can_assess0),
                    visible=True,
                    elem_id="assess-btn",
                )
                _gen["all_assets_btn"] = gr.Button(
                    _all_assets_btn_label(initial_lyrics),
                    variant="primary",
                    interactive=bool(_models_ok0 and _has_assess0),
                    visible=bool(_models_ok0 and _has_assess0),
                    elem_id="all-assets-btn",
                )
                _gen["cover_btn"] = gr.Button(
                    _cover_btn_label(),
                    variant="primary" if _can_cover(initial_song) else "secondary",
                    interactive=bool(_can_cover(initial_song)),
                    visible=bool(_models_ok0 and _has_assess0),
                    elem_id="cover-btn",
                )
                _gen["theme_btn"] = gr.Button(
                    _theme_btn_label(),
                    variant="primary" if _can_theme(initial_lyrics, initial_song) else "secondary",
                    interactive=bool(_can_theme(initial_lyrics, initial_song)),
                    visible=bool(_models_ok0 and _has_assess0),
                    elem_id="theme-btn",
                )
                _gen["run_btn"] = gr.Button(
                    _lyrics_btn_label(initial_lyrics),
                    variant="primary" if (can and _has_assess0) else "secondary",
                    interactive=bool(can and _has_assess0),
                    visible=bool(_models_ok0 and _has_assess0),
                    elem_id="run-btn",
                )
                _gen["stop_btn"] = gr.Button(
                    "Emergency Stop",
                    variant="stop",
                    visible=False,
                    elem_id="stop-btn",
                )

        # Dynamic galleries: Cover / Theme / Lyrics — each hidden until images exist
    th0 = _thumb_size_px()

    def _build_simple_gallery(prefix: str, n_slots: int, title: str, elem_id: str):
        """Cover / Theme gallery with Regenerate + Remove per slot (like Lyrics)."""
        with gr.Column(
            visible=False,
            elem_id=elem_id,
            elem_classes=["materials-thumbs"],
        ) as panel:
            header = gr.Button(
                title,
                variant="secondary",
                elem_id=f"{elem_id}-header",
                elem_classes=["materials-thumbs-heading"],
                size="sm",
            )
            cols, imgs, rows = [], [], []
            regen_btns, remove_btns = [], []
            for row_i in range(0, n_slots, THUMB_COLS):
                with gr.Row(visible=False, elem_classes=["thumb-row"]) as trow:
                    for j in range(THUMB_COLS):
                        with gr.Column(
                            visible=False,
                            scale=1,
                            min_width=max(64, th0 // 2),
                            elem_classes=["thumb-slot"],
                        ) as col:
                            img = gr.Image(
                                value=None,
                                label=None,
                                show_label=False,
                                height=th0,
                                interactive=False,
                                elem_classes=["thumb-img"],
                            )
                            with gr.Row(elem_classes=["thumb-btn-row"]):
                                btn = gr.Button(
                                    "Regenerate",
                                    visible=False,
                                    size="sm",
                                    elem_classes=["thumb-regen-btn"],
                                    min_width=48,
                                    scale=2,
                                )
                                rmv = gr.Button(
                                    "Remove",
                                    visible=False,
                                    size="sm",
                                    variant="stop",
                                    elem_classes=["thumb-remove-btn"],
                                    min_width=48,
                                    scale=1,
                                )
                            cols.append(col)
                            imgs.append(img)
                            regen_btns.append(btn)
                            remove_btns.append(rmv)
                rows.append(trow)
        _gen[f"{prefix}_panel"] = panel
        _gen[f"{prefix}_header"] = header
        _gen[f"{prefix}_cols"] = cols
        _gen[f"{prefix}_imgs"] = imgs
        _gen[f"{prefix}_rows"] = rows
        _gen[f"{prefix}_regen_btns"] = regen_btns
        _gen[f"{prefix}_remove_btns"] = remove_btns
        return header

    _build_simple_gallery("cover", COVER_SLOTS, "Cover Thumbnails", "cover-gallery")
    _build_simple_gallery("theme", THEME_SLOTS, "Theme Thumbnails", "theme-gallery")

    with gr.Column(
        visible=False,
        elem_id="materials-gallery",
        elem_classes=["materials-thumbs"],
    ) as _thumbs_panel:
        _gen["open_materials_folder"] = gr.Button(
            "Lyrics Thumbnails",
            variant="secondary",
            elem_id="materials-thumbs-header",
            elem_classes=["materials-thumbs-heading"],
            size="sm",
        )
        # Page switcher for L2/L3 (one still set per lyric line per page).
        # Hidden when Image Frequency is L1.
        _gen["lyrics_page_radio"] = gr.Radio(
            choices=_lyrics_page_choices(),
            value=_lyrics_page_label(),
            label="Lyrics Image Page",
            visible=False,
            interactive=True,
            elem_id="lyrics-page-radio",
        )
        _gen["thumbs_panel"] = _thumbs_panel
        _gen["thumb_cols"] = []
        _gen["thumb_imgs"] = []
        _gen["thumb_btns"] = []
        _gen["thumb_remove_btns"] = []
        _gen["thumb_rows"] = []
        for row_i in range(0, THUMB_SLOTS, THUMB_COLS):
            with gr.Row(visible=False, elem_classes=["thumb-row"]) as trow:
                for j in range(THUMB_COLS):
                    with gr.Column(
                        visible=False,
                        scale=1,
                        min_width=max(64, th0 // 2),
                        elem_classes=["thumb-slot"],
                    ) as col:
                        img = gr.Image(
                            value=None,
                            label=None,
                            show_label=False,
                            height=th0,
                            interactive=False,
                            elem_classes=["thumb-img"],
                        )
                        with gr.Row(elem_classes=["thumb-btn-row"]):
                            btn = gr.Button(
                                "Regenerate",
                                visible=False,
                                size="sm",
                                elem_classes=["thumb-regen-btn"],
                                min_width=48,
                                scale=2,
                            )
                            rmv = gr.Button(
                                "Remove",
                                visible=False,
                                size="sm",
                                variant="stop",
                                elem_classes=["thumb-remove-btn"],
                                min_width=48,
                                scale=1,
                            )
                        _gen["thumb_cols"].append(col)
                        _gen["thumb_imgs"].append(img)
                        _gen["thumb_btns"].append(btn)
                        _gen["thumb_remove_btns"].append(rmv)
            _gen["thumb_rows"].append(trow)



def _project_local_ref(ref_image: str, project_dir: str = "") -> str:
    """Copy ref into project when possible; always return app-local path or ''."""
    proj = (project_dir or configure.APP_STATE.get("current_project_folder") or "").strip()
    src = (ref_image or "").strip()
    if proj and Path(proj).is_dir():
        local = configure.ensure_project_reference(proj, src)
        if local:
            configure.update_generation({"reference_image_path": local})
            return local
    if src:
        local = configure.cache_reference_image(src)
        if local:
            configure.update_generation({"reference_image_path": local})
            return local
    return ""


# Serial regen worker — survives Gradio cancelling a click-generator when
# another Regenerate button is pressed (same outputs). Jobs live in
# named_regen_queue / regen_queue; UI generators only enqueue + poll.
_regen_worker_lock = threading.Lock()


def _mark_named_busy_global(kind: str, slot_idx: int, on: bool) -> None:
    key = f"{kind}:{int(slot_idx)}"
    busy = set(str(x) for x in (configure.APP_STATE.get("named_regen_busy") or []))
    if on:
        busy.add(key)
    else:
        busy.discard(key)
    configure.APP_STATE["named_regen_busy"] = sorted(busy)


def _execute_named_job(job: Dict[str, Any], ctx: Dict[str, Any]) -> str:
    kind = str(job.get("kind") or "cover")
    slot_idx = int(job.get("slot_idx", 0))
    proj = str(ctx.get("proj") or "")
    cfg = dict(ctx.get("cfg") or {})
    ref = str(ctx.get("ref") or "")
    path = str(job.get("path") or "")
    smap = _named_slot_map(kind, proj)
    if not path or not Path(path).is_file():
        path = smap.get(slot_idx, "")
    if not path:
        return f"No {kind} image in slot {slot_idx + 1}."
    song_name = Path(proj).name if proj else ""
    try:
        meta = configure.load_session_meta(Path(proj)) or {}
        song_name = (meta.get("label") or song_name).strip()
    except Exception:
        pass
    lyrics = ""
    try:
        lp = Path(proj) / "lyrics.txt"
        if lp.is_file():
            lyrics = lp.read_text(encoding="utf-8", errors="replace")
    except OSError:
        pass
    out = inference.regenerate_named_still(
        kind,
        path,
        cfg,
        song_name=song_name,
        lyrics=lyrics,
        reference_image=_project_local_ref(ref, proj),
        theme_index=slot_idx if kind == "theme" else 0,
    )
    return f"Regenerated {kind} still: {Path(out).name}"


def _execute_lyric_job(job: Any, ctx: Dict[str, Any]) -> str:
    """job is 0-based line index (legacy int) or dict with line_idx + variant."""
    if isinstance(job, dict):
        line_idx = int(job.get("line_idx", 0))
        variant = int(job.get("variant") or 1)
    else:
        line_idx = int(job)
        variant = int(ctx.get("variant") or 1)
    proj = str(ctx.get("proj") or "")
    cfg = dict(ctx.get("cfg") or {})
    ref = str(ctx.get("ref") or "")
    inference.regenerate_single_still(
        Path(proj), int(line_idx), cfg,
        reference_image=_project_local_ref(ref, proj),
        variant=variant,
    )
    return f"Regenerated lyrics still {int(line_idx) + 1} variant {variant}."


def _regen_worker_loop() -> None:
    print("[regen-worker] started", flush=True)
    configure.APP_STATE["generating"] = True
    configure.APP_STATE["session_status"] = "running"
    try:
        while True:
            if inference.is_cancel_requested():
                break
            nq = list(configure.APP_STATE.get("named_regen_queue") or [])
            lq = list(configure.APP_STATE.get("regen_queue") or [])
            if not nq and not lq:
                break
            ctx = dict(configure.APP_STATE.get("regen_worker_ctx") or {})
            if nq:
                job = nq.pop(0)
                configure.APP_STATE["named_regen_queue"] = nq
                jk = str(job.get("kind") or "cover")
                ji = int(job.get("slot_idx", 0))
                _mark_named_busy_global(jk, ji, True)
                try:
                    msg = _execute_named_job(job, ctx)
                    print(f"[regen-worker] {msg}", flush=True)
                except Exception as e:
                    print(f"[regen-worker] {jk} {ji + 1} failed: {e}", flush=True)
                finally:
                    _mark_named_busy_global(jk, ji, False)
                    configure.APP_STATE["image_gen_t0"] = None
                continue
            if lq:
                job = lq.pop(0)
                configure.APP_STATE["regen_queue"] = lq
                if isinstance(job, dict):
                    qi = int(job.get("line_idx", 0))
                else:
                    qi = int(job)
                q_line = qi + 1
                _unqueue_line(q_line)
                busy = set()
                for x in (configure.APP_STATE.get("regen_busy_lines") or []):
                    try:
                        busy.add(int(x))
                    except (TypeError, ValueError):
                        pass
                busy.add(q_line)
                configure.APP_STATE["regen_busy_lines"] = sorted(busy)
                try:
                    msg = _execute_lyric_job(job, ctx)
                    print(f"[regen-worker] {msg}", flush=True)
                except Exception as e:
                    print(f"[regen-worker] lyrics {q_line} failed: {e}", flush=True)
                finally:
                    busy = set()
                    for x in (configure.APP_STATE.get("regen_busy_lines") or []):
                        try:
                            busy.add(int(x))
                        except (TypeError, ValueError):
                            pass
                    busy.discard(q_line)
                    configure.APP_STATE["regen_busy_lines"] = sorted(busy)
                    configure.APP_STATE["image_gen_t0"] = None
    finally:
        configure.APP_STATE["generating"] = False
        configure.APP_STATE["session_status"] = "stopped"
        with _regen_worker_lock:
            configure.APP_STATE["regen_worker_alive"] = False
        print("[regen-worker] idle", flush=True)


def _ensure_regen_worker() -> None:
    with _regen_worker_lock:
        if configure.APP_STATE.get("regen_worker_alive"):
            return
        configure.APP_STATE["regen_worker_alive"] = True
    t = threading.Thread(target=_regen_worker_loop, name="named-regen-worker", daemon=True)
    t.start()


def _wire_create_events(status_box) -> None:
    def _resolve_ref_for_ui(path: str):
        """
        Resolve a reference path Gradio can serve:
        1) If a project folder is active → copy into output/<song>/reference.*
        2) Else → data/ref_cache/ref_<hash>.*
        Never returns an external path (avoids InvalidPathError).
        """
        p = (path or "").strip()
        if not p or not Path(p).is_file():
            # Prefer existing project reference when path blank/invalid
            proj = (configure.APP_STATE.get("current_project_folder") or "").strip()
            if proj:
                existing = configure.project_reference_path(proj)
                if existing:
                    return existing, True
            return "", False
        proj = (configure.APP_STATE.get("current_project_folder") or "").strip()
        if proj and Path(proj).is_dir():
            local = configure.ensure_project_reference(proj, p)
        else:
            local = configure.cache_reference_image(p)
        ok = bool(local and Path(local).is_file())
        return (local if ok else ""), ok

    def _browse_ref():
        path = _browse_file(_FILETYPES_IMAGE, "last_image_browse_dir")
        if not path:
            return gr.update(), gr.update()
        local, ok = _resolve_ref_for_ui(str(path))
        if ok:
            configure.update_generation({"reference_image_path": local})
        return (
            gr.update(value=local if ok else "", placeholder="Enter full path to image or Browse"),
            gr.update(value=local if ok else None, visible=ok),
        )

    def _remove_ref():
        configure.update_generation({"reference_image_path": ""})
        # Remove project-local reference.* if present
        proj = (configure.APP_STATE.get("current_project_folder") or "").strip()
        if proj and Path(proj).is_dir():
            try:
                for old in Path(proj).glob("reference.*"):
                    try:
                        old.unlink()
                    except OSError:
                        pass
            except OSError:
                pass
        return (
            gr.update(value="", placeholder="Enter full path to image or Browse"),
            gr.update(value=None, visible=False),
        )

    def _ref_image_change(path: str):
        local, ok = _resolve_ref_for_ui(path or "")
        configure.update_generation({"reference_image_path": local if ok else ""})
        return (
            gr.update(value=local if ok else (path or ""), placeholder="Enter full path to image or Browse"),
            gr.update(value=local if ok else None, visible=ok),
        )

    _gen["browse_ref"].click(
        _browse_ref,
        outputs=[_gen["ref_image"], _gen["ref_preview"]],
    )
    _gen["remove_ref"].click(
        _remove_ref,
        outputs=[_gen["ref_image"], _gen["ref_preview"]],
    )
    _gen["ref_image"].change(
        _ref_image_change,
        inputs=[_gen["ref_image"]],
        outputs=[_gen["ref_image"], _gen["ref_preview"]],
    )

    def _details_mode_change(mode: str):
        m = (mode or "").strip()
        show_settings = m == "Project Settings"
        show_nl = m == "Name and Lyrics"
        show_assess = m == "Song Assessment"
        show_ref = m == "Reference Character"
        assess = _load_assessment_text() if show_assess else gr.update()
        return (
            gr.update(visible=show_settings),
            gr.update(visible=show_nl),
            gr.update(visible=show_assess),
            gr.update(visible=show_ref),
            gr.update(value=assess) if show_assess else gr.update(),
        )

    if _gen.get("details_mode") is not None:
        _gen["details_mode"].change(
            _details_mode_change,
            inputs=[_gen["details_mode"]],
            outputs=[
                _gen["details_project_settings"],
                _gen["details_name_lyrics"],
                _gen["details_assessment"],
                _gen["details_reference"],
                _gen["assessment_view"],
            ],
        )

    def _save_assessment(text_val: str, song_name: str, lyrics: str, active_session_id: str):
        """Write the Assessment panel text to analysis.txt for the active project."""
        label = (song_name or "").strip()
        folder = (configure.APP_STATE.get("current_project_folder") or "").strip()
        if not folder and active_session_id:
            root = configure.get_output_dir()
            cand = Path(root) / str(active_session_id)
            if cand.is_dir():
                folder = str(cand)
        if not folder and label:
            # Ensure project dir exists so Save works before Run Assessment finishes
            try:
                from scripts.inference import ensure_project_dir, _slugify_folder_name
                slug = _slugify_folder_name(label, fallback="")
                if slug:
                    folder = str(ensure_project_dir(slug, sequential=False))
                    configure.APP_STATE["current_project_folder"] = folder
                    configure.APP_STATE["active_session_id"] = Path(folder).name
            except Exception as e:
                return f"Cannot create project folder: {e}", *_action_btn_updates(lyrics, song_name)
        if not folder:
            return (
                "Save Assessment: set a song name (or run assessment first) so a project folder exists.",
                *_action_btn_updates(lyrics, song_name),
            )
        path = Path(folder) / "analysis.txt"
        body = (text_val or "").strip()
        if not body:
            return "Assessment is empty — nothing saved.", *_action_btn_updates(lyrics, song_name)
        try:
            path.write_text(body if body.endswith("\n") else body + "\n", encoding="utf-8")
            try:
                (Path(folder) / "lyrics.txt").write_text(lyrics or "", encoding="utf-8")
            except OSError:
                pass
            configure.APP_STATE["current_project_folder"] = folder
            configure.APP_STATE["active_session_id"] = Path(folder).name
            msg = f"Assessment saved → {path}"
            return msg, *_action_btn_updates(lyrics, song_name)
        except OSError as e:
            return f"Save failed: {e}", *_action_btn_updates(lyrics, song_name)

    def _reload_assessment(lyrics: str, song_name: str):
        text_val = _load_assessment_text()
        return text_val, f"Reloaded assessment from disk.", *_action_btn_updates(lyrics, song_name)

    if _gen.get("save_assessment_btn") is not None:
        _gen["save_assessment_btn"].click(
            _save_assessment,
            inputs=[
                _gen["assessment_view"],
                _gen["song_name"],
                _gen["lyrics"],
                _gen["active_session_id"],
            ],
            outputs=[
                status_box,
                _gen["assess_btn"],
                _gen["all_assets_btn"],
                _gen["cover_btn"],
                _gen["theme_btn"],
                _gen["run_btn"],
                _gen["stop_btn"],
            ],
        )
    if _gen.get("reload_assessment_btn") is not None:
        _gen["reload_assessment_btn"].click(
            _reload_assessment,
            inputs=[_gen["lyrics"], _gen["song_name"]],
            outputs=[
                _gen["assessment_view"],
                status_box,
                _gen["assess_btn"],
                _gen["all_assets_btn"],
                _gen["cover_btn"],
                _gen["theme_btn"],
                _gen["run_btn"],
                _gen["stop_btn"],
            ],
        )

    def _ready_change(lyrics, song_name):
        assess_u, all_u, cover_u, theme_u, lyrics_u, stop_u = _action_btn_updates(
            lyrics, song_name, running=False
        )
        return assess_u, all_u, cover_u, theme_u, lyrics_u

    for _evt in ("change", "input", "blur"):
        try:
            getattr(_gen["lyrics"], _evt)(
                _ready_change,
                inputs=[_gen["lyrics"], _gen["song_name"]],
                outputs=[
                    _gen["assess_btn"], _gen["all_assets_btn"],
                    _gen["cover_btn"], _gen["theme_btn"], _gen["run_btn"],
                ],
            )
            getattr(_gen["song_name"], _evt)(
                _ready_change,
                inputs=[_gen["lyrics"], _gen["song_name"]],
                outputs=[
                    _gen["assess_btn"], _gen["all_assets_btn"],
                    _gen["cover_btn"], _gen["theme_btn"], _gen["run_btn"],
                ],
            )
        except Exception:
            pass

    # Song Lyrics: expand to full content height while focused; collapse to 12 on blur
    def _lyrics_focus(text):
        n = _lyrics_line_count(text)
        return gr.update(lines=n, max_lines=n)

    def _lyrics_blur(text):
        return gr.update(lines=12, max_lines=12)

    try:
        _gen["lyrics"].focus(
            _lyrics_focus, inputs=[_gen["lyrics"]], outputs=_gen["lyrics"]
        )
        _gen["lyrics"].blur(
            _lyrics_blur, inputs=[_gen["lyrics"]], outputs=_gen["lyrics"]
        )
    except Exception as e:
        print(f"[ui] lyrics focus/blur not bound: {e}", flush=True)

    def _stop():
        return inference.emergency_stop()

    _gen["stop_btn"].click(_stop, outputs=status_box)

    # Header click → current project folder, or output/ root if none
    def _open_materials_folder():
        return _open_project_folder()

    _gen["open_materials_folder"].click(_open_materials_folder, outputs=status_box)
    if _gen.get("cover_header") is not None:
        _gen["cover_header"].click(_open_materials_folder, outputs=status_box)
    if _gen.get("theme_header") is not None:
        _gen["theme_header"].click(_open_materials_folder, outputs=status_box)


    # ── Session sidebar helpers ─────────────────────────────────────────
    def _refresh_session_slots(active_id: str = "", expanded: bool = None):
        """
        Return updates for all session slot rows + id list + Start New visibility.
        Compact (expanded=False): numbered buttons only, no Delete / Delete All,
        Start New label = "New". Expanded: full labels, Delete visible, heading on.
        """
        if expanded is None:
            expanded = bool(configure.APP_STATE.get("sessions_sidebar_expanded", True))
        sessions = configure.list_sessions()
        ids = [s["id"] for s in sessions]
        row_updates = []
        sel_updates = []
        del_updates = []
        for i in range(_MAX_SESSION_SLOTS):
            if i < len(sessions):
                s = sessions[i]
                label = _session_row_label(s, expanded=expanded, index=i)
                is_active = (s["id"] == active_id)
                if is_active and expanded:
                    label = f"▶ {label}"
                elif is_active and not expanded:
                    label = f"•{i + 1}"
                row_updates.append(gr.update(visible=True))
                sel_updates.append(gr.update(value=label, visible=True))
                # Delete only when expanded — also collapse scale so it takes no space
                if expanded:
                    del_updates.append(gr.update(visible=True, scale=1, min_width=64))
                else:
                    del_updates.append(gr.update(visible=False, scale=0, min_width=0))
            else:
                row_updates.append(gr.update(visible=False))
                sel_updates.append(gr.update(value="", visible=False))
                del_updates.append(gr.update(visible=False, scale=0, min_width=0))
        show_new = bool(active_id) or configure.APP_STATE.get("session_status") in (
            "stopped", "done",
        )
        start_new = gr.update(
            visible=show_new,
            value="Start New Session" if expanded else "New",
        )
        heading = gr.update(value="**Session Slots**" if expanded else "")
        delete_all = gr.update(visible=bool(expanded))
        return (
            [ids]
            + row_updates
            + sel_updates
            + del_updates
            + [start_new, heading, delete_all]
        )

    _session_refresh_outputs = (
        [_gen["session_ids"]]
        + _gen["session_rows"]
        + _gen["session_select_btns"]
        + _gen["session_delete_btns"]
        + [
            _gen["start_new_session"],
            _gen["sessions_heading"],
            _gen["delete_all_sessions"],
        ]
    )

    def _toggle_sidebar(expanded: bool, active_id: str):
        expanded = not bool(expanded)
        configure.APP_STATE["sessions_sidebar_expanded"] = expanded
        btn_label = "--><--" if expanded else "<-->"
        col = gr.update(
            scale=1 if expanded else 0,
            min_width=200 if expanded else 48,
            elem_classes=["sessions-sidebar-expanded"] if expanded else ["sessions-sidebar-compact"],
        )
        # Re-label session buttons for compact/expanded
        refresh = _refresh_session_slots(active_id or "", expanded=expanded)
        return (expanded, gr.update(value=btn_label), col) + tuple(refresh)

    _gen["sidebar_toggle"].click(
        _toggle_sidebar,
        inputs=[_gen["sidebar_expanded"], _gen["active_session_id"]],
        outputs=[
            _gen["sidebar_expanded"],
            _gen["sidebar_toggle"],
            _gen["sidebar_col"],
        ] + _session_refresh_outputs,
    )

    def _select_session(idx: int, session_ids: list):
        if not session_ids or idx < 0 or idx >= len(session_ids):
            return (
                gr.update(), gr.update(), gr.update(), gr.update(),
                gr.update(), gr.update(), gr.update(),
                "",
                "",
            ) + tuple(_refresh_session_slots(""))
        sid = session_ids[idx]
        s = configure.get_session_by_id(sid)
        if not s:
            return (
                gr.update(), gr.update(), gr.update(), gr.update(),
                gr.update(), gr.update(), gr.update(),
                "Session not found.",
                "",
            ) + tuple(_refresh_session_slots(""))
        configure.APP_STATE["active_session_id"] = sid
        configure.APP_STATE["current_project_folder"] = s["path"]
        configure.APP_STATE["session_status"] = (
            "done" if s.get("phase") == "done" else "stopped"
        )
        imgs = s.get("image_paths") or configure.list_session_images(Path(s["path"]))
        configure.APP_STATE["generation_output_paths"] = list(imgs)
        configure.APP_STATE["thumb_expected_count"] = int(s.get("line_count") or len(imgs) or 0)
        configure.APP_STATE["thumb_generating_line"] = None
        configure.APP_STATE["regen_busy_lines"] = []
        configure.APP_STATE["thumb_queued_lines"] = []
        # Prefer stored lyrics; fall back to lyrics.txt
        lyrics = s.get("lyrics") or ""
        if not lyrics:
            try:
                lp = Path(s["path"]) / "lyrics.txt"
                if lp.exists():
                    lyrics = lp.read_text(encoding="utf-8", errors="replace")
            except OSError:
                pass
        song = s.get("song_name") or s.get("label") or sid
        # Strip serial suffix for the song-name field when possible
        # (folder may be "midnight_drive" or "midnight_drive_2")
        style = s.get("style") or configure.STYLE_DEFAULT
        steps = s.get("steps", configure.DEFAULT_STEPS)
        cfg_scale = s.get("cfg_scale", configure.DEFAULT_CFG)
        ref = s.get("reference_image") or ""
        # If ref is project-relative, resolve
        if ref and not Path(ref).is_file():
            cand = Path(s["path"]) / Path(ref).name
            if cand.is_file():
                ref = str(cand)
            else:
                for p in Path(s["path"]).glob("reference.*"):
                    ref = str(p)
                    break
        status = (
            f"Session loaded: {sid}  "
            f"({len(imgs)}/{s.get('line_count') or '?'} images, phase={s.get('phase')})"
        )
        # Prefer global generation size; session meta does not store size yet
        g = configure.load_generation()
        size_label = configure.normalize_image_size(
            str(g.get("imagegen_size") or configure.image_size_label_from_wh(
                int(g.get("imagegen_width") or configure.DEFAULT_WIDTH),
                int(g.get("imagegen_height") or configure.DEFAULT_HEIGHT),
            ))
        )
        main = (
            gr.update(value=song),
            gr.update(value=lyrics, lines=12, max_lines=12),
            gr.update(value=style if style in configure.STYLE_CHOICES else configure.STYLE_DEFAULT),
            gr.update(value=size_label),
            gr.update(value=steps),
            gr.update(value=cfg_scale),
            gr.update(value=ref),
            gr.update(value=(s.get("negative_prompt") if s.get("negative_prompt") is not None else configure.DEFAULT_NEGATIVE_PROMPT)),
            status,
            sid,
        )
        assess_u, all_u, cover_u, theme_u, lyrics_u, _stop_u = _action_btn_updates(
            lyrics, song, running=False
        )
        refresh = _refresh_session_slots(sid)
        return main + (assess_u, all_u, cover_u, theme_u, lyrics_u) + tuple(
            _all_gallery_updates(lyric_paths=list(imgs))
        ) + tuple(refresh)

    # Wire each select button with its index
    for i, btn in enumerate(_gen["session_select_btns"]):
        btn.click(
            lambda sid_list, i=i: _select_session(i, sid_list or []),
            inputs=[_gen["session_ids"]],
            outputs=[
                _gen["song_name"],
                _gen["lyrics"],
                _gen["style"],
                _gen["image_size"],
                _gen["steps"],
                _gen["cfg"],
                _gen["ref_image"],
                _gen["negative_prompt"],
                status_box,
                _gen["active_session_id"],
                _gen["assess_btn"],
                _gen["all_assets_btn"],
                _gen["cover_btn"],
                _gen["theme_btn"],
                _gen["run_btn"],
            ] + _thumb_panel_outputs() + _session_refresh_outputs,
        )

    def _delete_one(idx: int, session_ids: list, active_id: str):
        if not session_ids or idx < 0 or idx >= len(session_ids):
            return ("",) + tuple(_refresh_session_slots(active_id or ""))
        sid = session_ids[idx]
        ok = configure.delete_session(sid)
        new_active = "" if active_id == sid else (active_id or "")
        if active_id == sid:
            configure.APP_STATE["generation_output_paths"] = []
        msg = f"Deleted session: {sid}" if ok else f"Could not delete: {sid}"
        return (msg,) + tuple(_refresh_session_slots(new_active))

    for i, btn in enumerate(_gen["session_delete_btns"]):
        btn.click(
            lambda sid_list, active, i=i: _delete_one(i, sid_list or [], active or ""),
            inputs=[_gen["session_ids"], _gen["active_session_id"]],
            outputs=[status_box] + _session_refresh_outputs,
        )

    def _start_new():
        configure.APP_STATE["active_session_id"] = ""
        configure.APP_STATE["current_project_folder"] = ""
        configure.APP_STATE["session_status"] = "idle"
        configure.APP_STATE["generation_output_paths"] = []
        configure.APP_STATE["thumb_expected_count"] = 0
        configure.APP_STATE["thumb_generating_line"] = None
        configure.APP_STATE["regen_busy_lines"] = []
        configure.APP_STATE["thumb_queued_lines"] = []
        main = (
            gr.update(value=""),
            gr.update(value="", lines=12, max_lines=12),
            gr.update(value=configure.STYLE_DEFAULT),
            gr.update(value=configure.DEFAULT_IMAGE_SIZE),
            gr.update(value=configure.DEFAULT_STEPS),
            gr.update(value=configure.DEFAULT_CFG),
            gr.update(value=""),
            gr.update(value=configure.DEFAULT_NEGATIVE_PROMPT),
            "New session — enter song name & lyrics, then Generate.",
            "",
        )
        assess_u, all_u, cover_u, theme_u, lyrics_u, _stop_u = _action_btn_updates(
            "", "", running=False
        )
        return main + (assess_u, all_u, cover_u, theme_u, lyrics_u) + tuple(
            _all_gallery_updates(lyric_paths=[])
        ) + tuple(_refresh_session_slots(""))

    _gen["start_new_session"].click(
        _start_new,
        outputs=[
            _gen["song_name"],
            _gen["lyrics"],
            _gen["style"],
            _gen["image_size"],
            _gen["steps"],
            _gen["cfg"],
            _gen["ref_image"],
            _gen["negative_prompt"],
            status_box,
            _gen["active_session_id"],
            _gen["assess_btn"],
            _gen["all_assets_btn"],
            _gen["cover_btn"],
            _gen["theme_btn"],
            _gen["run_btn"],
        ] + _thumb_panel_outputs() + _session_refresh_outputs,
    )

    def _ask_delete_all():
        return (
            gr.update(visible=True),
            gr.update(visible=True),
            gr.update(visible=True),
            "Confirm: permanently delete ALL session folders under output/?",
        )

    def _cancel_delete_all():
        return (
            gr.update(visible=False, value=False),
            gr.update(visible=False),
            gr.update(visible=False),
            "Delete-all cancelled.",
        )

    def _do_delete_all(confirmed: bool):
        if not confirmed:
            return (
                gr.update(visible=True, value=False),
                gr.update(visible=True),
                gr.update(visible=True),
                "Tick the confirm box first.",
                gr.update(),
            ) + tuple(_all_gallery_updates()) + tuple(
                _refresh_session_slots(configure.APP_STATE.get("active_session_id") or "")
            )
        n = configure.delete_all_sessions()
        configure.APP_STATE["generation_output_paths"] = []
        return (
            gr.update(visible=False, value=False),
            gr.update(visible=False),
            gr.update(visible=False),
            f"Deleted {n} session(s).",
            "",
        ) + tuple(_all_gallery_updates(lyric_paths=[])) + tuple(_refresh_session_slots(""))

    _gen["delete_all_sessions"].click(
        _ask_delete_all,
        outputs=[
            _gen["confirm_delete_all"],
            _gen["confirm_delete_all_btn"],
            _gen["cancel_delete_all_btn"],
            status_box,
        ],
    )
    _gen["cancel_delete_all_btn"].click(
        _cancel_delete_all,
        outputs=[
            _gen["confirm_delete_all"],
            _gen["confirm_delete_all_btn"],
            _gen["cancel_delete_all_btn"],
            status_box,
        ],
    )
    _gen["confirm_delete_all_btn"].click(
        _do_delete_all,
        inputs=[_gen["confirm_delete_all"]],
        outputs=[
            _gen["confirm_delete_all"],
            _gen["confirm_delete_all_btn"],
            _gen["cancel_delete_all_btn"],
            status_box,
            _gen["active_session_id"],
        ] + _thumb_panel_outputs() + _session_refresh_outputs,
    )

    # Initial populate of session list when tab is built is handled at app load
    # via a dummy refresh bound after wiring (see build_app).

    def _run(
        lyrics, song_name, style, image_size, image_frequency, steps, cfg_scale, ref_image, hair_style, outfit_worn, negative_prompt, active_session_id,
        progress=gr.Progress(track_tqdm=False),
    ):
        # Idle restore vs running (hide generate trio, show Emergency Stop)
        idle_btns = _action_btn_updates(lyrics, song_name, running=False)
        run_btns = _action_btn_updates(lyrics, song_name, running=True)

        if not _can_create(lyrics, song_name):
            yield (
                "Song name, lyrics, and models are required — fill the song name "
                "and configure Encoder + Diffuser paths.",
                *idle_btns, active_session_id or "",
            ) + tuple(_all_gallery_updates()) + tuple(
                _refresh_session_slots(active_session_id or "")
            )
            return

        c = _cfg()
        cfg = dict(c)
        cfg["style"] = style
        cfg["project_label"] = (song_name or "").strip()
        size_label = configure.normalize_image_size(str(image_size or configure.DEFAULT_IMAGE_SIZE))
        w, h = configure.image_size_pixels(size_label)
        freq = configure.normalize_image_frequency(image_frequency)
        cfg["imagegen_width"] = w
        cfg["imagegen_height"] = h
        cfg["imagegen_size"] = size_label
        cfg["imagegen_frequency"] = freq
        cfg["imagegen_steps"] = int(steps or configure.DEFAULT_STEPS)
        cfg["imagegen_cfg_scale"] = float(cfg_scale or configure.DEFAULT_CFG)
        cfg["prompt_template"] = configure.prompt_template_for_style(style)
        cfg["negative_prompt"] = (
            (negative_prompt if negative_prompt is not None else configure.DEFAULT_NEGATIVE_PROMPT) or ""
        )
        cfg["hair_style"] = configure.normalize_hair_style(str(hair_style or ""))
        cfg["outfit_worn"] = configure.normalize_outfit(str(outfit_worn or ""))
        _g_sub = configure.load_generation()
        cfg["ref_gender"] = configure.normalize_gender(str(_g_sub.get("ref_gender") or ""))
        cfg["ref_bodyshape"] = configure.normalize_bodyshape(str(_g_sub.get("ref_bodyshape") or ""))

        configure.update_generation({
            "last_lyrics": lyrics,
            "project_label": (song_name or "").strip(),
            "reference_image_path": _project_local_ref(ref_image or ""),
            "hair_style": configure.normalize_hair_style(str(hair_style or "")),
            "outfit_worn": configure.normalize_outfit(str(outfit_worn or "")),
            "imagegen_width": w,
            "imagegen_height": h,
            "imagegen_size": size_label,
            "imagegen_frequency": freq,
            "imagegen_steps": cfg["imagegen_steps"],
            "imagegen_cfg_scale": cfg["imagegen_cfg_scale"],
        })
        configure.update_preferences({"style": style})

        # Resume if a session is selected and its folder still exists
        resume_folder = ""
        if active_session_id:
            s = configure.get_session_by_id(active_session_id)
            if s and Path(s["path"]).is_dir():
                resume_folder = s["path"]

        # Size gallery immediately (no_image placeholders for every lyric line)
        try:
            from scripts.inference import parse_lyrics, lyric_lines_only
            n_pre = len(lyric_lines_only(parse_lyrics(lyrics or "")))
            if n_pre > 0:
                configure.APP_STATE["thumb_expected_count"] = n_pre
        except Exception:
            pass
        configure.APP_STATE["thumb_generating_line"] = None
        configure.APP_STATE["regen_busy_lines"] = []
        configure.APP_STATE["thumb_queued_lines"] = []
        # Every lyric line without a still on disk is now listed for generation
        try:
            _n_q = int(configure.APP_STATE.get("thumb_expected_count") or 0)
            _have = _line_path_map(_list_project_images(resume_folder)) if resume_folder else {}
            configure.APP_STATE["thumb_queued_lines"] = [
                k for k in range(1, min(_n_q, THUMB_SLOTS) + 1) if k not in _have
            ]
        except Exception:
            configure.APP_STATE["thumb_queued_lines"] = []

        yield (
            _status_plain("[  0%] start  Starting lyrics slideshow pipeline…"),
            *run_btns, active_session_id or "",
        ) + tuple(_all_gallery_updates()) + tuple(
            _refresh_session_slots(active_session_id or "")
        )

        prog_q: queue.Queue = queue.Queue()
        result_holder: Dict[str, Any] = {}

        def cb(msg: str, frac: float, info: dict) -> None:
            configure.APP_STATE["status_last_msg"] = msg
            configure.APP_STATE["status_last_frac"] = frac
            configure.APP_STATE["status_last_info"] = info or {}
            prog_q.put((msg, frac, info or {}))

        def worker() -> None:
            try:
                result_holder["r"] = inference.run_materials_pipeline(
                    lyrics=lyrics,
                    cfg=cfg,
                    song_name=song_name or "",
                    reference_image=_project_local_ref(ref_image or ""),
                    progress_callback=cb,
                    resume_folder=resume_folder,
                )
            except Exception as e:
                result_holder["r"] = {
                    "success": False,
                    "message": f"Pipeline error: {e}",
                    "project_folder": "",
                    "image_count": 0,
                    "image_paths": [],
                    "session_id": active_session_id or "",
                }
                prog_q.put((f"ERROR: {e}", 1.0, {"phase": "error"}))

        th = threading.Thread(target=worker, daemon=True)
        th.start()

        status_line = _status_plain("[  0%] start  Starting materials pipeline…")
        last_yield = time.time()
        sid = active_session_id or ""
        configure.APP_STATE["generating"] = True
        # Pre-size the gallery to the full lyric line count (placeholders for each slot)
        try:
            from scripts.inference import parse_lyrics, lyric_lines_only
            n_lines = len(lyric_lines_only(parse_lyrics(lyrics or "")))
        except Exception:
            n_lines = 0
        if n_lines > 0:
            configure.APP_STATE["thumb_expected_count"] = n_lines
        configure.APP_STATE["thumb_generating_line"] = None
        configure.APP_STATE["regen_busy_lines"] = []
        configure.APP_STATE["thumb_queued_lines"] = []

        while th.is_alive() or not prog_q.empty():
            updated = False
            try:
                while True:
                    msg, frac, info = prog_q.get_nowait()
                    status_line = _status_from_progress(msg, frac, info)
                    # Track which still is currently generating (1-based)
                    phase = (info or {}).get("phase") or ""
                    line = (info or {}).get("line")
                    if phase in ("images", "regen") and line is not None:
                        try:
                            configure.APP_STATE["thumb_generating_line"] = int(line)
                            _unqueue_line(int(line))
                        except (TypeError, ValueError):
                            pass
                    elif phase in ("done", "error", "analysis", "prompts"):
                        if phase in ("done", "error"):
                            configure.APP_STATE["thumb_generating_line"] = None
                    try:
                        progress(float(frac), desc=msg)
                    except Exception:
                        pass
                    updated = True
            except queue.Empty:
                pass

            if updated or (time.time() - last_yield) > 0.5:
                if configure.APP_STATE.get("image_gen_t0"):
                    status_line = _status_from_progress(
                        configure.APP_STATE.get("status_last_msg") or "Generating still…",
                        float(configure.APP_STATE.get("status_last_frac") or 0.5),
                        configure.APP_STATE.get("status_last_info")
                        or {"phase": "images"},
                    )
                paths = _list_project_images(
                    configure.APP_STATE.get("current_project_folder") or ""
                )
                # Also merge any paths the pipeline reported
                for p in (configure.APP_STATE.get("generation_output_paths") or []):
                    if p and Path(p).exists() and p not in paths:
                        paths.append(p)
                sid = configure.APP_STATE.get("active_session_id") or sid
                yield (
                    status_line,
                    *run_btns, sid,
                ) + tuple(_all_gallery_updates(lyric_paths=paths)) + tuple(_refresh_session_slots(sid))
                last_yield = time.time()
            else:
                time.sleep(0.15)

        r = result_holder.get("r") or {}
        paths = r.get("image_paths") or configure.APP_STATE.get("generation_output_paths") or []
        paths = [p for p in paths if Path(p).exists()]
        msg = r.get("message") or "Done."
        sid = r.get("session_id") or configure.APP_STATE.get("active_session_id") or sid
        if r.get("success"):
            configure.update_generation({
                "last_project_folder": r.get("project_folder") or "",
            })
            configure.APP_STATE["active_session_id"] = sid
        configure.APP_STATE["generating"] = False
        # Run finished/stopped/failed: nothing from the batch stays queued
        configure.APP_STATE["thumb_queued_lines"] = []

        # Drain regenerate queue (queued while this run was busy)
        q = list(configure.APP_STATE.get("regen_queue") or [])
        configure.APP_STATE["regen_queue"] = []
        if q and not inference.is_cancel_requested():
            # Drop invalid jobs before running (int line_idx or {line_idx, variant})
            valid_q = []
            for job in q:
                try:
                    if isinstance(job, dict):
                        qi = int(job.get("line_idx", -1))
                        var = int(job.get("variant") or 1)
                        if qi < 0:
                            continue
                        valid_q.append({"line_idx": qi, "variant": var})
                    else:
                        qi = int(job)
                        if qi < 0:
                            continue
                        valid_q.append({"line_idx": qi, "variant": 1})
                except (TypeError, ValueError):
                    continue
            q = valid_q
            configure.APP_STATE["thumb_queued_lines"] = [j["line_idx"] + 1 for j in q]
            if q:
                msg = f"{msg}  Processing {len(q)} queued regenerate(s)…"
                yield (
                    msg, *idle_btns, sid,
                ) + tuple(_all_gallery_updates(lyric_paths=paths)) + tuple(_refresh_session_slots(sid))
            for job in q:
                if inference.is_cancel_requested():
                    break
                line_idx = int(job["line_idx"])
                var = int(job.get("variant") or 1)
                try:
                    configure.APP_STATE["generating"] = True
                    _unqueue_line(line_idx + 1)
                    proj = configure.APP_STATE.get("current_project_folder") or ""
                    cfg_now = configure.load_configuration()
                    gnow = configure.load_generation()
                    cfg_now["imagegen_steps"] = steps
                    cfg_now["imagegen_cfg_scale"] = cfg_scale
                    _sz = configure.normalize_image_size(str(image_size or configure.DEFAULT_IMAGE_SIZE))
                    _w, _h = configure.image_size_pixels(_sz)
                    _fq = configure.normalize_image_frequency(image_frequency)
                    cfg_now["imagegen_width"] = _w
                    cfg_now["imagegen_height"] = _h
                    cfg_now["imagegen_size"] = _sz
                    cfg_now["imagegen_frequency"] = _fq
                    cfg_now["negative_prompt"] = (
                        (negative_prompt if negative_prompt is not None else configure.DEFAULT_NEGATIVE_PROMPT) or ""
                    )
                    cfg_now["hair_style"] = configure.normalize_hair_style(str(gnow.get("hair_style") or ""))
                    cfg_now["outfit_worn"] = configure.normalize_outfit(str(gnow.get("outfit_worn") or ""))
                    cfg_now["ref_gender"] = configure.normalize_gender(str(gnow.get("ref_gender") or ""))
                    cfg_now["ref_bodyshape"] = configure.normalize_bodyshape(str(gnow.get("ref_bodyshape") or ""))
                    inference.regenerate_single_still(
                        Path(proj), line_idx, cfg_now,
                        reference_image=_project_local_ref(ref_image or ""),
                        variant=var,
                    )
                    paths = _list_project_images(proj)
                    yield (
                        f"Regenerated still {line_idx + 1} variant {var}.",
                        *idle_btns, sid,
                    ) + tuple(_all_gallery_updates(lyric_paths=paths)) + tuple(_refresh_session_slots(sid))
                except Exception as e:
                    yield (
                        f"Queued regenerate line {line_idx + 1} failed: {e}",
                        *idle_btns, sid,
                    ) + tuple(_all_gallery_updates(lyric_paths=paths)) + tuple(_refresh_session_slots(sid))
                finally:
                    configure.APP_STATE["generating"] = False

        configure.APP_STATE["thumb_generating_line"] = None
        configure.APP_STATE["regen_busy_lines"] = []
        configure.APP_STATE["thumb_queued_lines"] = []
        paths = _list_project_images(configure.APP_STATE.get("current_project_folder") or "")
        idle_btns = _action_btn_updates(lyrics, song_name, running=False)
        yield (
            msg,
            *idle_btns, sid,
        ) + tuple(_all_gallery_updates(lyric_paths=paths)) + tuple(_refresh_session_slots(sid))

    _run_outputs = [
        status_box,
        _gen["assess_btn"],
        _gen["all_assets_btn"],
        _gen["cover_btn"],
        _gen["theme_btn"],
        _gen["run_btn"],
        _gen["stop_btn"],
        _gen["active_session_id"],
    ] + _thumb_panel_outputs() + _session_refresh_outputs

    def _shared_gen_inputs():
        return [
            _gen["lyrics"],
            _gen["song_name"],
            _gen["style"],
            _gen["image_size"],
            _gen["image_frequency"],
            _gen["steps"],
            _gen["cfg"],
            _gen["ref_image"],
            _gen["hair_style"],
            _gen["outfit_worn"],
            _gen["negative_prompt"],
            _gen["active_session_id"],
        ]

    def _persist_subject_tokens(hair_style, outfit_worn, ref_gender, ref_bodyshape):
        configure.update_generation({
            "hair_style": configure.normalize_hair_style(str(hair_style or "")),
            "outfit_worn": configure.normalize_outfit(str(outfit_worn or "")),
            "ref_gender": configure.normalize_gender(str(ref_gender or "")),
            "ref_bodyshape": configure.normalize_bodyshape(str(ref_bodyshape or "")),
        })

    _subject_token_inputs = [
        _gen["hair_style"], _gen["outfit_worn"],
        _gen["ref_gender"], _gen["ref_bodyshape"],
    ]
    for _tok in ("hair_style", "outfit_worn", "ref_gender", "ref_bodyshape"):
        if _gen.get(_tok) is not None:
            _gen[_tok].change(
                _persist_subject_tokens,
                inputs=_subject_token_inputs,
                outputs=[],
            )

    def _on_image_frequency_change(image_frequency, lyrics, song_name, active_session_id):
        """Persist frequency and refresh Cover/Theme/Lyrics grids + action labels.

        Raising frequency expands empty no_image slots and switches buttons to
        "Complete … Images" when existing stills fall short. Lowering frequency
        slims the grids to the new expected count (extra files remain on disk).
        """
        freq = configure.normalize_image_frequency(image_frequency)
        configure.update_generation({"imagegen_frequency": freq})
        # Drop batch-pinned expectations so the dropdown is the display authority
        configure.APP_STATE["cover_slot_expected"] = 0
        configure.APP_STATE["theme_slot_expected"] = 0
        try:
            # Refresh lyrics expected from current lyrics × L
            n_lines = 0
            text = (lyrics or "").strip()
            if not text:
                proj = (configure.APP_STATE.get("current_project_folder") or "").strip()
                if proj and (Path(proj) / "lyrics.txt").is_file():
                    text = (Path(proj) / "lyrics.txt").read_text(
                        encoding="utf-8", errors="replace"
                    )
            if text.strip():
                from scripts.inference import parse_lyrics, lyric_lines_only
                n_lines = len(lyric_lines_only(parse_lyrics(text)))
            if n_lines > 0:
                # Grid slots = lyric lines only; L2/L3 variants use the page switcher
                configure.APP_STATE["thumb_expected_count"] = n_lines
            per = configure.frequency_lyrics_per_line(freq)
            # Keep current page when possible; clamp to new L range
            try:
                cur = int(configure.APP_STATE.get("lyrics_page") or 1)
            except (TypeError, ValueError):
                cur = 1
            configure.APP_STATE["lyrics_page"] = max(1, min(cur, max(1, per)))
        except Exception:
            pass
        pad_status = (
            f"Image frequency set to {freq} — "
            f"Cover {configure.frequency_cover_count(freq)}, "
            f"Theme {configure.frequency_theme_count(freq)}, "
            f"Lyrics×{configure.frequency_lyrics_per_line(freq)}."
        )
        return (
            pad_status,
            *_action_btn_updates(lyrics or "", song_name or "", running=False),
            active_session_id or "",
        ) + tuple(_all_gallery_updates()) + tuple(
            _refresh_session_slots(active_session_id or "")
        )

    if _gen.get("image_frequency") is not None:
        _gen["image_frequency"].change(
            _on_image_frequency_change,
            inputs=[
                _gen["image_frequency"],
                _gen["lyrics"],
                _gen["song_name"],
                _gen["active_session_id"],
            ],
            outputs=[
                status_box,
                _gen["assess_btn"],
                _gen["all_assets_btn"],
                _gen["cover_btn"],
                _gen["theme_btn"],
                _gen["run_btn"],
                _gen["stop_btn"],
                _gen["active_session_id"],
            ] + _thumb_panel_outputs() + _session_refresh_outputs,
        )

    def _on_lyrics_page_change(page_label, active_session_id):
        """Switch Lyrics Thumbnails to the selected variant page (L2/L3)."""
        label = str(page_label or "").strip()
        m = re.search(r"(\d+)", label)
        page = int(m.group(1)) if m else 1
        page = _set_lyrics_page(page)
        return (
            f"Showing lyrics stills — {_lyrics_page_label(page)} "
            f"(of {_lyrics_per_line_count()}).",
            active_session_id or "",
        ) + tuple(_all_gallery_updates()) + tuple(
            _refresh_session_slots(active_session_id or "")
        )

    if _gen.get("lyrics_page_radio") is not None:
        _gen["lyrics_page_radio"].change(
            _on_lyrics_page_change,
            inputs=[_gen["lyrics_page_radio"], _gen["active_session_id"]],
            outputs=[
                status_box,
                _gen["active_session_id"],
            ] + _thumb_panel_outputs() + _session_refresh_outputs,
        )

    _gen["run_btn"].click(
        _run,
        inputs=_shared_gen_inputs(),
        outputs=_run_outputs,
    )

    def _run_cover(
        lyrics, song_name, style, image_size, image_frequency, steps, cfg_scale, ref_image, hair_style, outfit_worn, negative_prompt, active_session_id,
        progress=gr.Progress(track_tqdm=False),
    ):
        idle_btns = _action_btn_updates(lyrics, song_name, running=False)
        run_btns = _action_btn_updates(lyrics, song_name, running=True)
        if not _can_cover(song_name):
            yield (
                _status_plain("Song name and models are required for a cover image."),
                *idle_btns, active_session_id or "",
            ) + tuple(_all_gallery_updates()) + tuple(
                _refresh_session_slots(active_session_id or "")
            )
            return
        c = _cfg()
        cfg = dict(c)
        cfg["style"] = style
        cfg["project_label"] = (song_name or "").strip()
        size_label = configure.normalize_image_size(str(image_size or configure.DEFAULT_IMAGE_SIZE))
        w, h = configure.image_size_pixels(size_label)
        freq = configure.normalize_image_frequency(image_frequency)
        cfg["imagegen_width"] = w
        cfg["imagegen_height"] = h
        cfg["imagegen_size"] = size_label
        cfg["imagegen_frequency"] = freq
        cfg["imagegen_steps"] = int(steps or configure.DEFAULT_STEPS)
        cfg["imagegen_cfg_scale"] = float(cfg_scale or configure.DEFAULT_CFG)
        cfg["prompt_template"] = configure.prompt_template_for_style(style)
        cfg["negative_prompt"] = (
            (negative_prompt if negative_prompt is not None else configure.DEFAULT_NEGATIVE_PROMPT) or ""
        )
        cfg["hair_style"] = configure.normalize_hair_style(str(hair_style or ""))
        cfg["outfit_worn"] = configure.normalize_outfit(str(outfit_worn or ""))
        _g_sub = configure.load_generation()
        cfg["ref_gender"] = configure.normalize_gender(str(_g_sub.get("ref_gender") or ""))
        cfg["ref_bodyshape"] = configure.normalize_bodyshape(str(_g_sub.get("ref_bodyshape") or ""))
        configure.update_generation({
            "project_label": (song_name or "").strip(),
            "reference_image_path": _project_local_ref(ref_image or ""),
            "hair_style": configure.normalize_hair_style(str(hair_style or "")),
            "outfit_worn": configure.normalize_outfit(str(outfit_worn or "")),
            "imagegen_width": w,
            "imagegen_height": h,
            "imagegen_size": size_label,
            "imagegen_frequency": freq,
            "imagegen_steps": cfg["imagegen_steps"],
            "imagegen_cfg_scale": cfg["imagegen_cfg_scale"],
        })
        configure.update_preferences({"style": style})
        resume_folder = ""
        if active_session_id:
            s = configure.get_session_by_id(active_session_id)
            if s and Path(s["path"]).is_dir():
                resume_folder = s["path"]
        yield (
            _status_from_progress("Generating cover image…", 0.0, {"phase": "cover", "line": 1, "total": 1}),
            *run_btns, active_session_id or "",
        ) + tuple(_all_gallery_updates()) + tuple(
            _refresh_session_slots(active_session_id or "")
        )
        prog_q: queue.Queue = queue.Queue()
        result_holder: Dict[str, Any] = {}

        def _cb(msg, frac, info=None):
            configure.APP_STATE["status_last_msg"] = msg
            configure.APP_STATE["status_last_frac"] = frac
            configure.APP_STATE["status_last_info"] = info or {}
            prog_q.put((_status_from_progress(msg, frac, info or {}), frac, info or {}))

        def _worker():
            try:
                result_holder["r"] = inference.generate_cover_image(
                    song_name or "",
                    cfg,
                    reference_image=_project_local_ref(ref_image or ""),
                    progress_callback=_cb,
                    resume_folder=resume_folder,
                    lyrics=lyrics or "",
                )
            except Exception as e:
                result_holder["r"] = {"success": False, "message": str(e)}
            finally:
                prog_q.put(None)

        threading.Thread(target=_worker, daemon=True).start()
        status_line = _status_from_progress(
            "Generating still…", 0.05, {"phase": "cover", "line": 1, "total": 1},
        )
        while True:
            try:
                item = prog_q.get(timeout=0.25)
            except queue.Empty:
                item = "__tick__"
            if item is None:
                break
            if item != "__tick__":
                status_line, _frac, _info = item
            elif configure.APP_STATE.get("image_gen_t0"):
                status_line = _status_from_progress(
                    configure.APP_STATE.get("status_last_msg") or "Generating still…",
                    float(configure.APP_STATE.get("status_last_frac") or 0.45),
                    configure.APP_STATE.get("status_last_info") or {"phase": "cover"},
                )
            else:
                continue
            sid = configure.APP_STATE.get("active_session_id") or active_session_id or ""
            yield (
                status_line, *run_btns, sid,
            ) + tuple(_all_gallery_updates()) + tuple(_refresh_session_slots(sid))
        r = result_holder.get("r") or {}
        msg = r.get("message") or "Cover done."
        sid = r.get("session_id") or configure.APP_STATE.get("active_session_id") or active_session_id or ""
        if r.get("success") and r.get("project_folder"):
            configure.APP_STATE["current_project_folder"] = r["project_folder"]
            configure.update_generation({"last_project_folder": r["project_folder"]})
        idle_btns = _action_btn_updates(lyrics, song_name, running=False)
        yield (
            _status_plain(msg), *idle_btns, sid,
        ) + tuple(_all_gallery_updates()) + tuple(_refresh_session_slots(sid))

    def _run_theme(
        lyrics, song_name, style, image_size, image_frequency, steps, cfg_scale, ref_image, hair_style, outfit_worn, negative_prompt, active_session_id,
        progress=gr.Progress(track_tqdm=False),
    ):
        idle_btns = _action_btn_updates(lyrics, song_name, running=False)
        run_btns = _action_btn_updates(lyrics, song_name, running=True)
        if not _can_theme(lyrics, song_name):
            yield (
                _status_plain("Song name, lyrics, and models are required for theme images."),
                *idle_btns, active_session_id or "",
            ) + tuple(_all_gallery_updates()) + tuple(
                _refresh_session_slots(active_session_id or "")
            )
            return
        c = _cfg()
        cfg = dict(c)
        cfg["style"] = style
        cfg["project_label"] = (song_name or "").strip()
        size_label = configure.normalize_image_size(str(image_size or configure.DEFAULT_IMAGE_SIZE))
        w, h = configure.image_size_pixels(size_label)
        freq = configure.normalize_image_frequency(image_frequency)
        cfg["imagegen_width"] = w
        cfg["imagegen_height"] = h
        cfg["imagegen_size"] = size_label
        cfg["imagegen_frequency"] = freq
        cfg["imagegen_steps"] = int(steps or configure.DEFAULT_STEPS)
        cfg["imagegen_cfg_scale"] = float(cfg_scale or configure.DEFAULT_CFG)
        cfg["prompt_template"] = configure.prompt_template_for_style(style)
        cfg["negative_prompt"] = (
            (negative_prompt if negative_prompt is not None else configure.DEFAULT_NEGATIVE_PROMPT) or ""
        )
        cfg["hair_style"] = configure.normalize_hair_style(str(hair_style or ""))
        cfg["outfit_worn"] = configure.normalize_outfit(str(outfit_worn or ""))
        _g_sub = configure.load_generation()
        cfg["ref_gender"] = configure.normalize_gender(str(_g_sub.get("ref_gender") or ""))
        cfg["ref_bodyshape"] = configure.normalize_bodyshape(str(_g_sub.get("ref_bodyshape") or ""))
        configure.update_generation({
            "last_lyrics": lyrics,
            "project_label": (song_name or "").strip(),
            "reference_image_path": _project_local_ref(ref_image or ""),
            "hair_style": configure.normalize_hair_style(str(hair_style or "")),
            "outfit_worn": configure.normalize_outfit(str(outfit_worn or "")),
            "imagegen_width": w,
            "imagegen_height": h,
            "imagegen_size": size_label,
            "imagegen_frequency": freq,
            "imagegen_steps": cfg["imagegen_steps"],
            "imagegen_cfg_scale": cfg["imagegen_cfg_scale"],
        })
        configure.update_preferences({"style": style})
        resume_folder = ""
        if active_session_id:
            s = configure.get_session_by_id(active_session_id)
            if s and Path(s["path"]).is_dir():
                resume_folder = s["path"]
        yield (
            _status_from_progress(
                "Analysing song & generating theme images…",
                0.0,
                {"phase": "theme"},
            ),
            *run_btns, active_session_id or "",
        ) + tuple(_all_gallery_updates()) + tuple(
            _refresh_session_slots(active_session_id or "")
        )
        prog_q: queue.Queue = queue.Queue()
        result_holder: Dict[str, Any] = {}

        def _cb(msg, frac, info=None):
            configure.APP_STATE["status_last_msg"] = msg
            configure.APP_STATE["status_last_frac"] = frac
            configure.APP_STATE["status_last_info"] = info or {}
            prog_q.put((_status_from_progress(msg, frac, info or {}), frac, info or {}))

        def _worker():
            try:
                result_holder["r"] = inference.generate_theme_images(
                    lyrics or "",
                    cfg,
                    song_name=song_name or "",
                    reference_image=_project_local_ref(ref_image or ""),
                    progress_callback=_cb,
                    resume_folder=resume_folder,
                )
            except Exception as e:
                result_holder["r"] = {"success": False, "message": str(e)}
            finally:
                prog_q.put(None)

        threading.Thread(target=_worker, daemon=True).start()
        while True:
            try:
                item = prog_q.get(timeout=0.2)
            except queue.Empty:
                item = "__tick__"
            if item is None:
                break
            if item != "__tick__":
                status_line, _frac, _info = item
            elif configure.APP_STATE.get("image_gen_t0"):
                status_line = _status_from_progress(
                    configure.APP_STATE.get("status_last_msg") or "Generating still…",
                    float(configure.APP_STATE.get("status_last_frac") or 0.45),
                    configure.APP_STATE.get("status_last_info") or {"phase": "theme"},
                )
            else:
                continue
            sid = configure.APP_STATE.get("active_session_id") or active_session_id or ""
            yield (
                status_line, *run_btns, sid,
            ) + tuple(_all_gallery_updates()) + tuple(_refresh_session_slots(sid))
        r = result_holder.get("r") or {}
        msg = r.get("message") or "Theme images done."
        sid = r.get("session_id") or configure.APP_STATE.get("active_session_id") or active_session_id or ""
        if r.get("success") and r.get("project_folder"):
            configure.APP_STATE["current_project_folder"] = r["project_folder"]
            configure.update_generation({"last_project_folder": r["project_folder"]})
        idle_btns = _action_btn_updates(lyrics, song_name, running=False)
        yield (
            _status_plain(msg), *idle_btns, sid,
        ) + tuple(_all_gallery_updates()) + tuple(_refresh_session_slots(sid))


    def _run_assessment(
        lyrics, song_name, style, image_size, image_frequency, steps, cfg_scale, ref_image, hair_style, outfit_worn, negative_prompt, active_session_id,
        progress=gr.Progress(track_tqdm=False),
    ):
        """Run song assessment only; enable generate buttons when analysis.txt is saved."""
        idle_btns = _action_btn_updates(lyrics, song_name, running=False)
        run_btns = _action_btn_updates(lyrics, song_name, running=True)
        # Pad Details Mode outputs: mode, settings, nl, assess, ref, assess_view
        _no_mode = (gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update())
        if not _models_configured():
            yield (
                _status_plain("Configure Encoder + Diffuser models on the Configuration tab first."),
                *idle_btns, active_session_id or "",
            ) + tuple(_all_gallery_updates()) + tuple(_refresh_session_slots(active_session_id or "")) + _no_mode
            return
        if not (song_name or "").strip() or not (lyrics or "").strip():
            yield (
                _status_plain("Song name and lyrics are required to run assessment."),
                *idle_btns, active_session_id or "",
            ) + tuple(_all_gallery_updates()) + tuple(_refresh_session_slots(active_session_id or "")) + _no_mode
            return

        configure.APP_STATE["generating"] = True
        configure.APP_STATE["session_status"] = "running"
        yield (
            _status_plain("Running song assessment…"),
            *run_btns, active_session_id or "",
        ) + tuple(_all_gallery_updates()) + tuple(_refresh_session_slots(active_session_id or "")) + _no_mode

        import queue
        import threading
        prog_q: queue.Queue = queue.Queue()
        result_holder: dict = {}

        def _cb(msg, frac=0.0, info=None):
            try:
                configure.APP_STATE["status_last_msg"] = msg
                configure.APP_STATE["status_last_frac"] = frac
                configure.APP_STATE["status_last_info"] = info or {}
                prog_q.put((_status_from_progress(msg, frac, info or {}), frac, info or {}))
            except Exception:
                pass

        def _worker():
            try:
                from scripts import inference as inf
                cfg = configure.load_configuration()
                size_label = configure.normalize_image_size(str(image_size or configure.DEFAULT_IMAGE_SIZE))
                w, h = configure.image_size_pixels(size_label)
                cfg["imagegen_width"] = w
                cfg["imagegen_height"] = h
                cfg["imagegen_size"] = size_label
                cfg["imagegen_frequency"] = configure.normalize_image_frequency(image_frequency)
                cfg["imagegen_steps"] = int(steps or configure.DEFAULT_STEPS)
                cfg["hair_style"] = configure.normalize_hair_style(str(hair_style or ""))
                cfg["outfit_worn"] = configure.normalize_outfit(str(outfit_worn or ""))
                cfg["imagegen_cfg_scale"] = float(cfg_scale or configure.DEFAULT_CFG)
                cfg["style"] = style or cfg.get("style") or configure.STYLE_LIGHT
                cfg["negative_prompt"] = (
                    (negative_prompt if negative_prompt is not None else configure.DEFAULT_NEGATIVE_PROMPT) or ""
                )
                label = (song_name or "").strip()
                from scripts.inference import ensure_project_dir, _slugify_folder_name, ensure_project_assessment
                slug = _slugify_folder_name(label, fallback="")
                resume = ""
                if active_session_id:
                    root = configure.get_output_dir()
                    cand = Path(root) / str(active_session_id)
                    if cand.is_dir():
                        resume = str(cand)
                project_dir = ensure_project_dir(slug, sequential=False, resume_path=resume or None)
                configure.APP_STATE["current_project_folder"] = str(project_dir)
                configure.APP_STATE["active_session_id"] = project_dir.name
                has_ref = bool(ref_image and Path(ref_image).is_file())
                analysis = ensure_project_assessment(
                    project_dir,
                    lyrics or "",
                    cfg,
                    has_character_ref=has_ref,
                    progress_callback=_cb,
                    force=True,
                )
                result_holder["r"] = {
                    "success": bool(analysis and (analysis.get("overall") or analysis.get("sections"))),
                    "message": "Assessment complete — Details Mode switched to Song Assessment. Edit if needed, then Save.",
                    "project_folder": str(project_dir),
                    "session_id": project_dir.name,
                    "assessment_text": (project_dir / "analysis.txt").read_text(encoding="utf-8", errors="replace")
                    if (project_dir / "analysis.txt").is_file() else "",
                }
            except Exception as e:
                import traceback
                traceback.print_exc()
                result_holder["r"] = {"success": False, "message": f"Assessment failed: {e}"}
            finally:
                try:
                    prog_q.put(None)
                except Exception:
                    pass

        th = threading.Thread(target=_worker, daemon=True)
        th.start()
        while True:
            try:
                item = prog_q.get(timeout=0.25)
            except queue.Empty:
                if not th.is_alive():
                    break
                continue
            if item is None:
                break
            status_line, frac, info = item
            yield (
                status_line, *run_btns, active_session_id or "",
            ) + tuple(_all_gallery_updates()) + tuple(_refresh_session_slots(active_session_id or "")) + _no_mode

        th.join(timeout=5)
        configure.APP_STATE["generating"] = False
        configure.APP_STATE["session_status"] = "stopped"
        r = result_holder.get("r") or {}
        msg = r.get("message") or "Assessment done."
        sid = r.get("session_id") or configure.APP_STATE.get("active_session_id") or active_session_id or ""
        if r.get("success") and r.get("project_folder"):
            configure.APP_STATE["current_project_folder"] = r["project_folder"]
            configure.update_generation({"last_project_folder": r["project_folder"]})
        idle_btns = _action_btn_updates(lyrics, song_name, running=False)
        # Auto-switch Details Mode → Song Assessment and load the text
        assess_text = r.get("assessment_text") or _load_assessment_text(
            r.get("project_folder") or ""
        )
        yield (
            _status_plain(msg), *idle_btns, sid,
        ) + tuple(_all_gallery_updates()) + tuple(_refresh_session_slots(sid)) + (
            gr.update(value="Song Assessment"),
            gr.update(visible=False),  # Project Settings
            gr.update(visible=False),  # Name and Lyrics
            gr.update(visible=True),   # Song Assessment
            gr.update(visible=False),  # Reference Character
            gr.update(value=assess_text),
        )

    _assess_outputs = _run_outputs + [
        _gen["details_mode"],
        _gen["details_project_settings"],
        _gen["details_name_lyrics"],
        _gen["details_assessment"],
        _gen["details_reference"],
        _gen["assessment_view"],
    ]

    _gen["assess_btn"].click(
        _run_assessment,
        inputs=_shared_gen_inputs(),
        outputs=_assess_outputs,
    )

    def _run_all_assets(
        lyrics, song_name, style, image_size, image_frequency, steps, cfg_scale, ref_image, hair_style, outfit_worn, negative_prompt, active_session_id,
        progress=gr.Progress(track_tqdm=False),
    ):
        """
        Generate (or re-generate) Cover + Theme + Lyrics stills.
        If the project already has any images, delete them first, then re-create
        using the current assessment / settings.
        """
        idle_btns = _action_btn_updates(lyrics, song_name, running=False)
        run_btns = _action_btn_updates(lyrics, song_name, running=True)
        if not _models_configured():
            yield (
                _status_plain("Configure Encoder + Diffuser models on the Configuration tab first."),
                *idle_btns, active_session_id or "",
            ) + tuple(_all_gallery_updates()) + tuple(_refresh_session_slots(active_session_id or ""))
            return
        if not _assessment_exists():
            yield (
                _status_plain("Run Assessment first, then Generate All Assets."),
                *idle_btns, active_session_id or "",
            ) + tuple(_all_gallery_updates()) + tuple(_refresh_session_slots(active_session_id or ""))
            return
        if not (song_name or "").strip():
            yield (
                _status_plain("Song name is required."),
                *idle_btns, active_session_id or "",
            ) + tuple(_all_gallery_updates()) + tuple(_refresh_session_slots(active_session_id or ""))
            return

        configure.APP_STATE["generating"] = True
        configure.APP_STATE["session_status"] = "running"
        assets_complete = _all_assets_complete(lyrics)
        label = "Re-Generate All Assets" if assets_complete else "Generate All Assets"
        yield (
            _status_plain(f"{label}…"),
            *run_btns, active_session_id or "",
        ) + tuple(_all_gallery_updates()) + tuple(_refresh_session_slots(active_session_id or ""))

        # Persist generation settings
        freq = configure.normalize_image_frequency(image_frequency)
        w, h = configure.image_size_pixels(image_size)
        cfg = configure.generation_config()
        cfg.update({
            "style": style or configure.STYLE_DEFAULT,
            "imagegen_size": configure.normalize_image_size(str(image_size or "")),
            "imagegen_width": w,
            "imagegen_height": h,
            "imagegen_frequency": freq,
            "imagegen_steps": int(steps or configure.DEFAULT_STEPS),
            "imagegen_cfg_scale": float(cfg_scale or configure.DEFAULT_CFG),
            "negative_prompt": negative_prompt or configure.DEFAULT_NEGATIVE_PROMPT,
            "hair_style": configure.normalize_hair_style(str(hair_style or "")),
            "outfit_worn": configure.normalize_outfit(str(outfit_worn or "")),
            "ref_gender": configure.normalize_gender(str(configure.load_generation().get("ref_gender") or "")),
            "ref_bodyshape": configure.normalize_bodyshape(str(configure.load_generation().get("ref_bodyshape") or "")),
        })
        configure.update_generation({
            "imagegen_size": cfg["imagegen_size"],
            "imagegen_width": w,
            "imagegen_height": h,
            "imagegen_frequency": freq,
            "imagegen_steps": cfg["imagegen_steps"],
            "imagegen_cfg_scale": cfg["imagegen_cfg_scale"],
            "hair_style": cfg["hair_style"],
            "outfit_worn": cfg["outfit_worn"],
        })
        configure.update_preferences({"style": cfg["style"]})

        # Resolve / ensure project folder
        folder = (configure.APP_STATE.get("current_project_folder") or "").strip()
        if not folder and active_session_id:
            cand = configure.get_output_dir() / str(active_session_id)
            if cand.is_dir():
                folder = str(cand)
        # Only wipe existing stills when every type is already complete (true re-gen)
        if assets_complete and folder:
            n_del = inference.clear_project_image_assets(Path(folder))
            print(f"[all-assets] cleared {n_del} image asset(s) from {folder}", flush=True)
            yield (
                _status_plain(f"Cleared {n_del} existing asset(s) — regenerating…"),
                *run_btns, active_session_id or "",
            ) + tuple(_all_gallery_updates()) + tuple(_refresh_session_slots(active_session_id or ""))

        import queue
        import threading
        prog_q: queue.Queue = queue.Queue()
        result_holder: dict = {"paths": [], "messages": []}

        def _cb(msg, frac=0.0, info=None):
            try:
                configure.APP_STATE["status_last_msg"] = msg
                configure.APP_STATE["status_last_frac"] = frac
                configure.APP_STATE["status_last_info"] = info or {}
                prog_q.put((_status_from_progress(msg, frac, info or {}), frac, info or {}))
            except Exception:
                pass

        def _worker():
            try:
                # 1) Cover
                r1 = inference.generate_cover_image(
                    song_name=song_name or "",
                    cfg=cfg,
                    reference_image=_project_local_ref(ref_image or ""),
                    progress_callback=_cb,
                    resume_folder=folder or "",
                    lyrics=lyrics or "",
                )
                if r1.get("project_folder"):
                    folder_local = r1["project_folder"]
                    configure.APP_STATE["current_project_folder"] = folder_local
                result_holder["messages"].append(r1.get("message") or "Cover done.")
                result_holder["paths"].extend(r1.get("image_paths") or [])
                if inference.is_cancel_requested():
                    return
                # 2) Theme
                r2 = inference.generate_theme_images(
                    lyrics=lyrics or "",
                    cfg=cfg,
                    song_name=song_name or "",
                    reference_image=_project_local_ref(ref_image or ""),
                    progress_callback=_cb,
                    resume_folder=folder or configure.APP_STATE.get("current_project_folder") or "",
                )
                result_holder["messages"].append(r2.get("message") or "Theme done.")
                result_holder["paths"].extend(r2.get("image_paths") or [])
                if inference.is_cancel_requested():
                    return
                # 3) Lyrics slideshow
                r3 = inference.run_materials_pipeline(
                    lyrics=lyrics or "",
                    cfg=cfg,
                    song_name=song_name or "",
                    reference_image=_project_local_ref(ref_image or ""),
                    progress_callback=_cb,
                    resume_folder=folder or configure.APP_STATE.get("current_project_folder") or "",
                )
                result_holder["messages"].append(r3.get("message") or "Lyrics done.")
                result_holder["paths"].extend(r3.get("image_paths") or [])
                result_holder["r"] = r3
            except Exception as e:
                import traceback
                traceback.print_exc()
                result_holder["r"] = {"success": False, "message": f"All-assets error: {e}"}
            finally:
                try:
                    prog_q.put(None)
                except Exception:
                    pass

        th = threading.Thread(target=_worker, daemon=True)
        th.start()
        while True:
            try:
                item = prog_q.get(timeout=0.25)
            except queue.Empty:
                if not th.is_alive():
                    break
                continue
            if item is None:
                break
            status_line, frac, info = item
            sid = configure.APP_STATE.get("active_session_id") or active_session_id or ""
            yield (
                status_line, *run_btns, sid,
            ) + tuple(_all_gallery_updates()) + tuple(_refresh_session_slots(sid))

        th.join(timeout=5)
        configure.APP_STATE["generating"] = False
        configure.APP_STATE["session_status"] = "stopped"
        r = result_holder.get("r") or {}
        msg = " · ".join(result_holder.get("messages") or []) or r.get("message") or "All assets done."
        sid = r.get("session_id") or configure.APP_STATE.get("active_session_id") or active_session_id or ""
        if r.get("project_folder"):
            configure.APP_STATE["current_project_folder"] = r["project_folder"]
            configure.update_generation({"last_project_folder": r["project_folder"]})
        idle_btns = _action_btn_updates(lyrics, song_name, running=False)
        yield (
            _status_plain(msg), *idle_btns, sid,
        ) + tuple(_all_gallery_updates()) + tuple(_refresh_session_slots(sid))

    _gen["all_assets_btn"].click(
        _run_all_assets,
        inputs=_shared_gen_inputs(),
        outputs=_run_outputs,
    )

    _gen["cover_btn"].click(
        _run_cover,
        inputs=_shared_gen_inputs(),
        outputs=_run_outputs,
    )
    _gen["theme_btn"].click(
        _run_theme,
        inputs=_shared_gen_inputs(),
        outputs=_run_outputs,
    )


    def _resolve_line_from_slot(line_idx: int, paths: list = None) -> int | None:
        """Map thumbnail slot → 0-based lyric line index.

        Gallery is sequential: slot 0 = line 1, slot 1 = line 2, …
        """
        if line_idx < 0 or line_idx >= THUMB_SLOTS:
            return None
        expected = _thumb_expected_count()
        if expected > 0 and line_idx >= expected:
            return None
        return line_idx

    def _mark_busy(line_no_1based: int, on: bool) -> None:
        busy = set(int(x) for x in (configure.APP_STATE.get("regen_busy_lines") or []))
        if on:
            busy.add(int(line_no_1based))
        else:
            busy.discard(int(line_no_1based))
        configure.APP_STATE["regen_busy_lines"] = sorted(busy)

    def _do_remove(line_idx: int, active_session_id):
        """Delete one still file; Generate Materials will refill gaps."""
        proj = (configure.APP_STATE.get("current_project_folder") or "").strip()
        if not proj or not Path(proj).is_dir():
            paths = _list_project_images()
            return (
                "No active project folder.",
                active_session_id or "",
            ) + tuple(_all_gallery_updates(lyric_paths=paths)) + tuple(
                _refresh_session_slots(active_session_id or "")
            )
        paths = _list_project_images(proj)
        resolved = _resolve_line_from_slot(line_idx, paths)
        if resolved is None:
            return (
                "Nothing to remove in that slot.",
                active_session_id or "",
            ) + tuple(_all_gallery_updates(lyric_paths=paths)) + tuple(
                _refresh_session_slots(active_session_id or "")
            )
        line_no = resolved + 1
        page = _lyrics_page_current()
        per = _lyrics_per_line_count()
        removed = []
        for oldf in list(Path(proj).iterdir()):
            if not oldf.is_file():
                continue
            if oldf.suffix.lower() not in (".png", ".jpg", ".jpeg", ".webp"):
                continue
            ln, var = _parse_still_line_variant(oldf.name)
            if ln != line_no:
                continue
            # L1 or single-file legacy: remove all for the line.
            # L2/L3: only the variant on the current Lyrics Image page.
            if per > 1 and var != page:
                continue
            try:
                oldf.unlink()
                removed.append(oldf.name)
            except OSError as e:
                print(f"[remove] {oldf}: {e}", flush=True)
        # Drop from APP_STATE paths
        kept = []
        for p in (configure.APP_STATE.get("generation_output_paths") or []):
            try:
                ln, var = _parse_still_line_variant(Path(p).name)
                if ln == line_no and (per <= 1 or var == page):
                    continue
            except Exception:
                pass
            kept.append(p)
        configure.APP_STATE["generation_output_paths"] = kept
        paths = _list_project_images(proj)
        try:
            configure.save_session_meta(Path(proj), {
                "images_done": len(paths),
            })
        except Exception:
            pass
        # Keep slot visible with no_image placeholder
        exp = int(configure.APP_STATE.get("thumb_expected_count") or 0)
        if line_no > exp:
            configure.APP_STATE["thumb_expected_count"] = line_no
        var_bit = f" (page {page} / variant {page})" if per > 1 else ""
        msg = (
            f"Removed still {line_no}{var_bit}"
            + (f" ({', '.join(removed)})" if removed else "")
            + ". Slot shows placeholder — Generate Materials or Regenerate to refill."
        )
        return (
            msg,
            active_session_id or "",
        ) + tuple(_all_gallery_updates(lyric_paths=paths)) + tuple(
            _refresh_session_slots(active_session_id or "")
        )

    def _do_regen(line_idx: int, image_size, image_frequency, steps, cfg_scale, ref_image, negative_prompt, active_session_id):
        """
        Regenerate ONE still. Puts a status-bar message immediately, then clears
        that slot and runs sd-cli for that line only (queue-aware).
        """
        # Immediate status so the user sees feedback on click before any I/O
        yield (
            f"Regenerate requested for still slot {int(line_idx) + 1}…",
            gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), active_session_id or "",
        ) + tuple(_all_gallery_updates()) + tuple(
            _refresh_session_slots(active_session_id or "")
        )

        proj = (configure.APP_STATE.get("current_project_folder") or "").strip()
        empty = (
            "No active project folder — load a session or Generate first.",
            gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), active_session_id or "",
        ) + tuple(_all_gallery_updates()) + tuple(
            _refresh_session_slots(active_session_id or "")
        )
        if not proj or not Path(proj).is_dir():
            yield empty
            return

        paths = _list_project_images(proj)
        resolved_idx = _resolve_line_from_slot(line_idx, paths)
        # Slot index IS the line index when gallery is sequential (slot 0 = line 1)
        if resolved_idx is None:
            expected = _thumb_expected_count()
            if 0 <= line_idx < expected:
                resolved_idx = line_idx
            else:
                yield (
                    "That thumbnail slot is outside the project line count.",
                    gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), active_session_id or "",
                ) + tuple(_all_gallery_updates(lyric_paths=paths)) + tuple(
                    _refresh_session_slots(active_session_id or "")
                )
                return

        # Bound against lyrics
        try:
            lyrics_path = Path(proj) / "lyrics.txt"
            n_lines = 0
            if lyrics_path.is_file():
                from scripts.inference import parse_lyrics, lyric_lines_only
                n_lines = len(lyric_lines_only(parse_lyrics(
                    lyrics_path.read_text(encoding="utf-8", errors="replace")
                )))
            if n_lines > 0 and not (0 <= resolved_idx < n_lines):
                yield (
                    f"Still index {resolved_idx + 1} is out of range (1..{n_lines}).",
                    gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), active_session_id or "",
                ) + tuple(_all_gallery_updates(lyric_paths=paths)) + tuple(
                    _refresh_session_slots(active_session_id or "")
                )
                return
        except Exception:
            pass

        line_no = resolved_idx + 1

        # Persist negative prompt on the project
        neg = (negative_prompt or "").strip()
        try:
            configure.save_session_meta(Path(proj), {"negative_prompt": neg})
            (Path(proj) / "negative_prompt.txt").write_text(neg + "\n", encoding="utf-8")
        except Exception:
            pass

        # Always enqueue + background worker (same pattern as cover/theme).
        # Keeps the existing still on disk until the new one is written so the
        # gallery never flashes "No Image". Concurrent Regenerate clicks only
        # enqueue; the worker runs sd-cli serially and survives Gradio cancelling
        # the click-generator when another button shares the same outputs.
        size_label = configure.normalize_image_size(str(image_size or configure.DEFAULT_IMAGE_SIZE))
        w, h = configure.image_size_pixels(size_label)
        freq = configure.normalize_image_frequency(image_frequency)
        cfg_now = configure.load_configuration()
        gnow = configure.load_generation()
        cfg_now["imagegen_steps"] = steps
        cfg_now["imagegen_cfg_scale"] = cfg_scale
        cfg_now["imagegen_width"] = w
        cfg_now["imagegen_height"] = h
        cfg_now["imagegen_size"] = size_label
        cfg_now["hair_style"] = configure.normalize_hair_style(str(gnow.get("hair_style") or ""))
        cfg_now["outfit_worn"] = configure.normalize_outfit(str(gnow.get("outfit_worn") or ""))
        cfg_now["ref_gender"] = configure.normalize_gender(str(gnow.get("ref_gender") or ""))
        cfg_now["ref_bodyshape"] = configure.normalize_bodyshape(str(gnow.get("ref_bodyshape") or ""))
        cfg_now["imagegen_frequency"] = freq
        cfg_now["negative_prompt"] = neg
        configure.update_generation({
            "imagegen_width": w,
            "imagegen_height": h,
            "imagegen_size": size_label,
            "imagegen_frequency": freq,
            "imagegen_steps": int(steps or configure.DEFAULT_STEPS),
            "imagegen_cfg_scale": float(cfg_scale or configure.DEFAULT_CFG),
        })
        page = _lyrics_page_current()
        per = _lyrics_per_line_count()
        variant = page if per > 1 else 1
        configure.APP_STATE["regen_worker_ctx"] = {
            "proj": proj,
            "cfg": cfg_now,
            "ref": ref_image or "",
            "variant": variant,
        }
        q = list(configure.APP_STATE.get("regen_queue") or [])
        job = {"line_idx": resolved_idx, "variant": variant}
        # Avoid duplicate identical jobs
        already = False
        for existing in q:
            if isinstance(existing, dict) and int(existing.get("line_idx", -1)) == resolved_idx and int(existing.get("variant") or 1) == variant:
                already = True
                break
            if existing == resolved_idx and variant == 1:
                already = True
                break
        if not already:
            q.append(job)
        configure.APP_STATE["regen_queue"] = q
        configure.APP_STATE["thumb_queued_lines"] = sorted(_queued_lines() | {line_no})
        _ensure_regen_worker()

        var_bit = f" page {page}" if per > 1 else ""
        yield (
            f"Queued regenerate for still {line_no}{var_bit}…",
            gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(),
            active_session_id or "",
        ) + tuple(_all_gallery_updates()) + tuple(
            _refresh_session_slots(active_session_id or "")
        )

        # Poll until THIS line leaves queued + busy
        last_state = None
        for _ in range(20000):
            busy_lines = set()
            for x in (configure.APP_STATE.get("regen_busy_lines") or []):
                try:
                    busy_lines.add(int(x))
                except (TypeError, ValueError):
                    pass
            # Also treat the active batch generating line as busy
            cur = configure.APP_STATE.get("thumb_generating_line")
            if cur is not None:
                try:
                    busy_lines.add(int(cur))
                except (TypeError, ValueError):
                    pass
            queued = _queued_lines()
            mine_busy = line_no in busy_lines
            mine_queued = (line_no in queued) and not mine_busy
            state = (mine_busy, mine_queued)
            if state != last_state:
                last_state = state
                if mine_busy:
                    msg = f"Regenerating still {line_no}…"
                elif mine_queued:
                    nq = len(configure.APP_STATE.get("regen_queue") or [])
                    msg = (
                        f"Queued still {line_no} "
                        f"(runs after current image work finishes — {nq} lyric job(s) waiting)."
                    )
                else:
                    msg = f"Regenerated still {line_no}."
                yield (
                    msg,
                    gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(),
                    active_session_id or "",
                ) + tuple(_all_gallery_updates()) + tuple(
                    _refresh_session_slots(active_session_id or "")
                )
            if not mine_busy and not mine_queued:
                break
            time.sleep(0.4)
        else:
            yield (
                f"Still {line_no} still running.",
                gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(),
                active_session_id or "",
            ) + tuple(_all_gallery_updates()) + tuple(
                _refresh_session_slots(active_session_id or "")
            )


    def _do_remove_named(kind: str, slot_idx: int, active_session_id):
        """Remove one cover or theme still by stable slot number (cover-NN / theme-NN)."""
        proj = (configure.APP_STATE.get("current_project_folder") or "").strip()
        smap = _named_slot_map(kind, proj)
        path = smap.get(int(slot_idx), "")
        if not proj or not Path(proj).is_dir() or not path:
            return (
                "Nothing to remove in that slot.",
                active_session_id or "",
            ) + tuple(_all_gallery_updates()) + tuple(
                _refresh_session_slots(active_session_id or "")
            )
        target = Path(path)
        try:
            if target.is_file():
                target.unlink()
                msg = f"Removed {kind} still: {target.name}"
            else:
                msg = f"{kind} file already missing."
        except OSError as e:
            msg = f"Could not remove {target.name}: {e}"
        return (
            msg,
            active_session_id or "",
        ) + tuple(_all_gallery_updates()) + tuple(
            _refresh_session_slots(active_session_id or "")
        )

    def _named_slot_key(kind: str, slot_idx: int) -> str:
        return f"{kind}:{int(slot_idx)}"

    def _mark_named_busy(kind: str, slot_idx: int, on: bool) -> None:
        key = _named_slot_key(kind, slot_idx)
        busy = set(str(x) for x in (configure.APP_STATE.get("named_regen_busy") or []))
        if on:
            busy.add(key)
        else:
            busy.discard(key)
        configure.APP_STATE["named_regen_busy"] = sorted(busy)

    def _enqueue_named_regen(kind: str, slot_idx: int, path: str = "") -> None:
        """Queue only — do not mark busy. Busy = actively generating.
        Store path so drain hits the correct file if the sorted list shifts.
        """
        q = list(configure.APP_STATE.get("named_regen_queue") or [])
        item = {
            "kind": kind,
            "slot_idx": int(slot_idx),
            "path": str(path or ""),
        }
        q = [x for x in q if not (x.get("kind") == kind and int(x.get("slot_idx", -1)) == int(slot_idx))]
        q.append(item)
        configure.APP_STATE["named_regen_queue"] = q

    def _build_named_cfg(image_size, steps, cfg_scale, negative_prompt):
        c = _cfg()
        cfg = dict(c)
        size_label = configure.normalize_image_size(str(image_size or configure.DEFAULT_IMAGE_SIZE))
        w, h = configure.image_size_pixels(size_label)
        cfg["imagegen_width"] = w
        cfg["imagegen_height"] = h
        cfg["imagegen_size"] = size_label
        cfg["imagegen_steps"] = int(steps or configure.DEFAULT_STEPS)
        cfg["imagegen_cfg_scale"] = float(cfg_scale or configure.DEFAULT_CFG)
        cfg["negative_prompt"] = (
            (negative_prompt if negative_prompt is not None else configure.DEFAULT_NEGATIVE_PROMPT) or ""
        )
        gnow = configure.load_generation()
        cfg["hair_style"] = configure.normalize_hair_style(str(gnow.get("hair_style") or ""))
        cfg["outfit_worn"] = configure.normalize_outfit(str(gnow.get("outfit_worn") or ""))
        cfg["ref_gender"] = configure.normalize_gender(str(gnow.get("ref_gender") or ""))
        cfg["ref_bodyshape"] = configure.normalize_bodyshape(str(gnow.get("ref_bodyshape") or ""))
        return cfg

    def _run_one_named_regen(
        kind: str, slot_idx: int, cfg, ref_image: str, proj: str, path: str = "",
    ) -> str:
        paths = _list_cover_images(proj) if kind == "cover" else _list_theme_images(proj)
        if path and (Path(path).is_file() or Path(path).parent.is_dir()):
            path = str(Path(path).resolve())
        else:
            if slot_idx < 0 or slot_idx >= len(paths):
                return f"No {kind} image in slot {slot_idx + 1}."
            path = paths[slot_idx]
        song_name = Path(proj).name
        try:
            meta = configure.load_session_meta(Path(proj)) or {}
            song_name = (meta.get("label") or song_name).strip()
        except Exception:
            pass
        lyrics = ""
        try:
            lp = Path(proj) / "lyrics.txt"
            if lp.is_file():
                lyrics = lp.read_text(encoding="utf-8", errors="replace")
        except OSError:
            pass
        out = inference.regenerate_named_still(
            kind,
            path,
            cfg,
            song_name=song_name,
            lyrics=lyrics,
            reference_image=_project_local_ref(ref_image or ""),
            theme_index=slot_idx if kind == "theme" else 0,
        )
        return f"Regenerated {kind} still: {Path(out).name}"

    def _do_regen_named(
        kind: str,
        slot_idx: int,
        image_size,
        image_frequency,
        steps,
        cfg_scale,
        ref_image,
        negative_prompt,
        active_session_id,
    ):
        """Enqueue a cover/theme regen. A background worker runs sd-cli serially
        so a second click cannot cancel the first generator mid-job."""
        pad = (
            gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(),
        )
        slot_idx = int(slot_idx)
        proj = (configure.APP_STATE.get("current_project_folder") or "").strip()
        smap = _named_slot_map(kind, proj)
        path = smap.get(slot_idx, "")
        if not proj or not Path(proj).is_dir() or not path:
            yield (
                f"No {kind} image in that slot.",
                *pad,
                active_session_id or "",
            ) + tuple(_all_gallery_updates()) + tuple(
                _refresh_session_slots(active_session_id or "")
            )
            return

        cfg = _build_named_cfg(image_size, steps, cfg_scale, negative_prompt)
        configure.APP_STATE["regen_worker_ctx"] = {
            "proj": proj,
            "cfg": cfg,
            "ref": ref_image or "",
        }
        _enqueue_named_regen(kind, slot_idx, path=path)
        _ensure_regen_worker()

        yield (
            f"Queued {kind} slot {slot_idx + 1}…",
            *pad,
            active_session_id or "",
        ) + tuple(_all_gallery_updates()) + tuple(
            _refresh_session_slots(active_session_id or "")
        )

        # Poll until THIS slot leaves queued+busy (worker may still be on others)
        last_state = None
        for _ in range(20000):
            busy = _named_busy_set(kind)
            queued = _named_queued_set(kind)
            mine_busy = slot_idx in busy
            mine_queued = slot_idx in queued and not mine_busy
            state = (mine_busy, mine_queued)
            if state != last_state:
                last_state = state
                if mine_busy:
                    msg = f"Regenerating {kind} {slot_idx + 1}…"
                elif mine_queued:
                    nq = len(configure.APP_STATE.get("named_regen_queue") or [])
                    msg = (
                        f"Queued {kind} slot {slot_idx + 1} "
                        f"(runs after current image work finishes — {nq} named job(s) waiting)."
                    )
                else:
                    msg = f"Regenerated {kind} still {slot_idx + 1}."
                yield (
                    msg,
                    *pad,
                    active_session_id or "",
                ) + tuple(_all_gallery_updates()) + tuple(
                    _refresh_session_slots(active_session_id or "")
                )
            if not mine_busy and not mine_queued:
                break
            time.sleep(0.4)
        else:
            yield (
                f"{kind} slot {slot_idx + 1} still running.",
                *pad,
                active_session_id or "",
            ) + tuple(_all_gallery_updates()) + tuple(
                _refresh_session_slots(active_session_id or "")
            )

    _regen_outputs = [
        status_box,
        _gen["assess_btn"],
        _gen["all_assets_btn"],
        _gen["cover_btn"],
        _gen["theme_btn"],
        _gen["run_btn"],
        _gen["stop_btn"],
        _gen["active_session_id"],
    ] + _thumb_panel_outputs() + _session_refresh_outputs

    _remove_outputs = [
        status_box,
        _gen["active_session_id"],
    ] + _thumb_panel_outputs() + _session_refresh_outputs

    # Wire each Regenerate / Remove button to ONE slot index only
    for i, btn in enumerate(_gen["thumb_btns"]):
        btn.click(
            functools.partial(_do_regen, i),
            inputs=[
                _gen["image_size"], _gen["image_frequency"], _gen["steps"], _gen["cfg"],
                _gen["ref_image"], _gen["negative_prompt"], _gen["active_session_id"],
            ],
            outputs=_regen_outputs,
            show_progress="minimal",
        )
    for i, btn in enumerate(_gen.get("thumb_remove_btns") or []):
        btn.click(
            functools.partial(_do_remove, i),
            inputs=[_gen["active_session_id"]],
            outputs=_remove_outputs,
            show_progress=False,
        )

    # Cover / Theme Regenerate + Remove (same pattern as Lyrics)
    for i, btn in enumerate(_gen.get("cover_regen_btns") or []):
        btn.click(
            functools.partial(_do_regen_named, "cover", i),
            inputs=[
                _gen["image_size"], _gen["image_frequency"], _gen["steps"], _gen["cfg"],
                _gen["ref_image"], _gen["negative_prompt"], _gen["active_session_id"],
            ],
            outputs=_regen_outputs,
            show_progress="minimal",
        )
    for i, btn in enumerate(_gen.get("cover_remove_btns") or []):
        btn.click(
            functools.partial(_do_remove_named, "cover", i),
            inputs=[_gen["active_session_id"]],
            outputs=_remove_outputs,
            show_progress=False,
        )
    for i, btn in enumerate(_gen.get("theme_regen_btns") or []):
        btn.click(
            functools.partial(_do_regen_named, "theme", i),
            inputs=[
                _gen["image_size"], _gen["image_frequency"], _gen["steps"], _gen["cfg"],
                _gen["ref_image"], _gen["negative_prompt"], _gen["active_session_id"],
            ],
            outputs=_regen_outputs,
            show_progress="minimal",
        )
    for i, btn in enumerate(_gen.get("theme_remove_btns") or []):
        btn.click(
            functools.partial(_do_remove_named, "theme", i),
            inputs=[_gen["active_session_id"]],
            outputs=_remove_outputs,
            show_progress=False,
        )

    # Populate sidebar once after UI is up
    def _initial_refresh():
        return _refresh_session_slots(configure.APP_STATE.get("active_session_id") or "")

    # Store for build_app to call via load event
    _gen["_initial_refresh"] = _initial_refresh
    _gen["_session_refresh_outputs"] = _session_refresh_outputs


# ---------------------------------------------------------------------------
# Prompting tab
# ---------------------------------------------------------------------------

_prompt: Dict[str, Any] = {}


def _build_prompting_tab() -> None:
    data = _prompting()
    gr.Markdown(
        "### Visual style prompt templates\n"
        "Each template is used per lyric line. `{line}` is replaced with the lyric text. "
        "Song/section analysis notes are injected automatically by the pipeline."
    )
    _prompt["light"] = gr.Textbox(
        label=configure.STYLE_LIGHT,
        lines=4,
        value=data.get(configure.STYLE_LIGHT, ""),
    )
    _prompt["dark"] = gr.Textbox(
        label=configure.STYLE_DARK,
        lines=4,
        value=data.get(configure.STYLE_DARK, ""),
    )
    _prompt["colorful"] = gr.Textbox(
        label=configure.STYLE_COLORFUL,
        lines=4,
        value=data.get(configure.STYLE_COLORFUL, ""),
    )
    with gr.Row():
        _prompt["save"] = gr.Button("Save Prompt Templates", variant="primary")
        _prompt["reset"] = gr.Button("Reset to Defaults")


def _wire_prompting_events(status_box) -> None:
    def _save(light, dark, colorful):
        configure.update_prompting({
            configure.STYLE_LIGHT: light,
            configure.STYLE_DARK: dark,
            configure.STYLE_COLORFUL: colorful,
        })
        return "Prompt templates saved."

    def _reset():
        defaults = configure._default_prompting()
        configure.save_prompting(defaults)
        return (
            defaults[configure.STYLE_LIGHT],
            defaults[configure.STYLE_DARK],
            defaults[configure.STYLE_COLORFUL],
            "Reset to defaults.",
        )

    _prompt["save"].click(
        _save,
        inputs=[_prompt["light"], _prompt["dark"], _prompt["colorful"]],
        outputs=status_box,
    )
    _prompt["reset"].click(
        _reset,
        outputs=[_prompt["light"], _prompt["dark"], _prompt["colorful"], status_box],
    )


# ---------------------------------------------------------------------------
# Configuration tab (models / backends)
# ---------------------------------------------------------------------------

_conf: Dict[str, Any] = {}


def _build_config_tab() -> None:
    cfg = _cfg()
    backends = configure.get_backend_choices()["all_choices"]

    def _backend_default(key: str) -> str:
        v = cfg.get(key, "CPU")
        return v if v in backends else "CPU"

    def _path_or_empty(key: str) -> str:
        return cfg.get(key, "") or ""

    def _load_default(*keys: str) -> str:
        for k in keys:
            v = cfg.get(k)
            if v:
                return configure.normalize_load_mode(str(v))
        return configure.DEFAULT_LOAD_MODE

    # ── Top row: threads ────────────────────────────────────────────────
    with gr.Row():
        _conf["worker_threads"] = gr.Slider(
            label="Threads Used (all heavy work)",
            minimum=configure.WORKER_THREADS_MIN,
            maximum=max(16, configure.get_cpu_info().get("cores_logical", 8)),
            step=1,
            value=int(cfg.get("worker_threads", configure.WORKER_THREADS_DEFAULT)),
            info="Shared across Thinking, Encoder, and ImageGen. Cores 0–1 stay free for OS/UI.",
        )

    gr.Markdown(
        "### Models & backends\n"
        "Each column is one role. Order top→bottom: **backend → model path → Browse → "
        "load mode / other settings**."
    )

    # ── Three columns: Thinking | Encoder | ImageGen ────────────────────
    with gr.Row(equal_height=False):
        # —— Thinking (Prompts) ——
        with gr.Column(scale=1):
            gr.Markdown("#### Thinking (Prompts)")
            _conf["thinking_backend"] = gr.Dropdown(
                label="Backend",
                choices=backends,
                value=_backend_default("thinking_backend"),
            )
            _conf["thinking_path"] = gr.Textbox(
                label="Model location",
                value=_path_or_empty("thinking_model_path"),
                placeholder=configure.MODEL_PATH_PLACEHOLDERS["thinking"],
                lines=2,
            )
            _conf["browse_thinking"] = gr.Button("Browse", size="sm")
            _conf["thinking_load_mode"] = gr.Dropdown(
                label="Prompts Load Mode",
                choices=configure.LOAD_MODE_CHOICES,
                value=_load_default("thinking_load_mode", "text_load_mode"),
            )
            _conf["thinking_gpu_layers"] = gr.Number(
                label="GPU layers (−1 = auto)",
                value=int(cfg.get("thinking_gpu_layers", cfg.get("text_gpu_layers", -1))),
                precision=0,
            )

        # —— Encoder ——
        with gr.Column(scale=1):
            gr.Markdown("#### Encoder")
            _conf["encoder_backend"] = gr.Dropdown(
                label="Backend",
                choices=backends,
                value=_backend_default("encoder_backend"),
            )
            _conf["encoder_path"] = gr.Textbox(
                label="Model location",
                value=_path_or_empty("encoder_model_path"),
                placeholder=configure.MODEL_PATH_PLACEHOLDERS["encoder"],
                lines=2,
            )
            _conf["browse_encoder"] = gr.Button("Browse", size="sm")
            _conf["encoder_load_mode"] = gr.Dropdown(
                label="Encoder Load Mode",
                choices=configure.LOAD_MODE_CHOICES,
                value=_load_default("encoder_load_mode", "text_load_mode"),
            )
            _conf["encoder_gpu_layers"] = gr.Number(
                label="GPU layers (−1 = auto)",
                value=int(cfg.get("encoder_gpu_layers", cfg.get("text_gpu_layers", -1))),
                precision=0,
            )

        # —— ImageGen (Flux) ——
        with gr.Column(scale=1):
            gr.Markdown("#### ImageGen (Flux)")
            _conf["imagegen_backend"] = gr.Dropdown(
                label="Backend",
                choices=backends,
                value=_backend_default("imagegen_backend"),
            )
            _conf["diffuser_path"] = gr.Textbox(
                label="Diffusion model",
                value=_path_or_empty("imagegen_model_path"),
                placeholder=configure.MODEL_PATH_PLACEHOLDERS.get(
                    "diffuser", "flux-2-klein-4b-*.gguf"
                ),
                lines=2,
            )
            _conf["browse_diffuser"] = gr.Button("Browse diffuser", size="sm")
            _conf["vae_path"] = gr.Textbox(
                label="VAE",
                value=_path_or_empty("vae_model_path"),
                placeholder=configure.MODEL_PATH_PLACEHOLDERS.get(
                    "vae", "flux2_ae.safetensors"
                ),
                lines=2,
                info="Must be Flux.2 VAE: flux2_ae.safetensors (from black-forest-labs/FLUX.2-dev). "
                     "Flux.1 / schnell ae.safetensors will fail with tensor shape errors.",
            )
            _conf["browse_vae"] = gr.Button("Browse VAE", size="sm")
            _conf["imagegen_load_mode"] = gr.Dropdown(
                label="ImageGen Load Mode",
                choices=configure.LOAD_MODE_CHOICES,
                value=_load_default("imagegen_load_mode"),
                info="sd-cli has no -lm flag; M-Lock still enforces same-device unload.",
            )
            _conf["imagegen_placement"] = gr.Dropdown(
                label="Placement (GPU)",
                choices=configure.PLACEMENT_CHOICES,
                value=configure.normalize_placement(
                    str(cfg.get("imagegen_placement", configure.DEFAULT_PLACEMENT))
                ),
            )

    _conf["save_btn"] = gr.Button("Save Configuration", variant="primary")


def _wire_config_events(status_box) -> None:
    def _b(key):
        def _fn():
            return _browse_file(_FILETYPES_MODEL, "last_model_browse_dir") or gr.update()
        return _fn

    _conf["browse_encoder"].click(_b("encoder"), outputs=_conf["encoder_path"])
    _conf["browse_thinking"].click(_b("thinking"), outputs=_conf["thinking_path"])
    _conf["browse_diffuser"].click(_b("diffuser"), outputs=_conf["diffuser_path"])
    _conf["browse_vae"].click(_b("vae"), outputs=_conf["vae_path"])

    def _save(
        workers,
        th_b, th_path, th_load, th_layers,
        enc_b, enc_path, enc_load, enc_layers,
        img_b, diff_path, vae_path, img_load, placement,
    ):
        th_layers_i = int(th_layers if th_layers is not None else -1)
        enc_layers_i = int(enc_layers if enc_layers is not None else -1)
        configure.update_configuration({
            "worker_threads": int(workers or configure.WORKER_THREADS_DEFAULT),
            "thinking_backend": th_b or "CPU",
            "thinking_model_path": th_path or "",
            "thinking_load_mode": configure.normalize_load_mode(str(th_load)),
            "thinking_gpu_layers": th_layers_i,
            "encoder_backend": enc_b or "CPU",
            "encoder_model_path": enc_path or "",
            "encoder_load_mode": configure.normalize_load_mode(str(enc_load)),
            "encoder_gpu_layers": enc_layers_i,
            "imagegen_backend": img_b or "CPU",
            "imagegen_model_path": diff_path or "",
            "vae_model_path": vae_path or "",
            "imagegen_load_mode": configure.normalize_load_mode(str(img_load)),
            "imagegen_placement": configure.normalize_placement(str(placement)),
            # Legacy shared keys kept in sync with Thinking / max layers
            "text_load_mode": configure.normalize_load_mode(str(th_load)),
            "text_gpu_layers": th_layers_i,
            "model_load_mode": configure.normalize_load_mode(str(th_load)),
        })
        return "Configuration saved."

    _conf["save_btn"].click(
        _save,
        inputs=[
            _conf["worker_threads"],
            _conf["thinking_backend"],
            _conf["thinking_path"],
            _conf["thinking_load_mode"],
            _conf["thinking_gpu_layers"],
            _conf["encoder_backend"],
            _conf["encoder_path"],
            _conf["encoder_load_mode"],
            _conf["encoder_gpu_layers"],
            _conf["imagegen_backend"],
            _conf["diffuser_path"],
            _conf["vae_path"],
            _conf["imagegen_load_mode"],
            _conf["imagegen_placement"],
        ],
        outputs=status_box,
    )


# ---------------------------------------------------------------------------
# Preferences tab
# ---------------------------------------------------------------------------

_pref: Dict[str, Any] = {}


def _build_pref_tab() -> None:
    prefs = _prefs()
    gr.Markdown("### Gallery")
    with gr.Row():
        _pref["max_thumbs"] = gr.Dropdown(
            label="Max thumbnails",
            choices=configure.MAX_THUMBNAIL_CHOICES,
            value=prefs.get("max_thumbnails", configure.DEFAULT_MAX_THUMBNAILS),
        )
        _pref["thumb_size"] = gr.Dropdown(
            label="Thumbnail size",
            choices=configure.INPUT_THUMBNAIL_CHOICES,
            value=prefs.get("input_thumbnail_size", configure.DEFAULT_INPUT_THUMBNAIL),
        )
    gr.Markdown("### Completion sounds")
    with gr.Row():
        _pref["bleep_section"] = gr.Checkbox(
            label="Bleep upon section completion",
            value=bool(prefs.get("bleep_section_completion", False)),
            info="Beep after project, analysis, prompts, and images stages.",
        )
        _pref["bleep_video"] = gr.Checkbox(
            label="Bleep upon batch completion",
            value=bool(prefs.get("bleep_video_completion", False)),
            info="Beep when all materials are ready. Both on → double bleep.",
        )
    _pref["save_btn"] = gr.Button("Save Preferences", variant="primary")


def _wire_pref_events(status_box) -> None:
    def _save(max_thumbs, thumb_size, bleep_section, bleep_video):
        configure.update_preferences({
            "max_thumbnails": int(max_thumbs),
            "input_thumbnail_size": int(thumb_size),
            "bleep_section_completion": bool(bleep_section),
            "bleep_video_completion": bool(bleep_video),
        })
        return "Preferences saved."

    _pref["save_btn"].click(
        _save,
        inputs=[
            _pref["max_thumbs"],
            _pref["thumb_size"],
            _pref["bleep_section"],
            _pref["bleep_video"],
        ],
        outputs=status_box,
    )


# ---------------------------------------------------------------------------
# Debug tab
# ---------------------------------------------------------------------------

_dbg: Dict[str, Any] = {}


def _collect_debug() -> str:
    try:
        L = []
        L.append("=== Lyrics-Materials Debug ===")
        L.append(f"Python : {sys.version}")
        cpu = configure.get_cpu_info()
        L.append(f"CPU    : {cpu.get('brand')}")
        L.append(f"Vendor : {cpu.get('vendor')}  arch={cpu.get('arch')}")
        L.append(f"Cores  : {cpu.get('cores_logical')} logical / {cpu.get('cores_physical')} physical")
        L.append(f"Threads: {cpu.get('default_threads')} default  (heavy work: {configure.HEAVY_THREADS})")
        L.append(f"AVX2   : {cpu.get('has_avx2')}   AVX512: {cpu.get('has_avx512')}")
        vk = configure.get_vulkan_info()
        L.append(f"Vulkan : available={vk.get('available')}  ver={vk.get('version')}")
        L.append(f"SDK    : {vk.get('sdk') or '(not set)'}")
        L.append(f"Enum by: {vk.get('enumerated_by')}")
        for d in vk.get("devices", []):
            free = int(d.get("vram_free_mb") or 0)
            total = int(d.get("vram_total_mb") or 0)
            floor = configure.free_vram_floor_mb(free)
            vram = f"  ({total} MiB total, {free} MiB free → safe budget {floor} MiB)"
            L.append(f"  {d['backend']}{d['index']}: {d['name']}{vram}")
        inst = configure.get_install_info()
        L.append("")
        L.append("--- Install (constants.ini) ---")
        L.append(f"Install type   : {inst.get('install_type')}  ({inst.get('backend_method')})")
        L.append(f"llama.cpp ref  : {inst.get('llama_ref') or '(unknown)'}")
        L.append(f"sd.cpp ref     : {inst.get('sd_ref') or '(unknown)'}")
        L.append(f"Heavy threads  : {inst.get('heavy_threads')}")
        bs = utilities.get_build_status()
        L.append("")
        L.append("--- Binaries ---")
        L.append(f"llama  : {bs['llama_path'] or 'MISSING'}")
        L.append(f"sd-cli : {bs['sd_path'] or 'MISSING'}")
        L.append(f"ffmpeg : {utilities.find_ffmpeg() or 'MISSING'}")
        return "\n".join(L)
    except Exception:
        return traceback.format_exc()


def _build_debug_tab() -> None:
    with gr.Row():
        _dbg["refresh"] = gr.Button("Refresh")
        _dbg["copy"] = gr.Button("Copy")
    _dbg["info"] = gr.Textbox(label="Debug Info", lines=16, value=_collect_debug(), interactive=False)


def _wire_debug_events(status_box) -> None:
    _dbg["refresh"].click(_collect_debug, outputs=_dbg["info"])

    def _copy(text):
        try:
            if sys.platform == "win32":
                proc = subprocess.Popen(["clip.exe"], stdin=subprocess.PIPE, text=True)
                proc.communicate(text, timeout=5)
                return "Copied."
            return "Clipboard only on Windows."
        except Exception as e:
            return str(e)

    _dbg["copy"].click(_copy, inputs=_dbg["info"], outputs=status_box)


# ---------------------------------------------------------------------------
# App assembly
# ---------------------------------------------------------------------------

def build_app():
    configure.ensure_data_dirs()
    css = """
#bottom-bar-wrap {
  position: sticky;
  bottom: 0;
  z-index: 200;
  margin-top: 0.75rem;
  padding-top: 0.15rem;
  padding-bottom: 0.25rem;
  background: var(--body-background-fill, #0b0f19);
  border-top: 1px solid #444;
}
#status-bar-heading {
  margin: 0 0 0.15rem 0 !important;
  padding: 0 !important;
  line-height: 1.2 !important;
}
#status-bar-heading h3,
#status-bar-heading p {
  margin: 0 !important;
  font-size: 0.95rem !important;
  font-weight: 600 !important;
}
#bottom-bar {
  display: flex !important;
  flex-direction: row !important;
  flex-wrap: nowrap !important;
  align-items: center !important;
  gap: 0.5rem !important;
  width: 100% !important;
}
#bottom-bar > * {
  margin-top: 0 !important;
  margin-bottom: 0 !important;
}
#exit-btn {
  min-height: 3.6rem !important;
  height: 3.6rem !important;
  background: #a93226 !important;
  border-color: #922b21 !important;
  color: #fff !important;
  font-weight: 700 !important;
  font-size: 1rem !important;
  flex: 0 0 auto !important;
  align-self: center !important;
}
#status-bar textarea,
#status-bar input,
#status-bar {
  min-height: 3.6rem !important;
  height: 3.6rem !important;
  max-height: 3.6rem !important;
  line-height: 3.2rem !important;
  overflow: hidden !important;
  white-space: nowrap !important;
  text-overflow: ellipsis !important;
  resize: none !important;
  flex: 1 1 auto !important;
}
#run-btn { font-weight: 700 !important; }
#sessions-sidebar {
  border-right: 1px solid #ccc;
  padding-right: 0.4rem;
  max-height: 78vh;
  overflow-y: auto;
}
/* Compact: hide all delete UI hard (Gradio sometimes leaves buttons visible) */
.sessions-sidebar-compact .session-delete-btn,
.sessions-sidebar-compact #delete-all-sessions,
.sessions-sidebar-compact #confirm-delete-all,
.sessions-sidebar-compact #confirm-delete-all + button {
  display: none !important;
  visibility: hidden !important;
  width: 0 !important;
  min-width: 0 !important;
  max-width: 0 !important;
  padding: 0 !important;
  margin: 0 !important;
  overflow: hidden !important;
}
.sessions-sidebar-compact .session-select-btn,
.sessions-sidebar-compact .session-select-btn button {
  width: 100% !important;
  min-width: 2rem !important;
  padding-left: 0.25rem !important;
  padding-right: 0.25rem !important;
  text-align: center !important;
  justify-content: center !important;
}
.sessions-sidebar-compact #sessions-toggle {
  font-size: 0.85rem !important;
  padding: 0.2rem 0.15rem !important;
}
.sessions-sidebar-compact #start-new-session {
  font-size: 0.8rem !important;
  padding: 0.2rem 0.15rem !important;
}
#sessions-toggle {
  width: 100% !important;
  font-family: monospace !important;
  letter-spacing: 0.05em;
}
.session-select-btn button,
.session-select-btn {
  text-align: left !important;
  justify-content: flex-start !important;
  white-space: nowrap !important;
  overflow: hidden !important;
  text-overflow: ellipsis !important;
  font-size: 0.85rem !important;
}
.session-delete-btn button,
.session-delete-btn {
  background: #c0392b !important;
  border-color: #a93226 !important;
  color: #fff !important;
  min-width: 58px !important;
  font-size: 0.75rem !important;
  padding: 0.2rem 0.4rem !important;
}
/* Sub-heading look (not a chrome button) — still clickable to open folder */
#materials-thumbs-header,
#materials-thumbs-header button,
.materials-thumbs-heading,
.materials-thumbs-heading button {
  width: 100% !important;
  background: transparent !important;
  border: none !important;
  box-shadow: none !important;
  outline: none !important;
  text-align: left !important;
  justify-content: flex-start !important;
  font-size: 1.15rem !important;
  font-weight: 600 !important;
  line-height: 1.3 !important;
  color: inherit !important;
  padding: 0.15rem 0 0.35rem 0 !important;
  margin: 0 0 0.25rem 0 !important;
  cursor: pointer !important;
  min-width: 0 !important;
}
#materials-thumbs-header:hover,
#materials-thumbs-header button:hover,
.materials-thumbs-heading:hover,
.materials-thumbs-heading button:hover {
  background: transparent !important;
  text-decoration: underline !important;
  opacity: 0.9 !important;
}
.materials-thumbs,
#materials-gallery,
#cover-gallery,
#theme-gallery {
  width: 100% !important;
  min-height: 0;
  overflow-y: auto;
  max-height: 70vh;
  margin-top: 0.35rem;
}
.materials-thumbs .thumb-row,
#materials-gallery .thumb-row {
  margin: 0 !important;
  gap: 0.25rem !important;
}
.materials-thumbs .thumb-slot,
#materials-gallery .thumb-slot {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 0.15rem;
  padding: 0.1rem;
}
/* Fit still to thumbnail BOX HEIGHT */
.materials-thumbs .thumb-img,
#materials-gallery .thumb-img {
  width: auto !important;
  max-width: 100% !important;
  height: auto !important;
  min-height: 0 !important;
  overflow: hidden !important;
}
.materials-thumbs .thumb-img img,
.materials-thumbs .thumb-img button:has(img) img,
#materials-gallery .thumb-img img,
#materials-gallery .thumb-img button:has(img) img {
  display: block !important;
  width: auto !important;
  max-width: 100% !important;
  height: 100% !important;
  max-height: 100% !important;
  object-fit: contain !important;
  object-position: center !important;
}
/* Hide Gradio Image toolbar on ALL galleries (Cover / Theme / Lyrics) */
.materials-thumbs .thumb-img .icon-button,
.materials-thumbs .thumb-img [class*="icon-button"],
.materials-thumbs .thumb-img .download,
.materials-thumbs .thumb-img .fullscreen,
.materials-thumbs .thumb-img .share,
.materials-thumbs .thumb-img .image-button-row,
.materials-thumbs .thumb-img .toolbar,
.materials-thumbs .thumb-img [aria-label="Download"],
.materials-thumbs .thumb-img [aria-label="Fullscreen"],
.materials-thumbs .thumb-img [aria-label="Share"],
.materials-thumbs .thumb-img button:not(:has(img)),
#materials-gallery .thumb-img .icon-button,
#materials-gallery .thumb-img [class*="icon-button"],
#materials-gallery .thumb-img .download,
#materials-gallery .thumb-img .fullscreen,
#materials-gallery .thumb-img .share,
#materials-gallery .thumb-img .image-button-row,
#materials-gallery .thumb-img .toolbar,
#materials-gallery .thumb-img [aria-label="Download"],
#materials-gallery .thumb-img [aria-label="Fullscreen"],
#materials-gallery .thumb-img [aria-label="Share"] {
  display: none !important;
}
.materials-thumbs .thumb-img button:has(img),
#materials-gallery .thumb-img button:has(img) {
  display: block !important;
  background: transparent !important;
  border: none !important;
  box-shadow: none !important;
  padding: 0 !important;
  width: 100% !important;
  height: auto !important;
  cursor: default !important;
}
#materials-gallery .thumb-regen-btn {
  width: 100% !important;
  min-width: 0 !important;
  font-size: 0.65rem !important;
  padding: 0.12rem 0.2rem !important;
}
#lyrics-box textarea {
  overflow-y: auto !important;
  transition: min-height 0.12s ease;
}
#run-btn button:disabled,
#run-btn:has(button:disabled) {
  opacity: 0.7 !important;
  cursor: not-allowed !important;
}
#materials-gallery .thumb-img img[src] {
  visibility: visible !important;
  opacity: 1 !important;
}
#delete-all-sessions {
  margin-top: 0.6rem;
  width: 100%;
}
#start-new-session {
  width: 100%;
  margin-bottom: 0.35rem;
}
"""
    _icon_href = ""
    try:
        _root = Path(__file__).resolve().parent.parent
        for _rel in ("images/program_icon.ico", "Images/program_icon.ico", "program_icon.ico"):
            _ip = _root / _rel
            if _ip.is_file():
                # file:// works in Qt WebEngine for local assets; also helps titlebar in some hosts
                _icon_href = _ip.as_uri()
                break
    except Exception:
        _icon_href = ""
    _head = f'<link rel="icon" href="{_icon_href}" type="image/x-icon">' if _icon_href else ""
    with gr.Blocks(title="Lyrics-Materials") as app:
        gr.Markdown("# Lyrics-Materials — Lyrics → Image Materials")
        with gr.Tabs():
            with gr.TabItem("Generation"):
                _build_create_tab()
            with gr.TabItem("Prompting"):
                _build_prompting_tab()
            with gr.TabItem("Configuration"):
                _build_config_tab()
            with gr.TabItem("Preferences"):
                _build_pref_tab()
            with gr.TabItem("Debug / Info"):
                _build_debug_tab()

        # Shared across ALL tabs: heading, then status | Exit on the SAME row
        with gr.Column(elem_id="bottom-bar-wrap"):
            gr.Markdown("### Status Bar", elem_id="status-bar-heading")
            with gr.Row(elem_id="bottom-bar"):
                shared_status = gr.Textbox(
                    value="Ready.",
                    show_label=False,
                    interactive=False,
                    container=False,
                    scale=9,
                    lines=1,
                    max_lines=1,
                    elem_id=configure.STATUS_BAR_KEY,
                )
                exit_btn = gr.Button(
                    "Exit Program",
                    variant="stop",
                    scale=1,
                    elem_id="exit-btn",
                    min_width=140,
                )

        exit_btn.click(_handle_exit_click)
        _wire_create_events(shared_status)
        _wire_prompting_events(shared_status)
        _wire_config_events(shared_status)
        _wire_pref_events(shared_status)
        _wire_debug_events(shared_status)

        # Populate session sidebar on first load
        if _gen.get("_initial_refresh") and _gen.get("_session_refresh_outputs"):
            app.load(
                _gen["_initial_refresh"],
                outputs=_gen["_session_refresh_outputs"],
            )

    return app, css, None, _head
