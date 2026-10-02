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
import shutil
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path
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


def _can_create(lyrics: str, song_name: str = "") -> bool:
    if not (lyrics or "").strip():
        return False
    if not (song_name or "").strip():
        return False
    if not _models_configured():
        return False
    return True


def _partial_project_state(lyrics: str = "") -> bool:
    """True when the active project has SOME but not ALL of its stills on disk."""
    try:
        folder = (configure.APP_STATE.get("current_project_folder") or "").strip()
        if not folder or not Path(folder).is_dir():
            return False
        n_lines = 0
        if (lyrics or "").strip():
            from scripts.inference import parse_lyrics, lyric_lines_only
            n_lines = len(lyric_lines_only(parse_lyrics(lyrics)))
        if n_lines <= 0:
            n_lines = int(configure.APP_STATE.get("thumb_expected_count") or 0)
        if n_lines <= 0:
            return False
        have = _line_path_map(_list_project_images(folder))
        n_have = len([k for k in have if 1 <= k <= n_lines])
        return 0 < n_have < n_lines
    except Exception:
        return False


def _run_btn_label(lyrics: str = "") -> str:
    """'Complete Materials' when resuming a partly finished project, else 'Generate Materials'."""
    return "Complete Materials" if _partial_project_state(lyrics) else "Generate Materials"


def _run_btn_updates(lyrics: str = "", song_name: str = ""):
    """
    One primary action button:
      ready, new / fully done project → Generate Materials (clickable)
      ready, project has some stills  → Complete Materials (same action)
      not ready                       → Configure the Pages First (disabled grey)
    Using a single control avoids Gradio dual-visibility races where both
    buttons could end up hidden.
    """
    ok = _can_create(lyrics, song_name)
    if ok:
        return gr.update(
            value=_run_btn_label(lyrics),
            interactive=True,
            variant="primary",
            visible=True,
        )
    return gr.update(
        value="Configure the Pages First",
        interactive=False,
        variant="secondary",
        visible=True,
    )


def _run_btn_for_project(proj: str) -> Any:
    """Run-button update for the active project (reads lyrics.txt + folder name)."""
    lyrics = ""
    try:
        lp = Path(proj) / "lyrics.txt"
        if lp.is_file():
            lyrics = lp.read_text(encoding="utf-8", errors="replace")
    except OSError:
        pass
    return _run_btn_updates(lyrics, Path(proj).name if proj else "")


def _lyrics_line_count(text: str) -> int:
    """Visible lines for Song Lyrics while focused (full content)."""
    n = len((text or "").splitlines()) or 1
    # +2 breathing room; clamp so the UI cannot explode
    return max(12, min(120, n + 2))


def _format_progress_line(msg: str, frac: float, info: dict) -> str:
    phase = (info or {}).get("phase", "")
    pct = int(round(max(0.0, min(1.0, float(frac or 0.0))) * 100))
    bits = [f"[{pct:3d}%]"]
    if phase:
        bits.append(str(phase))
    line_n = (info or {}).get("line")
    total = (info or {}).get("total")
    if line_n is not None and total is not None:
        bits.append(f"{line_n}/{total}")
    bits.append(msg)
    return " ".join(bits).replace("\n", " ").strip()[:240]


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
    """True only for materials stills named like 001-… / 001 - …. Excludes reference.*"""
    name = Path(path).name
    if name.lower().startswith("reference"):
        return False
    return bool(re.match(r"^\d{3}(?:\s*[-–—]|[-.])", name) or re.match(r"^\d{3}\.", name) or re.match(r"^\d{3}$", Path(path).stem))


_PLACEHOLDER_WARNED: set = set()

# kind -> file under images/ (user-facing spelling "qued" is intentional)
_PLACEHOLDER_FILES = {
    "no_image": "thumbnails_no_image.jpg",
    "queued": "thumbnails_qued_for_generation.jpg",
    "generating": "thumbnails_generating.jpg",
}


def _placeholder_thumb(kind: str) -> Optional[str]:
    """Absolute path to images/<placeholder>.jpg for no_image | queued | generating.

    Returns None if the file is missing (a one-time console warning is printed).
    """
    fname = _PLACEHOLDER_FILES.get(kind, f"thumbnails_{kind}.jpg")
    try:
        p = configure.get_images_dir() / fname
        if p.is_file():
            return str(p.resolve())
        if fname not in _PLACEHOLDER_WARNED:
            _PLACEHOLDER_WARNED.add(fname)
            print(f"  WARNING: placeholder not found: {p}", flush=True)
    except Exception:
        pass
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


def _line_path_map(paths: Optional[List[str]] = None) -> Dict[int, str]:
    """Map 1-based line number → absolute still path."""
    if paths is None:
        paths = _list_project_images()
    out: Dict[int, str] = {}
    for p in paths:
        if not p or not Path(p).is_file():
            continue
        m = re.match(r"^(\d+)", Path(p).name)
        if not m:
            continue
        out[int(m.group(1))] = str(Path(p).resolve())
    return out


def _thumb_expected_count() -> int:
    """How many sequential slots to show (line_count), capped at THUMB_SLOTS."""
    n = int(configure.APP_STATE.get("thumb_expected_count") or 0)
    if n <= 0:
        # Fall back to highest existing still number
        m = _line_path_map()
        if m:
            n = max(m.keys())
    return max(0, min(int(n), THUMB_SLOTS))


def _thumb_panel_updates(paths: Optional[List[str]] = None) -> List[Any]:
    """
    Gradio updates for the thumb grid (slot i = lyric line i+1):
      - has still on disk → that image
      - line is generating / regenerating now → thumbnails_generating.jpg
      - line is listed for generation, not started → thumbnails_qued_for_generation.jpg
      - expected but missing / removed, nothing pending → thumbnails_no_image.jpg
    Rows are shown for the expected line count (not only existing files).
    """
    by_line = _line_path_map(paths)
    expected = _thumb_expected_count()
    # Always show at least the highest existing line
    if by_line:
        expected = max(expected, min(max(by_line.keys()), THUMB_SLOTS))
    expected = min(expected, THUMB_SLOTS)

    busy_raw = configure.APP_STATE.get("regen_busy_lines") or []
    busy: set = set()
    for x in busy_raw:
        try:
            busy.add(int(x))
        except (TypeError, ValueError):
            pass
    # Current batch line (1-based) while full Generate is running
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

    updates: List[Any] = []
    for r in range(n_rows):
        row_start = r * THUMB_COLS  # 0-based slot
        # Show row if any slot in it is within expected range
        row_has = row_start < expected
        updates.append(gr.update(visible=row_has))

    for i in range(THUMB_SLOTS):
        line_no = i + 1
        if line_no > expected:
            updates.append(gr.update(visible=False))
            updates.append(gr.update(value=None))
            updates.append(gr.update(visible=False, value="Regenerate"))
            updates.append(gr.update(visible=False))
            continue

        real = by_line.get(line_no)
        # Priority: real still > generating now > queued > no_image.
        # A queued line is not "generating" until it is un-queued (starts).
        is_queued = (line_no in queued) and not real
        is_busy = (line_no in busy) and not real and not is_queued
        if real:
            value = real
        elif is_busy:
            value = gen_img
        elif is_queued:
            value = que_img
        else:
            value = no_img

        updates.append(gr.update(visible=True))
        updates.append(gr.update(value=value, height=th))
        # Regenerate/Remove only when a real still exists (not placeholders)
        # Regenerate allowed for existing OR missing (refill) slots; Remove only when file exists
        can_regen = not is_busy and not is_queued
        can_remove = bool(real) and not is_busy
        updates.append(gr.update(
            visible=True,
            interactive=can_regen,
            value="…" if (is_busy or is_queued) else "Regenerate",
        ))
        updates.append(gr.update(visible=True, interactive=can_remove))
    return updates


def _thumb_panel_outputs() -> List[Any]:
    """Ordered list of Gradio components for thumb-panel updates."""
    outs: List[Any] = []
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
    initial_lyrics = ""
    initial_style = prefs.get("style", configure.STYLE_LIGHT)
    initial_ref = ""
    initial_song = ""
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

        # ── Right: main generation controls ─────────────────────────────
        with gr.Column(scale=4, elem_id="gen-main-col"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### Project settings")
                    with gr.Row():
                        _gen["style"] = gr.Dropdown(
                            label="Visual style",
                            choices=configure.STYLE_CHOICES,
                            value=initial_style,
                            info="Influences prompt templates for still generation.",
                        )
                    gr.Markdown(
                        "### Generation defaults (Flux.2-klein-4B)\n"
                        "Image size is fixed at **768 × 512**."
                    )
                    with gr.Row():
                        _gen["steps"] = gr.Slider(
                            label="Steps",
                            minimum=1,
                            maximum=20,
                            step=1,
                            value=configure.DEFAULT_STEPS,
                        )
                        _gen["cfg"] = gr.Slider(
                            label="CFG",
                            minimum=0.5,
                            maximum=4.0,
                            step=0.1,
                            value=configure.DEFAULT_CFG,
                        )

                    gr.Markdown(
                        "### Optional reference image\n"
                        "If provided, the central character(s) in the stills are guided by this photo. "
                        "Without it, characters are invented purely from the lyrics."
                    )
                    with gr.Row():
                        _gen["ref_image"] = gr.Textbox(
                            label="Reference character image (optional)",
                            value=initial_ref,
                            interactive=True,
                            scale=4,
                        )
                        _gen["browse_ref"] = gr.Button("Browse", scale=1, min_width=90)

                with gr.Column(scale=1):
                    gr.Markdown(
                        "### Song & lyrics\n"
                        "Enter a short **song name** (used as the output folder under `output/`). "
                        "Spaces become underscores. Section headers (`[Intro]`, `[Chorus_1]`, …) "
                        "are context only — they do **not** get images. Blank lines are skipped.\n\n"
                        "Select a **session** on the left to resume, or leave blank and click "
                        "**Generate Materials** to start a new one."
                    )
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

            with gr.Row(elem_id="gen-action-row"):
                _gen["run_btn"] = gr.Button(
                    "Generate Materials" if can else "Configure the Pages First",
                    variant="primary" if can else "secondary",
                    interactive=bool(can),
                    visible=True,
                    elem_id="run-btn",
                )
                _gen["stop_btn"] = gr.Button(
                    "Emergency Stop",
                    variant="stop",
                    visible=False,
                    elem_id="stop-btn",
                )

        # Materials Thumbnails: FULL WIDTH under the sidebar+settings row
        # (left session column only affects Project Settings / Song Lyrics above)
    th0 = _thumb_size_px()
    with gr.Column(
        elem_id="materials-gallery",
        elem_classes=["materials-thumbs"],
    ) as _thumbs_panel:
        _gen["open_materials_folder"] = gr.Button(
            "Materials Thumbnails",
            variant="secondary",
            elem_id="materials-thumbs-header",
            elem_classes=["materials-thumbs-heading"],
            size="sm",
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
                    idx = row_i + j
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


def _wire_create_events(status_box: gr.Textbox) -> None:
    def _browse_ref():
        path = _browse_file(_FILETYPES_IMAGE, "last_image_browse_dir")
        return path or gr.update()

    _gen["browse_ref"].click(_browse_ref, outputs=_gen["ref_image"])

    def _ready_change(lyrics, song_name):
        return _run_btn_updates(lyrics, song_name)

    for _evt in ("change", "input", "blur"):
        try:
            getattr(_gen["lyrics"], _evt)(
                _ready_change,
                inputs=[_gen["lyrics"], _gen["song_name"]],
                outputs=_gen["run_btn"],
            )
            getattr(_gen["song_name"], _evt)(
                _ready_change,
                inputs=[_gen["lyrics"], _gen["song_name"]],
                outputs=_gen["run_btn"],
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
        style = s.get("style") or configure.STYLE_LIGHT
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
        main = (
            gr.update(value=song),
            gr.update(value=lyrics, lines=12, max_lines=12),
            gr.update(value=style if style in configure.STYLE_CHOICES else configure.STYLE_LIGHT),
            gr.update(value=steps),
            gr.update(value=cfg_scale),
            gr.update(value=ref),
            gr.update(value=(s.get("negative_prompt") if s.get("negative_prompt") is not None else configure.DEFAULT_NEGATIVE_PROMPT)),
            status,
            sid,
        )
        run_btn_u = _run_btn_updates(lyrics, song)
        refresh = _refresh_session_slots(sid)
        return main + (run_btn_u,) + tuple(_thumb_panel_updates(list(imgs))) + tuple(refresh)

    # Wire each select button with its index
    for i, btn in enumerate(_gen["session_select_btns"]):
        btn.click(
            lambda sid_list, i=i: _select_session(i, sid_list or []),
            inputs=[_gen["session_ids"]],
            outputs=[
                _gen["song_name"],
                _gen["lyrics"],
                _gen["style"],
                _gen["steps"],
                _gen["cfg"],
                _gen["ref_image"],
                _gen["negative_prompt"],
                status_box,
                _gen["active_session_id"],
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
            gr.update(value=configure.STYLE_LIGHT),
            gr.update(value=configure.DEFAULT_STEPS),
            gr.update(value=configure.DEFAULT_CFG),
            gr.update(value=""),
            gr.update(value=configure.DEFAULT_NEGATIVE_PROMPT),
            "New session — enter song name & lyrics, then Generate.",
            "",
        )
        return main + (_run_btn_updates("", ""),) + tuple(_thumb_panel_updates([])) + tuple(
            _refresh_session_slots("")
        )

    _gen["start_new_session"].click(
        _start_new,
        outputs=[
            _gen["song_name"],
            _gen["lyrics"],
            _gen["style"],
            _gen["steps"],
            _gen["cfg"],
            _gen["ref_image"],
            _gen["negative_prompt"],
            status_box,
            _gen["active_session_id"],
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
            ) + tuple(_thumb_panel_updates()) + tuple(
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
        ) + tuple(_thumb_panel_updates([])) + tuple(_refresh_session_slots(""))

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
        lyrics, song_name, style, steps, cfg_scale, ref_image, negative_prompt, active_session_id,
        progress=gr.Progress(track_tqdm=False),
    ):
        # During a run: keep the action button visible but disabled; show Stop
        # One or the other: never show Generate and Emergency Stop together
        hide_run = gr.update(visible=False)
        show_stop = gr.update(visible=True)
        show_run = _run_btn_updates(lyrics, song_name)  # restore ready/not-ready label
        hide_stop = gr.update(visible=False)

        if not _can_create(lyrics, song_name):
            yield (
                "Song name, lyrics, and models are required — fill the song name "
                "and configure Encoder + Diffuser paths.",
                _run_btn_updates(lyrics, song_name), hide_stop, active_session_id or "",
            ) + tuple(_thumb_panel_updates()) + tuple(
                _refresh_session_slots(active_session_id or "")
            )
            return

        c = _cfg()
        cfg = dict(c)
        cfg["style"] = style
        cfg["project_label"] = (song_name or "").strip()
        cfg["imagegen_width"] = configure.DEFAULT_WIDTH
        cfg["imagegen_height"] = configure.DEFAULT_HEIGHT
        cfg["imagegen_steps"] = int(steps or configure.DEFAULT_STEPS)
        cfg["imagegen_cfg_scale"] = float(cfg_scale or configure.DEFAULT_CFG)
        cfg["prompt_template"] = configure.prompt_template_for_style(style)
        cfg["negative_prompt"] = (
            (negative_prompt if negative_prompt is not None else configure.DEFAULT_NEGATIVE_PROMPT) or ""
        )

        configure.update_generation({
            "last_lyrics": lyrics,
            "project_label": (song_name or "").strip(),
            "reference_image_path": ref_image or "",
            "imagegen_width": configure.DEFAULT_WIDTH,
            "imagegen_height": configure.DEFAULT_HEIGHT,
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
            "[  0%] start  Starting materials pipeline…",
            hide_run, show_stop, active_session_id or "",
        ) + tuple(_thumb_panel_updates()) + tuple(
            _refresh_session_slots(active_session_id or "")
        )

        prog_q: queue.Queue = queue.Queue()
        result_holder: Dict[str, Any] = {}

        def cb(msg: str, frac: float, info: dict) -> None:
            prog_q.put((msg, frac, info or {}))

        def worker() -> None:
            try:
                result_holder["r"] = inference.run_materials_pipeline(
                    lyrics=lyrics,
                    cfg=cfg,
                    song_name=song_name or "",
                    reference_image=ref_image or "",
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

        status_line = "[  0%] start  Starting materials pipeline…"
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
                    status_line = _format_progress_line(msg, frac, info)
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
                    hide_run, show_stop, sid,
                ) + tuple(_thumb_panel_updates(paths)) + tuple(_refresh_session_slots(sid))
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
            # Drop invalid indices before running
            valid_q = []
            for qi in q:
                try:
                    qi = int(qi)
                except (TypeError, ValueError):
                    continue
                if qi < 0:
                    continue
                valid_q.append(qi)
            q = valid_q
            configure.APP_STATE["thumb_queued_lines"] = [qi + 1 for qi in q]
            if q:
                msg = f"{msg}  Processing {len(q)} queued regenerate(s)…"
                yield (
                    msg, show_run, hide_stop, sid,
                ) + tuple(_thumb_panel_updates(paths)) + tuple(_refresh_session_slots(sid))
            for line_idx in q:
                if inference.is_cancel_requested():
                    break
                try:
                    configure.APP_STATE["generating"] = True
                    _unqueue_line(int(line_idx) + 1)
                    proj = configure.APP_STATE.get("current_project_folder") or ""
                    cfg_now = configure.load_configuration()
                    cfg_now["imagegen_steps"] = steps
                    cfg_now["imagegen_cfg_scale"] = cfg_scale
                    cfg_now["negative_prompt"] = (
                        (negative_prompt if negative_prompt is not None else configure.DEFAULT_NEGATIVE_PROMPT) or ""
                    )
                    inference.regenerate_single_still(
                        Path(proj), int(line_idx), cfg_now, reference_image=ref_image or "",
                    )
                    paths = _list_project_images(proj)
                    yield (
                        f"Regenerated still {int(line_idx) + 1}.",
                        show_run, hide_stop, sid,
                    ) + tuple(_thumb_panel_updates(paths)) + tuple(_refresh_session_slots(sid))
                except Exception as e:
                    yield (
                        f"Queued regenerate line {int(line_idx) + 1} failed: {e}",
                        show_run, hide_stop, sid,
                    ) + tuple(_thumb_panel_updates(paths)) + tuple(_refresh_session_slots(sid))
                finally:
                    configure.APP_STATE["generating"] = False

        configure.APP_STATE["thumb_generating_line"] = None
        configure.APP_STATE["regen_busy_lines"] = []
        configure.APP_STATE["thumb_queued_lines"] = []
        paths = _list_project_images(configure.APP_STATE.get("current_project_folder") or "")
        show_run = _run_btn_updates(lyrics, song_name)  # label may now be Complete/Generate
        yield (
            msg,
            show_run, hide_stop, sid,
        ) + tuple(_thumb_panel_updates(paths)) + tuple(_refresh_session_slots(sid))

    _run_outputs = [
        status_box,
        _gen["run_btn"],
        _gen["stop_btn"],
        _gen["active_session_id"],
    ] + _thumb_panel_outputs() + _session_refresh_outputs

    _gen["run_btn"].click(
        _run,
        inputs=[
            _gen["lyrics"],
            _gen["song_name"],
            _gen["style"],
            _gen["steps"],
            _gen["cfg"],
            _gen["ref_image"],
            _gen["negative_prompt"],
            _gen["active_session_id"],
        ],
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
            ) + tuple(_thumb_panel_updates(paths)) + tuple(
                _refresh_session_slots(active_session_id or "")
            )
        paths = _list_project_images(proj)
        resolved = _resolve_line_from_slot(line_idx, paths)
        if resolved is None:
            return (
                "Nothing to remove in that slot.",
                active_session_id or "",
            ) + tuple(_thumb_panel_updates(paths)) + tuple(
                _refresh_session_slots(active_session_id or "")
            )
        line_no = resolved + 1
        removed = []
        for old in list(Path(proj).iterdir()):
            if not old.is_file():
                continue
            if old.suffix.lower() not in (".png", ".jpg", ".jpeg", ".webp"):
                continue
            m = re.match(r"^(\d+)", old.name)
            if m and int(m.group(1)) == line_no:
                try:
                    old.unlink()
                    removed.append(old.name)
                except OSError as e:
                    print(f"[remove] {old}: {e}", flush=True)
        # Drop from APP_STATE paths
        kept = []
        for p in (configure.APP_STATE.get("generation_output_paths") or []):
            try:
                m = re.match(r"^(\d+)", Path(p).name)
                if m and int(m.group(1)) == line_no:
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
        msg = (
            f"Removed still {line_no}"
            + (f" ({', '.join(removed)})" if removed else "")
            + ". Slot shows placeholder — Generate Materials or Regenerate to refill."
        )
        return (
            msg,
            active_session_id or "",
        ) + tuple(_thumb_panel_updates(paths)) + tuple(
            _refresh_session_slots(active_session_id or "")
        )

    def _do_regen(line_idx: int, steps, cfg_scale, ref_image, negative_prompt, active_session_id):
        """
        Regenerate ONE still. Clears that slot immediately, runs sd-cli for that
        line only, supports queueing other slots while one is running.
        show_progress is disabled on the button so Gradio does not spin every thumb.
        """
        proj = (configure.APP_STATE.get("current_project_folder") or "").strip()
        empty = (
            "No active project folder — load a session or Generate first.",
            gr.update(), gr.update(), active_session_id or "",
        ) + tuple(_thumb_panel_updates()) + tuple(
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
                    gr.update(), gr.update(), active_session_id or "",
                ) + tuple(_thumb_panel_updates(paths)) + tuple(
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
                    gr.update(), gr.update(), active_session_id or "",
                ) + tuple(_thumb_panel_updates(paths)) + tuple(
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

        # Queue if already generating
        if configure.APP_STATE.get("generating") or configure.APP_STATE.get("session_status") == "running":
            q = list(configure.APP_STATE.get("regen_queue") or [])
            if resolved_idx not in q:
                q.append(resolved_idx)
            configure.APP_STATE["regen_queue"] = q
            _mark_busy(line_no, True)
            configure.APP_STATE["thumb_queued_lines"] = sorted(_queued_lines() | {line_no})
            # Delete existing still now so UI shows blank for this slot only
            for old in list(Path(proj).iterdir()):
                if not old.is_file():
                    continue
                if old.suffix.lower() not in (".png", ".jpg", ".jpeg", ".webp"):
                    continue
                m = re.match(r"^(\d+)", old.name)
                if m and int(m.group(1)) == line_no:
                    try:
                        old.unlink()
                    except OSError:
                        pass
            paths = _list_project_images(proj)
            yield (
                f"Queued regenerate for still {line_no} "
                f"(runs after current work finishes).",
                gr.update(), gr.update(), active_session_id or "",
            ) + tuple(_thumb_panel_updates(paths)) + tuple(
                _refresh_session_slots(active_session_id or "")
            )
            return

        # Immediate: blank this slot, mark busy, delete file
        _mark_busy(line_no, True)
        for old in list(Path(proj).iterdir()):
            if not old.is_file():
                continue
            if old.suffix.lower() not in (".png", ".jpg", ".jpeg", ".webp"):
                continue
            m = re.match(r"^(\d+)", old.name)
            if m and int(m.group(1)) == line_no:
                try:
                    old.unlink()
                except OSError:
                    pass
        paths = _list_project_images(proj)
        yield (
            f"Regenerating still {line_no}…",
            gr.update(visible=False),
            gr.update(visible=True),
            active_session_id or "",
        ) + tuple(_thumb_panel_updates(paths)) + tuple(
            _refresh_session_slots(active_session_id or "")
        )

        configure.APP_STATE["generating"] = True
        configure.APP_STATE["session_status"] = "running"
        msg = f"Regenerated still {line_no}."
        try:
            cfg_now = configure.load_configuration()
            cfg_now["imagegen_steps"] = steps
            cfg_now["imagegen_cfg_scale"] = cfg_scale
            cfg_now["negative_prompt"] = neg
            inference.regenerate_single_still(
                Path(proj),
                int(resolved_idx),
                cfg_now,
                reference_image=ref_image or "",
            )
            configure.APP_STATE["session_status"] = "stopped"
        except Exception as e:
            configure.APP_STATE["session_status"] = "stopped"
            msg = f"Regenerate failed for still {line_no}: {e}"
            print(f"[regen] {msg}", flush=True)
        finally:
            configure.APP_STATE["generating"] = False
            _mark_busy(line_no, False)

        # Drain queue (other slots clicked while this one ran)
        q = list(configure.APP_STATE.get("regen_queue") or [])
        configure.APP_STATE["regen_queue"] = []
        extra = ""
        for qi in q:
            if inference.is_cancel_requested():
                break
            qi = int(qi)
            q_line = qi + 1
            _unqueue_line(q_line)
            _mark_busy(q_line, True)
            paths = _list_project_images(proj)
            yield (
                f"Regenerating still {q_line} (from queue)…",
                gr.update(visible=False),
                gr.update(visible=True),
                active_session_id or "",
            ) + tuple(_thumb_panel_updates(paths)) + tuple(
                _refresh_session_slots(active_session_id or "")
            )
            try:
                configure.APP_STATE["generating"] = True
                cfg_now = configure.load_configuration()
                cfg_now["imagegen_steps"] = steps
                cfg_now["imagegen_cfg_scale"] = cfg_scale
                cfg_now["negative_prompt"] = neg
                inference.regenerate_single_still(
                    Path(proj), qi, cfg_now, reference_image=ref_image or "",
                )
                extra += f" Also regenerated {q_line}."
            except Exception as e:
                extra += f" Queued {q_line} failed: {e}."
            finally:
                configure.APP_STATE["generating"] = False
                _mark_busy(q_line, False)

        configure.APP_STATE["thumb_queued_lines"] = []
        paths = _list_project_images(proj)
        yield (
            msg + extra,
            _run_btn_for_project(proj),
            gr.update(visible=False),
            active_session_id or "",
        ) + tuple(_thumb_panel_updates(paths)) + tuple(
            _refresh_session_slots(active_session_id or "")
        )

    _regen_outputs = [
        status_box,
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
                _gen["steps"], _gen["cfg"], _gen["ref_image"],
                _gen["negative_prompt"], _gen["active_session_id"],
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


def _wire_prompting_events(status_box: gr.Textbox) -> None:
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


def _wire_config_events(status_box: gr.Textbox) -> None:
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


def _wire_pref_events(status_box: gr.Textbox) -> None:
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


def _wire_debug_events(status_box: gr.Textbox) -> None:
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
#exit-btn {
  min-height: 3.6rem !important;
  height: 3.6rem !important;
  background: #a93226 !important;
  border-color: #922b21 !important;
  color: #fff !important;
  font-weight: 700 !important;
  font-size: 1rem !important;
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
#materials-gallery {
  width: 100% !important;
  min-height: 0;
  overflow-y: auto;
  max-height: 70vh;
  margin-top: 0.35rem;
}
#materials-gallery .thumb-row {
  margin: 0 !important;
  gap: 0.25rem !important;
}
#materials-gallery .thumb-slot {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 0.15rem;
  padding: 0.1rem;
}
/* Fit still to thumbnail BOX HEIGHT (768x512 → width follows height) */
#materials-gallery .thumb-img {
  width: auto !important;
  max-width: 100% !important;
  height: auto !important;
  min-height: 0 !important;
  overflow: hidden !important;
}
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
/* Hide Gradio Image toolbar only — do NOT hide the wrapper that contains <img>
   (Gradio 6 puts the still inside a button; hiding all buttons blanks the thumbs). */
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
                "Exit Program", variant="stop", scale=1, elem_id="exit-btn", min_width=140
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

    return app, css, None, None
