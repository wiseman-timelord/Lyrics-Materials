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


def _run_btn_updates(lyrics: str = "", song_name: str = ""):
    """
    One primary action button:
      ready  → Generate Materials (clickable)
      not    → Configure the Pages First (disabled grey)
    Using a single control avoids Gradio dual-visibility races where both
    buttons could end up hidden.
    """
    ok = _can_create(lyrics, song_name)
    if ok:
        return gr.update(
            value="Generate Materials",
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


def _list_project_images(project_dir: str = "") -> List[str]:
    """Sorted still paths for the active project (or APP_STATE paths)."""
    paths: List[str] = []
    folder = (project_dir or configure.APP_STATE.get("current_project_folder") or "").strip()
    if folder and Path(folder).is_dir():
        for p in sorted(Path(folder).glob("*.png")):
            paths.append(str(p))
        for p in sorted(Path(folder).glob("*.jpg")):
            paths.append(str(p))
    if not paths:
        for p in configure.APP_STATE.get("generation_output_paths") or []:
            if p and Path(p).exists():
                paths.append(str(p))
    # de-dupe preserve order
    seen = set()
    out: List[str] = []
    for p in paths:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out


def _thumb_panel_updates(paths: Optional[List[str]] = None) -> List[Any]:
    """
    Gradio updates for the thumb grid: each slot → (col visible, image value, button visible).
    Order matches _gen['thumb_cols'], _gen['thumb_imgs'], _gen['thumb_btns'].
    """
    if paths is None:
        paths = _list_project_images()
    # Stable order by leading line number ("001-…" or "001 - …")
    def _sort_key(p: str):
        name = Path(p).name
        m = re.match(r"^(\d+)", name)
        if m:
            return (0, int(m.group(1)), name.lower())
        return (1, 0, name.lower())
    paths = sorted([p for p in paths if p and Path(p).is_file()], key=_sort_key)
    th = _thumb_size_px()
    n = min(len(paths), THUMB_SLOTS)
    updates: List[Any] = []
    for i in range(THUMB_SLOTS):
        if i < n:
            # Absolute path string — Gradio Image accepts filepath values
            p = str(Path(paths[i]).resolve())
            updates.append(gr.update(visible=True))  # col
            updates.append(gr.update(value=p))  # img
            updates.append(gr.update(visible=True))  # btn
        else:
            updates.append(gr.update(visible=False))
            updates.append(gr.update(value=None))
            updates.append(gr.update(visible=False))
    return updates


def _thumb_panel_outputs() -> List[Any]:
    """Ordered list of Gradio components for thumb-panel updates."""
    outs: List[Any] = []
    for i in range(THUMB_SLOTS):
        outs.append(_gen["thumb_cols"][i])
        outs.append(_gen["thumb_imgs"][i])
        outs.append(_gen["thumb_btns"][i])
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

            with gr.Row():
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
                for row_i in range(0, THUMB_SLOTS, THUMB_COLS):
                    with gr.Row(elem_classes=["thumb-row"]):
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
                                btn = gr.Button(
                                    "Regenerate",
                                    visible=False,
                                    size="sm",
                                    elem_classes=["thumb-regen-btn"],
                                    min_width=max(64, int(th0 * 768 / 512) // 2),
                                )
                                _gen["thumb_cols"].append(col)
                                _gen["thumb_imgs"].append(img)
                                _gen["thumb_btns"].append(btn)


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
        main = (
            gr.update(value=""),
            gr.update(value="", lines=12, max_lines=12),
            gr.update(value=configure.STYLE_LIGHT),
            gr.update(value=configure.DEFAULT_STEPS),
            gr.update(value=configure.DEFAULT_CFG),
            gr.update(value=""),
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
        lyrics, song_name, style, steps, cfg_scale, ref_image, active_session_id,
        progress=gr.Progress(track_tqdm=False),
    ):
        # During a run: keep the action button visible but disabled; show Stop
        hide_run = gr.update(interactive=False, value="Generate Materials", visible=True)
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

        yield (
            "[  0%] start  Starting materials pipeline…",
            hide_run, show_stop, active_session_id or "",
        ) + tuple(_thumb_panel_updates([])) + tuple(
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

        while th.is_alive() or not prog_q.empty():
            updated = False
            try:
                while True:
                    msg, frac, info = prog_q.get_nowait()
                    status_line = _format_progress_line(msg, frac, info)
                    try:
                        progress(float(frac), desc=msg)
                    except Exception:
                        pass
                    updated = True
            except queue.Empty:
                pass

            if updated or (time.time() - last_yield) > 0.5:
                paths = [p for p in (configure.APP_STATE.get("generation_output_paths") or []) if Path(p).exists()]
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

        # Drain regenerate queue (queued while this run was busy)
        q = list(configure.APP_STATE.get("regen_queue") or [])
        configure.APP_STATE["regen_queue"] = []
        if q and not inference.is_cancel_requested():
            msg = f"{msg}  Processing {len(q)} queued regenerate(s)…"
            yield (
                msg, show_run, hide_stop, sid,
            ) + tuple(_thumb_panel_updates(paths)) + tuple(_refresh_session_slots(sid))
            for line_idx in q:
                if inference.is_cancel_requested():
                    break
                try:
                    configure.APP_STATE["generating"] = True
                    proj = configure.APP_STATE.get("current_project_folder") or ""
                    cfg_now = configure.load_configuration()
                    cfg_now["imagegen_steps"] = steps
                    cfg_now["imagegen_cfg_scale"] = cfg_scale
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

        paths = _list_project_images(configure.APP_STATE.get("current_project_folder") or "")
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
            _gen["active_session_id"],
        ],
        outputs=_run_outputs,
    )


    def _do_regen(line_idx: int, steps, cfg_scale, ref_image, active_session_id):
        """Regenerate one still, or queue if a generation is already running."""
        proj = (configure.APP_STATE.get("current_project_folder") or "").strip()
        if not proj or not Path(proj).is_dir():
            paths = _list_project_images()
            return (
                "No active project folder — load a session or Generate first.",
                gr.update(), gr.update(), active_session_id or "",
            ) + tuple(_thumb_panel_updates(paths)) + tuple(
                _refresh_session_slots(active_session_id or "")
            )

        # Map slot index to line number via sorted image filenames when possible
        paths = _list_project_images(proj)
        if line_idx < 0 or line_idx >= THUMB_SLOTS:
            return (
                "Invalid still index.",
                gr.update(), gr.update(), active_session_id or "",
            ) + tuple(_thumb_panel_updates(paths)) + tuple(
                _refresh_session_slots(active_session_id or "")
            )

        # Prefer line number from filename "001-…" / "001 - …" if the slot has an image
        resolved_idx = line_idx
        if line_idx < len(paths):
            name = Path(paths[line_idx]).name
            m = re.match(r"^(\d+)", name)
            if m:
                resolved_idx = int(m.group(1)) - 1
            else:
                resolved_idx = line_idx

        if configure.APP_STATE.get("generating") or configure.APP_STATE.get("session_status") == "running":
            q = list(configure.APP_STATE.get("regen_queue") or [])
            if resolved_idx not in q:
                q.append(resolved_idx)
            configure.APP_STATE["regen_queue"] = q
            return (
                f"Queued regenerate for still {resolved_idx + 1} "
                f"(will run after current generation finishes).",
                gr.update(), gr.update(), active_session_id or "",
            ) + tuple(_thumb_panel_updates(paths)) + tuple(
                _refresh_session_slots(active_session_id or "")
            )

        configure.APP_STATE["generating"] = True
        configure.APP_STATE["session_status"] = "running"
        try:
            cfg_now = configure.load_configuration()
            cfg_now["imagegen_steps"] = steps
            cfg_now["imagegen_cfg_scale"] = cfg_scale
            inference.regenerate_single_still(
                Path(proj),
                int(resolved_idx),
                cfg_now,
                reference_image=ref_image or "",
            )
            paths = _list_project_images(proj)
            configure.APP_STATE["session_status"] = "stopped"
            msg = f"Regenerated still {resolved_idx + 1}."
        except Exception as e:
            paths = _list_project_images(proj)
            configure.APP_STATE["session_status"] = "stopped"
            msg = f"Regenerate failed for still {resolved_idx + 1}: {e}"
            print(f"[regen] {msg}", flush=True)
        finally:
            configure.APP_STATE["generating"] = False

        # Drain any items queued while we were regenerating
        q = list(configure.APP_STATE.get("regen_queue") or [])
        configure.APP_STATE["regen_queue"] = []
        extra = ""
        for qi in q:
            if inference.is_cancel_requested():
                break
            try:
                configure.APP_STATE["generating"] = True
                cfg_now = configure.load_configuration()
                cfg_now["imagegen_steps"] = steps
                cfg_now["imagegen_cfg_scale"] = cfg_scale
                inference.regenerate_single_still(
                    Path(proj), int(qi), cfg_now, reference_image=ref_image or "",
                )
                extra += f" Also regenerated {int(qi) + 1}."
            except Exception as e:
                extra += f" Queued {int(qi) + 1} failed: {e}."
            finally:
                configure.APP_STATE["generating"] = False
        paths = _list_project_images(proj)
        return (
            msg + extra,
            gr.update(visible=True, interactive=True, value="Generate Materials"),
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

    # Wire each Regenerate button to ONE slot index only (partial, not shared lambda)
    for i, btn in enumerate(_gen["thumb_btns"]):
        btn.click(
            functools.partial(_do_regen, i),
            inputs=[_gen["steps"], _gen["cfg"], _gen["ref_image"], _gen["active_session_id"]],
            outputs=_regen_outputs,
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
  min-height: 120px;
  overflow-y: auto;
  max-height: 70vh;
}
#materials-gallery .thumb-slot {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 0.2rem;
  padding: 0.1rem;
}
#materials-gallery .thumb-img img,
#materials-gallery .thumb-img {
  object-fit: contain !important;
  max-width: 100% !important;
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
#materials-gallery .thumb-img {
  min-height: 64px !important;
}
#materials-gallery .thumb-img img {
  display: block !important;
  width: 100% !important;
  height: auto !important;
  max-height: 100% !important;
  object-fit: contain !important;
}
/* Do not hide the actual image element if Gradio wraps it oddly */
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
