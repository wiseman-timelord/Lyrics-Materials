"""
display.py - Gradio 6 UI for Lyrics-Slideshow.
Tabs: Generation | Prompting | Configuration | Preferences | Debug / Info
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
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

_FILETYPES_AUDIO = [
    ("Audio", "*.mp3 *.wav *.flac *.m4a *.ogg"),
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



# ---------------------------------------------------------------------------
# Audio staging for Gradio (never moves the user's original file)
# ---------------------------------------------------------------------------

_STAGED_AUDIO_NAME = "preview_audio"  # stem; suffix follows source


def _temp_audio_dir() -> Path:
    d = configure.get_data_dir() / "temp_audio"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _clear_staged_audio() -> None:
    """Remove previous preview copies under data/temp_audio/."""
    d = _temp_audio_dir()
    try:
        for f in d.iterdir():
            if f.is_file():
                try:
                    f.unlink()
                except OSError:
                    pass
    except OSError:
        pass


def stage_audio_for_player(source_path: str) -> Optional[str]:
    """
    Copy the selected track into data/temp_audio/ so Gradio can serve it.
    The original file is never moved or modified. Returns the staged path
    (or None if source is missing / unreadable).
    Supports mp3, wav, flac, m4a, ogg.
    """
    if not source_path:
        return None
    src = Path(source_path)
    if not src.is_file():
        return None
    suffix = src.suffix.lower() or ".mp3"
    if suffix not in (".mp3", ".wav", ".flac", ".m4a", ".ogg", ".aac", ".wma"):
        # still try — user may have an unusual but valid container
        pass
    dest = _temp_audio_dir() / f"{_STAGED_AUDIO_NAME}{suffix}"
    try:
        # Same path already staged (e.g. rebuild) — reuse
        if dest.exists() and dest.stat().st_size == src.stat().st_size:
            # Cheap check; also compare mtime
            if abs(dest.stat().st_mtime - src.stat().st_mtime) < 1.0:
                return str(dest)
        _clear_staged_audio()
        shutil.copy2(str(src), str(dest))
        return str(dest)
    except OSError as e:
        print(f"[audio] could not stage preview copy: {e}", flush=True)
        return None




def probe_audio_duration(path: str) -> float:
    """
    Return duration in seconds for mp3/wav/flac/…
    Tries: ffprobe → ffmpeg -i (Duration=) → wave module for .wav.
    Always logs which tool was used or why it failed.
    """
    if not path:
        return 0.0
    src = Path(path)
    if not src.is_file():
        print(f"[audio] probe: not a file: {path}", flush=True)
        return 0.0

    def _ok(d: float) -> float:
        try:
            d = float(d)
        except (TypeError, ValueError):
            return 0.0
        return d if d > 0.25 else 0.0

    def _run(cmd: list) -> subprocess.CompletedProcess:
        kwargs = dict(
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=45,
        )
        if sys.platform == "win32":
            # Hide console window; required for GUI apps spawning ffmpeg
            kwargs["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        return subprocess.run(cmd, **kwargs)

    # 1) ffprobe
    try:
        ffprobe = utilities.find_ffprobe()
        print(f"[audio] ffprobe path: {ffprobe}", flush=True)
        if ffprobe:
            r = _run([
                str(ffprobe), "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                str(src),
            ])
            out = (r.stdout or "").strip()
            if out:
                d = _ok(float(out.splitlines()[0].strip()))
                if d:
                    print(f"[audio] duration (ffprobe)={d:.2f}s  {src.name}", flush=True)
                    return d
            print(f"[audio] ffprobe no duration (rc={r.returncode}): {(r.stderr or '')[:200]}", flush=True)
    except Exception as e:
        print(f"[audio] ffprobe exception: {e}", flush=True)

    # 2) ffmpeg -i  (Duration on stderr; exit code is usually non-zero)
    try:
        ffmpeg = utilities.find_ffmpeg()
        print(f"[audio] ffmpeg path: {ffmpeg}", flush=True)
        if ffmpeg:
            r = _run([str(ffmpeg), "-hide_banner", "-i", str(src)])
            blob = (r.stderr or "") + "\n" + (r.stdout or "")
            m = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", blob)
            if not m:
                m = re.search(r"Duration:\s*(\d+):(\d+):(\d+)", blob)
            if m:
                h, mi, sec = int(m.group(1)), int(m.group(2)), float(m.group(3))
                d = _ok(h * 3600 + mi * 60 + sec)
                if d:
                    print(f"[audio] duration (ffmpeg)={d:.2f}s  {src.name}", flush=True)
                    return d
            # log a short excerpt so we can see what ffmpeg said
            excerpt = " | ".join(
                ln.strip() for ln in blob.splitlines()
                if ln.strip()
            )[:300]
            print(f"[audio] ffmpeg no Duration match: {excerpt}", flush=True)
    except Exception as e:
        print(f"[audio] ffmpeg exception: {e}", flush=True)

    # 3) wave module for PCM wav
    if src.suffix.lower() == ".wav":
        try:
            import wave
            with wave.open(str(src), "rb") as w:
                frames = w.getnframes()
                rate = w.getframerate() or 1
                d = _ok(float(frames) / float(rate))
                if d:
                    print(f"[audio] duration (wave)={d:.2f}s  {src.name}", flush=True)
                    return d
        except Exception as e:
            print(f"[audio] wave exception: {e}", flush=True)

    print(f"[audio] could not probe duration for {src}", flush=True)
    return 0.0



def format_mmss(seconds: float) -> str:
    s = max(0, int(round(float(seconds or 0))))
    return f"{s // 60}:{s % 60:02d}"


def audio_time_html(current: float = 0.0, total: float = 0.0) -> str:
    """HTML readout: current / total, with data-duration for the JS binder."""
    tot = max(0.0, float(total or 0.0))
    cur = max(0.0, float(current or 0.0))
    return (
        f'<div id="audio-time-readout" data-duration="{tot:.3f}">'
        f"{format_mmss(cur)} / {format_mmss(tot)}</div>"
    )


def _transport_html(current: float = 0.0, total: float = 0.0) -> str:
    """Time readout only (Play is Gradio's native Track Preview control)."""
    tot = max(0.0, float(total or 0.0))
    cur = max(0.0, float(current or 0.0))
    print(f"[ls-time][py] _transport_html current={cur:.3f} total={tot:.3f}")
    return (
        f'<div id="track-preview-controls">'
        f'<div id="audio-time-readout" data-duration="{tot:.3f}">'
        f"{format_mmss(cur)} / {format_mmss(tot)}</div></div>"
    )




def _open_output_folder() -> str:
    out_dir = configure.get_output_dir()
    try:
        subprocess.Popen(["explorer", str(out_dir)])
        return f"Opened: {out_dir}"
    except Exception as e:
        return f"ERROR: {e}"


# ---------------------------------------------------------------------------
# Generation tab
# ---------------------------------------------------------------------------

_gen: Dict[str, Any] = {}


def _models_configured() -> bool:
    """Encoder + diffusion paths must exist on disk. VAE is recommended but optional."""
    cfg = _cfg()
    enc = cfg.get("encoder_model_path", "")
    diff = cfg.get("imagegen_model_path", "")
    if not enc or not Path(enc).exists():
        return False
    if not diff or not Path(diff).exists():
        return False
    return True


def _can_create(lyrics: str, audio_path: str = "", markers_json: str = "") -> bool:
    """Ready when lyrics + models + audio + at least two markers are present."""
    if not (lyrics or "").strip():
        return False
    if not _models_configured():
        return False
    if not audio_path or not Path(audio_path).exists():
        return False
    # markers_json is a JSON list of {label, time}
    try:
        import json as _json
        markers = _json.loads(markers_json) if markers_json else []
        if not isinstance(markers, list) or len(markers) < 2:
            return False
    except Exception:
        return False
    return True


def _run_btn_update(lyrics: str, audio_path: str, markers_json: str) -> Any:
    return gr.update(visible=_can_create(lyrics, audio_path, markers_json))


def _format_progress_line(msg: str, frac: float, info: dict) -> str:
    """Single status-bar line (phase + counters)."""
    phase = (info or {}).get("phase", "")
    pct = int(round(max(0.0, min(1.0, float(frac or 0.0))) * 100))
    bits = [f"[{pct:3d}%]"]
    if phase:
        bits.append(str(phase))
    line_n = (info or {}).get("line")
    total = (info or {}).get("total")
    if line_n is not None and total is not None:
        bits.append(f"{line_n}/{total}")
    step = (info or {}).get("step")
    total_steps = (info or {}).get("total_steps")
    if step is not None and total_steps is not None:
        bits.append(f"step {step}/{total_steps}")
    bits.append(msg)
    # One line only for the status bar
    return " ".join(bits).replace("\n", " ").strip()[:240]


def _open_project_folder() -> str:
    """Open the current (or last) project folder in Explorer."""
    folder = (
        configure.APP_STATE.get("current_project_folder")
        or _gcfg().get("last_project_folder")
        or ""
    )
    if folder and Path(folder).is_dir():
        try:
            subprocess.Popen(["explorer", str(folder)])
            return f"Opened project folder: {folder}"
        except Exception as e:
            return f"ERROR: {e}"
    out = configure.get_output_dir()
    try:
        subprocess.Popen(["explorer", str(out)])
        return f"No active project — opened output root: {out}"
    except Exception as e:
        return f"ERROR: {e}"


def _build_create_tab() -> None:
    initial_lyrics = _gcfg().get("last_lyrics", "")
    g = _gcfg()
    prefs = _prefs()
    initial_style = prefs.get("style", configure.STYLE_LIGHT)
    initial_audio = g.get("last_audio_path", "") or ""
    initial_markers = g.get("last_markers") or []
    import json as _json
    initial_markers_json = _json.dumps(initial_markers)

    can = _can_create(initial_lyrics, initial_audio, initial_markers_json)

    with gr.Row():
        # ---- Left column: settings + audio path ----
        with gr.Column(scale=1):
            gr.Markdown("### Project settings")
            # Song length is detected from the audio file (hidden); not user-edited.
            _init_len = probe_audio_duration(initial_audio) if initial_audio else 0.0
            if _init_len <= 0:
                try:
                    _init_len = float(g.get("song_length_seconds") or 0)
                except (TypeError, ValueError):
                    _init_len = 0.0
            _gen["song_length"] = gr.Number(
                value=_init_len,
                visible=False,
                elem_id="song-length-hidden",
            )
            with gr.Row():
                _gen["video_format"] = gr.Dropdown(
                    label="Output container",
                    choices=configure.VIDEO_CHOICES,
                    value=prefs.get("video_format", configure.VIDEO_MP4),
                )
                _gen["resolution"] = gr.Dropdown(
                    label="Output resolution",
                    choices=configure.RESOLUTION_CHOICES,
                    value=g.get("output_resolution", configure.RESOLUTION_720P),
                )

            with gr.Row():
                _gen["style"] = gr.Dropdown(
                    label="Visual style",
                    choices=configure.STYLE_CHOICES,
                    value=initial_style,
                    info="Fade-in / fade-out colour between sections follows this theme.",
                )

            gr.Markdown("### Generation defaults (Flux.2-klein-4B)")
            with gr.Row():
                _gen["width"] = gr.Number(
                    label="Image width",
                    value=g.get("imagegen_width", configure.DEFAULT_WIDTH),
                    minimum=256,
                    maximum=2048,
                    step=64,
                )
                _gen["height"] = gr.Number(
                    label="Image height",
                    value=g.get("imagegen_height", configure.DEFAULT_HEIGHT),
                    minimum=256,
                    maximum=2048,
                    step=64,
                )
            with gr.Row():
                _gen["steps"] = gr.Slider(
                    label="Steps",
                    minimum=1,
                    maximum=20,
                    step=1,
                    value=g.get("imagegen_steps", configure.DEFAULT_STEPS),
                )
                _gen["cfg"] = gr.Slider(
                    label="CFG",
                    minimum=0.5,
                    maximum=4.0,
                    step=0.1,
                    value=g.get("imagegen_cfg_scale", configure.DEFAULT_CFG),
                )

            # Audio path — the Browse button now sits beside Track Preview below,
            # where the native Gradio Browse control is expected to be.
            _gen["audio_path"] = gr.Textbox(
                label="Audio file (required)",
                value=initial_audio,
                interactive=True,
            )

        # ---- Right column: lyrics + Rebuild Markers on same row ----
        with gr.Column(scale=1):
            gr.Markdown(
                "### Lyrics\n"
                "Paste full lyrics. Section headers (`[Intro]`, `[Chorus_1]`, `[Outro]`, "
                "`[Fade_Out]`, …) are detected automatically and become timeline markers."
            )
            with gr.Row(elem_id="lyrics-row"):
                _gen["lyrics"] = gr.Textbox(
                    label="Lyrics (one image per non-empty line)",
                    lines=14,
                    max_lines=36,
                    placeholder="Paste full lyrics here…\n[Intro]\nFirst line…\n…",
                    value=initial_lyrics,
                    elem_id="lyrics-box",
                    scale=5,
                )
                with gr.Column(scale=1, min_width=140):
                    gr.Markdown("&nbsp;", elem_id="lyrics-btn-spacer")
                    _gen["rebuild_markers"] = gr.Button(
                        "Rebuild Markers",
                        variant="secondary",
                        elem_id="rebuild-markers-btn",
                    )

    # ---- Full-width: preview player, then markers ----
    gr.Markdown(
        "### Audio preview & section markers\n"
        "Listen and scrub the track, then set marker times. "
        "Every section has a **Start** and **End**: images for that section play "
        "between them, with a style-colour fade-in at Start and fade-out at End. "
        "Gaps between an End and the next Start stay solid style colour "
        "(white / black / saturated purple)."
    )

    _initial_staged = (
        stage_audio_for_player(initial_audio)
        if initial_audio and Path(initial_audio).exists()
        else None
    )
    _init_dur = 0.0
    if initial_audio and Path(initial_audio).exists():
        _init_dur = probe_audio_duration(initial_audio)
    if _init_dur <= 0 and _initial_staged:
        _init_dur = probe_audio_duration(_initial_staged)

    with gr.Column(elem_id="audio-player-wrap"):
        with gr.Row(elem_id="track-preview-row"):
            with gr.Column(scale=5, min_width=0, elem_id="track-preview-col"):
                _gen["audio_player"] = gr.Audio(
                    label="Track Preview",
                    type="filepath",
                    value=_initial_staged,
                    interactive=True,
                    elem_id="audio-player",
                    sources=[],   # no upload/mic icons, no drag-and-drop
                    buttons=[],   # no download/share buttons (Gradio 6 name for this)
                )
                # Time readout: positioned inline with the "Track Preview" label
                # via CSS (see #track-transport), not stacked underneath it.
                _gen["transport"] = gr.HTML(
                    value=_transport_html(0.0, _init_dur),
                    elem_id="track-transport",
                )
            with gr.Column(scale=1, min_width=110, elem_id="track-preview-browse-col"):
                _gen["browse_audio"] = gr.Button(
                    "Browse", elem_id="track-preview-browse-btn"
                )

    # Hidden: live playback position (seconds) written by JS every 200ms
    _gen["playback_pos"] = gr.Textbox(
        value="0",
        visible=False,
        elem_id="playback-pos",
    )
    _gen["markers_json"] = gr.Textbox(
        value=initial_markers_json,
        visible=False,
        elem_id="markers-json",
    )
    _gen["marker_hint"] = gr.Markdown(
        value=_marker_hint_text(initial_markers),
    )

    gr.Markdown(
        "**Section markers** — edit the time, or click **Set Here** to use the "
        "current Preview/scrub position."
    )
    # Fixed slots so each marker has its own Set Here button.
    # 24 slots = up to 12 Start/End section pairs (typical songs need far fewer).
    _gen["m_rows"] = []
    _gen["m_labels"] = []
    _gen["m_times"] = []
    _gen["m_set_btns"] = []
    _init_n = len(initial_markers)
    for _i in range(MAX_MARKER_SLOTS):
        _vis = _i < _init_n
        _lab = initial_markers[_i]["label"] if _vis else ""
        _tim = float(initial_markers[_i].get("time", 0)) if _vis else 0.0
        with gr.Row(visible=_vis, elem_id=f"marker-row-{_i}") as _row:
            _lbl = gr.Textbox(
                value=_lab,
                label="Marker" if _i == 0 else None,
                show_label=(_i == 0),
                interactive=False,
                scale=3,
                elem_id=f"marker-label-{_i}",
            )
            _tnum = gr.Number(
                value=_tim,
                label="Time (s)" if _i == 0 else None,
                show_label=(_i == 0),
                minimum=0,
                step=0.01,
                precision=2,
                scale=2,
                elem_id=f"marker-time-{_i}",
            )
            _sbtn = gr.Button(
                "Set Position",
                scale=1,
                elem_id=f"marker-set-{_i}",
                elem_classes=["ls-set-position-btn"],
            )
        _gen["m_rows"].append(_row)
        _gen["m_labels"].append(_lbl)
        _gen["m_times"].append(_tnum)
        _gen["m_set_btns"].append(_sbtn)

    with gr.Row():
        _gen["run_btn"] = gr.Button(
            "Create Music Video",
            variant="primary",
            scale=2,
            visible=can,
        )
        _gen["stop_btn"] = gr.Button(
            "Emergency Stop",
            variant="stop",
            scale=1,
            visible=False,
        )

    _gen["open_project"] = gr.Button(
        "Slides Thumbnails (click to open project folder)",
        elem_id="slides-gallery-link",
    )
    _gen["gallery"] = gr.Gallery(
        label=None,
        show_label=False,
        columns=8,
        height=160,
        object_fit="contain",
        elem_id="slides-gallery",
    )



MAX_MARKER_SLOTS = 24


def _apply_markers_to_slots(markers: list):
    """Return Gradio updates for label/time/row visibility for all slots + json + hint."""
    import json as _json
    markers = markers or []
    updates = []
    # rows, labels, times visibility/values — order must match wire outputs
    for i in range(MAX_MARKER_SLOTS):
        if i < len(markers):
            lab = str(markers[i].get("label", ""))
            tim = float(markers[i].get("time", 0) or 0)
            updates.append(gr.update(visible=True))   # row
            updates.append(gr.update(value=lab))      # label
            updates.append(gr.update(value=tim))      # time
        else:
            updates.append(gr.update(visible=False))
            updates.append(gr.update(value=""))
            updates.append(gr.update(value=0.0))
    updates.append(_json.dumps(markers))
    updates.append(_marker_hint_text(markers))
    return updates


def _collect_markers_from_slots(*slot_values):
    """
    slot_values interleaves label, time for each visible slot (2 * MAX).
    Or we pass labels then times — see wire. Here: all labels then all times.
    """
    labels = slot_values[:MAX_MARKER_SLOTS]
    times = slot_values[MAX_MARKER_SLOTS:]
    out = []
    for lab, tim in zip(labels, times):
        lab = str(lab or "").strip()
        if not lab:
            continue
        try:
            t = float(tim)
        except (TypeError, ValueError):
            t = 0.0
        out.append({"label": lab, "time": max(0.0, t)})
    return out


def _marker_hint_text(markers) -> str:
    if not markers:
        return (
            "_No markers yet. Paste lyrics with section headers, set song length / audio, "
            "then click **Rebuild markers from lyrics**._"
        )
    labels = ", ".join(m.get("label", "?") for m in markers)
    return f"_Markers ({len(markers)}): {labels}_"



def _rebuild_markers_from_lyrics(lyrics: str, song_length: float, audio_path: str):
    """Parse section headers → default evenly-spaced markers; duration from audio."""
    import json as _json
    parsed = inference.parse_lyrics(lyrics or "")
    sections = inference.sections_with_lines(parsed)
    labels = inference.marker_plan_from_sections(sections)
    length = probe_audio_duration(audio_path) if audio_path else 0.0
    if length <= 0 and audio_path:
        staged_try = stage_audio_for_player(audio_path)
        if staged_try:
            length = probe_audio_duration(staged_try)
    if length <= 0:
        try:
            length = float(song_length or 0)
        except (TypeError, ValueError):
            length = 0.0
    if length <= 0:
        length = 180.0  # last-resort fallback only
        print("[audio] WARNING: using 180s fallback — duration probe failed", flush=True)
    markers = inference.default_marker_times(labels, length)
    staged = stage_audio_for_player(audio_path) if audio_path else None
    slot_updates = _apply_markers_to_slots(markers)
    return (
        *slot_updates,  # rows/labels/times x N, markers_json, hint
        gr.update(value=length),
        gr.update(value=staged),
        _transport_html(0.0, length),
    )


def _sync_markers_from_slots(*vals):
    """Any marker time/label edit → refresh markers_json + hint."""
    import json as _json
    markers = _collect_markers_from_slots(*vals)
    return _json.dumps(markers), _marker_hint_text(markers)



def _set_marker_here(index: int, playback_pos: str, *slot_vals):
    """Set marker at index to current playback position (seconds)."""
    markers = _collect_markers_from_slots(*slot_vals)
    try:
        pos = float(playback_pos or 0)
    except (TypeError, ValueError):
        pos = 0.0
    pos = max(0.0, pos)
    print(f"[ls-time][py] set_marker index={index} playback_pos={playback_pos!r} resolved={pos:.3f}")
    if 0 <= index < len(markers):
        markers[index]["time"] = round(pos, 3)
    return _apply_markers_to_slots(markers)


def _do_create(

    lyrics: str,
    song_length: float,
    style: str,
    audio_path: str,
    markers_json: str,
    video_format: str,
    resolution: str,
    width: float,
    height: float,
    steps: float,
    cfg_scale: float,
    progress=gr.Progress(track_tqdm=False),
):
    """
    Generator: Create hidden / Stop shown; streams one-line status updates;
    restores buttons when finished. Progress goes to the shared status bar only.
    """
    import threading
    import queue

    hide_run = gr.update(visible=False)
    show_stop = gr.update(visible=True)
    hide_stop = gr.update(visible=False)

    def _final_run_btn() -> Any:
        return gr.update(visible=_can_create(lyrics, audio_path, markers_json))

    if not _can_create(lyrics, audio_path, markers_json):
        msg = (
            "Cannot start: need lyrics, Encoder + Diffusion models, an audio file, "
            "and at least two section markers. "
            "Paste lyrics → Rebuild markers → adjust times → Create."
        )
        yield msg, [], _final_run_btn(), hide_stop
        return

    import json as _json
    try:
        markers = _json.loads(markers_json) if markers_json else []
    except Exception:
        markers = []
    markers = [m for m in markers if isinstance(m, dict) and m.get("label")]

    cfg = configure.generation_config()
    cfg["style"] = style
    cfg["markers"] = markers
    cfg["video_format"] = video_format
    cfg["output_resolution"] = resolution or configure.RESOLUTION_720P
    cfg["imagegen_width"] = int(width or configure.DEFAULT_WIDTH)
    cfg["imagegen_height"] = int(height or configure.DEFAULT_HEIGHT)
    cfg["imagegen_steps"] = int(steps or configure.DEFAULT_STEPS)
    cfg["imagegen_cfg_scale"] = float(cfg_scale or configure.DEFAULT_CFG)
    # Prompt template comes from prompting.json for the selected style
    cfg["prompt_template"] = configure.prompt_template_for_style(style)

    # Merge encoder / backend paths from configuration.json
    c = configure.load_configuration()
    for k in (
        "encoder_model_path", "mmproj_path", "thinking_model_path",
        "imagegen_model_path", "vae_model_path",
        "encoder_backend", "thinking_backend", "imagegen_backend",
        "worker_threads",
        "encoder_threads", "thinking_threads", "imagegen_threads",
        "imagegen_vulkan_device", "imagegen_placement",
        "model_load_mode", "text_load_mode", "encoder_load_mode", "thinking_load_mode", "imagegen_load_mode",
        "text_gpu_layers", "encoder_gpu_layers", "thinking_gpu_layers", "imagegen_gpu_layers",
    ):
        cfg[k] = c.get(k)

    configure.update_generation({
        "last_lyrics": lyrics,
        "song_length_seconds": float(song_length or 180),
        "last_audio_path": audio_path or "",
        "output_resolution": cfg["output_resolution"],
        "imagegen_width": cfg["imagegen_width"],
        "imagegen_height": cfg["imagegen_height"],
        "imagegen_steps": cfg["imagegen_steps"],
        "imagegen_cfg_scale": cfg["imagegen_cfg_scale"],
        "last_markers": markers,
    })
    configure.update_preferences({
        "style": style,
        "video_format": video_format,
    })

    yield "[  0%] start  Starting pipeline…", [], hide_run, show_stop

    prog_q: queue.Queue = queue.Queue()
    result_holder: Dict[str, Any] = {}

    def cb(msg: str, frac: float, info: dict) -> None:
        prog_q.put((msg, frac, info or {}))

    def worker() -> None:
        try:
            result_holder["r"] = inference.run_lyric_video_pipeline(
                lyrics=lyrics,
                song_length_seconds=float(song_length or 180),
                audio_path=audio_path or "",
                cfg=cfg,
                progress_callback=cb,
            )
        except Exception as e:
            result_holder["r"] = {
                "success": False,
                "message": f"Pipeline error: {e}",
                "video_path": "",
                "project_folder": "",
                "image_count": 0,
            }
            prog_q.put((f"ERROR: {e}", 1.0, {"phase": "error"}))

    th = threading.Thread(target=worker, daemon=True)
    th.start()

    status_line = "[  0%] start  Starting pipeline…"
    last_yield = time.time()

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

        if updated or (time.time() - last_yield) > 0.4:
            gallery: List[Any] = []
            proj = configure.APP_STATE.get("current_project_folder") or ""
            if proj and Path(proj).is_dir():
                gallery = sorted(Path(proj).glob("slide_*.png"))
            yield status_line, gallery, hide_run, show_stop
            last_yield = time.time()

        if th.is_alive():
            time.sleep(0.15)

    th.join(timeout=5)
    result = result_holder.get("r") or {
        "success": False,
        "message": "No result returned.",
        "project_folder": "",
    }

    gallery = []
    proj = result.get("project_folder") or configure.APP_STATE.get("current_project_folder") or ""
    if proj and Path(proj).is_dir():
        gallery = sorted(Path(proj).glob("slide_*.png"))
        configure.update_generation({"last_project_folder": str(proj)})

    if result.get("success"):
        status_line = f"[100%] done  {result.get('message', 'Done.')}"
    else:
        status_line = f"[100%] error  {result.get('message', 'Failed.')}"

    yield status_line, gallery, _final_run_btn(), hide_stop


def _on_emergency_stop():
    msg = inference.emergency_stop()
    return msg, gr.update(visible=False)


def _wire_create_events(status_box: gr.Textbox) -> None:
    def _on_browse_audio():
        path = _browse_file(_FILETYPES_AUDIO, "last_audio_browse_dir")
        staged = stage_audio_for_player(path) if path else None
        dur = probe_audio_duration(path) if path else 0.0
        if dur <= 0 and staged:
            dur = probe_audio_duration(staged)
        if path:
            configure.update_generation({
                "last_audio_path": path,
                "song_length_seconds": float(dur) if dur > 0 else 0.0,
            })
        return (
            path or "",
            gr.update(value=staged),
            gr.update(value=float(dur) if dur > 0 else 0.0),
            _transport_html(0.0, dur),
        )

    _gen["browse_audio"].click(
        _on_browse_audio,
        outputs=[
            _gen["audio_path"],
            _gen["audio_player"],
            _gen["song_length"],
            _gen["transport"],
        ],
    )
    _gen["open_project"].click(_open_project_folder, outputs=status_box)

    # Track Preview transport — pure client-side JS (no server round-trip)

    # Rebuild markers → fill slots
    _slot_outputs = []
    for i in range(MAX_MARKER_SLOTS):
        _slot_outputs.extend([
            _gen["m_rows"][i],
            _gen["m_labels"][i],
            _gen["m_times"][i],
        ])
    _slot_outputs.extend([_gen["markers_json"], _gen["marker_hint"]])

    _gen["rebuild_markers"].click(
        _rebuild_markers_from_lyrics,
        inputs=[_gen["lyrics"], _gen["song_length"], _gen["audio_path"]],
        outputs=_slot_outputs + [
            _gen["song_length"],
            _gen["audio_player"],
            _gen["transport"],
        ],
    )

    # Editing any time field re-syncs JSON
    _label_inputs = list(_gen["m_labels"])
    _time_inputs = list(_gen["m_times"])
    _all_slot_inputs = _label_inputs + _time_inputs

    for tcomp in _gen["m_times"]:
        tcomp.change(
            _sync_markers_from_slots,
            inputs=_all_slot_inputs,
            outputs=[_gen["markers_json"], _gen["marker_hint"]],
        )

    # Set Here per marker index
    def _make_set_handler(idx: int):
        def _handler(playback_pos, *slot_vals):
            return _set_marker_here(idx, playback_pos, *slot_vals)
        return _handler

    # Client JS reads Gradio's live clock (or media) at click time and
    # writes seconds into the hidden playback_pos field before Python runs.
    _set_pos_js = """
(pos, ...rest) => {
  try { console.warn('[ls-time] SetPos JS fired, incoming pos=', pos); } catch (e) {}
  let sec = 0;
  try {
    if (typeof window.__lsGetPlaybackSeconds === 'function') {
      sec = Number(window.__lsGetPlaybackSeconds()) || 0;
      try { console.warn('[ls-time] SetPos from __lsGetPlaybackSeconds=', sec); } catch (e) {}
    }
  } catch (e) { try { console.warn('[ls-time] SetPos __ls err', e); } catch (e2) {} }
  // Also try Gradio time#time directly
  if (!(sec > 0)) {
    try {
      const hosts = document.getElementsByTagName('gradio-app');
      const root = (hosts.length && hosts[0].shadowRoot) ? hosts[0].shadowRoot : document;
      const el = root.querySelector('time#time') || root.querySelector('.timestamps time');
      if (el) {
        const m = String(el.textContent || '').trim().match(/^([0-9]+):([0-9]{2})/);
        if (m) sec = parseInt(m[1], 10) * 60 + parseInt(m[2], 10);
      }
    } catch (e) {}
  }
  // Push into the hidden textbox so Python sees it
  try {
    const hosts = document.getElementsByTagName('gradio-app');
    const root = (hosts.length && hosts[0].shadowRoot) ? hosts[0].shadowRoot : document;
    const wrap = root.querySelector('#playback-pos');
    const input = wrap && wrap.querySelector('textarea, input');
    if (input) {
      input.value = (Math.max(0, sec)).toFixed(3);
      input.dispatchEvent(new Event('input', { bubbles: true }));
    }
  } catch (e) {}
  return [(Math.max(0, sec)).toFixed(3), ...rest];
}
"""
    for i, btn in enumerate(_gen["m_set_btns"]):
        btn.click(
            _make_set_handler(i),
            inputs=[_gen["playback_pos"]] + _all_slot_inputs,
            outputs=_slot_outputs,
            js=_set_pos_js,
        )

    for src in (_gen["lyrics"], _gen["audio_path"], _gen["markers_json"]):
        src.change(
            _run_btn_update,
            inputs=[_gen["lyrics"], _gen["audio_path"], _gen["markers_json"]],
            outputs=[_gen["run_btn"]],
        )

    _gen["run_btn"].click(
        _do_create,
        inputs=[
            _gen["lyrics"],
            _gen["song_length"],
            _gen["style"],
            _gen["audio_path"],
            _gen["markers_json"],
            _gen["video_format"],
            _gen["resolution"],
            _gen["width"],
            _gen["height"],
            _gen["steps"],
            _gen["cfg"],
        ],
        outputs=[
            status_box,
            _gen["gallery"],
            _gen["run_btn"],
            _gen["stop_btn"],
        ],
    )
    _gen["stop_btn"].click(
        _on_emergency_stop,
        outputs=[status_box, _gen["stop_btn"]],
    )



def _on_emergency_stop():
    msg = inference.emergency_stop()
    return msg, gr.update(visible=False)



# ---------------------------------------------------------------------------
# Prompting tab
# ---------------------------------------------------------------------------

_prm: Dict[str, Any] = {}


def _build_prompting_tab() -> None:
    data = _prompting()
    gr.Markdown(
        "### Visual style prompt templates\n"
        "These templates are used when generating image descriptions for each lyric line. "
        "`{line}` is replaced by the lyric text; `{style}` is available if you include it. "
        "Edit and save — the Generation page uses the template that matches the selected Visual style."
    )
    _prm["light"] = gr.Textbox(
        label=f"Style: {configure.STYLE_LIGHT}",
        lines=5,
        max_lines=12,
        value=data.get(configure.STYLE_LIGHT, configure.STYLE_PROMPT_TEMPLATES[configure.STYLE_LIGHT]),
    )
    _prm["dark"] = gr.Textbox(
        label=f"Style: {configure.STYLE_DARK}",
        lines=5,
        max_lines=12,
        value=data.get(configure.STYLE_DARK, configure.STYLE_PROMPT_TEMPLATES[configure.STYLE_DARK]),
    )
    _prm["colorful"] = gr.Textbox(
        label=f"Style: {configure.STYLE_COLORFUL}",
        lines=5,
        max_lines=12,
        value=data.get(configure.STYLE_COLORFUL, configure.STYLE_PROMPT_TEMPLATES[configure.STYLE_COLORFUL]),
    )
    with gr.Row():
        _prm["save_btn"] = gr.Button("Save Prompt Templates", variant="primary")
        _prm["reset_btn"] = gr.Button("Reset to defaults")


def _wire_prompting_events(status_box: gr.Textbox) -> None:
    def _save(light: str, dark: str, colorful: str) -> str:
        configure.update_prompting({
            configure.STYLE_LIGHT: (light or "").strip(),
            configure.STYLE_DARK: (dark or "").strip(),
            configure.STYLE_COLORFUL: (colorful or "").strip(),
        })
        return "Prompt templates saved."

    def _reset():
        defaults = configure._default_prompting()
        configure.save_prompting(defaults)
        return (
            defaults[configure.STYLE_LIGHT],
            defaults[configure.STYLE_DARK],
            defaults[configure.STYLE_COLORFUL],
            "Prompt templates reset to defaults.",
        )

    _prm["save_btn"].click(
        _save,
        inputs=[_prm["light"], _prm["dark"], _prm["colorful"]],
        outputs=status_box,
    )
    _prm["reset_btn"].click(
        _reset,
        outputs=[_prm["light"], _prm["dark"], _prm["colorful"], status_box],
    )


# ---------------------------------------------------------------------------
# Configuration tab
# ---------------------------------------------------------------------------

_cfg_w: Dict[str, Any] = {}


def _build_config_tab() -> None:
    """
    Layout:
      [Threads Used] full width
      [ Encoder | Thinking ] (2/3)   |   [ Diffuser ] (1/3)
      shared Load Mode + GPU Layers under text models
      [Save Configuration] full width
    """
    backends = configure.get_backend_choices()["all_choices"]
    cfg0 = _cfg()
    ph = configure.MODEL_PATH_PLACEHOLDERS

    def _backend_default(key: str) -> str:
        saved = cfg0.get(key, "")
        if saved in backends:
            return saved
        return backends[0] if backends else "CPU"

    def _path_or_empty(key: str) -> str:
        return (cfg0.get(key) or "").strip()

    def _text_load_mode() -> str:
        v = (
            cfg0.get("text_load_mode")
            or cfg0.get("encoder_load_mode")
            or cfg0.get("model_load_mode")
            or configure.DEFAULT_LOAD_MODE
        )
        return configure.normalize_load_mode(str(v))

    def _text_layers() -> int:
        for k in ("text_gpu_layers", "encoder_gpu_layers"):
            try:
                return int(cfg0.get(k, configure.DEFAULT_GPU_LAYERS))
            except (TypeError, ValueError):
                continue
        return configure.DEFAULT_GPU_LAYERS

    img_backend0 = _backend_default("imagegen_backend")
    enc_backend0 = _backend_default("encoder_backend")
    th_backend0 = _backend_default("thinking_backend")
    placement_visible = not str(img_backend0).upper().startswith("CPU")
    # Shared text GPU layers visible if either text backend is GPU
    text_gpu_vis = (
        not str(enc_backend0).upper().startswith("CPU")
        or not str(th_backend0).upper().startswith("CPU")
    )

    gr.Markdown(
        "### Models\n"
        "**Encoder** (Instruct / Uncensored) for project naming. "
        "**Thinking** (optional Q5) for visual prompts — falls back to Encoder if empty. "
        "**Diffuser** (Flux.2-klein Q8). "
        "mmproj is quarantined under `models/mmproj/` (not used for pure text).\n\n"
        "GPU layers **−1 = auto**: GGUF layer count × safe VRAM floor "
        "(free MiB rounded down to whole GB from install probe). "
        "This applies to the text models only — the Diffuser has no per-layer "
        "GPU control; its GPU placement is set entirely via **Diffuser Placement** below."
    )

    # ---- Threads Used (full width) ----
    logical = configure.get_cpu_info().get("cores_logical", 8) or 8
    max_usable = max(configure.WORKER_THREADS_MIN, logical - configure.AFFINITY_CORE_OFFSET)
    try:
        wt0 = int(cfg0.get("worker_threads", configure.WORKER_THREADS_DEFAULT))
    except (TypeError, ValueError):
        wt0 = configure.WORKER_THREADS_DEFAULT
    wt0 = max(configure.WORKER_THREADS_MIN, min(wt0, max_usable))
    gr.Markdown(
        "### Threads Used\n"
        "One setting for **all** multi-threaded heavy work. "
        f"Logical CPUs **0 and 1 stay free** for Windows/UI; workers use cores from 2 "
        f"(e.g. 8 threads → cores 2–9). Range on this machine: "
        f"{configure.WORKER_THREADS_MIN}–{max_usable}."
    )
    _cfg_w["worker_threads"] = gr.Slider(
        label="Threads Used",
        minimum=configure.WORKER_THREADS_MIN,
        maximum=max_usable,
        step=1,
        value=wt0,
    )

    # ---- Main row: text models 2/3 | diffuser 1/3 ----
    with gr.Row():
        with gr.Column(scale=2):
            gr.Markdown("#### Text models (Encoder + Thinking)")
            with gr.Row():
                with gr.Column(scale=1):
                    _cfg_w["enc_path"] = gr.Textbox(
                        label="Encoder model path",
                        value=_path_or_empty("encoder_model_path"),
                        placeholder=ph["encoder"],
                    )
                    _cfg_w["enc_browse"] = gr.Button("Browse encoder", scale=0)
                    _cfg_w["enc_backend"] = gr.Dropdown(
                        label="Encoder backend",
                        choices=backends,
                        value=enc_backend0,
                    )
                with gr.Column(scale=1):
                    _cfg_w["th_path"] = gr.Textbox(
                        label="Thinking model path (optional)",
                        value=_path_or_empty("thinking_model_path"),
                        placeholder=ph["thinking"],
                    )
                    _cfg_w["th_browse"] = gr.Button("Browse thinking", scale=0)
                    _cfg_w["th_backend"] = gr.Dropdown(
                        label="Thinking backend",
                        choices=backends,
                        value=th_backend0,
                    )
            gr.Markdown(
                "Shared settings for **both** text models (Load Mode + GPU layers)."
            )
            _cfg_w["text_load_mode"] = gr.Dropdown(
                label="Load mode (Encoder + Thinking)",
                choices=configure.LOAD_MODE_CHOICES,
                value=_text_load_mode(),
            )
            _cfg_w["text_gpu_layers"] = gr.Number(
                label="GPU layers (−1 = auto from safe VRAM)",
                value=_text_layers(),
                minimum=-1,
                maximum=configure.ENCODER_MAX_LAYERS,
                step=1,
                precision=0,
                visible=text_gpu_vis,
            )

        with gr.Column(scale=1):
            gr.Markdown("#### Diffuser (Flux.2)")
            _cfg_w["diff_path"] = gr.Textbox(
                label="Diffusion model path",
                value=_path_or_empty("imagegen_model_path"),
                placeholder=ph["diffuser"],
            )
            _cfg_w["diff_browse"] = gr.Button("Browse diffusion", scale=0)
            _cfg_w["vae_path"] = gr.Textbox(
                label="VAE path",
                value=_path_or_empty("vae_model_path"),
                placeholder=ph["vae"],
            )
            _cfg_w["vae_browse"] = gr.Button("Browse VAE", scale=0)
            _cfg_w["img_backend"] = gr.Dropdown(
                label="Image backend",
                choices=backends,
                value=img_backend0,
            )
            _cfg_w["img_load_mode"] = gr.Dropdown(
                label="Load mode",
                choices=configure.LOAD_MODE_CHOICES,
                value=configure.normalize_load_mode(
                    str(
                        cfg0.get("imagegen_load_mode")
                        or cfg0.get("model_load_mode")
                        or configure.DEFAULT_LOAD_MODE
                    )
                ),
            )
            # NOTE: sd-cli (stable-diffusion.cpp) has no per-layer GPU-offload
            # flag — unlike llama.cpp's -ngl, Flux/sd.cpp offload is all-or-
            # nothing per component (diffusion/te/vae), selected entirely via
            # --backend/--params-backend, i.e. the Placement dropdown below.
            # This field is kept (hidden) only so any previously saved
            # imagegen_gpu_layers value round-trips harmlessly; it is never
            # sent to sd-cli.
            _cfg_w["img_gpu_layers"] = gr.Number(
                label="GPU layers (−1 = auto)",
                value=int(cfg0.get("imagegen_gpu_layers", configure.DEFAULT_GPU_LAYERS) or -1),
                minimum=-1,
                maximum=max(configure.DIFFUSER_MAX_LAYERS, 99),
                step=1,
                precision=0,
                visible=False,
            )
            _cfg_w["placement"] = gr.Dropdown(
                label="Diffuser Placement",
                choices=configure.PLACEMENT_CHOICES,
                value=configure.normalize_placement(
                    cfg0.get("imagegen_placement", configure.DEFAULT_PLACEMENT)
                ),
                visible=placement_visible,
                info=(
                    "Gpu_Only = all on GPU. Split = diffusion GPU, TE+VAE CPU. "
                    "This is the only GPU offload control for the Diffuser — "
                    "there's no separate GPU-layers count for Flux/sd-cli."
                ),
            )

    _cfg_w["save_btn"] = gr.Button("Save Configuration", variant="primary")


def _save_config(*args):
    (
        worker_threads,
        enc_path, enc_b,
        th_path, th_b,
        text_lm, text_layers,
        diff, vae, img_b, img_lm, img_layers, placement,
    ) = args
    configure.quarantine_mmproj_files()
    try:
        wt = int(worker_threads)
    except (TypeError, ValueError):
        wt = configure.WORKER_THREADS_DEFAULT
    wt = max(configure.WORKER_THREADS_MIN, wt)
    tlm = configure.normalize_load_mode(str(text_lm or ""))
    try:
        tgl = int(text_layers if text_layers is not None else -1)
    except (TypeError, ValueError):
        tgl = -1
    configure.update_configuration({
        "worker_threads": wt,
        "encoder_threads": wt,
        "thinking_threads": wt,
        "imagegen_threads": wt,
        "encoder_model_path": (enc_path or "").strip(),
        "thinking_model_path": (th_path or "").strip(),
        "imagegen_model_path": (diff or "").strip(),
        "vae_model_path": (vae or "").strip(),
        "encoder_backend": enc_b,
        "thinking_backend": th_b,
        "imagegen_backend": img_b,
        "text_load_mode": tlm,
        "encoder_load_mode": tlm,
        "thinking_load_mode": tlm,
        "imagegen_load_mode": configure.normalize_load_mode(str(img_lm or "")),
        "model_load_mode": tlm,
        "text_gpu_layers": tgl,
        "encoder_gpu_layers": tgl,
        "thinking_gpu_layers": tgl,
        "imagegen_gpu_layers": int(img_layers if img_layers is not None else -1),
        "imagegen_placement": configure.normalize_placement(str(placement or "")),
    })
    cores = configure.affinity_core_list(wt)
    # Preview auto layer resolution for the encoder path if set
    auto_note = ""
    enc = (enc_path or "").strip()
    if tgl == -1 and enc:
        free = configure.vram_free_for_backend(str(enc_b or ""))
        resolved = configure.resolve_text_gpu_layers(enc, -1, str(enc_b or ""), free)
        floor = configure.free_vram_floor_mb(free)
        blocks = configure.model_layer_count(enc)
        auto_note = (
            f" Auto layers→{resolved}/{blocks} "
            f"(safe VRAM floor {floor} MiB)."
        )
    return (
        f"Configuration saved. Threads Used={wt} "
        f"(cores {cores[0]}–{cores[-1] if cores else '?'}).{auto_note}"
    )


def _text_gpu_visibility(enc_b: str, th_b: str):
    show = (
        (bool(enc_b) and not str(enc_b).upper().startswith("CPU"))
        or (bool(th_b) and not str(th_b).upper().startswith("CPU"))
    )
    return gr.update(visible=show)


def _placement_visibility(img_backend: str):
    show = bool(img_backend) and not str(img_backend).upper().startswith("CPU")
    return gr.update(visible=show)


def _wire_config_events(status_box: gr.Textbox) -> None:
    _cfg_w["enc_browse"].click(lambda: _browse_file(), outputs=_cfg_w["enc_path"])
    _cfg_w["th_browse"].click(lambda: _browse_file(), outputs=_cfg_w["th_path"])
    _cfg_w["diff_browse"].click(lambda: _browse_file(), outputs=_cfg_w["diff_path"])
    _cfg_w["vae_browse"].click(lambda: _browse_file(), outputs=_cfg_w["vae_path"])

    for key in ("enc_backend", "th_backend"):
        _cfg_w[key].change(
            _text_gpu_visibility,
            inputs=[_cfg_w["enc_backend"], _cfg_w["th_backend"]],
            outputs=[_cfg_w["text_gpu_layers"]],
        )
    _cfg_w["img_backend"].change(
        _placement_visibility,
        inputs=[_cfg_w["img_backend"]],
        outputs=[_cfg_w["placement"]],
    )

    _cfg_w["save_btn"].click(
        _save_config,
        inputs=[
            _cfg_w["worker_threads"],
            _cfg_w["enc_path"], _cfg_w["enc_backend"],
            _cfg_w["th_path"], _cfg_w["th_backend"],
            _cfg_w["text_load_mode"], _cfg_w["text_gpu_layers"],
            _cfg_w["diff_path"], _cfg_w["vae_path"],
            _cfg_w["img_backend"],
            _cfg_w["img_load_mode"], _cfg_w["img_gpu_layers"],
            _cfg_w["placement"],
        ],
        outputs=status_box,
    )


# ---------------------------------------------------------------------------
# Preferences tab
# ---------------------------------------------------------------------------

_pref: Dict[str, Any] = {}


def _build_pref_tab() -> None:
    prefs = _prefs()
    gr.Markdown("### Thumbnails")
    with gr.Row():
        _pref["max_thumbs"] = gr.Dropdown(
            label="How many thumbnails max to display",
            choices=configure.MAX_THUMBNAIL_CHOICES,
            value=prefs.get("max_thumbnails", configure.DEFAULT_MAX_THUMBNAILS),
        )
        _pref["thumb_size"] = gr.Dropdown(
            label="Size of thumbnails",
            choices=configure.INPUT_THUMBNAIL_CHOICES,
            value=prefs.get("input_thumbnail_size", configure.DEFAULT_INPUT_THUMBNAIL),
        )

    gr.Markdown("### Completion sounds")
    with gr.Row():
        _pref["bleep_section"] = gr.Checkbox(
            label="Bleep upon section completion",
            value=bool(prefs.get("bleep_section_completion", False)),
            info="Beep after project name, prompts, images, and video stages.",
        )
        _pref["bleep_video"] = gr.Checkbox(
            label="Bleep upon video completion",
            value=bool(prefs.get("bleep_video_completion", False)),
            info="Beep when the final video is ready. Both on → double bleep.",
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
        L.append("=== Lyrics-Slideshow Debug ===")
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
            vram = f"  ({total} MiB total, {free} MiB free → safe budget {floor} MiB / {floor // 1000} GB)"
            L.append(f"  {d['backend']}{d['index']}: {d['name']}{vram}")
        L.append("  (Safe budget = free MiB rounded down to whole GB; assumed available at runtime.)")
        inst = configure.get_install_info()
        L.append("")
        L.append("--- Install (constants.ini) ---")
        L.append(f"Install type   : {inst.get('install_type')}  ({inst.get('backend_method')})")
        L.append(f"llama.cpp ref  : {inst.get('llama_ref') or '(unknown)'}")
        L.append(f"sd.cpp ref     : {inst.get('sd_ref') or '(unknown)'}")
        L.append("Timing        : user section markers on audio (no Whisper)")
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

AUDIO_TIME_JS = """
() => {
  const D = (...a) => { try { console.error('[ls-time]', ...a); } catch (e) {} };
  D('BOOT — launch/load js entered');
  function gradioRoot() {
    const hosts = document.getElementsByTagName('gradio-app');
    const has = !!(hosts.length && hosts[0].shadowRoot);
    return has ? hosts[0].shadowRoot : document;
  }
  function fmt(t) {
    if (!isFinite(t) || t < 0) t = 0;
    t = Math.floor(Number(t) || 0);
    return Math.floor(t / 60) + ':' + String(t % 60).padStart(2, '0');
  }
  function parseMmSs(text) {
    if (!text) return null;
    const m = String(text).trim().match(/^([0-9]+):([0-9]{2})/);
    if (!m) return null;
    return parseInt(m[1], 10) * 60 + parseInt(m[2], 10);
  }
  function walkAll(node, fn, seen) {
    if (!node || seen.has(node)) return;
    seen.add(node);
    try { fn(node); } catch (e) {}
    if (node.shadowRoot) walkAll(node.shadowRoot, fn, seen);
    const kids = node.children;
    if (kids) for (let i = 0; i < kids.length; i++) walkAll(kids[i], fn, seen);
  }
  function findInTree(selector) {
    let found = null;
    walkAll(gradioRoot(), (n) => {
      if (found || !n.querySelector) return;
      try { const el = n.querySelector(selector); if (el) found = el; } catch (e) {}
    }, new Set());
    return found;
  }
  function inventory() {
    const info = {
      readout: !!findInTree('#audio-time-readout'),
      playbackPos: !!findInTree('#playback-pos'),
      audioPlayer: !!findInTree('#audio-player'),
      timeEl: !!findInTree('time#time'),
      durationEl: !!findInTree('time#duration'),
      timestamps: !!findInTree('.timestamps'),
      media: []
    };
    walkAll(gradioRoot(), (n) => {
      if (n.tagName === 'AUDIO' || n.tagName === 'VIDEO') {
        info.media.push({
          tag: n.tagName,
          src: String(n.currentSrc || n.src || '').slice(0, 60),
          t: n.currentTime,
          d: n.duration,
          paused: n.paused
        });
      }
    }, new Set());
    D('inventory', JSON.stringify(info));
    return info;
  }
  let lastCur = 0, lastDur = 0, tickCount = 0, lastLogTick = 0;
  function seedDur() {
    const legacy = findInTree('#audio-time-readout');
    if (legacy) {
      const d = parseFloat(legacy.getAttribute('data-duration') || '0');
      if (isFinite(d) && d > 0) lastDur = d;
    }
  }
  function setPlaybackPos(sec) {
    const wrap = findInTree('#playback-pos');
    if (!wrap) return;
    const input = wrap.querySelector('textarea, input');
    if (!input) return;
    const v = (Math.max(0, Number(sec) || 0)).toFixed(3);
    if (input.value !== v) {
      input.value = v;
      input.dispatchEvent(new Event('input', { bubbles: true }));
      input.dispatchEvent(new Event('change', { bubbles: true }));
    }
  }
  function paint(cur, dur, source) {
    cur = Math.max(0, Number(cur) || 0);
    if (isFinite(dur) && dur > 0) lastDur = dur; else dur = lastDur;
    lastCur = cur;
    const el = findInTree('#audio-time-readout');
    if (el) {
      el.textContent = fmt(cur) + ' / ' + fmt(dur || 0);
      if (dur > 0) el.setAttribute('data-duration', String(dur));
    }
    setPlaybackPos(cur);
  }
  function readGradioClock() {
    const curEl = findInTree('time#time') || findInTree('.timestamps time#time') || findInTree('.timestamps time');
    const durEl = findInTree('time#duration') || findInTree('.timestamps time#duration');
    if (durEl) {
      const d = parseMmSs(durEl.textContent);
      if (d !== null && d > 0) lastDur = d;
    }
    if (curEl) {
      const c = parseMmSs(curEl.textContent);
      if (c !== null) return { cur: c, text: curEl.textContent };
    }
    return null;
  }
  function readAnyMedia() {
    let best = null, bestScore = -1;
    walkAll(gradioRoot(), (n) => {
      if (n.tagName !== 'AUDIO' && n.tagName !== 'VIDEO') return;
      const src = n.currentSrc || n.src || '';
      if (!src) return;
      let score = 1;
      if (!n.paused) score += 10;
      if ((n.currentTime || 0) > 0) score += 5;
      if (isFinite(n.duration) && n.duration > 0) score += 2;
      if (score > bestScore) { bestScore = score; best = n; }
    }, new Set());
    return best;
  }
  function tick() {
    tickCount += 1;
    seedDur();
    const g = readGradioClock();
    if (g && g.cur !== null) {
      paint(g.cur, lastDur, 'gradio-clock');
      if (tickCount <= 5 || tickCount - lastLogTick >= 25) {
        D('tick', tickCount, 'via=gradio-clock', g.cur, g.text, 'dur', lastDur);
        lastLogTick = tickCount;
      }
      return;
    }
    const med = readAnyMedia();
    if (med) {
      const cur = isFinite(med.currentTime) ? med.currentTime : 0;
      const dur = (isFinite(med.duration) && med.duration > 0) ? med.duration : lastDur;
      paint(cur, dur, 'media');
      if (tickCount <= 5 || tickCount - lastLogTick >= 25) {
        D('tick', tickCount, 'via=media', cur, 'paused', med.paused);
        lastLogTick = tickCount;
      }
      return;
    }
    paint(lastCur, lastDur, 'stale');
    if (tickCount <= 10 || tickCount - lastLogTick >= 50) {
      D('tick', tickCount, 'via=NONE', lastCur, lastDur);
      inventory();
      lastLogTick = tickCount;
    }
  }
  window.__lsGetPlaybackSeconds = function () {
    const g = readGradioClock();
    if (g && g.cur !== null) return g.cur;
    const med = readAnyMedia();
    if (med && isFinite(med.currentTime)) return med.currentTime;
    return lastCur;
  };
  seedDur();
  paint(0, lastDur, 'init');
  inventory();
  setInterval(tick, 200);
  tick();
  D('interval armed, lastDur=', lastDur);
  try {
    new MutationObserver(function () { tick(); }).observe(gradioRoot(), {
      childList: true, subtree: true, characterData: true
    });
    D('MutationObserver attached');
  } catch (e) { D('MO failed', String(e)); }
}
"""

HEAD_TIME_SCRIPT = """<script>
(function(){
  try { console.error('[ls-time] HEAD SCRIPT RUNNING'); } catch (e) {}
  function boot() {
    if (window.__lsTimeBooted) return;
    window.__lsTimeBooted = true;
    try { console.error('[ls-time] HEAD boot()'); } catch (e) {}
    var D = function() {
      try { console.error.apply(console, ['[ls-time]'].concat([].slice.call(arguments))); } catch (e) {}
    };
    function gradioRoot() {
      var hosts = document.getElementsByTagName('gradio-app');
      return (hosts.length && hosts[0].shadowRoot) ? hosts[0].shadowRoot : document;
    }
    function fmt(t) {
      if (!isFinite(t) || t < 0) t = 0;
      t = Math.floor(Number(t) || 0);
      return Math.floor(t / 60) + ':' + String(t % 60).padStart(2, '0');
    }
    function parseMmSs(text) {
      if (!text) return null;
      var m = String(text).trim().match(/^([0-9]+):([0-9]{2})/);
      if (!m) return null;
      return parseInt(m[1], 10) * 60 + parseInt(m[2], 10);
    }
    function walkAll(node, fn, seen) {
      if (!node || seen.has(node)) return;
      seen.add(node);
      try { fn(node); } catch (e) {}
      if (node.shadowRoot) walkAll(node.shadowRoot, fn, seen);
      var kids = node.children;
      if (kids) for (var i = 0; i < kids.length; i++) walkAll(kids[i], fn, seen);
    }
    function findInTree(selector) {
      var found = null;
      walkAll(gradioRoot(), function (n) {
        if (found || !n.querySelector) return;
        try { var el = n.querySelector(selector); if (el) found = el; } catch (e) {}
      }, new Set());
      return found;
    }
    var lastCur = 0, lastDur = 0, tickCount = 0;
    function seedDur() {
      var legacy = findInTree('#audio-time-readout');
      if (legacy) {
        var d = parseFloat(legacy.getAttribute('data-duration') || '0');
        if (isFinite(d) && d > 0) lastDur = d;
      }
    }
    function setPlaybackPos(sec) {
      var wrap = findInTree('#playback-pos');
      if (!wrap) return;
      var input = wrap.querySelector('textarea, input');
      if (!input) return;
      var v = (Math.max(0, Number(sec) || 0)).toFixed(3);
      if (input.value !== v) {
        input.value = v;
        input.dispatchEvent(new Event('input', { bubbles: true }));
        input.dispatchEvent(new Event('change', { bubbles: true }));
      }
    }
    function paint(cur, dur) {
      cur = Math.max(0, Number(cur) || 0);
      if (isFinite(dur) && dur > 0) lastDur = dur; else dur = lastDur;
      lastCur = cur;
      var el = findInTree('#audio-time-readout');
      if (el) {
        el.textContent = fmt(cur) + ' / ' + fmt(dur || 0);
        if (dur > 0) el.setAttribute('data-duration', String(dur));
      }
      setPlaybackPos(cur);
    }
    function readGradioClock() {
      var curEl = findInTree('time#time') || findInTree('.timestamps time');
      var durEl = findInTree('time#duration');
      if (durEl) {
        var d = parseMmSs(durEl.textContent);
        if (d !== null && d > 0) lastDur = d;
      }
      if (curEl) {
        var c = parseMmSs(curEl.textContent);
        if (c !== null) return c;
      }
      return null;
    }
    function readAnyMedia() {
      var best = null, bestScore = -1;
      walkAll(gradioRoot(), function (n) {
        if (n.tagName !== 'AUDIO' && n.tagName !== 'VIDEO') return;
        var src = n.currentSrc || n.src || '';
        if (!src) return;
        var score = 1;
        if (!n.paused) score += 10;
        if ((n.currentTime || 0) > 0) score += 5;
        if (isFinite(n.duration) && n.duration > 0) score += 2;
        if (score > bestScore) { bestScore = score; best = n; }
      }, new Set());
      return best;
    }
    function tick() {
      tickCount += 1;
      seedDur();
      var g = readGradioClock();
      if (g !== null) { paint(g, lastDur); return; }
      var med = readAnyMedia();
      if (med) {
        paint(isFinite(med.currentTime) ? med.currentTime : 0,
              (isFinite(med.duration) && med.duration > 0) ? med.duration : lastDur);
        return;
      }
      paint(lastCur, lastDur);
      if (tickCount <= 8 || tickCount % 40 === 0) {
        D('tick', tickCount, 'NONE', lastCur, lastDur, 'timeEl', !!findInTree('time#time'));
      }
    }
    window.__lsGetPlaybackSeconds = function () {
      var g = readGradioClock();
      if (g !== null) return g;
      var med = readAnyMedia();
      if (med && isFinite(med.currentTime)) return med.currentTime;
      return lastCur;
    };
    seedDur();
    paint(0, lastDur);
    setInterval(tick, 200);
    tick();
    D('HEAD interval armed', lastDur);
  }
  boot();
  setTimeout(boot, 500);
  setTimeout(boot, 1500);
  setTimeout(boot, 3000);
})();
</script>"""


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
#slides-gallery-link {
  all: unset !important;
  display: inline-block !important;
  font-size: 1.1rem !important;
  font-weight: 600 !important;
  color: var(--body-text-color) !important;
  cursor: pointer !important;
  margin: 0.4rem 0 0.2rem 0 !important;
}
#slides-gallery-link:hover { text-decoration: underline !important; }
#slides-heading { display: none !important; }

#lyrics-row { align-items: flex-start !important; }
#rebuild-markers-btn {
  margin-top: 1.6rem !important;
  min-height: 2.6rem !important;
  white-space: normal !important;
}

/* Track Preview: sources=[] + buttons=[] on the gr.Audio already remove
   upload, microphone, download and share from the component. This is just
   a belt-and-braces backup in case a future Gradio version reintroduces
   any of them, and hides drag-and-drop affordance text. */
#audio-player button[aria-label*="Download" i],
#audio-player button[aria-label*="Share" i],
#audio-player button[aria-label*="Upload" i],
#audio-player button[aria-label*="Record" i],
#audio-player button[aria-label*="Microphone" i],
#audio-player a[download],
#audio-player [data-testid="download-btn"],
#audio-player [data-testid="share-btn"] {
  display: none !important;
}

/* Track Preview and its Browse button sit side by side */
#track-preview-row {
  align-items: flex-start !important;
  gap: 0.5rem !important;
}
#track-preview-col {
  position: relative !important;
}
#track-preview-browse-col {
  margin-top: 1.9rem !important;  /* clears the "Track Preview" label row */
}
#track-preview-browse-btn {
  min-height: 2.4rem !important;
}

/* Time readout floats inline with the "Track Preview" label, top-right of
   the box, instead of stacking as its own line underneath. */
#track-transport {
  position: absolute !important;
  top: 0.1rem !important;
  right: 0.6rem !important;
  z-index: 5 !important;
  pointer-events: none !important;
}
#track-preview-controls {
  display: flex !important;
  justify-content: flex-end !important;
  align-items: center !important;
  margin: 0 !important;
  padding: 0 !important;
}
#audio-time-readout {
  font-variant-numeric: tabular-nums;
  font-weight: 700;
  font-size: 0.85rem;
  color: var(--body-text-color);
  opacity: 0.95;
  background: var(--background-fill-primary);
  padding: 0.05rem 0.5rem;
  border-radius: 0.5rem;
  box-shadow: 0 0 0 1px var(--border-color-primary);
}

.ls-set-position-btn,
button.ls-set-position-btn,
[class*="ls-set-position"] {
  min-height: 3.4rem !important;
  height: 3.4rem !important;
  white-space: normal !important;
  line-height: 1.2 !important;
  font-weight: 600 !important;
}
[id^="marker-row-"] {
  align-items: stretch !important;
  margin-bottom: 0.2rem !important;
}
"""
    with gr.Blocks(title="Lyrics-Slideshow") as app:
        gr.Markdown("# Lyrics-Slideshow — Lyrics → Music Video Slideshow")
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

        try:
            app.load(fn=None, inputs=None, outputs=None, js=AUDIO_TIME_JS)
            print("[ls-time] registered app.load(js=AUDIO_TIME_JS) inside Blocks")
        except Exception as _e:
            print("[ls-time] app.load(js=) failed:", _e)


    return app, css, AUDIO_TIME_JS, HEAD_TIME_SCRIPT