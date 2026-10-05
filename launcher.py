#!/usr/bin/env python3
"""
launcher.py - Startup, shutdown, and main loop for Lyrics-Materials.
Hosts the Gradio UI inside a PyQt6 window.
"""
from __future__ import annotations

import platform
import socket
import sys
import time
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

try:
    import gradio as gr
except ImportError:
    print("ERROR: Gradio is not installed. Run Installation from the batch menu first.")
    sys.exit(1)

try:
    from PyQt6.QtCore import QUrl, QObject, pyqtSignal
    from PyQt6.QtGui import QIcon
    from PyQt6.QtWidgets import QApplication, QMainWindow
    from PyQt6.QtWebEngineWidgets import QWebEngineView
    from PyQt6.QtWebEngineCore import QWebEnginePage
except ImportError:
    print("ERROR: PyQt6 / PyQt6-WebEngine not installed. Run Installation first.")
    sys.exit(1)

import scripts.configure as configure
import scripts.utilities as utilities
import scripts.display as display

APP_TITLE = "Lyrics-Materials"
SERVER_NAME = "127.0.0.1"
SERVER_PORT = 7867


def _program_icon_path() -> Path | None:
    """Resolve images/program_icon.ico next to the app root (batch / launcher)."""
    candidates = [
        _PROJECT_ROOT / "images" / "program_icon.ico",
        _PROJECT_ROOT / "Images" / "program_icon.ico",
        _PROJECT_ROOT / "program_icon.ico",
        Path.cwd() / "images" / "program_icon.ico",
    ]
    for c in candidates:
        try:
            if c.is_file():
                return c.resolve()
        except OSError:
            continue
    return None


def _print_banner() -> None:
    cpu = configure.get_cpu_info()
    vk = configure.get_vulkan_info()
    bs = utilities.get_build_status()
    print(f"  Versioning: Python {platform.python_version()}; Gradio {gr.__version__}")
    print(f"\n  CPU     : {cpu.get('brand')}")
    print(f"  Threads : {configure.HEAVY_THREADS} (fixed for heavy work)")
    print(f"  Vulkan  : {vk.get('available')}")
    for d in vk.get("devices", []):
        print(f"    GPU{d['index']}: {d['name']}")
    if not bs["llama_built"] or not bs["sd_built"]:
        print("\n  NOTE: Backends not yet built. Run Installation.")
    else:
        print(f"\n  llama   : {bs['llama_path']}")
        print(f"  sd-cli  : {bs['sd_path']}")
    print(f"  ffmpeg  : {utilities.find_ffmpeg() or 'NOT FOUND'}")
    print()


def _wait_for_server(host: str, port: int, timeout: float = 25.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with socket.create_connection((host, port), timeout=0.5):
                return True
        except OSError:
            time.sleep(0.1)
    return False


class _ExitBridge(QObject):
    close_requested = pyqtSignal()


class _QuietPage(QWebEnginePage):
    def javaScriptConsoleMessage(self, level, message, line, source):
        if "Method not implemented" in str(message):
            return
        super().javaScriptConsoleMessage(level, message, line, source)


class AppWindow(QMainWindow):
    def __init__(self, url: str, geometry: dict, on_close) -> None:
        super().__init__()
        self._on_close = on_close
        self._closed_once = False
        self.setWindowTitle(APP_TITLE)
        icon_path = _program_icon_path()
        if icon_path is not None:
            icon = QIcon(str(icon_path))
            self.setWindowIcon(icon)
            # Taskbar / window chrome (Windows uses this with AppUserModelID)
            QApplication.instance().setWindowIcon(icon)
            print(f"  Window icon : {icon_path}")
        else:
            print("  Window icon : images/program_icon.ico not found (using default)")
        self._apply_saved_geometry(geometry)
        self.view = QWebEngineView(self)
        self.view.setPage(_QuietPage(self.view))
        self.setCentralWidget(self.view)
        self.view.load(QUrl(url))
        self.exit_bridge = _ExitBridge()
        self.exit_bridge.close_requested.connect(self.close)

    def _apply_saved_geometry(self, geometry: dict) -> None:
        self.resize(geometry["width"], geometry["height"])
        if geometry["x"] != configure.WINDOW_GEOMETRY_UNSET and \
           geometry["y"] != configure.WINDOW_GEOMETRY_UNSET:
            self.move(geometry["x"], geometry["y"])
        if geometry.get("maximized"):
            self.showMaximized()

    def current_geometry(self) -> dict:
        maximized = self.isMaximized()
        rect = self.normalGeometry() if maximized else self.geometry()
        return {
            "x": rect.x(), "y": rect.y(),
            "width": rect.width(), "height": rect.height(),
            "maximized": maximized,
        }

    def closeEvent(self, event) -> None:
        if self._closed_once:
            event.accept()
            return
        self._closed_once = True
        event.accept()
        self._on_close(self.current_geometry())


def _shutdown(window_geometry: dict, gradio_app) -> None:
    print("\n  Shutting down...")
    try:
        configure.save_window_geometry(
            x=window_geometry["x"], y=window_geometry["y"],
            width=window_geometry["width"], height=window_geometry["height"],
            maximized=window_geometry["maximized"],
        )
        print("  Window geometry saved.")
    except Exception as e:
        print(f"  WARNING: {e}")
    try:
        gradio_app.close(verbose=False)
        print("  Gradio server closed.")
    except Exception as e:
        print(f"  WARNING: {e}")
    print("  Goodbye.")
    import os
    os._exit(0)


def _set_windows_app_id() -> None:
    if platform.system() != "Windows":
        return
    try:
        import ctypes
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(
            f"WiseManTimeLord.{APP_TITLE}"
        )
    except Exception:
        pass


def main() -> None:
    _set_windows_app_id()
    configure.ensure_data_dirs()
    configure.init_session_state()
    _print_banner()

    blocks_app, _css, _js, _head_script = display.build_app()

    import warnings
    import os
    warnings.filterwarnings("ignore", message=".*HTTP_422_UNPROCESSABLE_ENTITY.*")
    # Quiet Gradio analytics; avoid share-link pressure when localhost probe is flaky
    os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")

    # Gradio only serves files under cwd / temp / allowed_paths.
    _allowed = [
        str(configure.get_data_dir()),
        str(configure.get_output_dir()),
        str(configure.get_models_dir()),
        str(configure.get_images_dir()),
        str(configure.get_ref_cache_dir()),
    ]
    launch_kwargs = dict(
        server_name=SERVER_NAME,
        server_port=SERVER_PORT,
        share=False,
        inbrowser=False,
        prevent_thread_lock=True,
        show_error=True,
        theme=gr.themes.Soft(),
        css=_css,
        allowed_paths=_allowed,
    )
    if _js:
        launch_kwargs["js"] = _js
    if _head_script:
        launch_kwargs["head"] = _head_script

    try:
        _server_app, local_url, _share_url = blocks_app.launch(**launch_kwargs)
    except ValueError as e:
        # Gradio 6 sometimes raises a misleading "localhost not accessible" error
        # when an internal startup fault occurs — retry once with an explicit URL root.
        print(f"  WARNING: Gradio launch issue ({e})")
        print("  Retrying launch…")
        launch_kwargs["server_name"] = "127.0.0.1"
        try:
            _server_app, local_url, _share_url = blocks_app.launch(**launch_kwargs)
        except Exception as e2:
            print(f"ERROR: Gradio failed to start: {e2}")
            raise

    if not _wait_for_server(SERVER_NAME, SERVER_PORT):
        print("ERROR: Gradio server did not come up in time.")
        sys.exit(1)

    qt_app = QApplication(sys.argv)
    qt_app.setApplicationName(APP_TITLE)

    saved_geometry = configure.get_window_geometry()
    window = AppWindow(
        local_url,
        saved_geometry,
        on_close=lambda geom: _shutdown(geom, blocks_app),
    )

    def _request_exit_from_gradio_thread() -> None:
        window.exit_bridge.close_requested.emit()

    display.set_exit_handler(_request_exit_from_gradio_thread)
    window.show()
    sys.exit(qt_app.exec())


if __name__ == "__main__":
    main()
