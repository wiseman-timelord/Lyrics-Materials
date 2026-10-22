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
# Prefer 7867; if busy (e.g. Image-Glamour / another Gradio), scan upward.
SERVER_PORT_CANDIDATES = list(range(7867, 7888)) + list(range(7860, 7867))


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


def _port_is_free(host: str, port: int) -> bool:
    """True when nothing is listening / bound on host:port."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind((host, port))
            return True
    except OSError:
        return False


def _pick_free_port(host: str = SERVER_NAME, preferred: int = SERVER_PORT) -> int:
    """
    Choose a free local port. Prefer `preferred`, then the candidate list.
    Falls back to OS-assigned ephemeral if everything in the list is taken.
    """
    ordered = [preferred] + [p for p in SERVER_PORT_CANDIDATES if p != preferred]
    for port in ordered:
        if _port_is_free(host, port):
            return port
    # Last resort: let the OS pick
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((host, 0))
        return int(s.getsockname()[1])


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
    # Prefer 7867; only hop ports when the socket is actually taken.
    chosen_port = _pick_free_port(SERVER_NAME, SERVER_PORT)
    if chosen_port != SERVER_PORT:
        print(f"  Port {SERVER_PORT} in use — binding to {chosen_port}.")

    launch_kwargs = dict(
        server_name="127.0.0.1",
        server_port=chosen_port,
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

    # Gradio 6 starts the HTTP server, then probes "is localhost reachable?".
    # That probe often false-fails (DNS localhost, timing, sibling apps) and
    # raises ValueError EVEN THOUGH the server is already listening. The older
    # launcher recovered by retrying once on the same port. Port-hopping made
    # things worse: it abandoned a live server and failed every candidate.
    #
    # Strategy:
    #   1) launch on chosen_port
    #   2) on the known ValueError — if the port answers, treat as success
    #   3) otherwise close + one same-port retry (matches pre-Techno behaviour)
    #   4) only then try the next free port (real bind conflicts)
    local_url = None
    last_err: Exception | None = None
    ports_to_try = [chosen_port] + [
        p for p in SERVER_PORT_CANDIDATES
        if p != chosen_port and _port_is_free(SERVER_NAME, p)
    ][:6]

    for attempt, port in enumerate(ports_to_try, 1):
        launch_kwargs["server_port"] = port
        launch_kwargs["server_name"] = "127.0.0.1"
        os.environ["GRADIO_SERVER_NAME"] = "127.0.0.1"
        os.environ["GRADIO_SERVER_PORT"] = str(port)

        def _try_launch() -> tuple:
            return blocks_app.launch(**launch_kwargs)

        def _server_alive() -> bool:
            return _wait_for_server("127.0.0.1", port, timeout=4.0)

        def _accept_live(reason: str) -> str:
            url = f"http://127.0.0.1:{port}/"
            print(f"  {reason}")
            print(f"  Using live server at {url}")
            return url

        try:
            if attempt > 1:
                print(f"  Trying port {port}…")
                try:
                    blocks_app.close(verbose=False)
                except Exception:
                    pass
                time.sleep(0.4)
            _server_app, local_url, _share_url = _try_launch()
            chosen_port = port
            last_err = None
            break
        except ValueError as e:
            last_err = e
            msg = str(e).lower()
            is_localhost_probe = (
                "localhost" in msg or "shareable link" in msg or "share=true" in msg
            )
            print(f"  WARNING: Gradio launch issue on :{port} ({e})")
            if is_localhost_probe and _server_alive():
                # Server is up; Gradio's post-start probe lied.
                local_url = _accept_live(
                    "Gradio localhost-probe failed, but the server is responding."
                )
                chosen_port = port
                last_err = None
                break
            if is_localhost_probe:
                # Same-port retry — this is what the working older launcher did.
                print("  Retrying launch on the same port…")
                try:
                    blocks_app.close(verbose=False)
                except Exception:
                    pass
                time.sleep(0.5)
                try:
                    _server_app, local_url, _share_url = _try_launch()
                    chosen_port = port
                    last_err = None
                    break
                except ValueError as e2:
                    last_err = e2
                    print(f"  WARNING: retry still raised ({e2})")
                    if _server_alive():
                        local_url = _accept_live(
                            "Retry raised, but the server is responding."
                        )
                        chosen_port = port
                        last_err = None
                        break
                    # Fall through to next port only if still dead
                except Exception as e2:
                    last_err = e2
                    print(f"  WARNING: retry failed ({e2})")
            # Non-probe ValueError: stop hopping
            if not is_localhost_probe:
                break
        except OSError as e:
            last_err = e
            print(f"  WARNING: bind failed on :{port} ({e})")
            # Real port conflict — try next free port
            continue
        except Exception as e:
            last_err = e
            print(f"  WARNING: Gradio failed on :{port} ({e})")
            if _server_alive():
                local_url = _accept_live("Exception after bind, server still up.")
                chosen_port = port
                last_err = None
                break

    if not local_url:
        print(f"ERROR: Gradio failed to start (last error: {last_err})")
        raise SystemExit(1) from last_err

    print(f"  Gradio listening on {local_url}")

    # Confirm the socket still answers before opening the Qt window
    if not _wait_for_server("127.0.0.1", chosen_port, timeout=10.0):
        print(f"ERROR: Gradio server did not come up on port {chosen_port} in time.")
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
