from __future__ import annotations

import argparse
import io
import json
import math
import os
import platform
import queue
import re
import shutil
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import webbrowser
from collections import deque
from datetime import datetime
from pathlib import Path
from tkinter import BOTH, END, LEFT, RIGHT, X, Button, Canvas, Checkbutton, Entry, Frame, Label, Listbox, PhotoImage, StringVar, Text, Tk, Toplevel, filedialog, messagebox, ttk

import psutil

from app_paths import ROOT
from game_profiles import PROFILES
from geometry_json import RECTANGLE, TRIANGLE, ROTATED_ELLIPSE, load_normalized_geometry
from generator_backend import GENERATOR_EXE, GENERATOR_JSON_SCAN_SECONDS, GENERATOR_POLL_SLEEP_SECONDS, GENERATOR_PREVIEW_SCAN_SECONDS, USER_SETTINGS_DIR, best_geometry_jsons, build_generator_command, build_generator_env, generated_jsons, generated_preview_files, generator_preview_path, load_settings, preprocess_input_image, write_custom_settings, write_user_settings_preset
from version import APP_DISPLAY_NAME, __version__, app_title


from app_config import (
    APP_DIR,
    PROBE_DIR,
    SESSION_PATH,
    MEMORY_SNAPSHOT_LIMIT_MB,
    PREVIEW_MAX,
    DETAILED_LOG_OUTPUT_LIMIT,
    DETAILED_LOG_MEMORY_LIMIT,
    FH6_AUTO_LOCATE_MAX_SECONDS,
    FH6_AUTO_LOCATE_TIMEOUT_SECONDS,
    UPDATE_VERSION_URL,
    UPDATE_CHANGELOG_URL,
    UPDATE_RELEASE_URL,
    UPDATE_CHECK_TIMEOUT_SECONDS,
    Theme,
)

from utils import load_cv2, load_pillow

from i18n import tr

LANGUAGES = {
    "English": "en",
    "Português (Brasil)": "pt-br",
    "中文": "zh",
    "中文 (繁體)": "zh-tw",
    "한국어": "ko",
}

ETA_MAX_PROGRESS_SAMPLES = 400
ETA_WINDOW_SECONDS = 45.0
ETA_MAX_HISTORY_SECONDS = 180.0
ETA_MIN_WINDOW_SECONDS = 4.0
ETA_MIN_WINDOW_LAYERS = 25
PREVIEW_JSON_SUPERSAMPLE = 2


# Removed: TEXT dictionary (now in i18n.py)


def ensure_dirs():
    PROBE_DIR.mkdir(parents=True, exist_ok=True)




def version_key(value):
    parts = []
    for part in re.findall(r"\d+", str(value)):
        parts.append(int(part))
    while len(parts) < 3:
        parts.append(0)
    return tuple(parts[:3])


def parse_version_source(source):
    match = re.search(r'__version__\s*=\s*"([^"]+)"', source or "")
    if not match:
        raise ValueError("remote version file did not contain __version__")
    return match.group(1).strip()


def fetch_text_url(url, timeout=UPDATE_CHECK_TIMEOUT_SECONDS):
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": f"{APP_DISPLAY_NAME}/{__version__}",
            "Accept": "text/plain,*/*",
        },
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        data = response.read(256 * 1024)
    return data.decode("utf-8", errors="replace")


def extract_changelog_section(changelog, version):
    text = (changelog or "").strip()
    if not text:
        return ""
    heading = re.compile(rf"^(#{{1,6}})\s+v?{re.escape(str(version))}\b.*$", re.I | re.M)
    match = heading.search(text)
    if not match:
        return text[:6000]
    next_heading = re.compile(rf"^{re.escape(match.group(1))}\s+\S+", re.M)
    next_match = next_heading.search(text, match.end())
    end = next_match.start() if next_match else len(text)
    return text[match.start():end].strip()[:6000]


def helper_command(helper_name):
    if getattr(sys, "frozen", False):
        return [sys.executable, "--helper", helper_name]
    return [sys.executable, APP_DIR / f"{helper_name}.py"]


def run_embedded_helper(helper_name, args):
    if helper_name == "fh6_probe":
        import fh6_probe

        previous_argv = sys.argv
        try:
            sys.argv = ["fh6_probe.py", *args]
            return fh6_probe.main()
        finally:
            sys.argv = previous_argv
    if helper_name == "main":
        import main as importer_main

        return importer_main.main(["main.py", *args])
    raise SystemExit(f"Unknown helper: {helper_name}")


def game_processes():
    names = {name.lower(): key for key, profile in PROFILES.items() for name in profile.process_names}
    found = []
    for proc in psutil.process_iter(["pid", "name", "exe"]):
        try:
            name = proc.info.get("name") or ""
            key = names.get(name.lower())
            if key:
                found.append({
                    "pid": proc.info["pid"],
                    "name": name,
                    "profile": key,
                    "label": f"{name} pid {proc.info['pid']}",
                })
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return found


def parse_hex_or_empty(value):
    value = str(value or "").strip()
    return value or None


def load_session_location():
    if not SESSION_PATH.exists():
        return None
    try:
        return json.loads(SESSION_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def clear_session_location():
    try:
        SESSION_PATH.unlink(missing_ok=True)
    except OSError:
        pass


def session_pid_is_live(session, game):
    try:
        pid = int(session.get("pid", -1))
        proc = psutil.Process(pid)
        profile = PROFILES.get(game)
        return bool(profile and proc.name().lower() in [name.lower() for name in profile.process_names])
    except (psutil.Error, TypeError, ValueError):
        return False


def session_matches_current_import(session, game, pid, layer_count):
    if not session:
        return False
    if str(session.get("layer_count", "")) != str(layer_count):
        return False
    if not session_pid_is_live(session, game):
        return False
    try:
        session_pid = int(session.get("pid", -1))
        return not pid or int(pid) == session_pid
    except (TypeError, ValueError):
        return False


# load_cv2 and load_pillow are now imported from utils


def preview_size_tuple(max_size=None):
    if max_size is None:
        return PREVIEW_MAX, PREVIEW_MAX
    if isinstance(max_size, (tuple, list)):
        if len(max_size) >= 2:
            width, height = max_size[0], max_size[1]
        elif len(max_size) == 1:
            width = height = max_size[0]
        else:
            width = height = PREVIEW_MAX
    else:
        width = height = max_size
    try:
        width = int(width)
        height = int(height)
    except (TypeError, ValueError):
        width = height = PREVIEW_MAX
    return max(1, width), max(1, height)


def preview_scale(width, height, max_size=None):
    max_w, max_h = preview_size_tuple(max_size)
    if width <= 0 or height <= 0:
        return 1.0
    return min(max_w / width, max_h / height, 1.0)


def resize_keep_aspect(image, max_size=None):
    loaded = load_cv2()
    if not loaded:
        return image
    cv2, _np = loaded
    height, width = image.shape[:2]
    scale = preview_scale(width, height, max_size)
    if scale < 1.0:
        resized_w = max(1, int(round(width * scale)))
        resized_h = max(1, int(round(height * scale)))
        image = cv2.resize(image, (resized_w, resized_h), interpolation=cv2.INTER_AREA)
    return image


def image_to_photo(image, max_size=None):
    loaded = load_cv2()
    if not loaded:
        return None
    cv2, _np = loaded
    image = resize_keep_aspect(image, max_size)
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        return None
    return encoded.tobytes()


def pil_to_photo(image, max_size=None):
    loaded = load_pillow()
    if not loaded:
        return None
    Image, _ImageDraw = loaded
    image = image.convert("RGB")
    image.thumbnail(preview_size_tuple(max_size), Image.Resampling.LANCZOS)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def render_source_image(path, max_size=None):
    loaded = load_cv2()
    if loaded:
        cv2, _np = loaded
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is not None:
            return image_to_photo(image, max_size)
    loaded = load_pillow()
    if not loaded:
        return None
    Image, _ImageDraw = loaded
    try:
        with Image.open(path) as image:
            return pil_to_photo(image, max_size)
    except Exception:
        return None


def _rotated_rect_corners(cx, cy, w, h, rot_deg, scale):
    """Return 4 (x, y) corner points for a rotated rectangle, scaled.

    Rotation convention matches geometry_optimize (R(theta)=[[cos,-sin],
    [sin,cos]]) so previews and the offline optimizer agree. (In-game angle
    behavior for square vinyls still needs a one-time fixture check, per S1.)
    """
    theta = math.radians(float(rot_deg))
    ct, st = math.cos(theta), math.sin(theta)
    hw, hh = float(w) / 2.0, float(h) / 2.0
    pts = []
    for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
        dx, dy = sx * hw, sy * hh
        px = (float(cx) + dx * ct - dy * st) * scale
        py = (float(cy) + dx * st + dy * ct) * scale
        pts.append((px, py))
    return pts


def _triangle_corners(cx, cy, w, h, rot_deg, scale):
    """Return 3 (x, y) corner points for an isoceles triangle stencil, scaled.

    Apex up at local (0, -h/2); base corners at (-+w/2, +h/2). Same rotation
    convention as _rotated_rect_corners / the optimizer triangle mask.
    """
    theta = math.radians(float(rot_deg))
    ct, st = math.cos(theta), math.sin(theta)
    hw, hh = float(w) / 2.0, float(h) / 2.0
    pts = []
    for dx, dy in ((0.0, -hh), (-hw, hh), (hw, hh)):
        px = (float(cx) + dx * ct - dy * st) * scale
        py = (float(cy) + dx * st + dy * ct) * scale
        pts.append((px, py))
    return pts


def render_geometry_json(path, max_size=None):
    pillow_preview = render_geometry_json_pillow(path, max_size)
    if pillow_preview:
        return pillow_preview
    loaded = load_cv2()
    if not loaded:
        return None
    cv2, np = loaded
    try:
        data = load_normalized_geometry(path)
        shapes = data["shapes"]
        image_w, image_h = [int(v) for v in shapes[0]["data"][2:]]
        bg_r, bg_g, bg_b, bg_a = [int(v) for v in shapes[0]["color"]]
        scale = preview_scale(image_w, image_h, max_size)
        preview_w = max(1, int(round(image_w * scale)))
        preview_h = max(1, int(round(image_h * scale)))
        preview = np.zeros((preview_h, preview_w, 3), np.uint8)
        if bg_a > 0:
            preview[:, :] = (bg_b, bg_g, bg_r)
        else:
            preview[:, :] = (38, 38, 38)
            tile = max(8, int(round(32 * scale)))
            for y in range(0, preview_h, tile):
                for x in range(0, preview_w, tile):
                    if ((x // tile) + (y // tile)) % 2 == 0:
                        preview[y:y + tile, x:x + tile] = (58, 58, 58)
        for shape in shapes[1:]:
            color = [int(v) for v in shape.get("color", [])]
            if len(color) == 4 and color[3] <= 0:
                continue
            r, g, b, _a = color
            shape_type = int(shape.get("type", 0))
            if shape_type == ROTATED_ELLIPSE:
                x, y, w, h, rot_deg = shape["data"]
                center = (int(round(float(x) * scale)), int(round(float(y) * scale)))
                axes = (max(1, int(round(float(h) * scale))), max(1, int(round(float(w) * scale))))
                preview = cv2.ellipse(preview, center, axes, -90 + float(rot_deg), 0.0, 360.0, (b, g, r), thickness=-1)
            elif shape_type == RECTANGLE:
                data = shape["data"]
                x, y, w, h = (float(v) for v in data[:4])
                rot_deg = float(data[4]) if len(data) >= 5 else 0.0
                if rot_deg:
                    pts = _rotated_rect_corners(x, y, w, h, rot_deg, scale)
                    poly = np.array([[int(round(px)), int(round(py))] for px, py in pts], np.int32)
                    preview = cv2.fillConvexPoly(preview, poly, (b, g, r))
                else:
                    x0 = int(round((x - w / 2) * scale))
                    y0 = int(round((y - h / 2) * scale))
                    x1 = int(round((x + w / 2) * scale))
                    y1 = int(round((y + h / 2) * scale))
                    preview = cv2.rectangle(preview, (x0, y0), (x1, y1), (b, g, r), thickness=-1)
            elif shape_type == TRIANGLE:
                data = shape["data"]
                x, y, w, h = (float(v) for v in data[:4])
                rot_deg = float(data[4]) if len(data) >= 5 else 0.0
                pts = _triangle_corners(x, y, w, h, rot_deg, scale)
                poly = np.array([[int(round(px)), int(round(py))] for px, py in pts], np.int32)
                preview = cv2.fillConvexPoly(preview, poly, (b, g, r))
        return image_to_photo(preview, max_size)
    except Exception:
        return None


def render_geometry_json_pillow(path, max_size=None):
    loaded = load_pillow()
    if not loaded:
        return None
    Image, ImageDraw = loaded
    try:
        data = load_normalized_geometry(path)
        shapes = data["shapes"]
        image_w, image_h = [int(v) for v in shapes[0]["data"][2:]]
        bg_r, bg_g, bg_b, bg_a = [int(v) for v in shapes[0]["color"]]
        scale = preview_scale(image_w, image_h, max_size)
        preview_w = max(1, int(round(image_w * scale)))
        preview_h = max(1, int(round(image_h * scale)))
        render_scale = scale * PREVIEW_JSON_SUPERSAMPLE
        render_w = max(1, preview_w * PREVIEW_JSON_SUPERSAMPLE)
        render_h = max(1, preview_h * PREVIEW_JSON_SUPERSAMPLE)
        if bg_a > 0:
            preview = Image.new("RGB", (render_w, render_h), (bg_r, bg_g, bg_b))
        else:
            preview = Image.new("RGB", (render_w, render_h), (38, 38, 38))
            draw_bg = ImageDraw.Draw(preview)
            tile = max(8, int(round(32 * render_scale)))
            for y in range(0, render_h, tile):
                for x in range(0, render_w, tile):
                    if ((x // tile) + (y // tile)) % 2 == 0:
                        draw_bg.rectangle((x, y, min(render_w, x + tile), min(render_h, y + tile)), fill=(58, 58, 58))
        draw = ImageDraw.Draw(preview)
        for shape in shapes[1:]:
            color = [int(v) for v in shape.get("color", [])]
            if len(color) == 4 and color[3] <= 0:
                continue
            r, g, b, _a = color
            shape_type = int(shape.get("type", 0))
            if shape_type == RECTANGLE:
                data = shape["data"]
                x, y, w, h = [float(v) for v in data[:4]]
                rot_deg = float(data[4]) if len(data) >= 5 else 0.0
                if rot_deg:
                    pts = _rotated_rect_corners(x, y, w, h, rot_deg, render_scale)
                    draw.polygon([(int(round(px)), int(round(py))) for px, py in pts], fill=(r, g, b))
                else:
                    x0 = int(round((x - w / 2) * render_scale))
                    y0 = int(round((y - h / 2) * render_scale))
                    x1 = int(round((x + w / 2) * render_scale))
                    y1 = int(round((y + h / 2) * render_scale))
                    draw.rectangle((x0, y0, x1, y1), fill=(r, g, b))
            elif shape_type == ROTATED_ELLIPSE:
                x, y, w, h, rot_deg = [float(v) for v in shape["data"]]
                draw_preview_ellipse_pillow(preview, x, y, w, h, rot_deg, (r, g, b), render_scale)
            elif shape_type == TRIANGLE:
                data = shape["data"]
                x, y, w, h = [float(v) for v in data[:4]]
                rot_deg = float(data[4]) if len(data) >= 5 else 0.0
                pts = _triangle_corners(x, y, w, h, rot_deg, render_scale)
                draw.polygon([(int(round(px)), int(round(py))) for px, py in pts], fill=(r, g, b))
        if PREVIEW_JSON_SUPERSAMPLE > 1:
            preview = preview.resize((preview_w, preview_h), Image.Resampling.LANCZOS)
        return pil_to_photo(preview)
    except Exception:
        return None


def _load_preview_numpy():
    """Lazy-load numpy for the fast preview blit. None if numpy is unavailable.

    numpy lives in requirements-preview.txt (optional), so the pure-Python
    fallback below must keep working when it is missing.
    """
    if not hasattr(_load_preview_numpy, "_cache"):
        try:
            import numpy as _np  # noqa: PLC0415

            _load_preview_numpy._cache = _np
        except Exception:
            _load_preview_numpy._cache = None
    return _load_preview_numpy._cache


def draw_preview_ellipse_pillow(image, x, y, w, h, rot_deg, color, scale):
    # Match the historical OpenCV preview path used before the one-file EXE:
    # cv2.ellipse(..., axes=(h, w), angle=-90+rot).
    width, height = image.size
    cx = float(x) * scale
    cy = float(y) * scale
    rx = max(float(h) * scale, 1.0)
    ry = max(float(w) * scale, 1.0)
    theta = (-90.0 + float(rot_deg)) * (math.pi / 180.0)
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    inv_rx2 = 1.0 / (rx * rx)
    inv_ry2 = 1.0 / (ry * ry)
    extent_x = math.sqrt(rx * rx * cos_t * cos_t + ry * ry * sin_t * sin_t)
    extent_y = math.sqrt(rx * rx * sin_t * sin_t + ry * ry * cos_t * cos_t)
    x_min = max(0, int(math.floor(cx - extent_x - 1)))
    x_max = min(width - 1, int(math.ceil(cx + extent_x + 1)))
    y_min = max(0, int(math.floor(cy - extent_y - 1)))
    y_max = min(height - 1, int(math.ceil(cy + extent_y + 1)))
    if x_min > x_max or y_min > y_max:
        return
    r, g, b = color

    # Fast path: vectorize the exact same rotated-ellipse test with numpy over
    # the shape's bbox, then blit the masked pixels back in one paste. Same math
    # (xr^2/rx^2 + yr^2/ry^2 <= 1) -> pixel-identical to the loop below.
    np = _load_preview_numpy()
    if np is not None:
        from PIL import Image  # noqa: PLC0415

        region = np.array(image.crop((x_min, y_min, x_max + 1, y_max + 1)))
        yy, xx = np.mgrid[y_min:y_max + 1, x_min:x_max + 1]
        dx = (xx + 0.5) - cx
        dy = (yy + 0.5) - cy
        xr = dx * cos_t + dy * sin_t
        yr = -dx * sin_t + dy * cos_t
        mask = xr * xr * inv_rx2 + yr * yr * inv_ry2 <= 1.0
        if mask.any():
            region[mask] = (r, g, b)
            image.paste(Image.fromarray(region, "RGB"), (x_min, y_min))
        return

    # Pure-Python fallback (no numpy installed).
    pixels = image.load()
    for yy in range(y_min, y_max + 1):
        dy = (float(yy) + 0.5) - cy
        for xx in range(x_min, x_max + 1):
            dx = (float(xx) + 0.5) - cx
            xr = dx * cos_t + dy * sin_t
            yr = -dx * sin_t + dy * cos_t
            if xr * xr * inv_rx2 + yr * yr * inv_ry2 <= 1.0:
                pixels[xx, yy] = (r, g, b)


class App:
    def __init__(self, initial_images):
        ensure_dirs()
        self.root = Tk()
        self.root.title(app_title())
        self.root.geometry("1420x900")
        self.root.minsize(1180, 780)
        self.root.configure(bg=Theme.BG)
        self.lang = "en"
        self.queue = queue.Queue()
        self.shutdown_event = threading.Event()
        self.active_processes = set()
        self.process_lock = threading.Lock()
        self.generation_lock = threading.Lock()
        self.generation_running = False
        self.current_generator_proc = None
        self.eta_samples = deque(maxlen=ETA_MAX_PROGRESS_SAMPLES)
        self.eta_display_time = None
        self.eta_smoothed_seconds_per_layer = None
        self.eta_display_remaining = None
        self.eta_max_layer_seen = None
        self.eta_recycle_notice_active = False
        self.closed = False
        self.settings = load_settings()
        self.images = [Path(path) for path in initial_images if Path(path).exists()]
        self.json_files = []
        self.outputs = []
        self.processes = []
        self.photo = None
        self.use_custom_settings = StringVar(value="0")
        self.custom_stop_at = StringVar()
        self.custom_max_resolution = StringVar()
        self.custom_random_samples = StringVar()
        self.custom_mutated_samples = StringVar()
        self.custom_save_at = StringVar()
        self.custom_max_threads = StringVar()
        self.custom_preprocess_mode = StringVar(value="none")
        self.translated = []
        self.detailed_log_lock = threading.Lock()
        self.detailed_log_lines = deque()
        self.detailed_log_chars = 0
        self.current_preview_request = None
        # Phase 2: monotonically increasing id so a stale off-thread JSON
        # render (superseded by a newer selection/resize) is dropped on arrival.
        self._preview_request_id = 0
        self.preview_resize_job = None
        self.update_state = {"status": "checking"}
        self.update_dialog = None
        self.update_check_started = False
        self.status = StringVar(value=tr(self.lang, "ready"))
        self.progress_text = StringVar(value="")
        self.selected_profile = StringVar()
        self.selected_game = StringVar(value="fh6")
        self.selected_pid = StringVar()
        self.layer_count = StringVar()
        self.snapshot_count = StringVar()
        self.current_count = StringVar()
        self.count_address = StringVar()
        self.table_address = StringVar()
        self.inspect_table_value = StringVar()
        self.runtime_folder = StringVar(value=str(ROOT))
        self.advanced_visible = False
        self._build()
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.refresh_processes()
        if self.settings:
            self.selected_profile.set(self.settings[min(2, len(self.settings) - 1)]["label"])
            self._update_setting_description()
        for image_path in list(self.images):
            self._load_existing_checkpoints_for_image(image_path)
        self._render_lists()
        self.log_line(tr(self.lang, "runtime_location").format(runtime=ROOT / "runtime", probe=PROBE_DIR.parent))
        self._poll_queue()
        self.root.after(1000, self.start_update_check)

    def _configure_styles(self):
        style = ttk.Style(self.root)
        try:
            style.theme_use("clam")
        except Exception:
            pass
        style.configure(".", background=Theme.BG, foreground=Theme.TEXT, fieldbackground=Theme.INPUT)
        style.configure("TFrame", background=Theme.BG)
        style.configure("TNotebook", background=Theme.BG, borderwidth=0)
        style.configure("Primary.TNotebook", background=Theme.BG, borderwidth=0, tabmargins=(0, 0, 0, 0))
        style.configure(
            "Primary.TNotebook.Tab",
            padding=(18, 8),
            font=("Segoe UI", 10, "bold"),
            background=Theme.PANEL_ALT,
            foreground=Theme.MUTED,
            borderwidth=0,
        )
        style.map(
            "Primary.TNotebook.Tab",
            background=[("selected", Theme.ACCENT_DARK), ("active", Theme.PANEL_ALT)],
            foreground=[("selected", "#ffffff"), ("active", Theme.TEXT)],
        )
        style.configure(
            "TLabelframe",
            background=Theme.PANEL,
            foreground=Theme.TEXT,
            bordercolor=Theme.BORDER,
            lightcolor=Theme.BORDER,
            darkcolor=Theme.BORDER,
        )
        style.configure(
            "TLabelframe.Label",
            background=Theme.PANEL,
            foreground=Theme.TEXT,
            font=("Segoe UI", 10, "bold"),
        )
        style.configure(
            "TCombobox",
            fieldbackground=Theme.INPUT,
            background=Theme.PANEL_ALT,
            foreground=Theme.TEXT,
            arrowcolor=Theme.TEXT,
            bordercolor=Theme.BORDER,
            lightcolor=Theme.BORDER,
            darkcolor=Theme.BORDER,
        )
        style.map(
            "TCombobox",
            fieldbackground=[("readonly", Theme.INPUT)],
            foreground=[("readonly", Theme.TEXT)],
            selectbackground=[("readonly", Theme.ACCENT_DARK)],
            selectforeground=[("readonly", "#ffffff")],
        )
        style.configure(
            "TScrollbar",
            background=Theme.PANEL_ALT,
            troughcolor=Theme.BG,
            bordercolor=Theme.BORDER,
            arrowcolor=Theme.TEXT,
        )

    def _register_process(self, proc):
        with self.process_lock:
            self.active_processes.add(proc)

    def _popen_registered(self, cmd, **kwargs):
        # Hold process_lock across Popen + registration so on_close cannot miss
        # a child process that starts while the window is closing.
        with self.process_lock:
            if self.shutdown_event.is_set() or self.closed:
                return None
            proc = subprocess.Popen(cmd, **kwargs)
            self.active_processes.add(proc)
            return proc

    def _unregister_process(self, proc):
        with self.process_lock:
            self.active_processes.discard(proc)

    def _terminate_process(self, proc):
        if proc.poll() is not None:
            return
        try:
            if os.name == "nt":
                subprocess.run(
                    ["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    creationflags=subprocess.CREATE_NO_WINDOW,
                    timeout=5,
                )
            else:
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass

    def _terminate_active_processes(self):
        with self.process_lock:
            processes = list(self.active_processes)
        for proc in processes:
            self._terminate_process(proc)

    def on_close(self):
        self.closed = True
        self.shutdown_event.set()
        self._terminate_active_processes()
        self.root.destroy()

    def _parent_bg(self, parent):
        try:
            return parent.cget("bg")
        except Exception:
            return Theme.PANEL

    def _label(self, parent, key, **kwargs):
        kwargs.setdefault("bg", self._parent_bg(parent))
        kwargs.setdefault("fg", Theme.TEXT)
        widget = Label(parent, text=tr(self.lang, key), **kwargs)
        self.translated.append((widget, key, "text"))
        return widget

    def _button(self, parent, key, command, **kwargs):
        kwargs.setdefault("bg", Theme.BUTTON)
        kwargs.setdefault("fg", Theme.TEXT)
        kwargs.setdefault("disabledforeground", Theme.MUTED)
        kwargs.setdefault("activebackground", Theme.BUTTON_ACTIVE)
        kwargs.setdefault("activeforeground", Theme.TEXT)
        kwargs.setdefault("relief", "flat")
        kwargs.setdefault("bd", 0)
        kwargs.setdefault("highlightthickness", 1)
        kwargs.setdefault("highlightbackground", Theme.BORDER)
        kwargs.setdefault("highlightcolor", Theme.ACCENT)
        kwargs.setdefault("padx", 8)
        kwargs.setdefault("pady", 3)
        widget = Button(parent, text=tr(self.lang, key), command=command, **kwargs)
        self.translated.append((widget, key, "text"))
        return widget

    def _mapped_label_color(self, color):
        value = str(color or "").lower()
        if value in ("black", "#000000", "#000", "systembuttontext", "systemwindowtext"):
            return Theme.TEXT
        if value in ("systemdisabledtext", "gray", "grey", "gray40", "grey40"):
            return Theme.MUTED
        if value in ("#555", "#555555", "gray50", "grey50"):
            return Theme.MUTED
        if value in ("#005a9e", "blue"):
            return Theme.ACCENT
        if value in ("#8a5300", "orange", "darkorange"):
            return Theme.WARN
        return color if color else Theme.TEXT

    def _apply_dark_theme_recursive(self, widget):
        try:
            if isinstance(widget, Frame):
                bg = Theme.BG if widget.master is self.root else Theme.PANEL
                widget.configure(bg=bg)
            elif isinstance(widget, Label):
                if widget in (getattr(self, "preview_label", None), getattr(self, "import_preview_label", None)):
                    widget.configure(bg=Theme.INPUT, fg=Theme.TEXT)
                elif widget is getattr(self, "update_indicator", None):
                    widget.configure(bg=Theme.PANEL)
                else:
                    widget.configure(bg=self._parent_bg(widget.master), fg=self._mapped_label_color(widget.cget("fg")))
            elif isinstance(widget, Button):
                widget.configure(
                    bg=Theme.BUTTON,
                    fg=Theme.TEXT,
                    disabledforeground=Theme.MUTED,
                    activebackground=Theme.BUTTON_ACTIVE,
                    activeforeground=Theme.TEXT,
                    relief="flat",
                    bd=0,
                    highlightbackground=Theme.BORDER,
                    highlightcolor=Theme.ACCENT,
                )
            elif isinstance(widget, Checkbutton):
                widget.configure(
                    bg=self._parent_bg(widget.master),
                    fg=Theme.TEXT,
                    disabledforeground=Theme.MUTED,
                    activebackground=self._parent_bg(widget.master),
                    activeforeground=Theme.TEXT,
                    selectcolor=Theme.INPUT,
                    relief="flat",
                    highlightbackground=Theme.BORDER,
                    highlightcolor=Theme.ACCENT,
                )
            elif isinstance(widget, Entry):
                widget.configure(
                    bg=Theme.INPUT,
                    fg=Theme.TEXT,
                    insertbackground=Theme.TEXT,
                    disabledbackground=Theme.PANEL_ALT,
                    disabledforeground=Theme.MUTED,
                    readonlybackground=Theme.INPUT,
                    relief="flat",
                    highlightthickness=1,
                    highlightbackground=Theme.BORDER,
                    highlightcolor=Theme.ACCENT,
                )
            elif isinstance(widget, Listbox):
                widget.configure(
                    bg=Theme.INPUT,
                    fg=Theme.TEXT,
                    selectbackground=Theme.ACCENT_DARK,
                    selectforeground="#ffffff",
                    highlightthickness=1,
                    highlightbackground=Theme.BORDER,
                    relief="flat",
                )
            elif isinstance(widget, Text):
                widget.configure(
                    bg=Theme.INPUT,
                    fg=Theme.TEXT,
                    insertbackground=Theme.TEXT,
                    selectbackground=Theme.ACCENT_DARK,
                    selectforeground="#ffffff",
                    highlightthickness=1,
                    highlightbackground=Theme.BORDER,
                    relief="flat",
                )
            elif isinstance(widget, Canvas):
                widget.configure(bg=Theme.PANEL, highlightthickness=0)
        except Exception:
            pass
        for child in widget.winfo_children():
            self._apply_dark_theme_recursive(child)

    def _build(self):
        self._configure_styles()
        header = Frame(self.root)
        header.pack(fill=X, padx=14, pady=(12, 6))

        # Title row: title on the left, language selector on the right
        title_row = Frame(header)
        title_row.pack(fill=X)
        self._label(title_row, "title", font=("Segoe UI", 18, "bold")).pack(side=LEFT)
        self.update_indicator = Label(
            title_row,
            text="",
            width=2,
            bg=Theme.BG,
            fg=Theme.WARN,
            font=("Segoe UI", 11, "bold"),
            cursor="hand2",
        )
        self.update_indicator.pack(side=RIGHT, padx=(8, 0))
        self.update_indicator.bind("<Button-1>", self.show_update_status)
        self.lang_combo = ttk.Combobox(title_row, values=list(LANGUAGES.keys()), state="readonly", width=10)
        self.lang_combo.set("English")
        self.lang_combo.pack(side=RIGHT)
        self.lang_combo.bind("<<ComboboxSelected>>", self._on_language)
        self._label(title_row, "language").pack(side=RIGHT, padx=(0, 6))

        # Subtitle row
        subtitle_row = Frame(header)
        subtitle_row.pack(fill=X)
        self._label(subtitle_row, "subtitle", fg=Theme.MUTED).pack(side=LEFT)

        process_bar = Frame(self.root)
        process_bar.pack(fill=X, padx=14, pady=(0, 8))
        self._label(process_bar, "process").pack(side=LEFT)
        self.process_combo = ttk.Combobox(process_bar, textvariable=self.selected_pid, state="readonly", width=44)
        self.process_combo.pack(side=LEFT, padx=8)
        self._button(process_bar, "refresh", self.refresh_processes).pack(side=LEFT)
        Label(process_bar, textvariable=self.status, anchor="e", bg=Theme.BG, fg=Theme.MUTED).pack(side=RIGHT)

        self.tabs = ttk.Notebook(self.root, style="Primary.TNotebook")
        self.tabs.pack(fill=BOTH, expand=True, padx=14, pady=(0, 8))
        self.generate_tab = Frame(self.tabs)
        self.import_tab = Frame(self.tabs)
        self.tools_tab = Frame(self.tabs)
        self.tutorial_tab = Frame(self.tabs)
        self.tabs.add(self.generate_tab, text=tr(self.lang, "generate_tab"))
        self.tabs.add(self.import_tab, text=tr(self.lang, "import_tab"))
        self.tabs.add(self.tutorial_tab, text=tr(self.lang, "tutorial_tab"))

        self._build_generate_tab()
        self._build_import_tab()
        self._build_tools_tab()
        self._build_tutorial_tab()
        self._build_log()
        self._apply_dark_theme_recursive(self.root)
        self.tabs.bind("<<NotebookTabChanged>>", self._schedule_preview_refresh)

    def _build_generate_tab(self):
        left_outer = Frame(self.generate_tab)
        left_outer.pack(side=LEFT, fill=BOTH, expand=True, padx=(0, 10), pady=10)
        right = Frame(self.generate_tab)
        right.pack(side=RIGHT, fill=BOTH, expand=True, pady=10)

        self._label(left_outer, "scroll_hint", anchor="w", justify=LEFT, fg="#8a5300").pack(fill=X, pady=(0, 6))
        scroll_area = Frame(left_outer)
        scroll_area.pack(fill=BOTH, expand=True, pady=(0, 8))
        left_canvas = Canvas(scroll_area, highlightthickness=0)
        left_scroll = ttk.Scrollbar(scroll_area, orient="vertical", command=left_canvas.yview)
        left_canvas.configure(yscrollcommand=left_scroll.set)
        left_canvas.pack(side=LEFT, fill=BOTH, expand=True)
        left_scroll.pack(side=RIGHT, fill="y")
        left = Frame(left_canvas)
        left_window = left_canvas.create_window((0, 0), window=left, anchor="nw")

        def _update_scroll_region(_event=None):
            left_canvas.configure(scrollregion=left_canvas.bbox("all"))

        def _match_canvas_width(event):
            left_canvas.itemconfigure(left_window, width=event.width)

        def _mousewheel(event):
            left_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

        def _bind_mousewheel(_event=None):
            left_canvas.bind_all("<MouseWheel>", _mousewheel)

        def _unbind_mousewheel(_event=None):
            left_canvas.unbind_all("<MouseWheel>")

        left.bind("<Configure>", _update_scroll_region)
        left_canvas.bind("<Configure>", _match_canvas_width)
        scroll_area.bind("<Enter>", _bind_mousewheel)
        scroll_area.bind("<Leave>", _unbind_mousewheel)

        step1 = ttk.LabelFrame(left, text=tr(self.lang, "generate_step_image"))
        self.translated.append((step1, "generate_step_image", "text"))
        step1.pack(fill=X, pady=(0, 6))
        row = Frame(step1)
        row.pack(fill=X, padx=10, pady=(6, 2))
        self._label(row, "images").pack(side=LEFT)
        self._button(row, "add_images", self.add_images).pack(side=RIGHT)
        self._button(row, "remove_image", self.remove_selected_image).pack(side=RIGHT, padx=8)
        self.image_list = Listbox(step1, height=3)
        self.image_list.pack(fill=X, padx=10, pady=(2, 8))
        self.image_list.bind("<<ListboxSelect>>", self._preview_selected_image)

        step2 = ttk.LabelFrame(left, text=tr(self.lang, "generate_step_quality"))
        self.translated.append((step2, "generate_step_quality", "text"))
        step2.pack(fill=X, pady=(0, 6))
        profile_row = Frame(step2)
        profile_row.pack(fill=X, padx=10, pady=(8, 4))
        self._label(profile_row, "quality").pack(side=LEFT)
        self.profile_combo = ttk.Combobox(
            profile_row,
            values=[item["label"] for item in self.settings],
            textvariable=self.selected_profile,
            state="readonly",
            width=32,
        )
        self.profile_combo.pack(side=LEFT, fill=X, expand=True, padx=(8, 0))
        self.profile_combo.bind("<<ComboboxSelected>>", self._update_setting_description)
        preset_actions = Frame(step2)
        preset_actions.pack(fill=X, padx=10, pady=(0, 6))
        self._button(preset_actions, "import_preset", self.import_preset).pack(side=LEFT)
        self._button(preset_actions, "open_preset_folder", self.open_preset_folder).pack(side=LEFT, padx=8)
        self.setting_description = Label(step2, text="", anchor="w", justify=LEFT, wraplength=500)
        self.setting_description.pack(fill=X, padx=10, pady=(0, 8))

        custom_section = ttk.LabelFrame(left, text=tr(self.lang, "custom_panel_title"))
        self.translated.append((custom_section, "custom_panel_title", "text"))
        custom_section.pack(fill=X, pady=(0, 6))
        self._label(custom_section, "custom_panel_hint", anchor="w", justify=LEFT, wraplength=540, fg="#005a9e").pack(fill=X, padx=10, pady=(6, 2))
        custom_toggle = Checkbutton(
            custom_section,
            text=tr(self.lang, "custom_settings"),
            variable=self.use_custom_settings,
            onvalue="1",
            offvalue="0",
            command=self._sync_custom_state,
        )
        custom_toggle.pack(anchor="w", padx=10, pady=(0, 2))
        self.translated.append((custom_toggle, "custom_settings", "text"))
        custom_grid = Frame(custom_section)
        custom_grid.pack(fill=X, padx=10, pady=(0, 6))
        self.custom_fields = []
        custom_specs = [
            ("custom_layers", self.custom_stop_at),
            ("custom_resolution", self.custom_max_resolution),
            ("custom_random", self.custom_random_samples),
            ("custom_mutated", self.custom_mutated_samples),
            ("custom_save_at", self.custom_save_at),
            ("custom_max_threads", self.custom_max_threads),
        ]
        for row_index, (key, variable) in enumerate(custom_specs):
            label = self._label(custom_grid, key, anchor="w")
            label.grid(row=row_index, column=0, sticky="w", pady=1, padx=(0, 8))
            entry = Entry(custom_grid, textvariable=variable, width=18)
            entry.grid(row=row_index, column=1, sticky="ew", pady=1)
            self.custom_fields.append(entry)
        custom_grid.columnconfigure(1, weight=1)
        preprocess_widget = self._field(custom_grid, "preprocess_mode", self.custom_preprocess_mode, row=len(custom_specs), values=["none", "luma_band", "edge_aware"], readonly=True)
        self.custom_fields.append(preprocess_widget)
        custom_actions = Frame(custom_section)
        custom_actions.pack(fill=X, padx=10, pady=(0, 8))
        self._button(custom_actions, "save_custom_preset", self.save_custom_preset).pack(side=LEFT)
        self._sync_custom_state()

        step3 = ttk.LabelFrame(left_outer, text=tr(self.lang, "generate_step_run"))
        self.translated.append((step3, "generate_step_run", "text"))
        step3.pack(fill=X)
        self._label(step3, "generate_step_run_hint", anchor="w", justify=LEFT, wraplength=540).pack(fill=X, padx=10, pady=(8, 4))
        actions = Frame(step3)
        actions.pack(fill=X, padx=10, pady=(4, 12))
        self.generate_button = self._button(actions, "start_generate", self.start_generate, font=("Segoe UI", 12, "bold"), height=2)
        self.generate_button.pack(side=LEFT, fill=X, expand=True)
        self.stop_generate_button = self._button(actions, "stop_generate", self.stop_generate, height=2, state="disabled")
        self.stop_generate_button.pack(side=LEFT, padx=8)
        self._button(actions, "open_output", self.open_output_folder, height=2).pack(side=LEFT)

        preview_header = Frame(right)
        preview_header.pack(fill=X)
        self._label(preview_header, "preview", anchor="w", font=("Segoe UI", 12, "bold")).pack(side=LEFT)
        self._label(preview_header, "preview_accuracy_note", anchor="e", justify=RIGHT, fg=Theme.WARN, wraplength=420).pack(side=RIGHT, fill=X, expand=True)
        self.preview_label = Label(right, text=tr(self.lang, "preview_hint"), bg="#202020", fg="#dddddd", width=60, height=24)
        self.preview_label.pack(fill=BOTH, expand=True, pady=6)
        self.preview_label.bind("<Configure>", self._schedule_preview_refresh)

    def _build_import_tab(self):
        left = Frame(self.import_tab)
        left.pack(side=LEFT, fill=BOTH, expand=True, padx=(0, 10), pady=10)
        right = Frame(self.import_tab)
        right.pack(side=RIGHT, fill=BOTH, expand=True, pady=10)

        step1 = ttk.LabelFrame(left, text=tr(self.lang, "step_game"))
        self.translated.append((step1, "step_game", "text"))
        step1.pack(fill=X, pady=(0, 10))
        self._label(step1, "step_game_hint", anchor="w", justify=LEFT, wraplength=540).pack(fill=X, padx=10, pady=(8, 4))
        game_row = Frame(step1)
        game_row.pack(fill=X, padx=10, pady=(0, 10))
        self._label(game_row, "game_profile").pack(side=LEFT)
        self.import_game_combo = ttk.Combobox(game_row, values=list(PROFILES.keys()), textvariable=self.selected_game, state="readonly", width=8)
        self.import_game_combo.pack(side=LEFT, padx=8)
        self._label(game_row, "pid").pack(side=LEFT)
        self.import_pid_entry = Entry(game_row, textvariable=self.selected_pid, width=30)
        self.import_pid_entry.pack(side=LEFT, padx=8)
        self._button(game_row, "refresh", self.refresh_processes).pack(side=LEFT)

        step2 = ttk.LabelFrame(left, text=tr(self.lang, "step_template"))
        self.translated.append((step2, "step_template", "text"))
        step2.pack(fill=X, pady=(0, 10))
        self._label(step2, "step_template_hint", anchor="w", justify=LEFT, wraplength=540).pack(fill=X, padx=10, pady=(8, 4))
        template_row = Frame(step2)
        template_row.pack(fill=X, padx=10, pady=(0, 10))
        self._label(template_row, "layer_count").pack(side=LEFT)
        self.layer_count_entry = Entry(template_row, textvariable=self.layer_count, width=18, font=("Segoe UI", 13))
        self.layer_count_entry.pack(side=LEFT, padx=8)

        step3 = ttk.LabelFrame(left, text=tr(self.lang, "step_json"))
        self.translated.append((step3, "step_json", "text"))
        step3.pack(fill=BOTH, expand=True, pady=(0, 10))
        self._label(step3, "step_json_hint", anchor="w", justify=LEFT, wraplength=540).pack(fill=X, padx=10, pady=(8, 4))
        row = Frame(step3)
        row.pack(fill=X, padx=10)
        self._label(row, "json_files").pack(side=LEFT)
        self._button(row, "add_json", self.add_json).pack(side=RIGHT)
        self._button(row, "remove_json", self.remove_selected_json).pack(side=RIGHT, padx=(8, 0))
        self._button(row, "use_outputs", self.use_generated_outputs).pack(side=RIGHT, padx=8)
        self.json_list = Listbox(step3, height=10)
        self.json_list.pack(fill=BOTH, expand=True, padx=10, pady=6)
        self.json_list.bind("<<ListboxSelect>>", self._preview_selected_json)

        step4 = ttk.LabelFrame(right, text=tr(self.lang, "step_import"))
        self.translated.append((step4, "step_import", "text"))
        step4.pack(fill=X, pady=(0, 10))
        self._label(step4, "step_import_hint", anchor="w", justify=LEFT, wraplength=500).pack(fill=X, padx=10, pady=(8, 4))
        self._label(step4, "easy_import_hint", anchor="w", justify=LEFT, wraplength=500, fg="#555").pack(fill=X, padx=10, pady=4)
        self._label(step4, "admin_note", anchor="w", justify=LEFT, wraplength=500, fg="#8a5300").pack(fill=X, padx=10, pady=4)
        actions = Frame(step4)
        actions.pack(fill=X, padx=10, pady=12)
        self._button(actions, "import_json", self.start_import, font=("Segoe UI", 13, "bold"), height=2).pack(side=LEFT, fill=X, expand=True)
        self.advanced_button = self._button(actions, "show_advanced", self.toggle_advanced)
        self.advanced_button.pack(side=LEFT, padx=(8, 0))

        self.advanced_frame = ttk.LabelFrame(right, text=tr(self.lang, "advanced_options"))
        self.translated.append((self.advanced_frame, "advanced_options", "text"))
        self._field(self.advanced_frame, "manual_count", self.count_address, row=0)
        self._field(self.advanced_frame, "manual_table", self.table_address, row=1)
        self._button(self.advanced_frame, "auto_locate", self.start_auto_locate).grid(row=2, column=0, columnspan=2, sticky="ew", pady=(8, 4))

        import_preview_header = Frame(right)
        import_preview_header.pack(fill=X, pady=(8, 0))
        self._label(import_preview_header, "import_preview", anchor="w", font=("Segoe UI", 12, "bold")).pack(side=LEFT)
        self._label(import_preview_header, "preview_accuracy_note", anchor="e", justify=RIGHT, fg=Theme.WARN, wraplength=420).pack(side=RIGHT, fill=X, expand=True)
        self.import_preview_label = Label(right, text=tr(self.lang, "preview_hint"), bg="#202020", fg="#dddddd", width=56, height=20)
        self.import_preview_label.pack(fill=BOTH, expand=True, pady=6)
        self.import_preview_label.bind("<Configure>", self._schedule_preview_refresh)

    def _build_tools_tab(self):
        form = Frame(self.tools_tab)
        form.pack(fill=X, padx=10, pady=10)
        self._field(form, "layer_count", self.layer_count, row=0)
        self._field(form, "snapshot_count", self.snapshot_count, row=1)
        self._field(form, "current_count", self.current_count, row=2)
        self._field(form, "table_address", self.inspect_table_value, row=3)
        runtime_entry = self._field(form, "runtime_folder", self.runtime_folder, row=4)
        runtime_entry.config(state="readonly")
        actions = Frame(self.tools_tab)
        actions.pack(fill=X, padx=10, pady=8)
        self._button(actions, "diagnose", self.start_diagnose).pack(side=LEFT)
        self._button(actions, "auto_locate", self.start_auto_locate).pack(side=LEFT, padx=6)
        self._button(actions, "save_snapshot", self.start_save_snapshot).pack(side=LEFT, padx=6)
        self._button(actions, "compare_snapshot", self.start_compare_snapshot).pack(side=LEFT, padx=6)
        self._button(actions, "inspect_table", self.start_inspect_table).pack(side=LEFT, padx=6)
        self._button(actions, "open_runtime_folder", self.open_runtime_folder).pack(side=LEFT, padx=6)

    def _build_tutorial_tab(self):
        self.tutorial_text = Text(self.tutorial_tab, wrap="word")
        self.tutorial_text.pack(fill=BOTH, expand=True, padx=10, pady=10)
        self._update_tutorial()

    def _build_log(self):
        row = Frame(self.root)
        row.pack(fill=X, padx=14)
        self._label(row, "logs", anchor="w").pack(side=LEFT)
        self._button(row, "export_logs", self.export_detailed_log).pack(side=RIGHT)
        self._label(row, "progress", anchor="e").pack(side=LEFT, padx=(18, 4))
        Label(row, textvariable=self.progress_text, anchor="w", fg="#005a9e").pack(side=LEFT, fill=X, expand=True)
        self.log = Text(self.root, height=9)
        self.log.pack(fill=BOTH, padx=14, pady=(0, 12))

    def _field(self, parent, key, variable, row, values=None, readonly=False):
        self._label(parent, key, anchor="w").grid(row=row, column=0, sticky="w", padx=(0, 8), pady=5)
        if values:
            widget = ttk.Combobox(parent, values=values, textvariable=variable, state="readonly" if readonly else "normal")
        else:
            widget = Entry(parent, textvariable=variable)
        widget.grid(row=row, column=1, sticky="ew", pady=5)
        parent.columnconfigure(1, weight=1)
        return widget

    def _on_language(self, _event=None):
        self.lang = LANGUAGES.get(self.lang_combo.get(), "en")
        for widget, key, option in self.translated:
            try:
                widget.config(**{option: tr(self.lang, key)})
            except Exception:
                pass
        self.tabs.tab(self.generate_tab, text=tr(self.lang, "generate_tab"))
        self.tabs.tab(self.import_tab, text=tr(self.lang, "import_tab"))
        self.tabs.tab(self.tutorial_tab, text=tr(self.lang, "tutorial_tab"))
        if self.photo is None:
            self.preview_label.config(text=tr(self.lang, "preview_hint"))
            if hasattr(self, "import_preview_label"):
                self.import_preview_label.config(text=tr(self.lang, "preview_hint"))
        if hasattr(self, "advanced_button"):
            self.advanced_button.config(text=tr(self.lang, "hide_advanced" if self.advanced_visible else "show_advanced"))
        self._update_tutorial()
        self.status.set(tr(self.lang, "ready"))

    def _update_tutorial(self):
        self.tutorial_text.config(state="normal")
        self.tutorial_text.delete("1.0", END)
        self.tutorial_text.insert(END, tr(self.lang, "tutorial"))
        self.tutorial_text.config(state="disabled")

    def _update_setting_description(self, _event=None):
        item = self._selected_setting()
        self.setting_description.config(text=item["description"] if item else "No settings profiles found.")
        if item and self.use_custom_settings.get() != "1":
            values = item.get("values", {})
            self.custom_stop_at.set(values.get("stopAt", "3000"))
            self.custom_max_resolution.set(values.get("maxResolution", "1200"))
            self.custom_random_samples.set(values.get("randomSamples", "3000"))
            self.custom_mutated_samples.set(values.get("mutatedSamples", "1000"))
            self.custom_save_at.set(values.get("saveAt", values.get("stopAt", "3000")))
            self.custom_max_threads.set(values.get("maxThreads", "0"))
            self.custom_preprocess_mode.set(values.get("preprocessMode", "none"))

    def _sync_custom_state(self):
        state = "normal" if self.use_custom_settings.get() == "1" else "disabled"
        for entry in getattr(self, "custom_fields", []):
            entry.config(state=state)
        if state == "disabled":
            self._update_setting_description()

    def _effective_setting(self):
        setting = self._selected_setting()
        if not setting or self.use_custom_settings.get() != "1":
            return setting
        return write_custom_settings(setting, self._custom_values())

    def _custom_values(self):
        custom = {
            "stopAt": self.custom_stop_at.get(),
            "maxResolution": self.custom_max_resolution.get(),
            "randomSamples": self.custom_random_samples.get(),
            "mutatedSamples": self.custom_mutated_samples.get(),
            "saveAt": self.custom_save_at.get(),
            "maxThreads": self.custom_max_threads.get(),
            "preprocessMode": self.custom_preprocess_mode.get(),
        }
        if not custom["saveAt"] and custom["stopAt"]:
            custom["saveAt"] = custom["stopAt"]
        return custom

    def save_custom_preset(self):
        setting = self._selected_setting()
        if not setting:
            self.log_line("No quality profile selected.")
            return
        USER_SETTINGS_DIR.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        output = filedialog.asksaveasfilename(
            title=tr(self.lang, "save_custom_preset"),
            initialdir=str(USER_SETTINGS_DIR),
            initialfile=f"user-preset-{timestamp}.ini",
            defaultextension=".ini",
            filetypes=[("INI settings", "*.ini"), ("All files", "*.*")],
        )
        if not output:
            return
        try:
            saved_path = write_user_settings_preset(setting, self._custom_values(), output)
        except OSError as exc:
            self.log_line(f"Failed to save preset: {exc}")
            return
        self._reload_settings(preferred_path=saved_path)
        self.log_line(tr(self.lang, "saved_preset").format(path=saved_path))

    def toggle_advanced(self):
        self.advanced_visible = not self.advanced_visible
        if self.advanced_visible:
            self.advanced_frame.pack(fill=X, pady=(0, 10))
        else:
            self.advanced_frame.pack_forget()
        self.advanced_button.config(text=tr(self.lang, "hide_advanced" if self.advanced_visible else "show_advanced"))

    def _selected_setting(self):
        label = self.selected_profile.get()
        for item in self.settings:
            if item["label"] == label:
                return item
        return self.settings[0] if self.settings else None

    def _reload_settings(self, preferred_path=None):
        previous = preferred_path
        if previous is None:
            current = self._selected_setting()
            previous = current.get("path") if current else None
        try:
            previous_resolved = Path(previous).resolve() if previous else None
        except OSError:
            previous_resolved = None
        self.settings = load_settings()
        values = [item["label"] for item in self.settings]
        if hasattr(self, "profile_combo"):
            self.profile_combo["values"] = values
        selected = None
        if previous_resolved:
            for item in self.settings:
                try:
                    if item["path"].resolve() == previous_resolved:
                        selected = item["label"]
                        break
                except OSError:
                    pass
        if selected is None and values:
            selected = values[min(2, len(values) - 1)]
        self.selected_profile.set(selected or "")
        self._update_setting_description()

    def _render_lists(self):
        self.image_list.delete(0, END)
        for path in self.images:
            self.image_list.insert(END, str(path))
        self.json_list.delete(0, END)
        for path in self.json_files:
            self.json_list.insert(END, str(path))

    def _add_json_paths(self, paths):
        added = 0
        for output in best_geometry_jsons(paths):
            output = Path(output)
            if output not in self.outputs:
                self.outputs.append(output)
            if output not in self.json_files:
                self.json_files.append(output)
                added += 1
        return added

    def _load_existing_checkpoints_for_image(self, image_path, log_to_queue=False):
        existing = best_geometry_jsons(generated_jsons(image_path))
        if not existing:
            return 0
        added = self._add_json_paths(existing[:1])
        if added:
            message = tr(self.lang, "existing_checkpoints_found").format(image=Path(image_path).name, count=len(existing))
            if log_to_queue:
                self.queue.put(("log", message))
            else:
                self.log_line(message)
        return added

    def _queue_generated_outputs(self, image_path, before):
        after = generated_jsons(image_path)
        new_outputs = best_geometry_jsons([path for path in after if path.resolve() not in before])
        if not new_outputs and after:
            new_outputs = best_geometry_jsons(after[:1])
        for output in new_outputs:
            if output not in self.outputs:
                self.outputs.append(output)
            if output not in self.json_files:
                self.json_files.append(output)
            self.queue.put(("log", f"Generated: {output}"))
        return new_outputs

    def log_line(self, message):
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        self._record_detail(f"UI: {message}")
        self.log.insert(END, f"[{timestamp}] {message}\n")
        self.log.see(END)

    def _record_detail(self, message):
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        text = str(message).rstrip()
        entry = f"[{timestamp}] {text}"
        with self.detailed_log_lock:
            self.detailed_log_lines.append(entry)
            self.detailed_log_chars += len(entry) + 1
            while self.detailed_log_chars > DETAILED_LOG_MEMORY_LIMIT and self.detailed_log_lines:
                removed = self.detailed_log_lines.popleft()
                self.detailed_log_chars -= len(removed) + 1

    def _format_command(self, cmd):
        return subprocess.list2cmdline([str(item) for item in cmd])

    def _diagnostic_log_header(self):
        try:
            profile = self._selected_setting()
            profile_name = profile["label"] if profile else ""
            profile_path = str(profile["path"]) if profile else ""
        except Exception:
            profile_name = ""
            profile_path = ""
        selected_pid = self.selected_pid_value()
        generator_exists = GENERATOR_EXE.exists()
        lines = [
            f"{APP_DISPLAY_NAME} detailed log",
            f"Generated at: {datetime.now().isoformat(timespec='seconds')}",
            f"App version: {__version__}",
            f"Python: {sys.version.replace(os.linesep, ' ')}",
            f"Platform: {platform.platform()}",
            f"Root: {ROOT}",
            f"Generator: {GENERATOR_EXE} exists={generator_exists}",
            f"Selected game: {self.selected_game.get()}",
            f"Selected PID: {selected_pid}",
            f"Selected process label: {self.selected_pid.get()}",
            f"Template layer count: {self.layer_count.get()}",
            f"Manual count address: {self.count_address.get()}",
            f"Manual table address: {self.table_address.get()}",
            f"Quality profile: {profile_name}",
            f"Quality profile path: {profile_path}",
            f"Custom settings enabled: {self.use_custom_settings.get()}",
            f"Images: {len(self.images)}",
            *[f"  image: {path}" for path in self.images],
            f"JSON files: {len(self.json_files)}",
            *[f"  json: {path}" for path in self.json_files],
            f"Generated outputs: {len(self.outputs)}",
            *[f"  output: {path}" for path in self.outputs],
        ]
        if SESSION_PATH.exists():
            try:
                session_text = SESSION_PATH.read_text(encoding="utf-8", errors="replace")
            except OSError as exc:
                session_text = f"<failed to read session: {exc}>"
            lines.extend(["Current FH6 session file:", session_text[:4000]])
        return "\n".join(lines).rstrip() + "\n"

    def _build_detailed_log_text(self):
        header = self._diagnostic_log_header()
        try:
            visible_log = self.log.get("1.0", END).strip()
        except Exception:
            visible_log = ""
        with self.detailed_log_lock:
            detail_log = "\n".join(self.detailed_log_lines).strip()
        body = "\n\n".join(
            section
            for section in (
                "=== Detailed Event Log ===\n" + (detail_log or "<empty>"),
                "=== Visible UI Log ===\n" + (visible_log or "<empty>"),
            )
            if section
        )
        marker = f"\n\n--- Log truncated to last {DETAILED_LOG_OUTPUT_LIMIT} characters ---\n"
        prefix = header + "\n"
        budget = DETAILED_LOG_OUTPUT_LIMIT - len(prefix)
        if budget <= len(marker):
            result = (prefix + body)[-DETAILED_LOG_OUTPUT_LIMIT:]
            return result if len(result) <= DETAILED_LOG_OUTPUT_LIMIT else result[-DETAILED_LOG_OUTPUT_LIMIT:]
        if len(body) > budget:
            body = marker + body[-(budget - len(marker)):]
        result = (prefix + body).rstrip()
        if len(result) >= DETAILED_LOG_OUTPUT_LIMIT:
            result = result[:DETAILED_LOG_OUTPUT_LIMIT - 1]
        return result + "\n"

    def export_detailed_log(self):
        initial = f"forza-painter-fh6-log-{datetime.now().strftime('%Y%m%d-%H%M%S')}.txt"
        output = filedialog.asksaveasfilename(
            title="Export detailed log",
            defaultextension=".txt",
            initialfile=initial,
            filetypes=[("Text log", "*.txt"), ("All files", "*.*")],
        )
        if not output:
            return
        text = self._build_detailed_log_text()
        try:
            Path(output).write_text(text, encoding="utf-8")
        except OSError as exc:
            self.log_line(f"Failed to export detailed log: {exc}")
            return
        self.log_line(f"Detailed log exported: {output} ({len(text)} chars, limit {DETAILED_LOG_OUTPUT_LIMIT}).")

    def start_update_check(self):
        if self.closed or self.update_check_started:
            return
        self.update_check_started = True
        threading.Thread(target=self._update_check_worker, daemon=True).start()

    def _update_check_worker(self):
        try:
            version_source = fetch_text_url(UPDATE_VERSION_URL)
            latest_version = parse_version_source(version_source)
        except (OSError, ValueError, urllib.error.URLError) as exc:
            self.queue.put(("update_failed", str(exc)))
            return

        changelog = ""
        try:
            changelog = fetch_text_url(UPDATE_CHANGELOG_URL)
        except (OSError, urllib.error.URLError) as exc:
            self._record_detail(f"Update changelog fetch failed: {exc}")

        payload = {
            "current": __version__,
            "latest": latest_version,
            "changelog": extract_changelog_section(changelog, latest_version),
        }
        if version_key(latest_version) > version_key(__version__):
            self.queue.put(("update_available", payload))
        else:
            self.queue.put(("update_current", payload))

    def _set_update_indicator(self, text="", color=Theme.WARN):
        if hasattr(self, "update_indicator"):
            self.update_indicator.config(text=text, fg=color)

    def _handle_update_failed(self, error):
        self.update_state = {"status": "failed", "error": error}
        self._set_update_indicator("!", Theme.WARN)
        self.log_line(f"Update check failed: {error}")

    def _handle_update_current(self, payload):
        self.update_state = {"status": "current", **payload}
        self._set_update_indicator("")
        self._record_detail(f"Update check OK: latest={payload.get('latest')}")

    def _handle_update_available(self, payload):
        self.update_state = {"status": "available", **payload}
        self._set_update_indicator("!", Theme.ACCENT)
        self.log_line(f"New version available: v{payload.get('latest')} (current v{__version__})")
        self.show_update_dialog(payload)

    def show_update_status(self, _event=None):
        status = self.update_state.get("status")
        if status == "failed":
            messagebox.showwarning(
                tr(self.lang, "update_check_failed_title"),
                tr(self.lang, "update_check_failed_message").format(error=self.update_state.get("error", "")),
                parent=self.root,
            )
        elif status == "available":
            self.show_update_dialog(self.update_state)

    def show_update_dialog(self, payload=None):
        payload = payload or self.update_state
        if self.update_dialog is not None and self.update_dialog.winfo_exists():
            self.update_dialog.lift()
            self.update_dialog.focus_force()
            return

        latest = payload.get("latest", "")
        changelog = payload.get("changelog") or "No changelog section was available."
        dialog = Toplevel(self.root)
        self.update_dialog = dialog
        dialog.title(tr(self.lang, "update_available_title"))
        dialog.configure(bg=Theme.BG)
        dialog.resizable(True, True)

        body = Frame(dialog, bg=Theme.BG)
        body.pack(fill=BOTH, expand=True, padx=16, pady=14)
        Label(
            body,
            text=tr(self.lang, "update_available_message").format(current=__version__, latest=latest),
            bg=Theme.BG,
            fg=Theme.TEXT,
            justify=LEFT,
            anchor="w",
            font=("Segoe UI", 11, "bold"),
        ).pack(fill=X, pady=(0, 10))
        Label(
            body,
            text=tr(self.lang, "changelog"),
            bg=Theme.BG,
            fg=Theme.MUTED,
            anchor="w",
        ).pack(fill=X)

        text_frame = Frame(body, bg=Theme.BG)
        text_frame.pack(fill=BOTH, expand=True, pady=(4, 12))
        changelog_text = Text(text_frame, width=80, height=18, wrap="word")
        scrollbar = ttk.Scrollbar(text_frame, orient="vertical", command=changelog_text.yview)
        changelog_text.configure(
            yscrollcommand=scrollbar.set,
            bg=Theme.INPUT,
            fg=Theme.TEXT,
            insertbackground=Theme.TEXT,
            selectbackground=Theme.ACCENT_DARK,
            selectforeground="#ffffff",
            highlightthickness=1,
            highlightbackground=Theme.BORDER,
            relief="flat",
        )
        changelog_text.pack(side=LEFT, fill=BOTH, expand=True)
        scrollbar.pack(side=RIGHT, fill="y")
        changelog_text.insert(END, changelog)
        changelog_text.config(state="disabled")

        actions = Frame(body, bg=Theme.BG)
        actions.pack(fill=X)

        def close_update_dialog():
            self.update_dialog = None
            dialog.destroy()

        def open_update_page():
            webbrowser.open(UPDATE_RELEASE_URL)
            close_update_dialog()

        Button(
            actions,
            text=tr(self.lang, "update_later"),
            command=close_update_dialog,
            bg=Theme.BUTTON,
            fg=Theme.TEXT,
            activebackground=Theme.BUTTON_ACTIVE,
            activeforeground=Theme.TEXT,
            relief="flat",
            bd=0,
            padx=12,
            pady=6,
        ).pack(side=RIGHT)
        Button(
            actions,
            text=tr(self.lang, "update_open_page"),
            command=open_update_page,
            bg=Theme.ACCENT_DARK,
            fg="#ffffff",
            activebackground=Theme.ACCENT,
            activeforeground="#ffffff",
            relief="flat",
            bd=0,
            padx=12,
            pady=6,
        ).pack(side=RIGHT, padx=(0, 8))

        dialog.protocol("WM_DELETE_WINDOW", close_update_dialog)

    def _reset_generation_eta(self):
        self.eta_samples.clear()
        self.eta_display_time = None
        self.eta_smoothed_seconds_per_layer = None
        self.eta_display_remaining = None
        self.eta_max_layer_seen = None
        self.eta_recycle_notice_active = False

    def _format_remaining_time(self, seconds):
        seconds = max(0, int(round(seconds)))
        hours, remainder = divmod(seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        suffix = {"zh": "剩余", "zh-tw": "剩餘", "ko": "남음"}.get(self.lang, "left")
        if hours:
            return f"{hours}h {minutes:02d}m {suffix}"
        if minutes:
            return f"{minutes}m {seconds:02d}s {suffix}"
        return f"{seconds}s {suffix}"

    def _progress_with_eta(self, friendly):
        match = re.match(r"Generated layer\s+(\d+)/(\d+)", friendly)
        if not match:
            return friendly
        current = int(match.group(1))
        total = int(match.group(2))
        now = time.time()
        if self.eta_max_layer_seen is not None and current <= self.eta_max_layer_seen:
            if current < self.eta_max_layer_seen and not self.eta_recycle_notice_active:
                self.eta_recycle_notice_active = True
                return tr(self.lang, "generator_recycled_layers").format(max_layer=self.eta_max_layer_seen, total=total)
            return None
        self.eta_max_layer_seen = current
        self.eta_recycle_notice_active = False
        self.eta_samples.append((current, now))
        while len(self.eta_samples) > 1 and now - self.eta_samples[0][1] > ETA_MAX_HISTORY_SECONDS:
            self.eta_samples.popleft()

        remaining_layers = max(0, total - current)
        if remaining_layers == 0:
            self.eta_display_remaining = 0
            self.eta_display_time = now
            eta_time = datetime.fromtimestamp(now).strftime("%H:%M:%S")
            return f"{friendly} | ETA {eta_time} ({self._format_remaining_time(0)})"

        if len(self.eta_samples) < 2:
            return friendly

        start_layer, start_time = self.eta_samples[0]
        for sample_layer, sample_time in self.eta_samples:
            if now - sample_time <= ETA_WINDOW_SECONDS:
                start_layer, start_time = sample_layer, sample_time
                break
        layer_delta = current - start_layer
        elapsed_seconds = now - start_time
        if layer_delta < ETA_MIN_WINDOW_LAYERS or elapsed_seconds < ETA_MIN_WINDOW_SECONDS:
            start_layer, start_time = self.eta_samples[0]
            layer_delta = current - start_layer
            elapsed_seconds = now - start_time
            if layer_delta < ETA_MIN_WINDOW_LAYERS or elapsed_seconds < ETA_MIN_WINDOW_SECONDS:
                return friendly

        seconds_per_layer = elapsed_seconds / layer_delta
        self.eta_smoothed_seconds_per_layer = seconds_per_layer
        measured_remaining = seconds_per_layer * remaining_layers
        if self.eta_display_remaining is None:
            self.eta_display_remaining = measured_remaining
        else:
            elapsed_since_display = now - self.eta_display_time if self.eta_display_time is not None else 0
            expected_remaining = max(0, self.eta_display_remaining - elapsed_since_display)
            if measured_remaining < expected_remaining * 0.65 or measured_remaining > expected_remaining * 1.6:
                self.eta_display_remaining = measured_remaining
            else:
                self.eta_display_remaining = expected_remaining * 0.35 + measured_remaining * 0.65
        self.eta_display_time = now
        remaining_seconds = self.eta_display_remaining
        eta_time = datetime.fromtimestamp(now + remaining_seconds).strftime("%H:%M:%S")
        return f"{friendly} | ETA {eta_time} ({self._format_remaining_time(remaining_seconds)})"

    def friendly_generator_line(self, line):
        text = (line or "").strip()
        if not text:
            return None
        progress = re.match(r"\[(\d+)/(\d+)\]\s+(.*)", text)
        if progress:
            current, total, detail = progress.groups()
            if "Added rotated ellipse" in detail:
                return f"Generated layer {current}/{total}"
            if "Saved geometry checkpoint" in detail:
                return f"Saved JSON checkpoint {current}/{total}"
            if "Saved preview snapshot" in detail:
                return f"Updated preview {current}/{total}"
            if "Step completed" in detail:
                return None
            return None
        if text.startswith("Loaded image:"):
            return text
        if text.startswith("Settings:"):
            return text
        if text.startswith("OpenCL: Selected device"):
            return text
        if text.startswith("Scoring mode:"):
            return text
        if text in ("FINISHED",):
            return text
        if "error" in text.lower() or "failed" in text.lower() or "panic" in text.lower():
            return text
        return None

    def queue_generator_message(self, friendly, last_message):
        if not friendly or friendly == last_message:
            return last_message
        if friendly.startswith("Generated layer "):
            message = self._progress_with_eta(friendly)
            if not message:
                return last_message
            self.queue.put(("progress", message))
            self.queue.put(("log", message))
            return friendly
        if friendly == "FINISHED":
            self.queue.put(("progress", friendly))
        self.queue.put(("log", friendly))
        return friendly

    def _int_setting(self, setting, key, default=0):
        try:
            return int(str(setting.get("values", {}).get(key, default)).strip())
        except (TypeError, ValueError):
            return default

    def _log_generation_load_warning(self, setting):
        stop_at = self._int_setting(setting, "stopAt")
        random_samples = self._int_setting(setting, "randomSamples")
        mutated_samples = self._int_setting(setting, "mutatedSamples")
        max_resolution = self._int_setting(setting, "maxResolution")
        if random_samples >= 200000 or mutated_samples >= 8000 or max_resolution >= 2000:
            self.queue.put((
                "log",
                "High quality generation selected: "
                f"layers={stop_at}, randomSamples={random_samples}, "
                f"mutatedSamples={mutated_samples}, maxResolution={max_resolution}. "
                "The first layer can take a long time before progress appears.",
            ))

    def _generator_exit_message(self, returncode):
        if returncode in (3221225477, -1073741819):
            return (
                "GPU generator crashed with Windows access violation 0xC0000005. "
                "This is usually an OpenCL/GPU driver or generator runtime crash, not an import error. "
                "Try a lower preset, lower Max resolution / Random samples, update the GPU driver, "
                "or convert the source image to a normal PNG/JPG path and retry."
            )
        if returncode == 3221226505:
            return (
                "GPU generator crashed with Windows stack buffer overrun 0xC0000409. "
                "Try updating the GPU driver, lowering the preset, or converting the image to PNG/JPG."
            )
        return f"Generator exited with code {returncode}."

    def add_images(self):
        files = filedialog.askopenfilenames(
            title="Choose images",
            filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp"), ("All files", "*.*")],
        )
        added_paths = []
        for item in files:
            path = Path(item)
            if path.exists() and path not in self.images:
                self.images.append(path)
                added_paths.append(path)
                self._load_existing_checkpoints_for_image(path)
        self._render_lists()
        if files:
            self.show_source_preview(Path(files[0]))
        if added_paths:
            existing_added = sum(1 for path in added_paths if generated_jsons(path))
            if existing_added:
                self.log_line(tr(self.lang, "cannot_resume_checkpoint"))

    def remove_selected_image(self):
        selection = list(self.image_list.curselection())
        if not selection:
            self.log_line(tr(self.lang, "no_image_selected"))
            return
        for index in sorted(selection, reverse=True):
            try:
                del self.images[index]
            except IndexError:
                pass
        self._render_lists()
        self.preview_label.config(image="", text=tr(self.lang, "preview_hint"))
        self.preview_label.image = None

    def _unique_preset_destination(self, source_path):
        USER_SETTINGS_DIR.mkdir(parents=True, exist_ok=True)
        stem = source_path.stem
        suffix = source_path.suffix or ".ini"
        candidate = USER_SETTINGS_DIR / f"{stem}{suffix}"
        index = 2
        while candidate.exists():
            candidate = USER_SETTINGS_DIR / f"{stem} ({index}){suffix}"
            index += 1
        return candidate

    def import_preset(self):
        files = filedialog.askopenfilenames(
            title="Import generator preset",
            filetypes=[("INI settings", "*.ini"), ("All files", "*.*")],
        )
        imported = []
        for item in files:
            source = Path(item)
            if not source.exists():
                continue
            destination = self._unique_preset_destination(source)
            try:
                shutil.copy2(source, destination)
                imported.append(destination)
            except OSError as exc:
                self.log_line(f"Failed to import preset {source}: {exc}")
        if imported:
            self._reload_settings(preferred_path=imported[-1])
            self.log_line(tr(self.lang, "imported_presets").format(count=len(imported)))

    def open_preset_folder(self):
        USER_SETTINGS_DIR.mkdir(parents=True, exist_ok=True)
        os.startfile(USER_SETTINGS_DIR)

    def add_json(self):
        files = filedialog.askopenfilenames(
            title="Choose geometry JSON",
            filetypes=[("Geometry JSON", "*.json"), ("All files", "*.*")],
        )
        for item in files:
            path = Path(item)
            if path.exists() and path not in self.json_files:
                self.json_files.append(path)
        self._render_lists()
        if files:
            self.show_json_preview(Path(files[0]))

    def remove_selected_json(self):
        selection = list(self.json_list.curselection())
        if not selection:
            self.log_line(tr(self.lang, "no_json_selected"))
            return
        for index in sorted(selection, reverse=True):
            try:
                del self.json_files[index]
            except IndexError:
                pass
        self._render_lists()
        if hasattr(self, "import_preview_label"):
            self.import_preview_label.config(image="", text=tr(self.lang, "preview_hint"))
            self.import_preview_label.image = None

    def use_generated_outputs(self):
        for path in self.outputs:
            if path.exists() and path not in self.json_files:
                self.json_files.append(path)
        self._render_lists()
        self.log_line(f"Added {len(self.outputs)} generated JSON file(s) to import list.")

    def _preview_selected_image(self, _event=None):
        selection = self.image_list.curselection()
        if selection:
            self.show_source_preview(self.images[selection[0]])

    def _preview_selected_json(self, _event=None):
        selection = self.json_list.curselection()
        if selection:
            self.show_json_preview(self.json_files[selection[0]])

    def _active_preview_label(self):
        if hasattr(self, "tabs") and hasattr(self, "import_preview_label"):
            try:
                if self.tabs.select() == str(self.import_tab):
                    return self.import_preview_label
            except Exception:
                pass
        return getattr(self, "preview_label", None)

    def _preview_bounds(self, label=None):
        label = label or self._active_preview_label()
        if label is None:
            return PREVIEW_MAX, PREVIEW_MAX
        try:
            self.root.update_idletasks()
            width = label.winfo_width()
            height = label.winfo_height()
        except Exception:
            width = height = 0
        if width <= 32 or height <= 32:
            return PREVIEW_MAX, PREVIEW_MAX
        return max(1, width - 16), max(1, height - 16)

    def _schedule_preview_refresh(self, _event=None):
        if not self.current_preview_request or self.closed:
            return
        if self.preview_resize_job is not None:
            try:
                self.root.after_cancel(self.preview_resize_job)
            except Exception:
                pass
        self.preview_resize_job = self.root.after(180, self._refresh_current_preview)

    def _refresh_current_preview(self):
        self.preview_resize_job = None
        request = self.current_preview_request
        if not request or self.closed:
            return
        kind, path = request
        path = Path(path)
        if not path.exists():
            return
        if kind == "json":
            # JSON geometry rasterization is the slow path; render off-thread.
            self._render_json_preview_async(path)
            return
        data = render_source_image(path, self._preview_bounds())
        self.show_preview(data)

    def _render_json_preview_async(self, path):
        """Rasterize a geometry JSON on a daemon thread, post bytes to the queue.

        render_geometry_json returns image bytes (no Tk objects), so only the
        final PhotoImage creation must stay on the main thread (in show_preview,
        reached via the 'preview_result' queue handler). A request id lets a
        newer selection/resize supersede an in-flight render.
        """
        self._preview_request_id += 1
        rid = self._preview_request_id
        bounds = self._preview_bounds()
        path = Path(path)

        def worker():
            try:
                data = render_geometry_json(path, bounds)
            except Exception:
                data = None
            if not self.closed:
                self.queue.put(("preview_result", (rid, data)))

        threading.Thread(target=worker, daemon=True).start()

    def show_json_preview(self, path):
        path = Path(path)
        self.current_preview_request = ("json", path)
        self._render_json_preview_async(path)

    def show_preview(self, data):
        if not data:
            self.current_preview_request = None
            message = tr(self.lang, "preview_unavailable")
            self.preview_label.config(image="", text=message, bg="#202020")
            self.preview_label.image = None
            if hasattr(self, "import_preview_label"):
                self.import_preview_label.config(image="", text=message, bg="#202020")
                self.import_preview_label.image = None
            return
        self.photo = data
        image = PhotoImage(data=data)
        self.preview_label.config(image=image, text="", bg="#202020")
        self.preview_label.image = image
        if hasattr(self, "import_preview_label"):
            import_image = PhotoImage(data=data)
            self.import_preview_label.config(image=import_image, text="", bg="#202020")
            self.import_preview_label.image = import_image

    def show_source_preview(self, path):
        path = Path(path)
        self.current_preview_request = ("source", path)
        self._preview_request_id += 1  # supersede any in-flight JSON render
        data = render_source_image(path, self._preview_bounds())
        if data:
            self.show_preview(data)
            return
        if Path(path).suffix.lower() in (".png", ".gif"):
            self.show_preview_file(path, remember=False)
            return
        self.show_preview(None)

    def show_preview_file(self, path, remember=True):
        path = Path(path)
        if remember:
            self.current_preview_request = ("file", path)
        self._preview_request_id += 1  # supersede any in-flight JSON render
        data = render_source_image(path, self._preview_bounds())
        if data:
            self.show_preview(data)
            return
        try:
            image = PhotoImage(file=str(path))
        except Exception:
            self.show_preview(None)
            return
        self.photo = image
        self.preview_label.config(image=image, text="", bg="#202020")
        self.preview_label.image = image
        if hasattr(self, "import_preview_label"):
            import_image = PhotoImage(file=str(path))
            self.import_preview_label.config(image=import_image, text="", bg="#202020")
            self.import_preview_label.image = import_image

    def refresh_processes(self):
        self.processes = game_processes()
        values = [item["label"] for item in self.processes]
        if not values:
            values = [tr(self.lang, "no_game")]
        self.process_combo["values"] = values
        if self.processes:
            self.selected_pid.set(values[0])
            self.selected_game.set(self.processes[0]["profile"])
        else:
            self.selected_pid.set("")

    def selected_pid_value(self):
        raw = self.selected_pid.get()
        match = re.search(r"pid\s+(\d+)", raw, re.I)
        if match:
            return int(match.group(1))
        try:
            return int(raw.strip())
        except ValueError:
            return None

    def _pid_matches_game(self, pid, game):
        profile = PROFILES.get(game)
        if not pid or not profile:
            return False
        try:
            process_name = psutil.Process(pid).name().lower()
        except psutil.Error:
            return False
        return process_name in [name.lower() for name in profile.process_names]

    def ensure_live_game_pid(self):
        game = self.selected_game.get() or "fh6"
        pid = self.selected_pid_value()
        if self._pid_matches_game(pid, game):
            return pid
        if pid:
            self.log_line(f"Selected game process pid {pid} is no longer running; refreshing process list.")
        self.refresh_processes()
        game = self.selected_game.get() or game
        pid = self.selected_pid_value()
        if self._pid_matches_game(pid, game):
            return pid
        self.log_line("No live supported game process is selected. Start FH6, then click Refresh.")
        return None

    def stop_generate(self):
        with self.generation_lock:
            if not self.generation_running:
                self.log_line(tr(self.lang, "no_generation_running"))
                return
            proc = self.current_generator_proc
        self.log_line(tr(self.lang, "stopping_generation"))
        self.shutdown_event.set()
        if proc is not None:
            self._terminate_process(proc)
        self.status.set(tr(self.lang, "stopped"))

    def start_generate(self):
        with self.generation_lock:
            if self.generation_running:
                self.log_line("Generation is already running. Wait for it to finish or use Stop current generation.")
                return
            self.generation_running = True
        if not self.images:
            with self.generation_lock:
                self.generation_running = False
            self.log_line("No images selected.")
            return
        setting = self._effective_setting()
        if not setting:
            with self.generation_lock:
                self.generation_running = False
            self.log_line("No quality profile selected.")
            return
        if not GENERATOR_EXE.exists():
            with self.generation_lock:
                self.generation_running = False
            self.log_line(f"Missing generator: {GENERATOR_EXE}")
            return
        self.shutdown_event.clear()
        self._reset_generation_eta()
        self.progress_text.set("")
        self.status.set(tr(self.lang, "running"))
        if hasattr(self, "generate_button"):
            self.generate_button.config(state="disabled")
        if hasattr(self, "stop_generate_button"):
            self.stop_generate_button.config(state="normal")
        threading.Thread(target=self._generate_worker, args=(setting,), daemon=True).start()

    def _generate_worker(self, setting):
        try:
            self.queue.put(("log", f"Selected profile: {setting['path'].name}"))
            self._log_generation_load_warning(setting)
            for image_path in list(self.images):
                if self.shutdown_event.is_set():
                    self.queue.put(("status", tr(self.lang, "stopped")))
                    return
                self._reset_generation_eta()
                input_image = preprocess_input_image(image_path, setting)
                if input_image != image_path:
                    self.queue.put(("log", f"Preprocessed image: {input_image}"))
                before = {path.resolve() for path in generated_jsons(input_image)}
                preview_path = generator_preview_path(input_image)
                if preview_path.exists():
                    try:
                        preview_path.unlink()
                    except OSError:
                        pass
                self.queue.put(("log", f"Generating: {image_path}"))
                self.queue.put(("preview_file", image_path))
                flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
                cmd = build_generator_command(input_image, setting)
                self._record_detail(f"GENERATOR COMMAND: {self._format_command(cmd)}")
                self.queue.put(("log", f"Running GPU generator with {setting['path'].name}"))
                if self.shutdown_event.is_set():
                    self.queue.put(("status", tr(self.lang, "stopped")))
                    return
                proc = self._popen_registered(
                    cmd,
                    cwd=ROOT,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    bufsize=1,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    creationflags=flags,
                    env=build_generator_env(),
                )
                if proc is None:
                    self.queue.put(("status", tr(self.lang, "stopped")))
                    return
                with self.generation_lock:
                    self.current_generator_proc = proc

                last_preview = None
                last_preview_mtime = None
                last_generator_message = None
                output_queue = queue.Queue()

                def _read_generator_output():
                    try:
                        for raw_line in proc.stdout:
                            self._record_detail(f"GENERATOR RAW: {raw_line.rstrip()}")
                            output_queue.put(raw_line)
                    finally:
                        output_queue.put(None)

                reader = threading.Thread(target=_read_generator_output, daemon=True)
                reader.start()

                def _drain_generator_output():
                    nonlocal last_generator_message
                    while True:
                        try:
                            raw_line = output_queue.get_nowait()
                        except queue.Empty:
                            break
                        if raw_line is None:
                            continue
                        friendly = self.friendly_generator_line(raw_line)
                        last_generator_message = self.queue_generator_message(friendly, last_generator_message)

                next_preview_scan = 0.0
                next_json_scan = 0.0

                try:
                    while proc.poll() is None:
                        if self.shutdown_event.is_set():
                            self._terminate_process(proc)
                            outputs = self._queue_generated_outputs(input_image, before)
                            for output in outputs:
                                self.queue.put(("log", tr(self.lang, "checkpoint_available_after_failure").format(path=output)))
                            if outputs:
                                self.queue.put(("render_lists", None))
                            self.queue.put(("status", tr(self.lang, "stopped")))
                            return
                        _drain_generator_output()
                        now = time.monotonic()
                        if now >= next_preview_scan:
                            next_preview_scan = now + GENERATOR_PREVIEW_SCAN_SECONDS
                            preview_files = generated_preview_files(input_image)
                            if preview_files:
                                newest_preview = preview_files[0]
                                preview_mtime = newest_preview.stat().st_mtime
                                if preview_mtime != last_preview_mtime:
                                    last_preview_mtime = preview_mtime
                                    self.queue.put(("preview_file", newest_preview))
                        if now >= next_json_scan:
                            next_json_scan = now + GENERATOR_JSON_SCAN_SECONDS
                            newest = generated_jsons(input_image)
                            if newest and newest[0] != last_preview:
                                last_preview = newest[0]
                        time.sleep(GENERATOR_POLL_SLEEP_SECONDS)
                    if self.shutdown_event.is_set():
                        return
                    reader.join(timeout=1)
                    _drain_generator_output()
                finally:
                    self._unregister_process(proc)
                    with self.generation_lock:
                        if self.current_generator_proc is proc:
                            self.current_generator_proc = None
                if proc.returncode != 0:
                    self._record_detail(f"GENERATOR EXIT: {proc.returncode}")
                    outputs = self._queue_generated_outputs(input_image, before)
                    for output in outputs:
                        self.queue.put(("log", tr(self.lang, "checkpoint_available_after_failure").format(path=output)))
                    if outputs:
                        self.queue.put(("render_lists", None))
                    self.queue.put(("log", self._generator_exit_message(proc.returncode)))
                    self.queue.put(("status", tr(self.lang, "failed")))
                    return
                self._record_detail("GENERATOR EXIT: 0")
                new_outputs = self._queue_generated_outputs(input_image, before)
                if not new_outputs:
                    self.queue.put(("log", "Generator finished but no JSON output was found."))
                    self.queue.put(("status", tr(self.lang, "failed")))
                    return
                for output in new_outputs:
                    preview_files = generated_preview_files(input_image)
                    if preview_files:
                        self.queue.put(("preview_file", preview_files[0]))
                    else:
                        self.queue.put(("preview_json", output))
            self.queue.put(("render_lists", None))
            self.queue.put(("status", tr(self.lang, "done")))
        except Exception as exc:
            self.queue.put(("log", f"Generator failed: {exc}"))
            self.queue.put(("status", tr(self.lang, "failed")))
        finally:
            self.queue.put(("generation_done", None))

    def open_output_folder(self):
        folder = None
        if self.outputs:
            folder = self.outputs[-1].parent
        elif self.images:
            folder = self.images[-1].parent
        if folder and folder.exists():
            os.startfile(folder)

    def open_runtime_folder(self):
        ROOT.mkdir(parents=True, exist_ok=True)
        os.startfile(ROOT)

    def run_subprocess(self, cmd, timeout=None):
        self._record_detail(f"HELPER COMMAND: {self._format_command(cmd)}")
        self.queue.put(("log", self._friendly_command_name(cmd)))
        flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        env = os.environ.copy()
        env.update({"FORZA_PAINTER_NO_ELEVATE": "1", "FORZA_PAINTER_NO_PAUSE": "1"})
        if self.shutdown_event.is_set():
            self._record_detail("HELPER EXIT: 130 before start")
            return 130
        proc = self._popen_registered(
            [str(x) for x in cmd],
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            creationflags=flags,
            env=env,
        )
        if proc is None:
            self._record_detail("HELPER EXIT: 130 no process")
            return 130
        started = time.time()
        try:
            while True:
                if self.shutdown_event.is_set():
                    self._terminate_process(proc)
                    self._record_detail("HELPER EXIT: 130 stopped")
                    return 130
                line = proc.stdout.readline()
                if line:
                    self._record_detail(f"HELPER RAW: {line.rstrip()}")
                    friendly = self._friendly_subprocess_line(line.rstrip())
                    if friendly:
                        self.queue.put(("log", friendly))
                if proc.poll() is not None:
                    break
                if timeout and time.time() - started > timeout:
                    self._terminate_process(proc)
                    self._record_detail(f"HELPER EXIT: 124 timeout after {timeout} seconds")
                    self.queue.put(("log", f"Timed out after {timeout} seconds."))
                    return 124
                time.sleep(0.05)
            if self.shutdown_event.is_set():
                self._record_detail("HELPER EXIT: 130 stopped after process exit")
                return 130
            for line in proc.stdout.read().splitlines():
                self._record_detail(f"HELPER RAW: {line.rstrip()}")
                friendly = self._friendly_subprocess_line(line.rstrip())
                if friendly:
                    self.queue.put(("log", friendly))
            self._record_detail(f"HELPER EXIT: {proc.returncode}")
            return proc.returncode
        finally:
            self._unregister_process(proc)

    def _friendly_command_name(self, cmd):
        joined = " ".join(str(x) for x in cmd)
        if "fh6_probe.py" in joined and "--auto-locate" in joined:
            return tr(self.lang, "locating")
        if "main.py" in joined:
            return tr(self.lang, "importing")
        return "Starting helper..."

    def _check_json_layer_fit(self, json_path, layer_count):
        try:
            from generator_backend import geometry_shape_count
            json_layers = geometry_shape_count(json_path)
            template_layers = int(layer_count)
        except Exception:
            return
        usable_layers = max(0, template_layers - 4)
        if json_layers and template_layers and json_layers > usable_layers:
            self.queue.put(("log", f"{tr(self.lang, 'json_needs_more_template_layers')} JSON={json_layers}, template={template_layers}, usable={usable_layers}"))
        if json_layers and usable_layers and json_layers < usable_layers * 0.75:
            self.queue.put(("log", f"{tr(self.lang, 'json_too_small')} JSON={json_layers}, usable={usable_layers}"))

    def _friendly_subprocess_line(self, line):
        if not line:
            return None
        raw = line.strip()
        lower = raw.lower()
        noisy_parts = (
            "base:",
            "candidate score=",
            "layout candidate",
            "table[",
            "ptr=0x",
            "count=0x",
            "tablefield=",
            "wrote fh6 session location",
            "fh6 layout-count scan checked",
            "process: forzahorizon",
            "current values:",
            "loaded ",
            "descriptor @",
            "info found:",
            "vtp found:",
        )
        if any(part in lower for part in noisy_parts):
            return None
        if "fast fh6 layer group candidates:" in lower:
            return tr(self.lang, "located")
        if "no safe fh6 layer group" in lower:
            return tr(self.lang, "safe_stop")
        if "auto-locating fh6 layer count/table" in lower:
            return tr(self.lang, "locating")
        if "cliverylayer table found" in lower:
            return tr(self.lang, "located")
        if "forza horizon 6 detected" in lower:
            return raw
        if raw.startswith("Writing layer") or raw == "DONE!" or raw.startswith("The ideal background color"):
            return raw
        if "openprocess" in lower or "error" in lower or "failed" in lower or "traceback" in lower:
            return raw
        if raw.startswith("<class 'SystemExit'>") or raw.startswith("SystemExit: 0"):
            return None
        return raw

    def start_auto_locate(self):
        pid = self.ensure_live_game_pid()
        layer_count = self.layer_count.get().strip()
        if not pid or not layer_count:
            self.log_line("PID and template layer count are required.")
            return
        self.status.set(tr(self.lang, "running"))
        threading.Thread(target=self._auto_locate_worker, args=(pid, layer_count), daemon=True).start()

    def _auto_locate_worker(self, pid, layer_count):
        clear_session_location()
        cmd = [
            *helper_command("fh6_probe"),
            "--game",
            self.selected_game.get() or "fh6",
            "--pid",
            str(pid),
            "--layer-count",
            str(layer_count),
            "--auto-locate",
            "--write-session",
            SESSION_PATH,
            "--limit-mb",
            str(MEMORY_SNAPSHOT_LIMIT_MB),
            "--max-matches",
            "500000",
            "--inspect-radius",
            "0x800",
            "--max-seconds",
            str(FH6_AUTO_LOCATE_MAX_SECONDS),
        ]
        self.queue.put(("log", tr(self.lang, "locating_wait")))
        code = self.run_subprocess(cmd, timeout=FH6_AUTO_LOCATE_TIMEOUT_SECONDS)
        located = False
        if code == 0 and SESSION_PATH.exists():
            session = load_session_location()
            if session_matches_current_import(session, self.selected_game.get() or "fh6", pid, layer_count):
                self.queue.put(("log", tr(self.lang, "located")))
                located = True
        self.queue.put(("status", tr(self.lang, "done") if located else tr(self.lang, "failed")))
        return located

    def start_import(self):
        if not self.json_files:
            self.log_line("No JSON files selected.")
            return
        layer_count = self.layer_count.get().strip()
        if not layer_count:
            self.log_line(tr(self.lang, "layer_count_required"))
            self.layer_count_entry.config(highlightbackground="red", highlightthickness=1)
            return
        self.layer_count_entry.config(highlightbackground=Theme.BORDER, highlightthickness=0)
        pid = self.ensure_live_game_pid()
        if not pid:
            return
        self.status.set(tr(self.lang, "running"))
        threading.Thread(target=self._import_worker, args=(pid,), daemon=True).start()

    def _import_worker(self, pid):
        game = self.selected_game.get() or "fh6"
        count_address = parse_hex_or_empty(self.count_address.get())
        table_address = parse_hex_or_empty(self.table_address.get())
        layer_count = self.layer_count.get().strip()
        if not count_address and not table_address and game == "fh6":
            clear_session_location()
            if pid and layer_count:
                self.queue.put(("log", tr(self.lang, "locating")))
                located = self._auto_locate_worker(pid, layer_count)
                session = load_session_location()
                if located and session_matches_current_import(session, game, pid, layer_count):
                    count_address = "0x{:x}".format(int(session["count_address"]))
                    table_address = "0x{:x}".format(int(session["table_address"]))
                else:
                    self.queue.put(("status", tr(self.lang, "failed")))
                    return
        for path in list(self.json_files):
            if game == "fh6" and layer_count:
                self._check_json_layer_fit(path, layer_count)
            cmd = [*helper_command("main"), "--game", game, "--no-preview"]
            if pid:
                cmd.extend(["--pid", str(pid)])
            if count_address:
                cmd.extend(["--layer-count-address", count_address])
            if table_address:
                cmd.extend(["--layer-table-address", table_address])
            if game == "fh6" and layer_count:
                cmd.extend(["--layer-count-value", str(layer_count)])
            cmd.append(path)
            code = self.run_subprocess(cmd)
            if code != 0:
                self.queue.put(("status", tr(self.lang, "failed")))
                return
        self.queue.put(("status", tr(self.lang, "done")))

    def start_diagnose(self):
        pid = self.ensure_live_game_pid()
        cmd = [*helper_command("main"), "--game", self.selected_game.get() or "fh6", "--diagnose"]
        if pid:
            cmd.extend(["--pid", str(pid)])
        self.status.set(tr(self.lang, "running"))
        threading.Thread(target=lambda: self._run_command_worker(cmd, 120), daemon=True).start()

    def start_save_snapshot(self):
        pid = self.ensure_live_game_pid()
        count = self.snapshot_count.get().strip() or self.layer_count.get().strip()
        if not pid or not count:
            self.log_line("PID and snapshot layer count are required.")
            return
        output_path = PROBE_DIR / f"memory-count-{count}.jsonl"
        cmd = [
            *helper_command("fh6_probe"),
            "--game",
            self.selected_game.get() or "fh6",
            "--pid",
            str(pid),
            "--layer-count",
            str(count),
            "--save-memory-snapshot",
            output_path,
            "--limit-mb",
            str(MEMORY_SNAPSHOT_LIMIT_MB),
        ]
        self.status.set(tr(self.lang, "running"))
        threading.Thread(target=lambda: self._run_command_worker(cmd, 360), daemon=True).start()

    def start_compare_snapshot(self):
        pid = self.ensure_live_game_pid()
        previous = self.snapshot_count.get().strip()
        current = self.current_count.get().strip() or self.layer_count.get().strip()
        if not pid or not previous or not current:
            self.log_line("PID, snapshot layer count, and current layer count are required.")
            return
        snapshot_path = PROBE_DIR / f"memory-count-{previous}.jsonl"
        candidates_path = PROBE_DIR / f"memory-count-{previous}-to-{current}-candidates.json"
        intersect_path = PROBE_DIR / f"memory-count-{int(previous) - 1}-to-{previous}-candidates.json"
        cmd = [
            *helper_command("fh6_probe"),
            "--game",
            self.selected_game.get() or "fh6",
            "--pid",
            str(pid),
            "--layer-count",
            str(current),
            "--compare-memory-snapshot",
            snapshot_path,
            "--write-candidates",
            candidates_path,
            "--max-matches",
            "50000",
        ]
        if intersect_path.exists():
            cmd.extend(["--intersect-candidates", intersect_path])
        self.status.set(tr(self.lang, "running"))
        threading.Thread(target=lambda: self._run_command_worker(cmd, 360), daemon=True).start()

    def start_inspect_table(self):
        pid = self.ensure_live_game_pid()
        table = self.inspect_table_value.get().strip()
        count = self.layer_count.get().strip()
        if not pid or not table or not count:
            self.log_line("PID, layer count, and table address are required.")
            return
        cmd = [
            *helper_command("fh6_probe"),
            "--game",
            self.selected_game.get() or "fh6",
            "--pid",
            str(pid),
            "--layer-count",
            str(count),
            "--inspect-table",
            table,
            "--inspect-layers",
            "12",
        ]
        self.status.set(tr(self.lang, "running"))
        threading.Thread(target=lambda: self._run_command_worker(cmd, 60), daemon=True).start()

    def _run_command_worker(self, cmd, timeout):
        code = self.run_subprocess(cmd, timeout=timeout)
        self.queue.put(("status", tr(self.lang, "done") if code == 0 else tr(self.lang, "failed")))

    def _poll_queue(self):
        if self.closed:
            return
        while True:
            try:
                kind, payload = self.queue.get_nowait()
            except queue.Empty:
                break
            if kind == "log":
                self.log_line(payload)
            elif kind == "progress":
                self.progress_text.set(payload)
            elif kind == "status":
                self.status.set(payload)
            elif kind == "generation_done":
                stopped = self.shutdown_event.is_set()
                with self.generation_lock:
                    self.generation_running = False
                    self.current_generator_proc = None
                if hasattr(self, "generate_button"):
                    self.generate_button.config(state="normal")
                if hasattr(self, "stop_generate_button"):
                    self.stop_generate_button.config(state="disabled")
                if stopped and not self.closed:
                    self.progress_text.set(tr(self.lang, "generation_stopped"))
                    self.status.set(tr(self.lang, "stopped"))
                    self.log_line(tr(self.lang, "generation_stopped"))
            elif kind == "preview":
                self.show_preview(payload)
            elif kind == "preview_result":
                rid, data = payload
                if not self.closed and rid == self._preview_request_id:
                    self.show_preview(data)
            elif kind == "preview_json":
                self.show_json_preview(payload)
            elif kind == "preview_file":
                self.show_preview_file(payload)
            elif kind == "render_lists":
                self._render_lists()
            elif kind == "update_failed":
                self._handle_update_failed(payload)
            elif kind == "update_current":
                self._handle_update_current(payload)
            elif kind == "update_available":
                self._handle_update_available(payload)
        if not self.closed:
            self.root.after(100, self._poll_queue)

    def run(self):
        self.root.mainloop()


def main():
    if len(sys.argv) >= 3 and sys.argv[1] == "--helper":
        run_embedded_helper(sys.argv[2], sys.argv[3:])
        return
    parser = argparse.ArgumentParser(description=f"Standalone {APP_DISPLAY_NAME} desktop app.")
    parser.add_argument("--version", action="version", version=f"{APP_DISPLAY_NAME} {__version__}")
    parser.add_argument("images", nargs="*", help="Optional image files to preload.")
    args = parser.parse_args()
    App(args.images).run()


if __name__ == "__main__":
    main()
