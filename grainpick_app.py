"""
GrainPick App — Desktop application for thin section grain extraction
=====================================================================
Run:
    python grainpick_app.py

No terminal commands needed after launch.
"""

import os
import sys
import json
import time
import threading
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import cv2
import numpy as np
from PIL import Image, ImageTk

# ── SAM import ───────────────────────────────────────────────────────────────
try:
    import torch
    from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
    SAM_AVAILABLE = True
except ImportError:
    SAM_AVAILABLE = False

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_CHECKPOINT = "./checkpoints/sam_vit_h_4b8939.pth"
MODEL_TYPE         = "vit_h"
OUTPUT_DIR = str(Path(__file__).parent / "grains")
EXPORT_SIZE        = (1080, 1080)
EXPORT_PADDING     = 6

SAM_AUTO_CONFIG = dict(
    points_per_side=16,          # reduced from 32 — less CPU, still good coverage
    pred_iou_thresh=0.88,
    stability_score_thresh=0.92,
    crop_n_layers=0,             # disabled crop layers — major CPU saving
    crop_n_points_downscale_factor=2,
    min_mask_region_area=800,    # larger minimum — filters tiny fragments
)

MINERALS = [
    "Quartz", "Feldspar", "Plagioclase", "K-Feldspar",
    "Olivine", "Pyroxene", "Amphibole", "Biotite",
    "Muscovite", "Calcite", "Dolomite", "Hornblende",
    "Garnet", "Epidote", "Chlorite", "Magnetite",
    "Pyrite", "Zircon", "Apatite", "Titanite",
    "Tourmaline", "Kyanite", "Sillimanite", "Andalusite",
    "Staurolite",
]

COLORS = {
    "auto":     (220, 180, 60),
    "active":   (60, 220, 255),
    "accepted": (60, 220, 60),
    "rejected": (80,  80, 200),
    "pos_pt":   (0,  255,   0),
    "neg_pt":   (0,    0, 255),
}

# ─────────────────────────────────────────────────────────────────────────────
# Data
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Grain:
    mask:     np.ndarray
    bbox:     tuple
    grain_id: int
    source:   str
    accepted: bool = False
    rejected: bool = False

    @property
    def area(self):
        return int(self.mask.sum())


@dataclass
class LoadedImage:
    path:    str
    mode:    str        # "PPL" or "XPL"
    name:    str        # display name
    grains:  list = None
    image_rgb: np.ndarray = None

    def __post_init__(self):
        self.grains = self.grains or []


# ─────────────────────────────────────────────────────────────────────────────
# Segmentation backend (same engine as grainpick.py)
# ─────────────────────────────────────────────────────────────────────────────

class SegmentationBackend:
    def __init__(self, checkpoint: str, device: str = "cpu"):
        self.predictor      = None
        self.auto_generator = None
        self.device         = device

        if SAM_AVAILABLE and os.path.exists(checkpoint):
            sam = sam_model_registry[MODEL_TYPE](checkpoint=checkpoint)
            sam.to(device=device)
            self.predictor      = SamPredictor(sam)
            self.auto_generator = SamAutomaticMaskGenerator(sam, **SAM_AUTO_CONFIG)

    def auto_segment(self, image_rgb: np.ndarray) -> list:
        if self.auto_generator:
            image_rgb = image_rgb.astype("uint8")
            masks_data = self.auto_generator.generate(image_rgb)
            grains = []
            for i, m in enumerate(masks_data):
                mask = m["segmentation"]
                x, y, w, h = m["bbox"]
                grains.append(Grain(
                    mask=mask.astype(bool),
                    bbox=(int(x), int(y), int(w), int(h)),
                    grain_id=i, source="auto",
                ))
            return grains
        return self._watershed(image_rgb)

    def _watershed(self, image_rgb: np.ndarray) -> list:
        gray   = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
        blur   = cv2.GaussianBlur(gray, (5, 5), 0)
        thresh = cv2.adaptiveThreshold(blur, 255,
                     cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                     cv2.THRESH_BINARY_INV, 15, 3)
        kernel  = np.ones((3, 3), np.uint8)
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
        sure_bg = cv2.dilate(opening, kernel, iterations=3)
        dist    = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
        _, sure_fg = cv2.threshold(dist, 0.4 * dist.max(), 255, 0)
        sure_fg    = sure_fg.astype(np.uint8)
        unknown    = cv2.subtract(sure_bg, sure_fg)
        _, markers = cv2.connectedComponents(sure_fg)
        markers += 1
        markers[unknown == 255] = 0
        img_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        markers = cv2.watershed(img_bgr, markers)
        grains  = []
        for lid in np.unique(markers):
            if lid <= 1: continue
            mask = (markers == lid)
            if mask.sum() < SAM_AUTO_CONFIG["min_mask_region_area"]: continue
            ys, xs = np.where(mask)
            x, y   = int(xs.min()), int(ys.min())
            w, h   = int(xs.max() - x), int(ys.max() - y)
            grains.append(Grain(mask=mask, bbox=(x, y, w, h),
                                grain_id=len(grains), source="auto"))
        return grains

    def predict_from_points(self, image_rgb, pos_pts, neg_pts):
        if not self.predictor or not pos_pts:
            return None
        self.predictor.set_image(image_rgb.astype("uint8"))
        pts  = np.array(pos_pts + neg_pts)
        lbs  = np.array([1]*len(pos_pts) + [0]*len(neg_pts))
        masks, scores, _ = self.predictor.predict(
            point_coords=pts, point_labels=lbs, multimask_output=True)
        return masks[np.argmax(scores)].astype(bool)


# ─────────────────────────────────────────────────────────────────────────────
# Exporter — organised folder structure
# ─────────────────────────────────────────────────────────────────────────────

class GrainExporter:
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.manifest   = []
        self.counters   = {}   # (mineral, mode) -> count

    def _next_id(self, mineral: str, mode: str) -> int:
        key = (mineral, mode)
        self.counters[key] = self.counters.get(key, 0) + 1
        return self.counters[key]

    def export_grain(self, grain: Grain, image_rgb: np.ndarray,
                     mineral: str, mode: str) -> str:
        """Save grain to grains/Mineral/PPL|XPL/Mineral_PPL_NNNN.png"""
        H, W   = image_rgb.shape[:2]
        x, y, w, h = grain.bbox
        pad    = EXPORT_PADDING
        x1, y1 = max(0, x - pad), max(0, y - pad)
        x2, y2 = min(W, x + w + pad), min(H, y + h + pad)

        crop_rgb  = image_rgb[y1:y2, x1:x2]
        crop_mask = grain.mask[y1:y2, x1:x2]

        # Black background, grain pixels only
        result = np.zeros_like(crop_rgb)
        result[crop_mask] = crop_rgb[crop_mask]

        pil = Image.fromarray(result, "RGB")
        pil = pil.resize(EXPORT_SIZE, Image.LANCZOS)

        # Build path: grains/Olivine/PPL/
        folder = self.output_dir / mineral / mode
        folder.mkdir(parents=True, exist_ok=True)

        n     = self._next_id(mineral, mode)
        fname = f"{mineral}_{mode}_{n:04d}.png"
        fpath = folder / fname
        pil.save(fpath)

        self.manifest.append({
            "file":     str(fpath),
            "mineral":  mineral,
            "mode":     mode,
            "grain_id": grain.grain_id,
            "source":   grain.source,
            "area_px":  grain.area,
        })
        return str(fpath)

    def save_manifest(self):
        mpath = self.output_dir / "manifest.json"
        with open(mpath, "w") as f:
            json.dump(self.manifest, f, indent=2)

    def total_saved(self):
        return sum(self.counters.values())

    def count_for(self, mineral, mode):
        return self.counters.get((mineral, mode), 0)


# ─────────────────────────────────────────────────────────────────────────────
# PPL / XPL popup dialog
# ─────────────────────────────────────────────────────────────────────────────

class ModeDialog(tk.Toplevel):
    def __init__(self, parent, filename):
        super().__init__(parent)
        self.result = None
        self.title("Image mode")
        self.resizable(False, False)
        self.grab_set()
        self.configure(bg="#161b22")

        tk.Label(self, text="What mode is this image?",
                 font=("Helvetica", 13, "bold"),
                 fg="#e6edf3", bg="#161b22", pady=10).pack()
        tk.Label(self, text=filename,
                 font=("Helvetica", 11), fg="#8b949e",
                 bg="#161b22", pady=2).pack()

        btn_frame = tk.Frame(self, pady=16, bg="#161b22")
        btn_frame.pack()

        def make_btn(parent, text, color, pick_val):
            f = tk.Frame(parent, bg=color, padx=2, pady=2)
            f.pack(side="left", padx=10)
            lbl = tk.Label(f, text=text, width=10,
                           font=("Helvetica", 13, "bold"),
                           bg=color, fg="#ffffff",
                           padx=20, pady=12, cursor="hand2")
            lbl.pack()
            lbl.bind("<Button-1>", lambda e: self._pick(pick_val))
            lbl.bind("<Enter>",    lambda e: lbl.config(bg=self._hl(color)))
            lbl.bind("<Leave>",    lambda e: lbl.config(bg=color))
            f.bind("<Button-1>",   lambda e: self._pick(pick_val))

        make_btn(btn_frame, "PPL", "#1f6feb", "PPL")
        make_btn(btn_frame, "XPL", "#238636", "XPL")

        self.protocol("WM_DELETE_WINDOW", lambda: self._pick(None))
        self.update_idletasks()
        # Centre on parent
        x = parent.winfo_x() + (parent.winfo_width()  - self.winfo_width())  // 2
        y = parent.winfo_y() + (parent.winfo_height() - self.winfo_height()) // 2
        self.geometry(f"+{x}+{y}")
        self.wait_window()

    def _hl(self, color):
        try:
            h = color.lstrip("#")
            r,g,b = int(h[0:2],16), int(h[2:4],16), int(h[4:6],16)
            return f"#{min(255,r+40):02x}{min(255,g+40):02x}{min(255,b+40):02x}"
        except:
            return color

    def _pick(self, val):
        self.result = val
        self.destroy()


# ─────────────────────────────────────────────────────────────────────────────
# Main Application
# ─────────────────────────────────────────────────────────────────────────────

class GrainPickApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("GrainPick")
        self.geometry("1280x820")
        self.configure(bg="#1a1a1a")

        # State
        self.loaded_images: list[LoadedImage] = []
        self.current_img_idx  = -1
        self.current_grain_idx = 0
        self.pos_points       = []
        self.neg_points       = []
        self.active_mask      = None
        self.manual_mode      = False
        self.group_mode       = False
        self.grouped_indices  = set()   # indices of grains selected for grouping
        self.zoom             = 1.0
        self.pan_x            = 0.0
        self.pan_y            = 0.0
        self._drag_start      = None
        self._tk_image        = None

        # Backend & exporter
        self.backend  = None
        self.exporter = GrainExporter(OUTPUT_DIR)
        self._render_cache = None   # cached blended image for fast redraws
        self._cache_key    = None   # (img_idx, grain_idx, n_grains) — invalidate when changed

        self._build_ui()
        self._load_sam_async()

    def _btn(self, parent, text, bg, fg, command, font=("Helvetica", 11),
             pady=6, padx=6, bold=False, store=None):
        """Mac-safe button using Label — bypasses macOS colour override on tk.Button."""
        fnt = (font[0], font[1], "bold") if bold else font
        lbl = tk.Label(parent, text=text, bg=bg, fg=fg,
                       font=fnt, cursor="hand2",
                       padx=padx, pady=pady)
        lbl.bind("<Button-1>", lambda e: command())
        lbl.bind("<Enter>",    lambda e: lbl.config(bg=self._lighten(bg)))
        lbl.bind("<Leave>",    lambda e: lbl.config(bg=bg))
        if store:
            setattr(self, store, lbl)
        return lbl

    def _lighten(self, hex_color):
        try:
            h = hex_color.lstrip("#")
            r, g, b = int(h[0:2],16), int(h[2:4],16), int(h[4:6],16)
            r, g, b = min(255,r+30), min(255,g+30), min(255,b+30)
            return f"#{r:02x}{g:02x}{b:02x}"
        except Exception:
            return hex_color

    # ── UI Build ──────────────────────────────────────────────────────────────

    def _build_ui(self):
        # ── Colour palette ──
        self.C = {
            "bg":        "#0d1117",   # darkest background
            "panel":     "#161b22",   # sidebar / panel bg
            "toolbar":   "#1c2128",   # toolbar bg
            "border":    "#30363d",   # separator / border
            "green":     "#3fb950",   # primary green accent
            "green_dk":  "#238636",   # darker green (save button)
            "green_hi":  "#56d364",   # bright green (hover / highlight)
            "red":       "#da3633",   # reject / danger
            "red_dk":    "#b91c1c",   # darker red
            "amber":     "#d29922",   # warnings / mode label
            "blue":      "#1f6feb",   # auto-detect button
            "blue_lt":   "#388bfd",   # lighter blue
            "muted":     "#8b949e",   # muted text
            "text":      "#e6edf3",   # primary text
            "input_bg":  "#21262d",   # input / entry bg
            "selected":  "#1f4e2e",   # listbox selection
        }
        C = self.C

        self.configure(bg=C["bg"])

        # ── Top bar ──
        topbar = tk.Frame(self, bg=C["panel"], height=48)
        topbar.pack(fill="x", side="top")
        topbar.pack_propagate(False)

        tk.Label(topbar, text="⬡ GrainPick", fg=C["green"], bg=C["panel"],
                 font=("Helvetica", 15, "bold")).pack(side="left", padx=16)

        self.sam_status = tk.Label(topbar, text="⏳ Loading SAM...",
                                   fg=C["amber"], bg=C["panel"],
                                   font=("Helvetica", 11))
        self.sam_status.pack(side="left", padx=8)

        self._btn(topbar, "＋  Add Images", C["green_dk"], "#ffffff",
                  self._add_images, font=("Helvetica",11),
                  bold=True, padx=14, pady=6).pack(side="right", padx=16, pady=10)

        # ── Main layout ──
        main = tk.Frame(self, bg=C["bg"])
        main.pack(fill="both", expand=True)

        self._build_sidebar(main)
        self._build_canvas(main)
        self._build_right_panel(main)

    def _build_sidebar(self, parent):
        C = self.C
        sidebar = tk.Frame(parent, bg=C["panel"], width=210)
        sidebar.pack(side="left", fill="y")
        sidebar.pack_propagate(False)

        # Header
        hdr = tk.Frame(sidebar, bg=C["toolbar"], height=32)
        hdr.pack(fill="x")
        hdr.pack_propagate(False)
        tk.Label(hdr, text="LOADED IMAGES", fg=C["muted"], bg=C["toolbar"],
                 font=("Helvetica", 9, "bold")).pack(side="left", padx=12, pady=8)

        # Scrollable image list with close buttons
        list_frame = tk.Frame(sidebar, bg=C["panel"])
        list_frame.pack(fill="both", expand=True, padx=4, pady=4)

        scrollbar = tk.Scrollbar(list_frame, bg=C["panel"],
                                  troughcolor=C["bg"], width=8)
        scrollbar.pack(side="right", fill="y")

        self.img_canvas = tk.Canvas(list_frame, bg=C["panel"],
                                     highlightthickness=0,
                                     yscrollcommand=scrollbar.set)
        self.img_canvas.pack(side="left", fill="both", expand=True)
        scrollbar.config(command=self.img_canvas.yview)

        self.img_list_inner = tk.Frame(self.img_canvas, bg=C["panel"])
        self.img_canvas.create_window((0,0), window=self.img_list_inner, anchor="nw")
        self.img_list_inner.bind("<Configure>",
            lambda e: self.img_canvas.configure(
                scrollregion=self.img_canvas.bbox("all")))

        # Divider
        tk.Frame(sidebar, bg=C["border"], height=1).pack(fill="x")

        # Stats
        stats_frame = tk.Frame(sidebar, bg=C["panel"])
        stats_frame.pack(fill="x", pady=10, padx=12)
        tk.Label(stats_frame, text="TOTAL SAVED", fg=C["muted"],
                 bg=C["panel"], font=("Helvetica", 9, "bold")).pack(anchor="w")
        self.total_label = tk.Label(stats_frame, text="0",
                                    fg=C["green"], bg=C["panel"],
                                    font=("Helvetica", 28, "bold"))
        self.total_label.pack(anchor="w")
        tk.Label(stats_frame, text="grains", fg=C["muted"],
                 bg=C["panel"], font=("Helvetica", 10)).pack(anchor="w")

    def _build_canvas(self, parent):
        C = self.C
        canvas_frame = tk.Frame(parent, bg=C["bg"])
        canvas_frame.pack(side="left", fill="both", expand=True)

        # Toolbar
        toolbar = tk.Frame(canvas_frame, bg=C["toolbar"], height=44)
        toolbar.pack(fill="x")
        toolbar.pack_propagate(False)

        self._btn(toolbar, "⚡  Auto-detect", C["blue"], "#ffffff",
                  lambda: self._run_auto_segment(), font=("Helvetica",11),
                  bold=True, padx=12, pady=8,
                  store="auto_btn").pack(side="left", padx=8, pady=8)

        self._btn(toolbar, "☝  Click to pick", C["input_bg"], C["text"],
                  lambda: self._toggle_manual(), font=("Helvetica",11),
                  padx=10, pady=8,
                  store="manual_btn").pack(side="left", padx=4, pady=8)

        self._btn(toolbar, "⊞  Group grains", C["input_bg"], C["text"],
                  lambda: self._toggle_group_mode(), font=("Helvetica",11),
                  padx=10, pady=8,
                  store="group_btn").pack(side="left", padx=4, pady=8)

        # Zoom buttons on right
        for label, factor in [("＋ Zoom", 1.2), ("－ Zoom", 1/1.2), ("⊙ Reset", None)]:
            cmd = self._reset_zoom if factor is None else lambda f=factor: self._zoom(f)
            self._btn(toolbar, label, C["input_bg"], C["text"],
                      cmd, font=("Helvetica",11),
                      padx=10, pady=8).pack(side="right", padx=4, pady=8)

        # Canvas
        self.canvas = tk.Canvas(canvas_frame, bg="#0a0c0f",
                                 cursor="crosshair", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<ButtonPress-1>",   self._on_left_click)
        self.canvas.bind("<ButtonPress-3>",   self._on_right_click)
        self.canvas.bind("<ButtonPress-2>",   self._pan_start)
        self.canvas.bind("<B2-Motion>",       self._pan_move)
        self.canvas.bind("<MouseWheel>",      self._on_scroll)
        self.canvas.bind("<Shift-MouseWheel>",self._on_scroll_x)

        # Status bar
        status_frame = tk.Frame(canvas_frame, bg=C["toolbar"], height=28)
        status_frame.pack(fill="x")
        status_frame.pack_propagate(False)
        self.status_bar = tk.Label(status_frame, text="Load an image to begin",
                                   bg=C["toolbar"], fg=C["muted"],
                                   font=("Helvetica", 10), anchor="w", padx=12)
        self.status_bar.pack(fill="both", expand=True)

    def _build_right_panel(self, parent):
        C = self.C
        panel = tk.Frame(parent, bg=C["panel"], width=230)
        panel.pack(side="right", fill="y")
        panel.pack_propagate(False)

        def section_header(text):
            hdr = tk.Frame(panel, bg=C["toolbar"], height=28)
            hdr.pack(fill="x")
            hdr.pack_propagate(False)
            tk.Label(hdr, text=text, fg=C["muted"], bg=C["toolbar"],
                     font=("Helvetica", 9, "bold")).pack(side="left", padx=12, pady=6)

        def divider():
            tk.Frame(panel, bg=C["border"], height=1).pack(fill="x", pady=2)

        # Mode indicator
        section_header("IMAGE MODE")
        mode_frame = tk.Frame(panel, bg=C["panel"])
        mode_frame.pack(fill="x", padx=12, pady=8)
        self.mode_label = tk.Label(mode_frame, text="— None loaded —",
                                   fg=C["amber"], bg=C["panel"],
                                   font=("Helvetica", 14, "bold"))
        self.mode_label.pack(anchor="w")

        divider()

        # Grain navigation
        section_header("GRAIN NAVIGATION")
        nav_frame = tk.Frame(panel, bg=C["panel"])
        nav_frame.pack(fill="x", padx=12, pady=8)

        nav_btns = tk.Frame(nav_frame, bg=C["panel"])
        nav_btns.pack(fill="x")
        self._btn(nav_btns, "← Prev", C["input_bg"], C["text"],
                  self._prev_grain, font=("Helvetica",11),
                  padx=6, pady=8).pack(side="left", expand=True, fill="x", padx=(0,2))
        self._btn(nav_btns, "Next →", C["input_bg"], C["text"],
                  self._next_grain, font=("Helvetica",11),
                  padx=6, pady=8).pack(side="right", expand=True, fill="x", padx=(2,0))

        self.grain_counter = tk.Label(panel, text="0 / 0",
                                      fg=C["green"], bg=C["panel"],
                                      font=("Helvetica", 12, "bold"))
        self.grain_counter.pack(pady=4)

        divider()

        # Mineral selector
        section_header("MINERAL LABEL")
        mineral_frame = tk.Frame(panel, bg=C["panel"])
        mineral_frame.pack(fill="x", padx=12, pady=8)

        tk.Label(mineral_frame, text="Select mineral:", fg=C["muted"],
                 bg=C["panel"], font=("Helvetica", 10)).pack(anchor="w", pady=(0,4))

        self.mineral_var = tk.StringVar(value=MINERALS[0])
        style = ttk.Style()
        style.theme_use("default")
        style.configure("Green.TCombobox",
                        fieldbackground=C["input_bg"],
                        background=C["input_bg"],
                        foreground=C["text"],
                        selectbackground=C["selected"],
                        selectforeground=C["green_hi"],
                        arrowcolor=C["green"])
        mineral_menu = ttk.Combobox(mineral_frame,
                                    textvariable=self.mineral_var,
                                    values=MINERALS,
                                    font=("Helvetica", 12),
                                    style="Green.TCombobox",
                                    state="readonly")
        mineral_menu.pack(fill="x", pady=(0, 8))

        tk.Label(mineral_frame, text="Or type new mineral:", fg=C["muted"],
                 bg=C["panel"], font=("Helvetica", 10)).pack(anchor="w", pady=(0,4))
        self.custom_mineral = tk.Entry(mineral_frame,
                                       font=("Helvetica", 12),
                                       bg=C["input_bg"], fg=C["text"],
                                       insertbackground=C["green"],
                                       relief="flat", bd=4)
        self.custom_mineral.pack(fill="x")

        divider()

        # Action buttons
        section_header("ACTIONS")
        action_frame = tk.Frame(panel, bg=C["panel"])
        action_frame.pack(fill="x", padx=12, pady=10)

        self._btn(action_frame, "✓   Save Grain", C["green_dk"], "#ffffff",
                  self._save_grain, font=("Helvetica",13),
                  bold=True, padx=6, pady=12).pack(fill="x", pady=(0,6))

        self._btn(action_frame, "✕   Reject Grain", C["red_dk"], "#ffffff",
                  self._reject_grain, font=("Helvetica",12),
                  padx=6, pady=10).pack(fill="x", pady=(0,4))

        self._btn(action_frame, "↺   Clear Clicks", C["input_bg"], C["text"],
                  self._clear_manual_state, font=("Helvetica",11),
                  padx=6, pady=8).pack(fill="x")

        divider()

        # Session stats
        section_header("SESSION")
        stats_frame = tk.Frame(panel, bg=C["panel"])
        stats_frame.pack(fill="x", padx=12, pady=8)
        tk.Label(stats_frame, text="Saved this session:",
                 fg=C["muted"], bg=C["panel"],
                 font=("Helvetica", 10)).pack(anchor="w")
        self.session_label = tk.Label(stats_frame, text="0 grains",
                                      fg=C["green"], bg=C["panel"],
                                      font=("Helvetica", 16, "bold"))
        self.session_label.pack(anchor="w", pady=2)

        self._btn(panel, "📂  Open Output Folder", C["input_bg"], C["green_hi"],
                  self._open_output_folder, font=("Helvetica",10),
                  padx=6, pady=8).pack(fill="x", padx=12, pady=8)

    # ── SAM loading ───────────────────────────────────────────────────────────

    def _load_sam_async(self):
        def _load():
            try:
                self.backend = SegmentationBackend(DEFAULT_CHECKPOINT, device="cpu")
                if self.backend.auto_generator:
                    self.after(0, lambda: self.sam_status.config(
                        text="✓ SAM ready", fg=self.C["green"]))
                    self.after(0, lambda: self.auto_btn.config(bg=self.C["blue"]))
                else:
                    self.after(0, lambda: self.sam_status.config(
                        text="⚠ SAM not found — using fallback", fg=self.C["amber"]))
                    self.after(0, lambda: self.auto_btn.config(bg=self.C["blue"]))
            except Exception as e:
                self.after(0, lambda: self.sam_status.config(
                    text=f"✕ Error: {e}", fg="#f87171"))
        threading.Thread(target=_load, daemon=True).start()

    # ── Image management ──────────────────────────────────────────────────────

    def _rebuild_img_list(self):
        """Rebuild the sidebar image list with close buttons."""
        C = self.C
        for w in self.img_list_inner.winfo_children():
            w.destroy()

        for i, li in enumerate(self.loaded_images):
            is_active = (i == self.current_img_idx)
            row_bg = C["selected"] if is_active else C["panel"]

            row = tk.Frame(self.img_list_inner, bg=row_bg, pady=4)
            row.pack(fill="x", padx=2, pady=1)

            # Mode badge
            mode_col = C["blue_lt"] if li.mode == "PPL" else C["green_hi"]
            tk.Label(row, text=li.mode, bg=mode_col, fg="#000000",
                     font=("Helvetica", 8, "bold"),
                     padx=4, pady=1).pack(side="left", padx=(4,4))

            # Filename — click to select
            name = li.name[:18] + "…" if len(li.name) > 20 else li.name
            lbl = tk.Label(row, text=name, bg=row_bg,
                           fg=C["green_hi"] if is_active else C["text"],
                           font=("Helvetica", 10), cursor="hand2", anchor="w")
            lbl.pack(side="left", fill="x", expand=True)
            lbl.bind("<Button-1>", lambda e, idx=i: self._switch_image(idx))
            row.bind("<Button-1>",  lambda e, idx=i: self._switch_image(idx))

            # Close ✕ button
            close = tk.Label(row, text="✕", bg=row_bg,
                             fg=C["muted"], font=("Helvetica", 10),
                             cursor="hand2", padx=6)
            close.pack(side="right")
            close.bind("<Button-1>", lambda e, idx=i: self._remove_image(idx))
            close.bind("<Enter>",    lambda e, w=close: w.config(fg="#f87171"))
            close.bind("<Leave>",    lambda e, w=close, bg=row_bg: w.config(fg=C["muted"]))

    def _remove_image(self, idx):
        """Remove an image from the loaded list."""
        self.loaded_images.pop(idx)
        # Adjust current index
        if not self.loaded_images:
            self.current_img_idx = -1
            self.canvas.delete("all")
            self._update_status("Load an image to begin")
        elif self.current_img_idx >= len(self.loaded_images):
            self.current_img_idx = len(self.loaded_images) - 1
            self._switch_image(self.current_img_idx)
        elif self.current_img_idx == idx:
            self._switch_image(self.current_img_idx)
        self._rebuild_img_list()

    def _add_images(self):
        paths = filedialog.askopenfilenames(
            title="Select thin section images",
            filetypes=[("Images", "*.jpg *.jpeg *.png *.tif *.tiff *.bmp"), ("All", "*.*")]
        )
        for path in paths:
            fname  = Path(path).name
            dialog = ModeDialog(self, fname)
            mode   = dialog.result
            if mode is None:
                continue
            img_rgb = np.array(Image.open(path).convert("RGB")).astype("uint8")
            li = LoadedImage(path=path, mode=mode, name=fname, image_rgb=img_rgb)
            self.loaded_images.append(li)

        if self.loaded_images and self.current_img_idx == -1:
            self.current_img_idx = 0
            self._switch_image(0)

        self._rebuild_img_list()

    def _on_img_select(self, event):
        pass  # handled by row click in _rebuild_img_list

    def _switch_image(self, idx):
        self.current_img_idx   = idx
        self.current_grain_idx = 0
        self._clear_manual_state()
        self._render_cache = None
        self._cache_key    = None
        li = self.loaded_images[idx]
        self.mode_label.config(
            text=li.mode,
            fg=self.C["blue_lt"] if li.mode == "PPL" else self.C["green_hi"]
        )
        self.zoom  = 1.0
        self.pan_x = 0.0
        self.pan_y = 0.0
        self._update_status(f"Loaded: {li.name}  [{li.mode}]  —  Press Auto-detect to segment grains")
        self._rebuild_img_list()
        self._draw()

    # ── Auto segmentation ─────────────────────────────────────────────────────

    def _run_auto_segment(self):
        if self.current_img_idx < 0:
            messagebox.showinfo("No image", "Please load an image first.")
            return
        li = self.loaded_images[self.current_img_idx]
        self.auto_btn.config(text="⏳ Segmenting...")
        self._update_status("Running SAM segmentation — this takes 30–90s, please wait...")

        def _work():
            grains = self.backend.auto_segment(li.image_rgb)
            li.grains = grains
            self.after(0, self._on_segment_done)

        threading.Thread(target=_work, daemon=True).start()

    def _on_segment_done(self):
        li = self.loaded_images[self.current_img_idx]
        self.current_grain_idx = 0
        self.auto_btn.config(text="⚡  Auto-detect")
        self._update_status(f"Found {len(li.grains)} grains. Use ← → to navigate. Save or reject each.")
        self._render_cache = None
        self._draw()
        self._update_grain_counter()

    # ── Manual mode ───────────────────────────────────────────────────────────

    def _toggle_group_mode(self):
        self.group_mode = not self.group_mode
        self.grouped_indices.clear()
        if self.group_mode:
            self.group_btn.config(bg=self.C["amber"], fg="#000000")
            self.group_btn.config(text="⊞  Group ON")
            self._update_status("Group mode: click grains to select them, then Save Grain to merge & export as one")
        else:
            self.group_btn.config(bg=self.C["input_bg"], fg=self.C["text"])
            self.group_btn.config(text="⊞  Group grains")
            self._update_status("Group mode off")
        self._render_cache = None
        self._draw()
        if self.manual_mode:
            self.manual_btn.config(bg=self.C["green_dk"], fg="#ffffff")
            self.manual_btn.config(text="☝  Manual ON")
            self._update_status("Manual: left-click grain centre → boundary appears instantly. Right-click to exclude regions. Save when happy.")
        else:
            self._clear_manual_state()
            self.manual_btn.config(bg=self.C["input_bg"], fg=self.C["text"])
            self.manual_btn.config(text="☝  Click to pick")

    def _clear_manual_state(self):
        self.pos_points  = []
        self.neg_points  = []
        self.active_mask = None
        self._draw()

    # ── Canvas drawing ────────────────────────────────────────────────────────

    def _draw(self):
        if self.current_img_idx < 0:
            return
        li = self.loaded_images[self.current_img_idx]
        img = li.image_rgb
        H, W = img.shape[:2]

        # ── Cache the base overlay (all grain outlines + current highlight) ──
        # Only recompute when grains or active grain changes, not on every zoom/pan
        cache_key = (
            self.current_img_idx,
            self.current_grain_idx,
            len(li.grains),
            tuple(g.accepted for g in li.grains),
            tuple(g.rejected for g in li.grains),
            self.active_mask is not None,
            len(self.pos_points),
            len(self.neg_points),
            frozenset(self.grouped_indices),
        )

        if cache_key != self._cache_key:
            overlay = img.copy()

            # Draw all grain outlines
            for i, g in enumerate(li.grains):
                if i in self.grouped_indices:
                    col = (220, 140, 20)   # amber = grouped
                elif g.accepted:   col = COLORS["accepted"]
                elif g.rejected:   col = COLORS["rejected"]
                else:              col = COLORS["auto"]
                m = g.mask.astype(np.uint8) * 255
                cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(overlay, cnts, -1, col, 1)

            # Highlight current grain
            if li.grains and 0 <= self.current_grain_idx < len(li.grains):
                if not (self.pos_points or self.neg_points) and self.active_mask is None:
                    g = li.grains[self.current_grain_idx]
                    m = g.mask.astype(np.uint8) * 255
                    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    cv2.fillPoly(overlay, cnts, COLORS["active"])
                    cv2.drawContours(overlay, cnts, -1, (255, 255, 255), 2)

            # Manual mask
            if self.active_mask is not None:
                m = self.active_mask.astype(np.uint8) * 255
                cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.fillPoly(overlay, cnts, COLORS["active"])
                cv2.drawContours(overlay, cnts, -1, (255, 255, 255), 2)

            # Click points
            for px, py in self.pos_points:
                cv2.circle(overlay, (px, py), 6, COLORS["pos_pt"], -1)
                cv2.circle(overlay, (px, py), 6, (255, 255, 255), 1)
            for px, py in self.neg_points:
                cv2.circle(overlay, (px, py), 6, COLORS["neg_pt"], -1)
                cv2.circle(overlay, (px, py), 6, (255, 255, 255), 1)

            self._render_cache = cv2.addWeighted(overlay, 0.45, img, 0.55, 0)
            self._cache_key    = cache_key

        blended = self._render_cache

        # ── Zoom & pan (fast — just crops the cached render) ──
        cw = self.canvas.winfo_width()  or 800
        ch = self.canvas.winfo_height() or 600
        new_w = max(1, int(W * self.zoom))
        new_h = max(1, int(H * self.zoom))
        big   = cv2.resize(blended, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # pan_x/pan_y are in image space — convert to screen space for cropping
        ox = int(self.pan_x * self.zoom)
        oy = int(self.pan_y * self.zoom)
        ox = max(0, min(ox, max(0, new_w - cw)))
        oy = max(0, min(oy, max(0, new_h - ch)))

        crop = big[oy:oy+ch, ox:ox+cw]
        if crop.shape[0] < ch or crop.shape[1] < cw:
            canvas_arr = np.zeros((ch, cw, 3), dtype=np.uint8)
            canvas_arr[:crop.shape[0], :crop.shape[1]] = crop
            crop = canvas_arr

        rgb_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        pil_img  = Image.fromarray(rgb_crop)
        self._tk_image = ImageTk.PhotoImage(pil_img)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor="nw", image=self._tk_image)

    # ── Mouse events ──────────────────────────────────────────────────────────

    def _canvas_to_img(self, cx, cy):
        """Convert canvas pixel coordinates to original image coordinates."""
        # pan_x/pan_y are in image-space pixels
        ix = int(cx / self.zoom + self.pan_x)
        iy = int(cy / self.zoom + self.pan_y)
        return ix, iy

    def _img_to_canvas(self, ix, iy):
        """Convert image coordinates back to canvas coordinates (for dot feedback)."""
        cx = int((ix - self.pan_x) * self.zoom)
        cy = int((iy - self.pan_y) * self.zoom)
        return cx, cy

    def _draw_click_dot(self, cx, cy, color):
        """Draw a visible dot on the canvas at the click position."""
        r = 6
        self.canvas.create_oval(cx-r, cy-r, cx+r, cy+r,
                                 fill=color, outline="white", width=1)

    def _on_left_click(self, event):
        if self.current_img_idx < 0: return
        ix, iy = self._canvas_to_img(event.x, event.y)
        li = self.loaded_images[self.current_img_idx]
        H, W = li.image_rgb.shape[:2]
        if not (0 <= ix < W and 0 <= iy < H):
            self._update_status("Click landed outside image — try zooming in first")
            return

        if self.group_mode:
            for i, g in enumerate(li.grains):
                if 0 <= iy < g.mask.shape[0] and 0 <= ix < g.mask.shape[1]:
                    if g.mask[iy, ix]:
                        if i in self.grouped_indices:
                            self.grouped_indices.discard(i)
                        else:
                            self.grouped_indices.add(i)
                        n = len(self.grouped_indices)
                        self._update_status(f"Group mode: {n} grain{'s' if n!=1 else ''} selected — press Save Grain to merge & export")
                        self._render_cache = None
                        self._draw()
                        return
            self._update_status("No grain found at click — try clicking directly on a grain outline")

        elif self.manual_mode or self.pos_points or self.neg_points:
            self.pos_points.append((ix, iy))
            # Draw immediate dot feedback before SAM responds
            self._draw_click_dot(event.x, event.y, "#00ff00")
            self._update_status(f"Point added at ({ix},{iy}) — SAM is updating boundary...")
            self._update_mask_from_points()

        else:
            self._select_grain_at(ix, iy)

    def _on_right_click(self, event):
        if self.current_img_idx < 0: return
        ix, iy = self._canvas_to_img(event.x, event.y)
        li = self.loaded_images[self.current_img_idx]
        H, W = li.image_rgb.shape[:2]
        if not (0 <= ix < W and 0 <= iy < H): return
        self.neg_points.append((ix, iy))
        self._draw_click_dot(event.x, event.y, "#ff4444")
        self._update_status(f"Exclusion point added at ({ix},{iy}) — SAM is updating boundary...")
        self._update_mask_from_points()

    def _select_grain_at(self, ix, iy):
        li = self.loaded_images[self.current_img_idx]
        for i, g in enumerate(li.grains):
            if 0 <= iy < g.mask.shape[0] and 0 <= ix < g.mask.shape[1]:
                if g.mask[iy, ix]:
                    self.current_grain_idx = i
                    self._update_grain_counter()
                    self._render_cache = None
                    self._draw()
                    return
        # Clicked outside all auto grains — start a paint selection
        if self.manual_mode:
            self.pos_points.append((ix, iy))
            self._update_mask_from_points()

    def _update_mask_from_points(self):
        li = self.loaded_images[self.current_img_idx]
        if not self.pos_points:
            self.active_mask = None
            self._draw()
            return

        def _work():
            H, W = li.image_rgb.shape[:2]

            # ── Fast local SAM: crop a window around the click ──
            # Much faster than running SAM on the full image
            if self.backend and self.backend.predictor:
                px, py = self.pos_points[-1]

                # Crop a 512x512 window around the click point
                crop_size = 512
                x1 = max(0, px - crop_size // 2)
                y1 = max(0, py - crop_size // 2)
                x2 = min(W, x1 + crop_size)
                y2 = min(H, y1 + crop_size)
                x1 = max(0, x2 - crop_size)
                y1 = max(0, y2 - crop_size)

                crop = li.image_rgb[y1:y2, x1:x2].astype("uint8")

                # Translate all points into crop coordinates
                local_pos = [(p[0]-x1, p[1]-y1) for p in self.pos_points
                             if x1 <= p[0] < x2 and y1 <= p[1] < y2]
                local_neg = [(p[0]-x1, p[1]-y1) for p in self.neg_points
                             if x1 <= p[0] < x2 and y1 <= p[1] < y2]

                if not local_pos:
                    self.active_mask = None
                    self.after(0, self._draw)
                    return

                self.backend.predictor.set_image(crop)
                pts  = np.array(local_pos + local_neg)
                lbs  = np.array([1]*len(local_pos) + [0]*len(local_neg))
                masks, scores, _ = self.backend.predictor.predict(
                    point_coords=pts, point_labels=lbs,
                    multimask_output=True)
                local_mask = masks[np.argmax(scores)].astype(bool)

                # Paste local mask back into full image space
                full_mask = np.zeros((H, W), dtype=bool)
                full_mask[y1:y2, x1:x2] = local_mask
                self.active_mask = full_mask

            else:
                # Fallback flood fill if SAM not available
                img = li.image_rgb.astype("uint8")
                gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                blur = cv2.GaussianBlur(gray, (5, 5), 0)
                combined = np.zeros((H, W), dtype=np.uint8)
                for px, py in self.pos_points:
                    if not (0 <= px < W and 0 <= py < H): continue
                    flood_mask = np.zeros((H+2, W+2), dtype=np.uint8)
                    cv2.floodFill(blur.copy(), flood_mask, (px, py), 255,
                                  loDiff=(35,), upDiff=(35,),
                                  flags=cv2.FLOODFILL_MASK_ONLY)
                    combined = cv2.bitwise_or(combined, flood_mask[1:-1, 1:-1])
                self.active_mask = combined.astype(bool)

            self.after(0, self._draw)

        threading.Thread(target=_work, daemon=True).start()

    def _pan_start(self, event):
        self._drag_start = (event.x, event.y)

    def _pan_move(self, event):
        if not self._drag_start: return
        dx = (event.x - self._drag_start[0]) / self.zoom
        dy = (event.y - self._drag_start[1]) / self.zoom
        self.pan_x -= dx
        self.pan_y -= dy
        self._drag_start = (event.x, event.y)
        self._draw()

    def _on_scroll(self, event):
        if self.zoom > 1.05:
            dy = -event.delta / self.zoom / 3
            self.pan_y += dy
            self._draw()
        else:
            factor = 1.15 if event.delta > 0 else 1/1.15
            self._zoom(factor, pivot=(event.x, event.y))

    def _on_scroll_x(self, event):
        if self.zoom > 1.05:
            dx = -event.delta / self.zoom / 3
            self.pan_x += dx
            self._draw()

    def _zoom(self, factor, pivot=None):
        if pivot:
            ix, iy = self._canvas_to_img(*pivot)
        self.zoom = max(0.2, min(20.0, self.zoom * factor))
        if pivot:
            self.pan_x = ix - pivot[0] / self.zoom
            self.pan_y = iy - pivot[1] / self.zoom
        self._draw()

    def _reset_zoom(self):
        self.zoom  = 1.0
        self.pan_x = 0.0
        self.pan_y = 0.0
        self._draw()

    # ── Grain navigation ──────────────────────────────────────────────────────

    def _next_grain(self):
        if self.current_img_idx < 0: return
        li = self.loaded_images[self.current_img_idx]
        if li.grains:
            self.current_grain_idx = min(len(li.grains)-1, self.current_grain_idx+1)
            self._clear_manual_state()
            self._update_grain_counter()
            self._draw()

    def _prev_grain(self):
        if self.current_img_idx < 0: return
        self.current_grain_idx = max(0, self.current_grain_idx - 1)
        self._clear_manual_state()
        self._update_grain_counter()
        self._draw()

    def _update_grain_counter(self):
        li = self.loaded_images[self.current_img_idx] if self.current_img_idx >= 0 else None
        total = len(li.grains) if li else 0
        self.grain_counter.config(text=f"{self.current_grain_idx+1} / {total}")

    # ── Save / reject ─────────────────────────────────────────────────────────

    def _get_mineral(self) -> str:
        custom = self.custom_mineral.get().strip()
        return custom if custom else self.mineral_var.get()

    def _save_grain(self):
        if self.current_img_idx < 0:
            messagebox.showinfo("No image", "Load an image first.")
            return
        li      = self.loaded_images[self.current_img_idx]
        mineral = self._get_mineral()
        if not mineral:
            messagebox.showwarning("No mineral", "Please select or type a mineral name.")
            return

        # ── Group mode — merge selected grains into one ──
        if self.group_mode and self.grouped_indices:
            merged_mask = np.zeros(li.image_rgb.shape[:2], dtype=bool)
            for idx in self.grouped_indices:
                if idx < len(li.grains):
                    merged_mask |= li.grains[idx].mask
                    li.grains[idx].accepted = True
            ys, xs = np.where(merged_mask)
            if not len(xs):
                messagebox.showwarning("Empty group", "No region selected.")
                return
            x, y = int(xs.min()), int(ys.min())
            w, h = int(xs.max()-x), int(ys.max()-y)
            new_id = max((g.grain_id for g in li.grains), default=-1)+1
            grain  = Grain(mask=merged_mask, bbox=(x,y,w,h),
                           grain_id=new_id, source="group", accepted=True)
            li.grains.append(grain)
            path = self.exporter.export_grain(grain, li.image_rgb, mineral, li.mode)
            self._update_status(f"Saved merged group ({len(self.grouped_indices)} grains) → {path}")
            self._show_toast(f"✓  {mineral} ({li.mode}) — {len(self.grouped_indices)} grains merged!", color=self.C["amber"])
            self.grouped_indices.clear()
            self._render_cache = None
            self._update_counts()
            self._draw()
            return

        # ── Manual selection ──
        if self.active_mask is not None:
            mask = self.active_mask
            ys, xs = np.where(mask)
            if not len(xs):
                messagebox.showwarning("Empty mask", "No region selected.")
                return
            x, y = int(xs.min()), int(ys.min())
            w, h = int(xs.max()-x), int(ys.max()-y)
            new_id = max((g.grain_id for g in li.grains), default=-1)+1
            grain  = Grain(mask=mask, bbox=(x,y,w,h), grain_id=new_id,
                           source="manual", accepted=True)
            li.grains.append(grain)
            self._clear_manual_state()

        # ── Auto grain ──
        elif li.grains:
            grain = li.grains[self.current_grain_idx]
            if grain.accepted:
                messagebox.showinfo("Already saved", "This grain is already saved.")
                return
            grain.accepted = True
        else:
            messagebox.showinfo("No grain", "No grain selected.")
            return

        path = self.exporter.export_grain(grain, li.image_rgb, mineral, li.mode)
        self._update_status(f"Saved → {path}")
        self._update_counts()
        self._render_cache = None
        self._draw()
        self._show_toast(f"✓  {mineral} ({li.mode}) saved!", color=self.C["green_dk"])
        self._next_grain()

    def _reject_grain(self):
        if self.current_img_idx < 0: return
        li = self.loaded_images[self.current_img_idx]
        if self.active_mask is not None:
            self._clear_manual_state()
            return
        if li.grains:
            li.grains[self.current_grain_idx].rejected = True
            self._next_grain()
        self._draw()

    def _update_counts(self):
        total = self.exporter.total_saved()
        self.total_label.config(text=str(total))
        self.session_label.config(text=f"{total} grains")
        self.exporter.save_manifest()

    # ── Utilities ─────────────────────────────────────────────────────────────

    def _update_status(self, msg: str):
        self.status_bar.config(text=msg)

    def _show_toast(self, message: str, color: str = "#238636", duration: int = 2500):
        """Show a floating toast notification that fades away after duration ms."""
        toast = tk.Toplevel(self)
        toast.overrideredirect(True)       # no title bar
        toast.attributes("-topmost", True) # always on top
        toast.attributes("-alpha", 1.0)

        # Position — bottom right of main window
        self.update_idletasks()
        wx = self.winfo_x() + self.winfo_width()  - 320
        wy = self.winfo_y() + self.winfo_height() - 80
        toast.geometry(f"300x50+{wx}+{wy}")

        tk.Label(toast, text=message,
                 bg=color, fg="#ffffff",
                 font=("Helvetica", 13, "bold"),
                 padx=16, pady=12).pack(fill="both", expand=True)

        def _fade(alpha=1.0):
            if alpha <= 0:
                toast.destroy()
                return
            toast.attributes("-alpha", alpha)
            toast.after(50, _fade, alpha - 0.05)

        # Start fade after duration ms
        toast.after(duration, _fade)

    def _open_output_folder(self):
        path = str(Path(OUTPUT_DIR).resolve())
        os.makedirs(path, exist_ok=True)
        if sys.platform == "darwin":
            os.system(f'open "{path}"')
        elif sys.platform == "win32":
            os.startfile(path)
        else:
            os.system(f'xdg-open "{path}"')


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app = GrainPickApp()
    app.mainloop()