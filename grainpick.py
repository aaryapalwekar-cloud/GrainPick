"""
GrainPick — Automated grain boundary detection & extractor for thin section images
============================================================
Usage:
    python grainpick.py --image path/to/thin_section.jpg [--output ./grains] [--checkpoint ./checkpoints/sam_vit_h_4b8939.pth]

Controls (interactive window):
    Left-click          Add a positive point (include this in grain)
    Right-click         Add a negative point (exclude from grain)
    A                   Accept current grain & save
    R                   Reject / skip current grain
    N                   New manual selection (clear points, pick new grain)
    U                   Undo last point
    S                   Save all accepted grains now
    Q / Esc             Quit and save
    Mouse wheel         Zoom in/out
    Middle-click drag   Pan
"""

import argparse
import os
import sys
import json
import time
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np
from PIL import Image

# ── SAM import (graceful error) ──────────────────────────────────────────────
try:
    import torch
    from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
    SAM_AVAILABLE = True
except ImportError:
    SAM_AVAILABLE = False
    print("[WARN] segment-anything or torch not installed.")
    print("       Run:  bash setup.sh   to install dependencies.")
    print("       Falling back to OpenCV-only mode (watershed segmentation).\n")


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_CHECKPOINT = "./checkpoints/sam_vit_h_4b8939.pth"
MODEL_TYPE = "vit_h"

# SAM automatic mask generator settings — tuned for thin sections
# Increase points_per_side for denser sampling on fine-grained rocks
SAM_AUTO_CONFIG = dict(
    points_per_side=32,          # grid density; raise to 64 for very fine grains
    pred_iou_thresh=0.86,        # confidence threshold; lower = more masks
    stability_score_thresh=0.90,
    crop_n_layers=1,
    crop_n_points_downscale_factor=2,
    min_mask_region_area=500,    # px² — filters out tiny noise; raise for coarser grains
)

COLORS = {
    "accepted": (80, 220, 80),
    "rejected": (80, 80, 220),
    "active":   (80, 220, 255),
    "auto":     (220, 180, 80),
    "point_pos":(0, 255, 0),
    "point_neg":(0, 0, 255),
}

EXPORT_PADDING = 6   # px padding around each grain crop


# ─────────────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Grain:
    mask: np.ndarray          # bool H×W
    bbox: tuple               # (x, y, w, h) in original image coords
    grain_id: int
    source: str               # "auto" | "manual"
    label: str = ""           # optional mineral label — editable later
    accepted: bool = False
    rejected: bool = False

    @property
    def area(self):
        return int(self.mask.sum())


# ─────────────────────────────────────────────────────────────────────────────
# Segmentation backend
# ─────────────────────────────────────────────────────────────────────────────

class SegmentationBackend:
    def __init__(self, checkpoint: str, device: str = "auto"):
        self.predictor = None
        self.auto_generator = None
        self._device = self._resolve_device(device)

        if SAM_AVAILABLE and os.path.exists(checkpoint):
            print(f"[INFO] Loading SAM ({MODEL_TYPE}) on {self._device} ...")
            t0 = time.time()
            sam = sam_model_registry[MODEL_TYPE](checkpoint=checkpoint)
            sam.to(device=self._device)
            self.predictor = SamPredictor(sam)
            self.auto_generator = SamAutomaticMaskGenerator(sam, **SAM_AUTO_CONFIG)
            print(f"[INFO] SAM loaded in {time.time()-t0:.1f}s")
        else:
            if SAM_AVAILABLE:
                print(f"[WARN] Checkpoint not found at {checkpoint}")
            print("[INFO] Using OpenCV watershed fallback.")

    def _resolve_device(self, device):
        if device != "auto":
            return device
        if SAM_AVAILABLE:
            if torch.cuda.is_available():
                return "cuda"
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
        return "cpu"

    def auto_segment(self, image_rgb: np.ndarray) -> list[Grain]:
        """Run automatic segmentation on the full image."""
        if self.auto_generator is not None:
            return self._sam_auto(image_rgb)
        return self._watershed_auto(image_rgb)

    def _sam_auto(self, image_rgb: np.ndarray) -> list[Grain]:
        print("[INFO] Running SAM automatic segmentation (may take 30-90s)...")
        t0 = time.time()
        masks_data = self.auto_generator.generate(image_rgb)
        print(f"[INFO] SAM found {len(masks_data)} candidate regions in {time.time()-t0:.1f}s")

        grains = []
        for i, m in enumerate(masks_data):
            mask = m["segmentation"]
            x, y, w, h = m["bbox"]  # SAM returns xywh
            grains.append(Grain(
                mask=mask.astype(bool),
                bbox=(int(x), int(y), int(w), int(h)),
                grain_id=i,
                source="auto",
            ))
        return grains

    def _watershed_auto(self, image_rgb: np.ndarray) -> list[Grain]:
        """Fallback: OpenCV watershed segmentation."""
        print("[INFO] Running watershed segmentation fallback...")
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)

        # Adaptive threshold
        thresh = cv2.adaptiveThreshold(
            blur, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 15, 3
        )

        # Morphological cleanup
        kernel = np.ones((3, 3), np.uint8)
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
        sure_bg = cv2.dilate(opening, kernel, iterations=3)
        dist = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
        _, sure_fg = cv2.threshold(dist, 0.4 * dist.max(), 255, 0)
        sure_fg = sure_fg.astype(np.uint8)
        unknown = cv2.subtract(sure_bg, sure_fg)

        _, markers = cv2.connectedComponents(sure_fg)
        markers += 1
        markers[unknown == 255] = 0

        img_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        markers = cv2.watershed(img_bgr, markers)

        grains = []
        for label_id in np.unique(markers):
            if label_id <= 1:
                continue
            mask = (markers == label_id)
            if mask.sum() < SAM_AUTO_CONFIG["min_mask_region_area"]:
                continue
            ys, xs = np.where(mask)
            x, y = int(xs.min()), int(ys.min())
            w, h = int(xs.max() - x), int(ys.max() - y)
            grains.append(Grain(
                mask=mask,
                bbox=(x, y, w, h),
                grain_id=len(grains),
                source="auto",
            ))

        print(f"[INFO] Watershed found {len(grains)} candidate regions")
        return grains

    def predict_from_points(self, image_rgb: np.ndarray,
                             pos_points: list, neg_points: list) -> Optional[np.ndarray]:
        """Given click points, predict a single grain mask."""
        if self.predictor is None:
            return self._watershed_single(image_rgb, pos_points)

        self.predictor.set_image(image_rgb)
        all_points = pos_points + neg_points
        all_labels = [1] * len(pos_points) + [0] * len(neg_points)

        if not all_points:
            return None

        pts = np.array(all_points)
        lbs = np.array(all_labels)

        masks, scores, _ = self.predictor.predict(
            point_coords=pts,
            point_labels=lbs,
            multimask_output=True,
        )
        # Pick highest-scoring mask
        best = masks[np.argmax(scores)]
        return best.astype(bool)

    def _watershed_single(self, image_rgb, pos_points):
        """Flood-fill fallback for single click."""
        if not pos_points:
            return None
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        x, y = pos_points[-1]
        seed_val = int(blur[y, x])
        lo, hi = max(0, seed_val - 30), min(255, seed_val + 30)
        mask = np.zeros((gray.shape[0] + 2, gray.shape[1] + 2), dtype=np.uint8)
        cv2.floodFill(blur.copy(), mask, (x, y), 255,
                      loDiff=(lo,), upDiff=(255 - hi,))
        return mask[1:-1, 1:-1].astype(bool)


# ─────────────────────────────────────────────────────────────────────────────
# Export
# ─────────────────────────────────────────────────────────────────────────────

class GrainExporter:
    def __init__(self, output_dir: str, source_name: str):
        self.output_dir = Path(output_dir)
        self.source_name = Path(source_name).stem
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.manifest = []

    def export_grain(self, grain: Grain, image_rgb: np.ndarray) -> str:
        """Save one grain as a PNG with transparency mask."""
        H, W = image_rgb.shape[:2]
        x, y, w, h = grain.bbox
        pad = EXPORT_PADDING

        # Padded crop bounds (clamped to image)
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(W, x + w + pad)
        y2 = min(H, y + h + pad)

        crop_rgb  = image_rgb[y1:y2, x1:x2]
        crop_mask = grain.mask[y1:y2, x1:x2]

        # RGBA — alpha = grain mask
        rgba = np.dstack([crop_rgb, (crop_mask * 255).astype(np.uint8)])
        pil  = Image.fromarray(rgba, "RGBA")

        fname = f"{self.source_name}_grain_{grain.grain_id:04d}.png"
        fpath = self.output_dir / fname
        pil.save(fpath)

        meta = {
            "file": fname,
            "grain_id": grain.grain_id,
            "source": grain.source,
            "label": grain.label,
            "bbox": [x1, y1, x2 - x1, y2 - y1],
            "area_px": grain.area,
        }
        self.manifest.append(meta)
        return str(fpath)

    def save_manifest(self):
        mpath = self.output_dir / f"{self.source_name}_manifest.json"
        with open(mpath, "w") as f:
            json.dump(self.manifest, f, indent=2)
        print(f"[INFO] Manifest saved → {mpath}")


# ─────────────────────────────────────────────────────────────────────────────
# Interactive viewer
# ─────────────────────────────────────────────────────────────────────────────

class GrainPickerUI:
    def __init__(self, image_path: str, grains: list[Grain],
                 backend: SegmentationBackend, exporter: GrainExporter):
        self.image_rgb  = np.array(Image.open(image_path).convert("RGB"))
        self.grains     = grains
        self.backend    = backend
        self.exporter   = exporter

        self.current_idx   = 0
        self.active_mask   = None
        self.pos_points    = []
        self.neg_points    = []
        self.saved_count   = 0

        # View state
        self.zoom    = 1.0
        self.pan_x   = 0
        self.pan_y   = 0
        self._panning = False
        self._pan_start = (0, 0)

        self.win = "GrainPick — [A]ccept  [R]eject  [N]ew  [U]ndo  [S]ave  [Q]uit"
        cv2.namedWindow(self.win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.win, 1200, 800)
        cv2.setMouseCallback(self.win, self._mouse_cb)

    # ── Coordinate helpers ───────────────────────────────────────────────────

    def _screen_to_img(self, sx, sy):
        ix = int((sx / self.zoom) + self.pan_x)
        iy = int((sy / self.zoom) + self.pan_y)
        return ix, iy

    def _img_to_screen(self, ix, iy):
        sx = int((ix - self.pan_x) * self.zoom)
        sy = int((iy - self.pan_y) * self.zoom)
        return sx, sy

    # ── Mouse callback ───────────────────────────────────────────────────────

    def _mouse_cb(self, event, x, y, flags, param):
        if event == cv2.EVENT_MBUTTONDOWN:
            self._panning = True
            self._pan_start = (x, y)

        elif event == cv2.EVENT_MBUTTONUP:
            self._panning = False

        elif event == cv2.EVENT_MOUSEMOVE and self._panning:
            dx = (x - self._pan_start[0]) / self.zoom
            dy = (y - self._pan_start[1]) / self.zoom
            self.pan_x -= dx
            self.pan_y -= dy
            self._pan_start = (x, y)
            self._draw()

        elif event == cv2.EVENT_MOUSEWHEEL:
            # flags > 0 = scroll up = zoom in (works on some systems)
            factor = 1.15 if flags > 0 else 1 / 1.15
            ix, iy = self._screen_to_img(x, y)
            self.zoom = max(0.2, min(20.0, self.zoom * factor))
            self.pan_x = ix - x / self.zoom
            self.pan_y = iy - y / self.zoom
            self._draw()

        elif event == cv2.EVENT_MOUSEHWHEEL:
            # Mac trackpad horizontal scroll — use for zoom too
            factor = 1.15 if flags > 0 else 1 / 1.15
            ix, iy = self._screen_to_img(x, y)
            self.zoom = max(0.2, min(20.0, self.zoom * factor))
            self.pan_x = ix - x / self.zoom
            self.pan_y = iy - y / self.zoom
            self._draw()

        elif event == cv2.EVENT_LBUTTONDOWN:
            ix, iy = self._screen_to_img(x, y)
            H, W = self.image_rgb.shape[:2]
            if 0 <= ix < W and 0 <= iy < H:
                self.pos_points.append((ix, iy))
                self._update_manual_mask()

        elif event == cv2.EVENT_RBUTTONDOWN:
            ix, iy = self._screen_to_img(x, y)
            H, W = self.image_rgb.shape[:2]
            if 0 <= ix < W and 0 <= iy < H:
                self.neg_points.append((ix, iy))
                self._update_manual_mask()

    # ── Mask prediction ──────────────────────────────────────────────────────

    def _update_manual_mask(self):
        if not self.pos_points:
            self.active_mask = None
        else:
            self.active_mask = self.backend.predict_from_points(
                self.image_rgb, self.pos_points, self.neg_points
            )
        self._draw()

    # ── Drawing ──────────────────────────────────────────────────────────────

    def _draw(self):
        img = self.image_rgb.copy()
        H, W = img.shape[:2]
        overlay = img.copy()

        # Draw all auto grains (dim)
        for g in self.grains:
            if g.accepted:
                col = COLORS["accepted"]
            elif g.rejected:
                col = COLORS["rejected"]
            else:
                col = COLORS["auto"]
            # Draw contour only for performance
            m = g.mask.astype(np.uint8) * 255
            cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(overlay, cnts, -1, col, 1)

        # Active grain highlight
        if self.current_idx < len(self.grains) and not (self.pos_points or self.neg_points):
            g = self.grains[self.current_idx]
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
            cv2.circle(overlay, (px, py), 5, COLORS["point_pos"], -1)
            cv2.circle(overlay, (px, py), 5, (255, 255, 255), 1)
        for px, py in self.neg_points:
            cv2.circle(overlay, (px, py), 5, COLORS["point_neg"], -1)
            cv2.circle(overlay, (px, py), 5, (255, 255, 255), 1)

        # Blend overlay
        img = cv2.addWeighted(overlay, 0.45, img, 0.55, 0)

        # Apply zoom & pan
        new_w = max(1, int(W * self.zoom))
        new_h = max(1, int(H * self.zoom))
        big = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        win_h, win_w = 800, 1200
        ox = int(self.pan_x * self.zoom)
        oy = int(self.pan_y * self.zoom)
        ox = max(0, min(ox, new_w - 1))
        oy = max(0, min(oy, new_h - 1))

        crop = big[oy:oy + win_h, ox:ox + win_w]
        if crop.shape[0] < win_h or crop.shape[1] < win_w:
            canvas = np.zeros((win_h, win_w, 3), dtype=np.uint8)
            canvas[:crop.shape[0], :crop.shape[1]] = crop
            crop = canvas

        # HUD
        accepted = sum(1 for g in self.grains if g.accepted)
        rejected = sum(1 for g in self.grains if g.rejected)
        remaining = len(self.grains) - accepted - rejected
        manual_active = bool(self.pos_points or self.neg_points)

        hud_lines = [
            f"Auto grains: {len(self.grains)}   Accepted: {accepted}   Rejected: {rejected}   Remaining: {remaining}",
            f"Saved: {self.saved_count}   Zoom: {self.zoom:.1f}x",
            "Manual mode ACTIVE — L-click: include   R-click: exclude   [A] accept   [N] clear" if manual_active
            else f"Auto grain {self.current_idx + 1}/{len(self.grains)} — [A] accept   [R] reject   [→/←] next/prev   [N] manual",
        ]
        y_hud = 20
        for line in hud_lines:
            cv2.putText(crop, line, (12, y_hud),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(crop, line, (12, y_hud),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 240, 200), 1, cv2.LINE_AA)
            y_hud += 20

        cv2.imshow(self.win, crop)

    # ── Accept / reject logic ────────────────────────────────────────────────

    def _accept_current(self):
        if self.pos_points and self.active_mask is not None:
            # Manual grain — create a new Grain object
            mask = self.active_mask
            ys, xs = np.where(mask)
            if len(xs) == 0:
                return
            x, y = int(xs.min()), int(ys.min())
            w, h = int(xs.max() - x), int(ys.max() - y)
            new_id = max((g.grain_id for g in self.grains), default=-1) + 1
            g = Grain(mask=mask, bbox=(x, y, w, h),
                      grain_id=new_id, source="manual", accepted=True)
            self.grains.append(g)
            path = self.exporter.export_grain(g, self.image_rgb)
            self.saved_count += 1
            print(f"[SAVE] Manual grain {new_id} → {path}")
            self._clear_manual()
        elif self.current_idx < len(self.grains):
            g = self.grains[self.current_idx]
            if not g.accepted:
                g.accepted = True
                g.rejected = False
                path = self.exporter.export_grain(g, self.image_rgb)
                self.saved_count += 1
                print(f"[SAVE] Grain {g.grain_id} ({g.source}) → {path}")
            self.current_idx = min(self.current_idx + 1, len(self.grains) - 1)

        self._draw()

    def _reject_current(self):
        if self.current_idx < len(self.grains):
            g = self.grains[self.current_idx]
            g.rejected = True
            g.accepted = False
            self.current_idx = min(self.current_idx + 1, len(self.grains) - 1)
        self._draw()

    def _clear_manual(self):
        self.pos_points = []
        self.neg_points = []
        self.active_mask = None

    def _undo(self):
        if self.neg_points:
            self.neg_points.pop()
        elif self.pos_points:
            self.pos_points.pop()
        self._update_manual_mask()

    # ── Main loop ────────────────────────────────────────────────────────────

    def run(self):
        print(f"\n[INFO] {len(self.grains)} auto-detected grains loaded.")
        print("[INFO] Opening interactive viewer...\n")
        self._draw()

        while True:
            key = cv2.waitKey(50) & 0xFF

            if key == ord('a') or key == ord('A'):
                self._accept_current()

            elif key == ord('r') or key == ord('R'):
                if not (self.pos_points or self.neg_points):
                    self._reject_current()
                else:
                    self._clear_manual()
                    self._draw()

            elif key == ord('n') or key == ord('N'):
                self._clear_manual()
                self._draw()

            elif key == ord('u') or key == ord('U'):
                self._undo()

            elif key == ord('s') or key == ord('S'):
                self.exporter.save_manifest()
                print(f"[INFO] {self.saved_count} grains saved so far.")

            elif key == 81 or key == ord(','):   # left arrow / ,
                self.current_idx = max(0, self.current_idx - 1)
                self._clear_manual()
                self._draw()

            elif key == 83 or key == ord('.'):   # right arrow / .
                self.current_idx = min(len(self.grains) - 1, self.current_idx + 1)
                self._clear_manual()
                self._draw()

            elif key == ord('=') or key == ord('+'):   # zoom in
                self.zoom = min(20.0, self.zoom * 1.2)
                self._draw()

            elif key == ord('-'):                       # zoom out
                self.zoom = max(0.2, self.zoom / 1.2)
                self._draw()

            elif key == ord('0'):                       # reset zoom
                self.zoom = 1.0
                self.pan_x = 0
                self.pan_y = 0
                self._draw()

            elif key in (ord('q'), ord('Q'), 27):   # Q or Esc
                break

            # Check window closed
            if cv2.getWindowProperty(self.win, cv2.WND_PROP_VISIBLE) < 1:
                break

        cv2.destroyAllWindows()
        self.exporter.save_manifest()
        print(f"\n[DONE] Session complete. {self.saved_count} grains exported.")
        print(f"       Output directory: {self.exporter.output_dir}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="GrainPick — thin section grain extractor")
    parser.add_argument("--image",      required=True,              help="Path to thin section image")
    parser.add_argument("--output",     default="./grains",         help="Output directory for grain crops")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT, help="SAM model checkpoint path")
    parser.add_argument("--device",     default="auto",             help="cpu / cuda / mps / auto")
    parser.add_argument("--skip-auto",  action="store_true",        help="Skip auto-segmentation, manual only")
    args = parser.parse_args()

    if not os.path.exists(args.image):
        print(f"[ERROR] Image not found: {args.image}")
        sys.exit(1)

    # Load backend
    backend  = SegmentationBackend(args.checkpoint, args.device)
    exporter = GrainExporter(args.output, args.image)

    # Auto-segment
    if args.skip_auto:
        grains = []
        print("[INFO] Skipping auto-segmentation. Use clicks to pick grains.")
    else:
        image_rgb = np.array(Image.open(args.image).convert("RGB"))
        grains = backend.auto_segment(image_rgb)

    # Launch UI
    ui = GrainPickerUI(args.image, grains, backend, exporter)
    ui.run()


if __name__ == "__main__":
    main()