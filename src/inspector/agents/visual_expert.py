"""Visual expert agent: training-free patch comparison with a frozen DINOv3 (DINOv2 fallback).

Algorithm
---------
1. Resize query ``I`` and reference ``R`` to a multiple of the patch size and extract per-patch
   features (L2-normalized).
2. For every query patch, anomaly = ``1 - max_j cos(f_I[p], f_R[j])`` (distance to the closest
   reference patch, so pose shifts are tolerated).
3. Upsample to the image size -> heatmap; the argmax gives the 3x3 grid cell and a zoomed crop.
4. ``confidence_score`` = peak distance linearly rescaled to [0, 1] with calibration constants
   from ``Settings`` (tune on the dev split only).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from inspector.agents.base import BaseAgent
from inspector.config.settings import Settings
from inspector.schemas.agent_outputs import GridCell, VisualExpertOutput
from inspector.schemas.state import InspectionState

_GRID: list[list[GridCell]] = [
    ["top left", "top center", "top right"],
    ["middle left", "center", "middle right"],
    ["bottom left", "bottom center", "bottom right"],
]
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def grid_cell(y: int, x: int, h: int, w: int) -> GridCell:
    """Map a pixel (y, x) of an h x w image to its 3x3 cell name."""
    return _GRID[min(2, y * 3 // h)][min(2, x * 3 // w)]


def anomaly_map(q: np.ndarray, r: np.ndarray) -> np.ndarray:
    """Per-patch min cosine distance. ``q`` (Nq, D), ``r`` (Nr, D) -> (Nq,) in [0, 2]."""
    q = q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-8)
    r = r / (np.linalg.norm(r, axis=1, keepdims=True) + 1e-8)
    return 1.0 - (q @ r.T).max(axis=1)


class VisualExpert(BaseAgent):
    name = "visual_expert"

    def __init__(self, settings: Settings, out_dir: Path):
        self.s = settings
        self.out_dir = Path(out_dir) / "expert_artifacts"
        self.model: Any = None
        self.patch = 16
        self.n_prefix = 1  # CLS (+ register tokens for DINOv3)
        self.size = 448    # input side in pixels (multiple of 14 and 16 after rounding)
        self.torch: Any = None
        self.device = "cpu"

    # ---- lifecycle -----------------------------------------------------------
    def load(self) -> None:
        import torch
        from transformers import AutoModel

        self.torch = torch
        self.device = self.s.device if (self.s.device != "cuda" or torch.cuda.is_available()) else "cpu"
        dtype = getattr(torch, self.s.dtype) if self.device == "cuda" else torch.float32
        last_err: Exception | None = None
        for model_id in (self.s.expert_model, self.s.expert_fallback_model):
            try:  # DINOv3 is gated: if the license was not accepted we fall back to DINOv2
                self.model = AutoModel.from_pretrained(model_id, torch_dtype=dtype).to(self.device).eval()
                self.model_id = model_id
                break
            except Exception as e:  # noqa: BLE001
                last_err = e
        if self.model is None:
            raise RuntimeError(f"Could not load expert model: {last_err!r}")
        cfg = self.model.config
        ps = cfg.patch_size
        self.patch = ps if isinstance(ps, int) else ps[0]
        self.n_prefix = 1 + int(getattr(cfg, "num_register_tokens", 0))
        self.size = (448 // self.patch) * self.patch
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def unload(self) -> None:
        self.model = None
        super().unload()

    # ---- feature extraction --------------------------------------------------
    def _features(self, img: Image.Image) -> np.ndarray:
        arr = (np.asarray(img.convert("RGB").resize((self.size, self.size), Image.BICUBIC), dtype=np.float32) / 255.0 - _MEAN) / _STD
        x = self.torch.from_numpy(arr).permute(2, 0, 1)[None].to(self.device, self.model.dtype)
        with self.torch.no_grad():
            h = self.model(pixel_values=x).last_hidden_state[0]
        return h[self.n_prefix :].float().cpu().numpy()  # (grid*grid, D)

    # ---- agent ---------------------------------------------------------------
    def _run(self, state: InspectionState, rec: dict[str, Any]) -> InspectionState:
        query = Image.open(state.query_image_path).convert("RGB")
        ref = Image.open(state.reference_image_path).convert("RGB")
        out = self.compute(query, ref, state.image_id)
        rec["model"] = getattr(self, "model_id", "?")
        rec["output"] = out.model_dump()
        state.visual_evidence = out
        return state

    def compute(self, query: Image.Image, ref: Image.Image, image_id: str) -> VisualExpertOutput:
        """Pure image-in / evidence-out step (also usable from a notebook)."""
        dist = anomaly_map(self._features(query), self._features(ref))
        g = self.size // self.patch
        amap = dist.reshape(g, g)
        w, h = query.size
        up = np.asarray(Image.fromarray(amap.astype(np.float32), mode="F").resize((w, h), Image.BICUBIC))
        py, px = np.unravel_index(int(up.argmax()), up.shape)
        peak = float(amap.max())
        score = float(np.clip((peak - self.s.anomaly_norm_low) / (self.s.anomaly_norm_high - self.s.anomaly_norm_low), 0, 1))

        stem = image_id.replace("/", "__")
        heat_path = self.out_dir / f"{stem}_heatmap.png"
        crop_path = self.out_dir / f"{stem}_crop.png"
        norm = np.clip((up - up.min()) / (np.ptp(up) + 1e-8), 0, 1)
        overlay = np.asarray(query, dtype=np.float32).copy()
        overlay[..., 0] = np.clip(overlay[..., 0] * (1 - 0.6 * norm) + 255 * 0.6 * norm, 0, 255)
        Image.fromarray(overlay.astype(np.uint8)).save(heat_path)

        cw, ch = max(w // 3, 32), max(h // 3, 32)
        x0 = int(np.clip(px - cw // 2, 0, max(w - cw, 0)))
        y0 = int(np.clip(py - ch // 2, 0, max(h - ch, 0)))
        query.crop((x0, y0, x0 + cw, y0 + ch)).resize((cw * 2, ch * 2), Image.BICUBIC).save(crop_path)

        return VisualExpertOutput(
            heatmap_path=str(heat_path),
            candidate_region=grid_cell(int(py), int(px), h, w),
            confidence_score=score,
            crop_path=str(crop_path),
        )
