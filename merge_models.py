"""
merge_models.py
===============
Merges a YOLOv8s model (extra training data) into a YOLOv8m model (target
architecture) using two complementary strategies:

  1. DETECTION-HEAD WEIGHT BLENDING
     The detection head of YOLOv8m and YOLOv8s share the same output tensor
     shapes (they both predict the same 11 classes with the same 4-box +
     11-class structure).  We linearly interpolate compatible weight tensors
     from s into m so the m model benefits from what s learned.

  2. ENSEMBLE INFERENCE WRAPPER  (EnsembleYOLO)
     At inference time, both models are run on every frame and their raw
     bounding-box predictions are merged with weighted NMS.  This gives the
     full benefit of v8s extra training data at no architectural cost.

Usage
-----
  python merge_models.py --alpha 0.25

  alpha = 0   => pure v8m (no blending)
  alpha = 0.25 => 75% v8m + 25% v8s  (recommended default)
  alpha = 0.5 => equal blend
  alpha = 1   => pure v8s weights in v8m architecture (not recommended)

Outputs
-------
  weights/merged_v8m_from_s.pt  -- blended v8m weights (strategy 1)
"""

import os, sys, copy, logging, argparse
import torch

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
WEIGHTS_DIR = os.path.join(BASE_DIR, "weights")
DATA_UTIL   = os.path.join(BASE_DIR, "data.yml & utility file")

MODEL_M_PATH = next(
    (p for p in [
        os.path.join(WEIGHTS_DIR, "best.pt"),
        os.path.join(BASE_DIR, "final_best_v8m.pt"),
    ] if os.path.isfile(p)),
    None,
)
MODEL_S_PATH = next(
    (p for p in [
        os.path.join(DATA_UTIL, "final_best_v8s.pt"),
        os.path.join(BASE_DIR, "final_best_v8s.pt"),
    ] if os.path.isfile(p)),
    None,
)
OUTPUT_PATH = os.path.join(WEIGHTS_DIR, "merged_v8m_from_s.pt")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("merge_models")


# ===========================================================================
# Strategy 1 -- Detection-head weight blending
# ===========================================================================

def _state_dict(model_path: str) -> dict:
    ck = torch.load(model_path, map_location="cpu", weights_only=False)
    if isinstance(ck, dict):
        m = ck.get("model") or ck.get("ema")
        if m is not None:
            return m.float().state_dict()
        if "state_dict" in ck:
            return ck["state_dict"]
        return ck
    return ck.float().state_dict()


def _blend_state_dicts(sd_m: dict, sd_s: dict, alpha: float = 0.25) -> dict:
    """
    blended_key = (1 - alpha) * sd_m[key]  +  alpha * sd_s[key]
    Only tensors present in both dicts AND with matching shapes are blended.
    Integer tensors (e.g. batch-norm running stats) are kept from v8m.
    """
    blended = copy.deepcopy(sd_m)
    matched = skipped_shape = skipped_key = 0

    for key in sd_m:
        if key not in sd_s:
            skipped_key += 1
            continue
        t_m = sd_m[key]
        t_s = sd_s[key]
        if t_m.shape != t_s.shape:
            skipped_shape += 1
            continue
        if not t_m.is_floating_point():
            continue   # keep v8m integer tensors as-is
        blended[key] = (1.0 - alpha) * t_m + alpha * t_s
        matched += 1

    log.info(
        "Blend summary: %d tensors blended | %d skipped (shape mismatch) "
        "| %d skipped (key absent in v8s)",
        matched, skipped_shape, skipped_key,
    )
    return blended


def blend_and_save(alpha: float = 0.25) -> str:
    if MODEL_M_PATH is None:
        raise FileNotFoundError(
            "YOLOv8m weights not found.  Expected:\n"
            f"  {os.path.join(WEIGHTS_DIR, 'best.pt')}\n"
            f"  {os.path.join(BASE_DIR, 'final_best_v8m.pt')}"
        )
    if MODEL_S_PATH is None:
        raise FileNotFoundError(
            "YOLOv8s weights not found.  Expected:\n"
            f"  {os.path.join(DATA_UTIL, 'final_best_v8s.pt')}"
        )

    log.info("Loading v8m  <- %s", MODEL_M_PATH)
    ck_m = torch.load(MODEL_M_PATH, map_location="cpu", weights_only=False)

    log.info("Loading v8s  <- %s", MODEL_S_PATH)
    sd_s = _state_dict(MODEL_S_PATH)

    # Extract v8m model object
    if isinstance(ck_m, dict):
        model_obj = ck_m.get("model") or ck_m.get("ema")
    else:
        model_obj = ck_m

    sd_m = model_obj.float().state_dict()

    # Inspect how many layers actually overlap (informational)
    keys_m = set(sd_m.keys())
    keys_s = set(sd_s.keys())
    log.info(
        "v8m layers: %d | v8s layers: %d | common keys: %d",
        len(keys_m), len(keys_s), len(keys_m & keys_s),
    )

    log.info("Blending weights (alpha=%.2f)...", alpha)
    blended_sd = _blend_state_dicts(sd_m, sd_s, alpha=alpha)

    # Load blended state dict back into v8m model object
    missing, unexpected = model_obj.float().load_state_dict(blended_sd, strict=False)
    if missing:
        log.warning("Missing keys after load: %s", missing[:5])
    if unexpected:
        log.warning("Unexpected keys after load: %s", unexpected[:5])

    # Rebuild checkpoint
    if isinstance(ck_m, dict):
        out_ck = {**ck_m, "model": model_obj}
        out_ck.pop("ema", None)
        out_ck.pop("optimizer", None)
        out_ck["epoch"] = -1
        out_ck["date"]  = "merged_v8m_from_s"
    else:
        out_ck = model_obj

    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    torch.save(out_ck, OUTPUT_PATH)
    log.info("Saved merged model -> %s", OUTPUT_PATH)
    return OUTPUT_PATH


# ===========================================================================
# Strategy 2 -- EnsembleYOLO (drop-in inference wrapper)
# ===========================================================================

class EnsembleYOLO:
    """
    Drop-in replacement for ultralytics.YOLO that runs v8m AND v8s on every
    frame and merges detections with weighted NMS.

    Parameters
    ----------
    model_m_path : path to YOLOv8m .pt file
    model_s_path : path to YOLOv8s .pt file
    weight_m     : confidence multiplier for v8m boxes (default 1.0)
    weight_s     : confidence multiplier for v8s boxes (default 0.8)
    iou_merge    : IoU threshold for NMS when merging boxes
    """

    def __init__(
        self,
        model_m_path=None,
        model_s_path=None,
        weight_m: float = 1.0,
        weight_s: float = 0.8,
        iou_merge: float = 0.45,
    ):
        from ultralytics import YOLO

        mp = model_m_path or MODEL_M_PATH
        sp = model_s_path or MODEL_S_PATH

        if mp is None or not os.path.isfile(mp):
            raise FileNotFoundError(f"v8m weights not found: {mp}")
        if sp is None or not os.path.isfile(sp):
            raise FileNotFoundError(f"v8s weights not found: {sp}")

        log.info("EnsembleYOLO: loading v8m <- %s", mp)
        self.model_m = YOLO(mp)
        log.info("EnsembleYOLO: loading v8s <- %s", sp)
        self.model_s = YOLO(sp)

        self.weight_m  = weight_m
        self.weight_s  = weight_s
        self.iou_merge = iou_merge

        # Mimic attributes detector.py reads from a YOLO object
        self.names = self.model_m.names
        self.model = self.model_m.model

    def predict(self, source, **kwargs):
        import torchvision.ops as tv_ops
        res_m = self.model_m.predict(source, **kwargs)
        res_s = self.model_s.predict(source, **kwargs)
        return [self._merge(rm, rs) for rm, rs in zip(res_m, res_s)]

    def __call__(self, source, **kwargs):
        return self.predict(source, **kwargs)

    def _merge(self, rm, rs):
        def _unpack(boxes, w):
            if boxes is None or len(boxes) == 0:
                return None, None, None
            return boxes.xyxy.cpu(), boxes.conf.cpu() * w, boxes.cls.cpu()

        xyxy_m, conf_m, cls_m = _unpack(rm.boxes, self.weight_m)
        xyxy_s, conf_s, cls_s = _unpack(rs.boxes, self.weight_s)

        xys   = [t for t in (xyxy_m, xyxy_s) if t is not None]
        confs = [t for t in (conf_m, conf_s)  if t is not None]
        clss  = [t for t in (cls_m,  cls_s)   if t is not None]

        if not xys:
            return rm

        import torchvision.ops as tv_ops
        all_xy   = torch.cat(xys,   dim=0)
        all_conf = torch.cat(confs, dim=0)
        all_cls  = torch.cat(clss,  dim=0)

        keep = tv_ops.batched_nms(
            all_xy.float(), all_conf.float(), all_cls.long(), self.iou_merge
        )

        merged_data = torch.cat(
            [all_xy[keep], all_conf[keep].unsqueeze(1), all_cls[keep].unsqueeze(1)],
            dim=1,
        )

        result = copy.deepcopy(rm)
        result.boxes.data = merged_data.to(rm.boxes.data.device)
        return result

    # pass-through helpers so detector.py does not break
    def to(self, device):
        self.model_m.to(device)
        self.model_s.to(device)
        return self

    def eval(self):
        self.model_m.model.eval()
        self.model_s.model.eval()
        return self

    def export(self, *a, **kw):
        return self.model_m.export(*a, **kw)

    def val(self, *a, **kw):
        return self.model_m.val(*a, **kw)


# ===========================================================================
# CLI
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Merge YOLOv8s training knowledge into YOLOv8m."
    )
    parser.add_argument(
        "--alpha", type=float, default=0.25,
        help="Blend factor 0=pure v8m .. 1=pure v8s.  Default: 0.25",
    )
    parser.add_argument(
        "--strategy", choices=["blend", "ensemble", "both"], default="both",
        help="blend=save blended .pt | ensemble=test EnsembleYOLO | both (default)",
    )
    parser.add_argument(
        "--test-image", default=None,
        help="Image path to test ensemble inference output (optional).",
    )
    args = parser.parse_args()

    if args.strategy in ("blend", "both"):
        out = blend_and_save(alpha=args.alpha)
        print(f"\n[OK] Blended v8m weights saved -> {out}")
        print(
            "     In config.py add this path to _candidates so the app finds it:\n"
            f"       os.path.join(WEIGHTS_DIR, 'merged_v8m_from_s.pt')\n"
        )

    if args.strategy in ("ensemble", "both") and args.test_image:
        log.info("Quick inference test with EnsembleYOLO on: %s", args.test_image)
        ens = EnsembleYOLO()
        results = ens.predict(args.test_image, conf=0.35, iou=0.45)
        for r in results:
            n = len(r.boxes) if r.boxes else 0
            log.info("  Detections: %d boxes", n)

    elif args.strategy in ("ensemble", "both"):
        print(
            "\n[INFO] EnsembleYOLO class is ready.\n"
            "       Import it in detector.py:\n"
            "         from merge_models import EnsembleYOLO\n"
            "       Then replace:  model = YOLO(MODEL_PATH)\n"
            "       With:          model = EnsembleYOLO()\n"
        )


if __name__ == "__main__":
    main()
