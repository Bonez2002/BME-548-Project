# FULL BENCHMARK (FIXED + SINGLE THRESHOLD + CELLPOSE + ANTI-SLEEP)

import os
from glob import glob
import time
import numpy as np
import pandas as pd
import tifffile
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

from cellpose import models

import sys
import subprocess
import atexit
import ctypes
from scipy.ndimage import label
from scipy.spatial.distance import directed_hausdorff
from scipy.ndimage import distance_transform_edt

# ============================================================
# CONFIG
# ============================================================

CONFIG = {
    "models_root_dir": r"S:/Lab Data/Python 3.14.2/Machine Learning and Imaging/Final Model/Modified Models",
    "input_images_dir": r"S:/Lab Data/Python 3.14.2/Machine Learning and Imaging/Final Model/Fake Images/synthetic_volumes",
    "ground_truth_dir": r"S:/Lab Data/Python 3.14.2/Machine Learning and Imaging/Final Model/Fake Images/synthetic_masks",
    "output_csv_dir": r"S:/Lab Data/Python 3.14.2/Machine Learning and Imaging/Final Model/Results Fake Images",

    # 🔥 USER-CONTROLLED THRESHOLD
    "threshold": 0.05,

    "patch_size": 128,
    "batch_size": 64,
    "use_amp": True,

    "device": "cuda" if torch.cuda.is_available() else "cpu",

    # CELLPOSE
    "cellpose_do_3D": True,
    "cellpose_z_axis": 0,
    "cellpose_batch_size": 16,
    "cellpose_tile_overlap": 0.1,
    "cellpose_volume_batch_size": 16,
}

device = torch.device(CONFIG["device"])
print("Using device:", device)

# ============================================================
# ANTI-SLEEP
# ============================================================

class AntiSleep:
    def __init__(self):
        self.proc = None
        atexit.register(self.stop)

    def start(self):
        try:
            if sys.platform.startswith("win"):
                ctypes.windll.kernel32.SetThreadExecutionState(
                    0x80000000 | 0x00000001 | 0x00000002
                )
                print("Anti-sleep enabled (Windows)")
            elif sys.platform == "darwin":
                self.proc = subprocess.Popen(["caffeinate"])
                print("Anti-sleep enabled (macOS)")
            else:
                self.proc = subprocess.Popen(["systemd-inhibit", "sleep", "999999"])
                print("Anti-sleep enabled (Linux)")
        except Exception:
            print("Warning: Anti-sleep failed")

    def stop(self):
        try:
            if self.proc is not None:
                self.proc.terminate()
        except Exception:
            pass

# ============================================================
# NORMALIZATION (MATCH TRAINING)
# ============================================================

def normalize_volume(img):
    img = img.astype(np.float32)

    if img.max() == img.min():
        return np.zeros_like(img, dtype=np.float32)

    p1, p99 = np.percentile(img, (1, 99))

    if (p99 - p1) < 1e-6:
        return np.zeros_like(img, dtype=np.float32)

    img = (img - p1) / (p99 - p1 + 1e-6)
    return np.clip(img, 0, 1)

# ============================================================
# MODEL
# ============================================================

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.InstanceNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.InstanceNorm3d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.seq(x)

class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.seq = nn.Sequential(nn.MaxPool3d(2), DoubleConv(in_ch, out_ch))
    def forward(self, x): return self.seq(x)

class Up(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_ch//2, in_ch//2, 2, stride=2)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        diffZ = x2.size(2) - x1.size(2)
        diffY = x2.size(3) - x1.size(3)
        diffX = x2.size(4) - x1.size(4)

        x1 = F.pad(x1, (
            diffX//2, diffX-diffX//2,
            diffY//2, diffY-diffY//2,
            diffZ//2, diffZ-diffZ//2
        ))

        return self.conv(torch.cat([x2, x1], dim=1))

class UNet3D(nn.Module):
    def __init__(self, in_channels, base_filters, out_channels):
        super().__init__()
        f = base_filters

        self.inc = DoubleConv(in_channels, f)
        self.down1 = Down(f, f*2)
        self.down2 = Down(f*2, f*4)
        self.down3 = Down(f*4, f*8)
        self.down4 = Down(f*8, f*8)

        self.up1 = Up(f*16, f*4)
        self.up2 = Up(f*8, f*2)
        self.up3 = Up(f*4, f)
        self.up4 = Up(f*2, f)

        self.outc = nn.Conv3d(f, out_channels, 1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        return self.outc(x)

# ============================================================
# LOAD MODEL
# ============================================================

def load_model(path):
    model = UNet3D(1, 12, 1)
    model.load_state_dict(torch.load(path, map_location=device), strict=False)
    model = model.to(device)
    model.eval()
    return model

# ============================================================
# SLIDING WINDOW
# ============================================================

def sliding_window_inference(model, volume):

    _, Z, Y, X = volume.shape
    p = CONFIG["patch_size"]
    stride = p // 2

    patches = []
    coords = []

    for z in range(0, Z, stride):
        for y in range(0, Y, stride):
            for x in range(0, X, stride):

                z0 = min(z, Z - p)
                y0 = min(y, Y - p)
                x0 = min(x, X - p)

                patch = volume[:, z0:z0+p, y0:y0+p, x0:x0+p]
                patches.append(patch)
                coords.append((z0, y0, x0))

    output = torch.zeros((1, Z, Y, X), device=device)
    count = torch.zeros_like(output)

    with torch.no_grad():
        for i in range(0, len(patches), CONFIG["batch_size"]):

            batch = np.stack(patches[i:i+CONFIG["batch_size"]]).astype(np.float32)
            coords_batch = coords[i:i+CONFIG["batch_size"]]

            inp = torch.from_numpy(batch).to(device)

            if CONFIG["use_amp"] and device.type == "cuda":
                with torch.amp.autocast("cuda"):
                    logits = model(inp)
            else:
                logits = model(inp)

            probs = torch.sigmoid(logits)

            for b in range(len(coords_batch)):
                z0, y0, x0 = coords_batch[b]
                output[:, z0:z0+p, y0:y0+p, x0:x0+p] += probs[b]
                count[:, z0:z0+p, y0:y0+p, x0:x0+p] += 1

    output /= count.clamp(min=1)
    return output.cpu().numpy()[0]

# ============================================================
# METRICS (UNCHANGED)
# ============================================================

def compute_metrics(pred, gt):
    pred = pred.astype(bool)
    gt = gt.astype(bool)

    TP = np.count_nonzero(pred & gt)
    TN = np.count_nonzero(~pred & ~gt)
    FP = np.count_nonzero(pred & ~gt)
    FN = np.count_nonzero(~pred & gt)

    accuracy = (TP + TN) / (TP + TN + FP + FN + 1e-8)
    dice = (2 * TP) / (2 * TP + FP + FN + 1e-8)
    iou = TP / (TP + FP + FN + 1e-8)
    precision = TP / (TP + FP + 1e-8)
    recall = TP / (TP + FN + 1e-8)

    return accuracy, dice, iou, precision, recall

def compute_object_f1_fast(pred, gt, iou_thresh=0.5):
    pred_lab, num_pred = label(pred)
    gt_lab, num_gt = label(gt)

    if num_pred == 0 and num_gt == 0:
        return 1.0
    if num_pred == 0 or num_gt == 0:
        return 0.0

    pred_flat = pred_lab.ravel()
    gt_flat = gt_lab.ravel()

    max_pred = num_pred + 1
    max_gt = num_gt + 1

    overlap = np.bincount(
        pred_flat * max_gt + gt_flat,
        minlength=max_pred * max_gt
    ).reshape(max_pred, max_gt)

    overlap = overlap[1:, 1:]

    pred_sizes = overlap.sum(axis=1)
    gt_sizes = overlap.sum(axis=0)

    union = pred_sizes[:, None] + gt_sizes[None, :] - overlap
    iou = overlap / (union + 1e-8)

    matched_gt = set()
    TP = 0

    for p in range(iou.shape[0]):
        g = np.argmax(iou[p])
        if iou[p, g] >= iou_thresh and g not in matched_gt:
            TP += 1
            matched_gt.add(g)

    FP = num_pred - TP
    FN = num_gt - TP

    precision = TP / (TP + FP + 1e-8)
    recall = TP / (TP + FN + 1e-8)

    return 2 * precision * recall / (precision + recall + 1e-8)

def compute_hausdorff_fast(pred, gt):
    pred = pred.astype(bool)
    gt = gt.astype(bool)

    if not pred.any() or not gt.any():
        return np.nan

    dt_gt = distance_transform_edt(~gt)
    dt_pred = distance_transform_edt(~pred)

    d1 = dt_gt[pred].max()
    d2 = dt_pred[gt].max()

    return max(d1, d2)

# ============================================================
# RUN U-NET (TIMING ADDED ONLY)
# ============================================================

def run_unet(path, name):

    model = load_model(path)

    print(f"{name} → using threshold: {CONFIG['threshold']}")

    image_paths = sorted(glob(os.path.join(CONFIG["input_images_dir"], "*.tif")))

    rows = []
    start = time.time()

    for p in tqdm(image_paths, desc=name):

        img_start = time.time()

        fname = os.path.basename(p)
        gt_path = os.path.join(CONFIG["ground_truth_dir"], fname)
        if not os.path.exists(gt_path): continue

        vol = normalize_volume(tifffile.imread(p))
        gt = tifffile.imread(gt_path)

        if vol.ndim == 3:
            vol = vol[None]

        probs = sliding_window_inference(model, vol)
        pred = probs > CONFIG["threshold"]

        acc, dice, iou, prec, rec = compute_metrics(pred, gt)

        obj_f1 = compute_object_f1_fast(pred, gt)
        hd = compute_hausdorff_fast(pred, gt)

        img_time = time.time() - img_start

        rows.append((name, fname, acc, dice, iou, prec, rec, obj_f1, hd, img_time))

    df = pd.DataFrame(rows, columns=[
        "Model","Image","Accuracy","Dice","IoU","Precision","Recall",
        "Object_F1","Hausdorff","Time_sec"
    ])

    sm = df.groupby("Model").mean(numeric_only=True).reset_index()
    sm["Total_Time_sec"] = time.time() - start

    return df, sm

# ============================================================
# CELLPOSE (TIMING ADDED ONLY)
# ============================================================

def run_cellpose():

    print("\nRunning CellposeSAM (COMBINED)")

    model = models.CellposeModel(
        gpu=(device.type == "cuda"),
        pretrained_model="cpsam"
    )

    image_paths = sorted(glob(os.path.join(CONFIG["input_images_dir"], "*.tif")))

    rows = []
    start = time.time()

    batch_vols, batch_names, batch_gts = [], [], []

    for p in tqdm(image_paths, desc="CellposeSAM"):

        fname = os.path.basename(p)
        gt_path = os.path.join(CONFIG["ground_truth_dir"], fname)
        if not os.path.exists(gt_path):
            continue

        vol = tifffile.imread(p)
        gt = tifffile.imread(gt_path)

        if vol.ndim == 4:
            vol = vol[0]

        vol = vol.astype(np.float32)

        batch_vols.append(vol)
        batch_names.append(fname)
        batch_gts.append(gt)

        if len(batch_vols) == CONFIG["cellpose_volume_batch_size"]:

            batch_start = time.time()

            cp_out = model.eval(
                batch_vols,
                do_3D=CONFIG.get("cellpose_do_3D", True),
                z_axis=CONFIG.get("cellpose_z_axis", 0),
                batch_size=CONFIG.get("cellpose_batch_size", 64),
                resample=False,
                augment=False,
                tile_overlap=CONFIG.get("cellpose_tile_overlap", 0.025)
            )

            batch_time = time.time() - batch_start
            per_img_time = batch_time / len(batch_vols)

            masks_list = cp_out[0]

            for i in range(len(batch_vols)):

                pred = masks_list[i] > 0
                gt = batch_gts[i]

                if pred.shape != gt.shape:
                    continue

                acc, dice, iou, prec, rec = compute_metrics(pred, gt)

                obj_f1 = compute_object_f1_fast(pred, gt)
                hd = compute_hausdorff_fast(pred, gt)

                rows.append((
                    "CellposeSAM", batch_names[i],
                    acc, dice, iou, prec, rec,
                    obj_f1, hd, per_img_time
                ))

            batch_vols, batch_names, batch_gts = [], [], []

    df = pd.DataFrame(rows, columns=[
        "Model","Image",
        "Accuracy","Dice","IoU","Precision","Recall",
        "Object_F1","Hausdorff","Time_sec"
    ])

    sm = df.groupby("Model").mean(numeric_only=True).reset_index()
    sm["Total_Time_sec"] = time.time() - start

    return df, sm

# ============================================================
# MAIN (UNCHANGED)
# ============================================================

def get_model_paths():
    model_dirs = sorted(glob(os.path.join(CONFIG["models_root_dir"], "*")))
    return [(os.path.basename(d), os.path.join(d,"best_model.pth"))
            for d in model_dirs if os.path.exists(os.path.join(d,"best_model.pth"))]

if __name__ == "__main__":

    anti = AntiSleep()
    anti.start()

    os.makedirs(CONFIG["output_csv_dir"], exist_ok=True)

    all_per_image = []
    all_summary = []

    for name,path in get_model_paths():
        df,sm = run_unet(path,name)
        all_per_image.append(df)
        all_summary.append(sm)

    df_cp, sm_cp = run_cellpose()
    all_per_image.append(df_cp)
    all_summary.append(sm_cp)

    pd.concat(all_per_image).to_csv(os.path.join(CONFIG["output_csv_dir"],"per_image.csv"),index=False)
    pd.concat(all_summary).to_csv(os.path.join(CONFIG["output_csv_dir"],"summary.csv"),index=False)

    anti.stop()

    print("\nDone.")