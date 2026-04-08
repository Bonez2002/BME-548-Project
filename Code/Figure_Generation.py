"""
THRESHOLD ANALYSIS — AUTO-BATCHED DICE SWEEP
============================================

This script performs a fast, exhaustive sweep of segmentation thresholds
(0 → 1) on both the training set and an external calibration dataset.

For every threshold it computes the Dice coefficient using the model's
predicted probability maps (via sliding-window 3D inference).

Key features:
- Full 3D U-Net with InstanceNorm (the "All_Great" model)
- Overlapping-patch inference that works on arbitrarily large volumes
- Automatic GPU batch-size estimation based on available VRAM
- Vectorized threshold evaluation (extremely fast even with 50+ thresholds)
- Prevents Windows/macOS/Linux from sleeping during long runs
- Saves clean CSV files for easy plotting in Figure_Generation notebooks

No training — only inference + metric collection.
"""

import os
import sys
import subprocess
import atexit
from glob import glob

import numpy as np
import pandas as pd
import tifffile
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# ANTI-SLEEP (keeps the machine awake during multi-hour runs)
# ============================================================

class AntiSleep:
    """
    Cross-platform helper that prevents the computer from entering sleep mode.
    Uses Windows kernel calls, macOS caffeinate, or Linux systemd-inhibit.
    Automatically cleaned up on script exit.
    """
    def __init__(self):
        self.proc = None
        atexit.register(self.stop)

    def start(self):
        try:
            if sys.platform.startswith("win"):
                import ctypes
                # ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_AWAYMODE_REQUIRED
                ctypes.windll.kernel32.SetThreadExecutionState(
                    0x80000000 | 0x00000001 | 0x00000002
                )
                print("Anti-sleep enabled (Windows)")
            elif sys.platform == "darwin":
                self.proc = subprocess.Popen(["caffeinate"])
                print("Anti-sleep enabled (macOS)")
            else:
                # Linux
                self.proc = subprocess.Popen(["systemd-inhibit", "sleep", "999999"])
                print("Anti-sleep enabled (Linux)")
        except Exception:
            print("Warning: Anti-sleep failed to start")

    def stop(self):
        try:
            if self.proc is not None:
                self.proc.terminate()
        except Exception:
            pass


# ============================================================
# CONFIG
# ============================================================

CONFIG = {
    # Path to the best 3D U-Net model (All_Great checkpoint)
    "model_path": r"S:/Lab Data/Python 3.14.2/Machine Learning and Imaging/Final Model/Modified Models/All_Great/best_model.pth",

    # Training set (used to evaluate how well the model generalizes to its own data)
    "train_images_dir": r"S:/Lab Data/Python 3.14.2/Machine Learning and Imaging/Final Model/Images/All_Great/images",
    "train_masks_dir": r"S:/Lab Data/Python 3.14.2/Machine Learning and Imaging/Final Model/Images/All_Great/masks",

    # External calibration dataset (Fluo-N3DH-SIM+ — a public 3D nuclei benchmark)
    "calibration_images_dir": r"D:/Downloads/Fluo-N3DH-SIM+/Fluo-N3DH-SIM+/01",
    "calibration_masks_dir": r"D:/Downloads/Fluo-N3DH-SIM+/Fluo-N3DH-SIM+/01_GT/TRA",

    # Where to save the resulting CSV files (for later plotting)
    "output_dir": r"S:/Lab Data/Python 3.14.2/Machine Learning and Imaging/Threshold_Analysis",

    # Sliding-window parameters
    "patch_size": 128,           # 128×128×128 patches (fits comfortably in GPU memory)
    "threshold_steps": 50,       # Number of thresholds to test (0.0 → 1.0)
}

# Make sure output folder exists
os.makedirs(CONFIG["output_dir"], exist_ok=True)


# ============================================================
# MODEL ARCHITECTURE (3D U-Net with InstanceNorm)
# ============================================================

class DoubleConv(nn.Module):
    """Two consecutive 3D convolutions + InstanceNorm + ReLU"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.seq(x)


class Down(nn.Module):
    """Max-pool + DoubleConv (encoder block)"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.seq = nn.Sequential(nn.MaxPool3d(2), DoubleConv(in_ch, out_ch))

    def forward(self, x):
        return self.seq(x)


class Up(nn.Module):
    """Upsample + skip connection + DoubleConv (decoder block)"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_ch//2, in_ch//2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        # Pad the upsampled feature map to exactly match the skip connection size
        dz = x2.size(2) - x1.size(2)
        dy = x2.size(3) - x1.size(3)
        dx = x2.size(4) - x1.size(4)

        x1 = F.pad(x1, (dx//2, dx-dx//2, dy//2, dy-dy//2, dz//2, dz-dz//2))

        return self.conv(torch.cat([x2, x1], dim=1))


class UNet3D(nn.Module):
    """
    Standard 3D U-Net with 4 encoder/decoder levels.
    Base feature count = 12 channels (very lightweight for 3D).
    """
    def __init__(self, in_ch=1, base=12):
        super().__init__()
        self.inc   = DoubleConv(in_ch, base)
        self.down1 = Down(base,      base*2)
        self.down2 = Down(base*2,    base*4)
        self.down3 = Down(base*4,    base*8)
        self.down4 = Down(base*8,    base*8)
        self.up1   = Up(base*16,     base*4)
        self.up2   = Up(base*8,      base*2)
        self.up3   = Up(base*4,      base)
        self.up4   = Up(base*2,      base)
        self.outc  = nn.Conv3d(base, 1, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x,  x3)
        x = self.up3(x,  x2)
        x = self.up4(x,  x1)
        return self.outc(x)


# ============================================================
# HELPERS
# ============================================================

def safe_read_tiff(path):
    """Read a TIFF file using memory mapping when possible (much faster for large 3D stacks)."""
    try:
        return tifffile.memmap(path)
    except:
        return tifffile.imread(path)


def normalize(img):
    """
    Robust normalization: 1st–99th percentile stretch + clip to [0, 1].
    Handles constant images gracefully.
    """
    img = img.astype(np.float32)
    if img.max() == img.min():
        return np.zeros_like(img)
    p1, p99 = np.percentile(img, (1, 99))
    img = (img - p1) / (p99 - p1 + 1e-6)
    return np.clip(img, 0, 1)


# ============================================================
# SLIDING WINDOW INFERENCE (overlapping patches)
# ============================================================

def sliding_window(model, volume, patch, device):
    """
    Run full-volume inference using overlapping 3D patches.
    Returns probability map (after sigmoid) of the exact same shape as input.
    """
    model.eval()
    _, Z, Y, X = volume.shape

    stride = patch // 2                     # 50% overlap → smooth blending
    out = torch.zeros((1, Z, Y, X), device=device)
    count = torch.zeros_like(out)           # keeps track of how many times each voxel was predicted

    with torch.no_grad():
        for z in range(0, Z, stride):
            for y in range(0, Y, stride):
                for x in range(0, X, stride):

                    # Clamp patch to volume boundaries
                    z0 = min(z, Z - patch)
                    y0 = min(y, Y - patch)
                    x0 = min(x, X - patch)

                    p = volume[:, z0:z0+patch, y0:y0+patch, x0:x0+patch]
                    p = p.unsqueeze(0).to(device, dtype=torch.float32)

                    probs = torch.sigmoid(model(p))[0]          # shape: (1, Zp, Yp, Xp)

                    out[:, z0:z0+patch, y0:y0+patch, x0:x0+patch] += probs
                    count[:, z0:z0+patch, y0:y0+patch, x0:x0+patch] += 1

    # Average overlapping predictions
    return (out / count.clamp(min=1)).cpu()


# ============================================================
# AUTO BATCH SIZE (GPU memory aware)
# ============================================================

def estimate_optimal_batch_size(model, img_paths, mask_paths, device, max_test=6):
    """
    Quick one-volume forward pass to measure VRAM usage and estimate
    how many full volumes can safely be processed in one batch.
    """
    if device.type != "cuda":
        return 1

    print("\nEstimating optimal batch size...")

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    # Run a single volume through the pipeline
    img = normalize(safe_read_tiff(img_paths[0]))
    if img.ndim == 3:
        img = img[None]
    img = torch.from_numpy(img.astype(np.float32))

    with torch.no_grad():
        _ = sliding_window(model, img, CONFIG["patch_size"], device)

    used = torch.cuda.max_memory_allocated(device)
    total = torch.cuda.get_device_properties(device).total_memory

    if used == 0:
        return 1

    # Aim for 90% of total VRAM
    target = total * 0.9
    est_batch = int(target // used)
    est_batch = max(1, min(est_batch, max_test))

    print(f"Auto-detected batch size: {est_batch} volumes")
    torch.cuda.empty_cache()
    return est_batch


# ============================================================
# FAST THRESHOLD SWEEP (BATCHED VOLUMES + VECTORIZED THRESHOLDS)
# ============================================================

def run_threshold_sweep_fast_batched(model, img_paths, mask_paths, name, volume_batch_size):
    """
    Main workhorse:
    1. Processes volumes in batches
    2. Runs sliding-window inference for each volume
    3. Flattens everything and evaluates 50 thresholds in pure NumPy (very fast)
    4. Computes Dice for every threshold
    5. Saves results as CSV
    """
    thresholds = np.linspace(0, 1, CONFIG["threshold_steps"]).astype(np.float32)

    TP = np.zeros(len(thresholds), dtype=np.float64)
    FP = np.zeros(len(thresholds), dtype=np.float64)
    FN = np.zeros(len(thresholds), dtype=np.float64)

    for i in tqdm(range(0, len(img_paths), volume_batch_size), desc=name):
        batch_imgs = img_paths[i:i+volume_batch_size]
        batch_masks = mask_paths[i:i+volume_batch_size]

        probs_list = []
        masks_list = []

        for img_p, mask_p in zip(batch_imgs, batch_masks):
            img = normalize(safe_read_tiff(img_p))
            mask = (safe_read_tiff(mask_p) > 0).astype(np.float32)

            if img.ndim == 3:
                img = img[None]

            img = torch.from_numpy(img.astype(np.float32))

            # Full-volume probability map
            probs = sliding_window(model, img, CONFIG["patch_size"], device)

            probs_list.append(probs.numpy().reshape(-1))
            masks_list.append(mask.reshape(-1))

        # Concatenate all voxels from this batch of volumes
        probs_flat = np.concatenate(probs_list)
        mask_flat = np.concatenate(masks_list)

        # Vectorized threshold evaluation (shape: [n_thresholds, n_voxels])
        pred = probs_flat[None, :] > thresholds[:, None]
        target = mask_flat[None, :] > 0.5

        TP += np.sum(pred & target, axis=1)
        FP += np.sum(pred & ~target, axis=1)
        FN += np.sum(~pred & target, axis=1)

        # Clean up immediately to keep memory low
        del probs_list, masks_list, probs_flat, mask_flat, pred, target

    # Dice coefficient for every threshold
    dice = (2 * TP + 1e-6) / (2 * TP + FP + FN + 1e-6)

    df = pd.DataFrame({
        "threshold": thresholds,
        "dice": dice
    })

    out_path = os.path.join(CONFIG["output_dir"], f"{name}.csv")
    df.to_csv(out_path, index=False)

    print(f"Saved: {out_path}")
    return df


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    anti = AntiSleep()
    anti.start()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")

    # Load the trained 3D U-Net
    model = UNet3D().to(device).float()
    model.load_state_dict(torch.load(CONFIG["model_path"], map_location=device))
    model.eval()

    # Gather file lists (sorted so image ↔ mask pairing is guaranteed)
    train_imgs = sorted(glob(os.path.join(CONFIG["train_images_dir"], "*.tif")))
    train_masks = sorted(glob(os.path.join(CONFIG["train_masks_dir"], "*.tif")))

    cal_imgs = sorted(glob(os.path.join(CONFIG["calibration_images_dir"], "*.tif")))
    cal_masks = sorted(glob(os.path.join(CONFIG["calibration_masks_dir"], "*.tif")))

    # Decide how many volumes we can process in one go
    volume_batch_size = estimate_optimal_batch_size(
        model, train_imgs, train_masks, device
    )

    try:
        # Run the full threshold sweep on training data
        df_train = run_threshold_sweep_fast_batched(
            model, train_imgs, train_masks,
            "threshold_vs_dice_training",
            volume_batch_size
        )

        # Run the same sweep on the external calibration dataset
        df_cal = run_threshold_sweep_fast_batched(
            model, cal_imgs, cal_masks,
            "threshold_vs_dice_calibration",
            volume_batch_size
        )

        # Simple average of the two curves (often used for final threshold selection)
        df_comb = df_train.copy()
        df_comb["dice"] = (df_train["dice"] + df_cal["dice"]) / 2.0
        df_comb.to_csv(os.path.join(CONFIG["output_dir"], "threshold_vs_dice_combined.csv"), index=False)

        print("\nAll threshold sweeps completed successfully!")

    finally:
        anti.stop()