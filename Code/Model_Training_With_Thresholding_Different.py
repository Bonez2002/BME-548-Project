# train_3d_unet_gpu_optimized_cellpatch_full_with_threshold.py

import os
import sys
import random
import subprocess
import atexit
from glob import glob

import numpy as np
import tifffile
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# ============================================================
# CONFIG
# ============================================================
CONFIG = {
    "cell_patch_ratio": 0.75,
    "background_penalty_strength": 0.8,

    "load_pretrained": True,
    "pretrained_model_path": r"S:/Lab Data/Python 3.14.2/Machine Learning and Imaging/Final Model/Modified Models/All_Great/best_model.pth",

    "datasets": [
        {
            "name": "All_Great",
            "train_images_dir": "S:/Lab Data/Python 3.14.2/Machine Learning and Imaging/Final Model/Images/All_Great/images",
            "train_masks_dir": "S:/Lab Data/Python 3.14.2/Machine Learning and Imaging/Final Model/Images/All_Great/masks",
            "val_images_dir": "",
            "val_masks_dir": ""
        },
        {
            "name": "All_Moderate",
            "train_images_dir": "S:/Lab Data/Python 3.14.2/Machine Learning and Imaging/Final Model/Images/All_Moderate/images",
            "train_masks_dir": "S:/Lab Data/Python 3.14.2/Machine Learning and Imaging/Final Model/Images/All_Moderate/masks",
            "val_images_dir": "",
            "val_masks_dir": ""
        },
        {
            "name": "All_Poor",
            "train_images_dir": "S:/Lab Data/Python 3.14.2/Machine Learning and Imaging/Final Model/Images/All_Poor/images",
            "train_masks_dir": "S:/Lab Data/Python 3.14.2/Machine Learning and Imaging/Final Model/Images/All_Poor/masks",
            "val_images_dir": "",
            "val_masks_dir": ""
        },
        {
            "name": "HighSNR_Great",
            "train_images_dir": "S:/Lab Data/Python 3.14.2/Machine Learning and Imaging/Final Model/Images/HighSNR_Great/images",
            "train_masks_dir": "S:/Lab Data/Python 3.14.2/Machine Learning and Imaging/Final Model/Images/HighSNR_Great/masks",
            "val_images_dir": "",
            "val_masks_dir": ""
        },
        {
            "name": "HighSNR_Moderate",
            "train_images_dir": "S:/Lab Data/Python 3.14.2/Machine Learning and Imaging/Final Model/Images/HighSNR_Moderate/images",
            "train_masks_dir": "S:/Lab Data/Python 3.14.2/Machine Learning and Imaging/Final Model/Images/HighSNR_Moderate/masks",
            "val_images_dir": "",
            "val_masks_dir": ""
        },
        {
            "name": "HighSNR_Poor",
            "train_images_dir": "S:/Lab Data/Python 3.14.2/Machine Learning and Imaging/Final Model/Images/HighSNR_Poor/images",
            "train_masks_dir": "S:/Lab Data/Python 3.14.2/Machine Learning and Imaging/Final Model/Images/HighSNR_Poor/masks",
            "val_images_dir": "",
            "val_masks_dir": ""
        },
        {
            "name": "LowSNR_Great",
            "train_images_dir": "S:/Lab Data/Python 3.14.2/Machine Learning and Imaging/Final Model/Images/LowSNR_Great/images",
            "train_masks_dir": "S:/Lab Data/Python 3.14.2/Machine Learning and Imaging/Final Model/Images/LowSNR_Great/masks",
            "val_images_dir": "",
            "val_masks_dir": ""
        },
        {
            "name": "LowSNR_Moderate",
            "train_images_dir": "S:/Lab Data/Python 3.14.2/Machine Learning and Imaging/Final Model/Images/LowSNR_Moderate/images",
            "train_masks_dir": "S:/Lab Data/Python 3.14.2/Machine Learning and Imaging/Final Model/Images/LowSNR_Moderate/masks",
            "val_images_dir": "",
            "val_masks_dir": ""
        },
        {
            "name": "LowSNR_Poor",
            "train_images_dir": "S:/Lab Data/Python 3.14.2/Machine Learning and Imaging/Final Model/Images/LowSNR_Poor/images",
            "train_masks_dir": "S:/Lab Data/Python 3.14.2/Machine Learning and Imaging/Final Model/Images/LowSNR_Poor/masks",
            "val_images_dir": "",
            "val_masks_dir": ""
        },
    ],

    "output_root_dir": "S:/Lab Data/Python 3.14.2/Machine Learning and Imaging/Final Model/Modified Models",

    "val_fraction": 0.2,
    "random_seed": 15,

    "base_filters": 12,

    "epochs": 512,
    "batch_size": 16,
    "patch_size": 128,  
    "patches_per_volume": 8,
    "lr": 5e-4,

    "early_stopping_patience": 64,
    "early_stopping_min_delta": 1e-4,

    "num_workers": 1,
    "persistent_workers": True,
    "prefetch_factor": 1,
    "pin_memory": True,

    "use_amp": True,

    # NEW
    "threshold_steps": 50,

    # ============================
    # CALIBRATION (ADDED)
    # ============================
    "calibration_images_dir": "D:/Downloads/Fluo-N3DH-SIM+/Fluo-N3DH-SIM+/01",
    "calibration_masks_dir": "D:/Downloads/Fluo-N3DH-SIM+/Fluo-N3DH-SIM+/01_GT/TRA",
    "num_calibration_samples": 25,
    "use_calibration_for_threshold": True,
    "calibration_weight": 1.0,
    # ============================
}


# ============================================================
# SEED
# ============================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_seed(CONFIG["random_seed"])

# ============================================================
# SAFE TIFF LOADER
# ============================================================

def safe_read_tiff(path):
    try:
        return tifffile.memmap(path)
    except Exception:
        return tifffile.imread(path)

def normalize_volume(img):
    img = img.astype(np.float32)

    # handle constant images safely
    if img.max() == img.min():
        return np.zeros_like(img, dtype=np.float32)

    p1, p99 = np.percentile(img, (1, 99))

    if (p99 - p1) < 1e-6:
        return np.zeros_like(img, dtype=np.float32)

    img = (img - p1) / (p99 - p1 + 1e-6)
    img = np.clip(img, 0, 1)

    return img.astype(np.float32)

# ============================================================
# LOAD MODEL WEIGHTS
# ============================================================

def load_model_weights(model, path):
    if not path:
        print("WARNING: pretrained_model_path is empty")
        return

    if not os.path.exists(path):
        print(f"WARNING: pretrained model not found: {path}")
        return

    print(f"Loading pretrained model: {path}")
    state = torch.load(path, map_location=device)
    model.load_state_dict(state, strict=False)


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
                import ctypes
                ctypes.windll.kernel32.SetThreadExecutionState(
                    0x80000000 | 0x00000001
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
# MODEL
# ============================================================

class DoubleConv(nn.Module):
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
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.seq = nn.Sequential(
            nn.MaxPool3d(2),
            DoubleConv(in_ch, out_ch),
        )

    def forward(self, x):
        return self.seq(x)


class Up(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_ch // 2, in_ch // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        diff_z = x2.size(2) - x1.size(2)
        diff_y = x2.size(3) - x1.size(3)
        diff_x = x2.size(4) - x1.size(4)

        x1 = F.pad(
            x1,
            (
                diff_x // 2, diff_x - diff_x // 2,
                diff_y // 2, diff_y - diff_y // 2,
                diff_z // 2, diff_z - diff_z // 2,
            ),
        )

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class UNet3D(nn.Module):
    def __init__(self, in_ch, base):
        super().__init__()
        self.inc = DoubleConv(in_ch, base)
        self.down1 = Down(base, base * 2)
        self.down2 = Down(base * 2, base * 4)
        self.down3 = Down(base * 4, base * 8)
        self.down4 = Down(base * 8, base * 8)

        self.up1 = Up(base * 16, base * 4)
        self.up2 = Up(base * 8, base * 2)
        self.up3 = Up(base * 4, base)
        self.up4 = Up(base * 2, base)

        self.outc = nn.Conv3d(base, 1, kernel_size=1)

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
# BETTER LOSS: FOCAL TVERSKY (DROP-IN REPLACEMENT)
# ============================================================

class FocalTverskyLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=0.5, gamma=0.75):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def forward(self, logits, target):
        probs = torch.sigmoid(logits)
        target = (target > 0.5).float()

        dims = (1, 2, 3, 4)

        TP = (probs * target).sum(dims)
        FP = (probs * (1 - target)).sum(dims)
        FN = ((1 - probs) * target).sum(dims)

        tversky = (TP + 1e-6) / (
            TP + self.alpha * FN + self.beta * FP + 1e-6
        )

        return ((1 - tversky) ** self.gamma).mean()


# ============================================================
# DATASET
# ============================================================

class PatchDataset(Dataset):
    def __init__(self, imgs, masks, patch_size, patches_per_volume, cell_patch_ratio):
        self.imgs = imgs
        self.masks = masks
        self.patch_size = int(patch_size)
        self.patches_per_volume = int(patches_per_volume)
        self.cell_patch_ratio = float(cell_patch_ratio)

        self.cache = {}
        self.length = len(imgs) * patches_per_volume

    def __len__(self):
        return self.length

    def load(self, idx):
        if idx in self.cache:
            return self.cache[idx]

        img = safe_read_tiff(self.imgs[idx])
        mask = safe_read_tiff(self.masks[idx])

        img = normalize_volume(img)

        mask = (mask > 0)

        # 🔥 PRECOMPUTE ONCE
        coords = np.argwhere(mask)

        edge = np.logical_xor(mask, np.pad(mask, 1)[1:-1,1:-1,1:-1])
        edge_coords = np.argwhere(edge)

        if img.ndim == 3:
            img = img[None]

        self.cache[idx] = (img, mask, coords, edge_coords)
        return self.cache[idx]

    def get_random_start(self, size, patch):
        if size <= patch:
            return 0
        return random.randint(0, size - patch)

    def get_cell_center(self, mask):
        coords = np.argwhere(mask)
        if len(coords) == 0:
            return None
        return coords[np.random.randint(0, len(coords))]

    def get_cell_patch_start(self, center, shape, patch):
        zc, yc, xc = int(center[0]), int(center[1]), int(center[2])
        z_max = max(0, shape[0] - patch)
        y_max = max(0, shape[1] - patch)
        x_max = max(0, shape[2] - patch)

        z = min(max(zc - patch // 2, 0), z_max)
        y = min(max(yc - patch // 2, 0), y_max)
        x = min(max(xc - patch // 2, 0), x_max)

        return z, y, x

    def __getitem__(self, idx):
        vidx = random.randint(0, len(self.imgs) - 1)
        img, mask, coords, edge_coords = self.load(vidx)

        _, z_dim, y_dim, x_dim = img.shape
        p = self.patch_size

        # ============================================================
        # SMART SAMPLING (CELLS + EDGES)
        # ============================================================

        use_cell_patch = (random.random() < self.cell_patch_ratio)

        if use_cell_patch:
            center = self.get_cell_center(mask)
        else:
            # bias toward edges instead of empty space
            edge = np.logical_xor(mask, np.pad(mask, 1)[1:-1,1:-1,1:-1])
            coords = edge_coords

            if len(coords) > 0:
                center = coords[np.random.randint(0, len(coords))]
            else:
                center = None

        if use_cell_patch:
            center = coords[np.random.randint(0, len(coords))] if len(coords) > 0 else None
            if center is not None:
                z, y, x = self.get_cell_patch_start(center, mask.shape, p)
            else:
                z = self.get_random_start(z_dim, p)
                y = self.get_random_start(y_dim, p)
                x = self.get_random_start(x_dim, p)
        else:
            z = self.get_random_start(z_dim, p)
            y = self.get_random_start(y_dim, p)
            x = self.get_random_start(x_dim, p)

        img_patch = img[:, z:z+p, y:y+p, x:x+p]
        mask_patch = mask[z:z+p, y:y+p, x:x+p]

        img_patch = np.asarray(img_patch, dtype=np.float32)
        mask_patch = np.asarray(mask_patch, dtype=np.float32)

        img_patch, mask_patch = augment_3d(img_patch, mask_patch)

        # 🔥 FIX: force contiguous memory
        img_patch = np.ascontiguousarray(img_patch)
        mask_patch = np.ascontiguousarray(mask_patch)

        return (
            torch.from_numpy(img_patch),
            torch.from_numpy(mask_patch[None]),
)

# ============================================================
# CALIBRATION HELPERS (ADDED)
# ============================================================

def load_calibration_data():
    if not CONFIG.get("use_calibration_for_threshold", False):
        return [], []

    img_dir = CONFIG["calibration_images_dir"]
    mask_dir = CONFIG["calibration_masks_dir"]

    if not os.path.exists(img_dir) or not os.path.exists(mask_dir):
        print("Calibration folders not found — skipping.")
        return [], []

    imgs = sorted(glob(os.path.join(img_dir, "*.tif")))
    masks = sorted(glob(os.path.join(mask_dir, "*.tif")))

    if len(imgs) == 0 or len(masks) == 0:
        print("No calibration data found.")
        return [], []

    if len(imgs) != len(masks):
        raise RuntimeError("Calibration images/masks mismatch")

    n = min(CONFIG["num_calibration_samples"], len(imgs))
    idxs = np.random.choice(len(imgs), n, replace=False)

    imgs = [imgs[i] for i in idxs]
    masks = [masks[i] for i in idxs]

    print(f"Using {len(imgs)} calibration samples")

    return imgs, masks


def compute_probs_for_dataset(model, img_paths, mask_paths):
    probs_list = []
    targets_list = []

    for img_path, mask_path in tqdm(
        list(zip(img_paths, mask_paths)),
        total=len(img_paths),
        desc="Calibration inference"
    ):
        img = safe_read_tiff(img_path)
        img = normalize_volume(img)

        mask = safe_read_tiff(mask_path)

        mask = (mask > 0).astype(np.float32)

        if img.ndim == 3:
            img = img[None]

        img = torch.from_numpy(img.astype(np.float32))

        probs = sliding_window_inference(
            model,
            img,
            CONFIG["patch_size"],
            device
        )

        probs_list.append(probs.cpu())
        targets_list.append(torch.from_numpy(mask[None]))

    return probs_list, targets_list

# ============================================================
# SOFT DICE (NO THRESHOLD BIAS)
# ============================================================

def dice_soft(logits, target):
    probs = torch.sigmoid(logits)
    target = (target > 0.5).float()

    inter = (probs * target).sum()
    denom = probs.sum() + target.sum()

    return ((2.0 * inter + 1e-6) / (denom + 1e-6)).item()


# ============================================================
# NEW: THRESHOLD UTILITIES
# ============================================================

def dice_at_threshold(probs, target, thresh):
    pred = (probs > thresh).float()
    target = (target > 0.5).float()

    inter = (pred * target).sum()
    denom = pred.sum() + target.sum()

    return ((2.0 * inter + 1e-6) / (denom + 1e-6)).item()


def find_best_threshold(model, val_loader, out_dir):
    print("\nRunning threshold sweep...")

    model.eval()

    thresholds = np.linspace(0.05, 0.999, CONFIG["threshold_steps"])
    dice_scores = []

    all_probs = []
    all_targets = []

    with torch.no_grad():
        for xb, yb in tqdm(val_loader, desc="Collecting predictions"):
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            if device.type == "cuda":
                xb = xb.contiguous(memory_format=torch.channels_last_3d)

            logits = model(xb)
            probs = torch.sigmoid(logits)

            all_probs.append(probs.cpu())
            all_targets.append(yb.cpu())

    all_probs = torch.cat(all_probs, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    for t in tqdm(thresholds, desc="Sweeping thresholds"):
        dices = []
        for i in range(all_probs.shape[0]):
            d = dice_at_threshold(all_probs[i], all_targets[i], t)
            dices.append(d)
        dice_scores.append(np.mean(dices))

    best_idx = int(np.argmax(dice_scores))
    best_thresh = thresholds[best_idx]

    # save plot
    plt.figure()
    plt.plot(thresholds, dice_scores)
    plt.xlabel("Threshold")
    plt.ylabel("Dice")
    plt.title("Threshold vs Dice")
    plt.grid(True)
    plt.savefig(os.path.join(out_dir, "threshold_vs_dice.png"))
    plt.close()

    # save threshold
    with open(os.path.join(out_dir, "best_threshold.txt"), "w") as f:
        f.write(str(best_thresh))

    print(f"Best threshold: {best_thresh:.4f}")

# ============================================================
# NEW: FULL-VOLUME INFERENCE + THRESHOLD SWEEP (ADDED ONLY)
# ============================================================

def sliding_window_inference(model, volume, patch_size, device):
    model.eval()

    _, Z, Y, X = volume.shape
    p = patch_size
    stride = p // 2  # overlap for smoother stitching

    output = torch.zeros((1, Z, Y, X), dtype=torch.float32, device=device)
    count_map = torch.zeros_like(output)

    with torch.no_grad():
        for z in range(0, Z, stride):
            for y in range(0, Y, stride):
                for x in range(0, X, stride):

                    z0 = min(z, Z - p)
                    y0 = min(y, Y - p)
                    x0 = min(x, X - p)

                    patch = volume[:, z0:z0+p, y0:y0+p, x0:x0+p]
                    patch = patch.unsqueeze(0).to(device)

                    if device.type == "cuda":
                        patch = patch.contiguous(memory_format=torch.channels_last_3d)

                    logits = model(patch)
                    probs = torch.sigmoid(logits)[0]

                    output[:, z0:z0+p, y0:y0+p, x0:x0+p] += probs
                    count_map[:, z0:z0+p, y0:y0+p, x0:x0+p] += 1

    output /= count_map.clamp(min=1)
    return output

def find_best_threshold_full(model, val_imgs, val_masks, out_dir):
    print("\nRunning FULL-VOLUME threshold sweep (STREAMING + CALIBRATION)...")

    model.eval()

    thresholds = np.linspace(0.00, 1.0, CONFIG["threshold_steps"])
    dice_scores = np.zeros(len(thresholds), dtype=np.float64)

    val_weight = float(1.0 - CONFIG["calibration_weight"])
    cal_weight = float(CONFIG["calibration_weight"])

    # ----------------------------------------
    # LOAD CALIBRATION PATHS (NOT DATA)
    # ----------------------------------------
    cal_imgs, cal_masks = load_calibration_data()

    val_count = 0
    cal_count = 0

    with torch.no_grad():

        # ======================================
        # VALIDATION LOOP
        # ======================================
        for img_path, mask_path in tqdm(
            list(zip(val_imgs, val_masks)),
            total=len(val_imgs),
            desc="Validation sweep"
        ):
            img = normalize_volume(safe_read_tiff(img_path))
            mask = (safe_read_tiff(mask_path) > 0).astype(np.float32)

            if img.ndim == 3:
                img = img[None]

            img = torch.from_numpy(img.astype(np.float32))

            probs = sliding_window_inference(
                model,
                img,
                CONFIG["patch_size"],
                device
            ).cpu()

            target = torch.from_numpy(mask[None])

            # downsample (optional but recommended)
            probs = probs[:, ::2, ::2, ::2]
            target = target[:, ::2, ::2, ::2]

            for i, t in enumerate(thresholds):
                pred = (probs > t).float()

                inter = (pred * target).sum()
                denom = pred.sum() + target.sum()

                dice = (2.0 * inter + 1e-6) / (denom + 1e-6)
                dice_scores[i] += val_weight * dice.item()

            val_count += 1

            del probs, target, pred
            if device.type == "cuda":
                torch.cuda.empty_cache()

        # ======================================
        # CALIBRATION LOOP (SEPARATE PASS)
        # ======================================
        if len(cal_imgs) > 0:
            for img_path, mask_path in tqdm(
                list(zip(cal_imgs, cal_masks)),
                total=len(cal_imgs),
                desc="Calibration sweep"
            ):
                img = normalize_volume(safe_read_tiff(img_path))
                mask = (safe_read_tiff(mask_path) > 0).astype(np.float32)

                if img.ndim == 3:
                    img = img[None]

                img = torch.from_numpy(img.astype(np.float32))

                probs = sliding_window_inference(
                    model,
                    img,
                    CONFIG["patch_size"],
                    device
                ).cpu()

                target = torch.from_numpy(mask[None])

                probs = probs[:, ::2, ::2, ::2]
                target = target[:, ::2, ::2, ::2]

                for i, t in enumerate(thresholds):
                    pred = (probs > t).float()

                    inter = (pred * target).sum()
                    denom = pred.sum() + target.sum()

                    dice = (2.0 * inter + 1e-6) / (denom + 1e-6)
                    dice_scores[i] += cal_weight * dice.item()

                cal_count += 1

                del probs, target, pred
                if device.type == "cuda":
                    torch.cuda.empty_cache()

    # ----------------------------------------
    # CONVERT SUM TO WEIGHTED MEAN
    # ----------------------------------------
    normalizer = 0.0
    if val_count > 0:
        normalizer += val_weight * val_count
    if cal_count > 0:
        normalizer += cal_weight * cal_count

    if normalizer > 0:
        dice_scores /= normalizer

    best_idx = int(np.argmax(dice_scores))
    best_thresh = thresholds[best_idx]

    plt.figure()
    plt.plot(thresholds, dice_scores)
    plt.xlabel("Threshold")
    plt.ylabel("Dice (Val + Calibration)")
    plt.grid(True)
    plt.savefig(os.path.join(out_dir, "threshold_vs_dice_combined_stream.png"))
    plt.close()

    with open(os.path.join(out_dir, "best_threshold_combined_stream.txt"), "w") as f:
        f.write(str(best_thresh))

    print(f"Best COMBINED threshold: {best_thresh:.4f}")

# ============================================================
# NEW: ROC + PRECISION-RECALL ANALYSIS (FULL VOLUME)
# ============================================================

def compute_roc_pr_full_lowmem(model, val_imgs, val_masks, out_dir, num_bins=512):
    print("\nRunning ROC + PR analysis (LOW MEMORY)...")

    model.eval()

    # histogram bins
    bin_edges = np.linspace(0.0, 1.0, num_bins + 1)
    tp_hist = np.zeros(num_bins, dtype=np.float64)
    fp_hist = np.zeros(num_bins, dtype=np.float64)

    # --------------------------------------------------------
    # STREAM THROUGH VOLUMES (NO STORAGE)
    # --------------------------------------------------------
    for img_path, mask_path in tqdm(
        list(zip(val_imgs, val_masks)),
        total=len(val_imgs),
        desc="Streaming inference (ROC/PR)"
    ):
        img = safe_read_tiff(img_path)
        img = normalize_volume(img)

        mask = safe_read_tiff(mask_path)

        mask = (mask > 0).astype(np.uint8)

        if img.ndim == 3:
            img = img[None]

        img = torch.from_numpy(img.astype(np.float32))

        probs = sliding_window_inference(
            model,
            img,
            CONFIG["patch_size"],
            device
        ).cpu().numpy()

        probs = probs.astype(np.float32).ravel()
        mask = mask.astype(np.float32).ravel()

        # bin indices
        inds = np.clip(
            np.searchsorted(bin_edges, probs, side="right") - 1,
            0,
            num_bins - 1
        ).astype(np.int32)

        # accumulate histograms
        tp_hist += np.bincount(
            inds,
            weights=mask,
            minlength=num_bins
        )

        fp_hist += np.bincount(
            inds,
            weights=(1 - mask),
            minlength=num_bins
        )

    # --------------------------------------------------------
    # CUMULATIVE SUM (HIGH → LOW threshold)
    # --------------------------------------------------------
    tp_cum = np.cumsum(tp_hist[::-1])
    fp_cum = np.cumsum(fp_hist[::-1])

    total_pos = tp_cum[-1] + 1e-8
    total_neg = fp_cum[-1] + 1e-8

    tpr = tp_cum / total_pos
    fpr = fp_cum / total_neg

    precision = tp_cum / (tp_cum + fp_cum + 1e-8)
    recall = tpr

    # --------------------------------------------------------
    # AUC
    # --------------------------------------------------------
    roc_auc = np.trapezoid(tpr, fpr)
    # sort by recall (required for correct AUC)
    order = np.argsort(recall)
    recall_sorted = recall[order]
    precision_sorted = precision[order]

    pr_auc = np.trapezoid(precision_sorted, recall_sorted)

    # --------------------------------------------------------
    # BEST THRESHOLDS
    # --------------------------------------------------------
    thresholds = bin_edges[:-1][::-1]

    # ROC (Youden J)
    j_scores = tpr - fpr
    best_roc_thresh = thresholds[np.argmax(j_scores)]

    # PR (F1)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    best_pr_thresh = thresholds[np.argmax(f1)]

    # --------------------------------------------------------
    # PLOTS
    # --------------------------------------------------------
    plt.figure()
    plt.plot(fpr, tpr)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve (AUC = {roc_auc:.4f})")
    plt.grid(True)
    plt.savefig(os.path.join(out_dir, "roc_curve_lowmem.png"))
    plt.close()

    plt.figure()
    plt.plot(recall, precision)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"PR Curve (AUC = {pr_auc:.4f})")
    plt.grid(True)
    plt.savefig(os.path.join(out_dir, "pr_curve_lowmem.png"))
    plt.close()

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------
    with open(os.path.join(out_dir, "roc_pr_metrics_lowmem.txt"), "w") as f:
        f.write(f"ROC AUC: {roc_auc:.6f}\n")
        f.write(f"PR AUC: {pr_auc:.6f}\n")
        f.write(f"Best ROC threshold: {best_roc_thresh:.6f}\n")
        f.write(f"Best PR threshold (F1): {best_pr_thresh:.6f}\n")

    print(f"ROC AUC: {roc_auc:.4f}")
    print(f"PR AUC: {pr_auc:.4f}")
    print(f"Best ROC threshold: {best_roc_thresh:.4f}")
    print(f"Best PR threshold: {best_pr_thresh:.4f}")


# ============================================================
# TRAIN
# ============================================================

def train_dataset(ds):
    print(f"\n===== Training {ds['name']} =====")

    out_dir = os.path.join(CONFIG["output_root_dir"], ds["name"])
    os.makedirs(out_dir, exist_ok=True)

    imgs = sorted(glob(os.path.join(ds["train_images_dir"], "*.tif")))
    masks = sorted(glob(os.path.join(ds["train_masks_dir"], "*.tif")))

    if len(imgs) == 0:
        print(f"WARNING: no images found for {ds['name']}")
        return
    if len(masks) == 0:
        print(f"WARNING: no masks found for {ds['name']}")
        return
    if len(imgs) != len(masks):
        raise RuntimeError(
            f"Image/mask count mismatch for {ds['name']}: "
            f"{len(imgs)} images vs {len(masks)} masks"
        )

    split = max(1, int(len(imgs) * CONFIG["val_fraction"]))

    val_imgs = imgs[:split]
    val_masks = masks[:split]
    train_imgs = imgs[split:]
    train_masks = masks[split:]

    if len(train_imgs) == 0:
        raise RuntimeError(
            f"No training images remain after split for {ds['name']}. "
            f"Need more files or lower val_fraction."
        )

    sample = tifffile.imread(train_imgs[0])
    in_ch = 1 if sample.ndim == 3 else sample.shape[0] if sample.shape[0] <= 4 else sample.shape[-1]

    model = UNet3D(in_ch, CONFIG["base_filters"]).to(device)
    if device.type == "cuda":
        model = model.to(memory_format=torch.channels_last_3d)

    if CONFIG["load_pretrained"]:
        load_model_weights(model, CONFIG["pretrained_model_path"])

    train_dataset_obj = PatchDataset(
        train_imgs,
        train_masks,
        CONFIG["patch_size"],
        CONFIG["patches_per_volume"],
        CONFIG["cell_patch_ratio"],
    )

    val_dataset_obj = PatchDataset(
        val_imgs,
        val_masks,
        CONFIG["patch_size"],
        6,  # more coverage
        CONFIG["cell_patch_ratio"],
    )

    train_loader = DataLoader(
        train_dataset_obj,
        batch_size=CONFIG["batch_size"],
        shuffle=True,
        num_workers=CONFIG["num_workers"],
        pin_memory=CONFIG["pin_memory"],
        persistent_workers=CONFIG["persistent_workers"] if CONFIG["num_workers"] > 0 else False,
        prefetch_factor=CONFIG["prefetch_factor"] if CONFIG["num_workers"] > 0 else None,
        drop_last=False,
    )

    val_loader = DataLoader(
        val_dataset_obj,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=CONFIG["pin_memory"],
    )

    opt = torch.optim.AdamW(model.parameters(), lr=CONFIG["lr"])
    loss_fn = nn.BCEWithLogitsLoss(
    pos_weight=torch.tensor([8.0], device=device)
    )

    scaler = torch.amp.GradScaler("cuda") if (CONFIG["use_amp"] and device.type == "cuda") else None

    best_val = -1.0
    no_improve = 0

    for epoch in range(CONFIG["epochs"]):
        model.train()
        running_loss = 0.0

        pbar = tqdm(train_loader, desc=f"{ds['name']} Epoch {epoch+1}/{CONFIG['epochs']}")
        for xb, yb in pbar:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            if device.type == "cuda":
                xb = xb.contiguous(memory_format=torch.channels_last_3d)

            opt.zero_grad(set_to_none=True)

            if scaler is not None:
                with torch.amp.autocast("cuda"):
                    pred = model(xb)
                    loss = loss_fn(pred, yb)

                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            else:
                pred = model(xb)
                loss = loss_fn(pred, yb)
                loss.backward()
                opt.step()

            running_loss += loss.item()
            avg_loss = running_loss / max(1, (pbar.n + 1))
            pbar.set_postfix(loss=f"{avg_loss:.4f}")

        train_loss = running_loss / max(1, len(train_loader))

        model.eval()
        val_dices = []

        with torch.no_grad():
            for img_path, mask_path in zip(val_imgs, val_masks):

                img = safe_read_tiff(img_path)
                img = normalize_volume(img)

                mask = safe_read_tiff(mask_path)

                if img.ndim == 3:
                    img = img[None]

                img = torch.from_numpy(img.astype(np.float32))

                probs = sliding_window_inference(
                    model,
                    img,
                    CONFIG["patch_size"],
                    device
                )

                # 🔥 LOWER THRESHOLD (VERY IMPORTANT)
                pred = (probs > 0.2).float()
                target = torch.from_numpy((mask > 0)[None].astype(np.float32)).to(device)

                inter = (pred * target).sum()
                denom = pred.sum() + target.sum()

                dice = (2 * inter + 1e-6) / (denom + 1e-6)
                val_dices.append(dice.item())

                # DEBUG (once)
                print(
                    f"[DEBUG] prob mean={probs.mean().item():.4f}, "
                    f"max={probs.max().item():.4f}"
                )
                break  # only print once

        val_dice = float(np.mean(val_dices))

        if device.type == "cuda":
            used_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
            reserved_mb = torch.cuda.max_memory_reserved(device) / (1024 ** 2)
            print(
                f"Train Loss: {train_loss:.4f} | "
                f"Val Dice: {val_dice:.4f} | "
                f"Max GPU Alloc: {used_mb:.1f} MB | "
                f"Max GPU Reserved: {reserved_mb:.1f} MB"
            )
            torch.cuda.reset_peak_memory_stats(device)
        else:
            print(f"Train Loss: {train_loss:.4f} | Val Dice: {val_dice:.4f}")

        if val_dice > best_val + CONFIG["early_stopping_min_delta"]:
            best_val = val_dice
            no_improve = 0
            torch.save(model.state_dict(), os.path.join(out_dir, "best_model.pth"))
            print("Saved best model")
        else:
            no_improve += 1
            print(f"No improvement ({no_improve}/{CONFIG['early_stopping_patience']})")

        torch.save(model.state_dict(), os.path.join(out_dir, "last_model.pth"))

        if no_improve >= CONFIG["early_stopping_patience"]:
            print("Early stopping triggered")
            break

    print(f"Finished: {ds['name']}")
    print(f"Best Val Dice: {best_val:.4f}")

    # ========================================================
    # NEW: THRESHOLD SEARCH (ADDED ONLY)
    # ========================================================

    model.load_state_dict(torch.load(os.path.join(out_dir, "best_model.pth"), map_location=device))
    find_best_threshold(model, val_loader, out_dir)
    find_best_threshold_full(model, val_imgs, val_masks, out_dir)
    compute_roc_pr_full_lowmem(model, val_imgs, val_masks, out_dir)

# ============================================================
# AUGMENTATION (FAST + LOW RAM)
# ============================================================

def augment_3d(img, mask):
    # Random flips
    if random.random() < 0.5:
        img = img[:, ::-1]
        mask = mask[::-1]
    if random.random() < 0.5:
        img = img[:, :, ::-1]
        mask = mask[:, ::-1]
    if random.random() < 0.5:
        img = img[:, :, :, ::-1]
        mask = mask[:, :, ::-1]

    # Random intensity scaling
    if random.random() < 0.5:
        scale = 0.9 + 0.2 * random.random()
        shift = 0.1 * (random.random() - 0.5)
        img = img * scale + shift

    # Noise
    if random.random() < 0.3:
        noise = np.random.normal(0, 0.02, size=img.shape).astype(np.float32)
        img = img + noise

    return img, mask


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    anti = AntiSleep()
    anti.start()

    # ============================================================
    # DEVICE
    # ============================================================

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("\n==============================")
    print("Device Information")
    print("==============================")
    if device.type == "cuda":
        print("Using CUDA")
        print("GPU:", torch.cuda.get_device_name(0))
        print(
            "Total Memory:",
            round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2),
            "GB",
        )
        torch.backends.cudnn.benchmark = True
    else:
        print("Using CPU")
    print("==============================\n")

    try:
        for ds in CONFIG["datasets"]:
            train_dataset(ds)
    finally:
        anti.stop()

    print("\nAll datasets finished.")