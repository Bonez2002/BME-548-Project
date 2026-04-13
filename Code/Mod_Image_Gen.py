"""
MODIFIED IMAGE GENERATOR — PSF Simulation + Convolution
========================================================

This script does TWO things:

1. Generates realistic 3D confocal PSFs for three microscope objectives
   (Great, Moderate, Poor) using the Gibson-Lanni model (psfmodels library).

2. Applies each PSF to EVERY 3D image in a source folder using fast FFT convolution.

CHANGES MADE (per your request):
- Normalization now happens FIRST on the clean input image:
  → Min-max normalization across the WHOLE 3D volume (global min & max)
  → This brings every image to [0, 1] range before blurring
- PSF convolution is then applied to the normalized image
- Final scaling to 16-bit (0–65535) is done directly on the blurred result
  (no re-stretching based on blurred max). This preserves the realistic
  peak-intensity loss caused by the PSF (spherical aberration, depth, etc.).

Result:
- Clean PSF .tif files (for inspection or later use)
- A set of blurred ("realistic") image folders, one per objective:
    Great_Objective/
    Moderate_Objective/
    Poor_Objective/

Perfect for creating synthetic training data that matches your actual microscope’s
optical imperfections.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import tifffile
import psfmodels as psfm
from scipy.signal import fftconvolve   # Fast 3D convolution


# ==========================================================
# ===================== USER SETTINGS ======================
# ==========================================================

# Toggle interactive display and saving
SAVE_PSF = True

# Base output folder
OUTPUT_FOLDER = "S:/Lab Data/Python 3.14.2/Machine Learning and Imaging/Images_Mod/dataset-all/images"

# >>> SOURCE FOLDER CONTAINING CLEAN 3D IMAGES TO BE BLURRED
INPUT_IMAGE_FOLDER = "S:/Lab Data/Python 3.14.2/Machine Learning and Imaging/Images/dataset-all/images"

# Hoechst 33342 wavelengths (microns)
EXCITATION_WAVELENGTH_UM = 0.405
EMISSION_WAVELENGTH_UM   = 0.460

# Sample refractive index (water-like)
NS_SAMPLE = 1.33

# Manufacturer’s nominal coverslip thickness
TG0_DESIGN_COVERSLIP_UM = 170.0


# ==========================================================
# ===================== PSF GRID SETTINGS ==================
# ==========================================================

# Voxel size of the generated PSF (same for all objectives)
COMMON_VOXEL_XY_UM = 0.08
COMMON_VOXEL_Z_UM  = 0.25

# PSF volume size (must be odd numbers for centering)
PSF_SIZE_XY_PX = 127
PSF_SIZE_Z_PX  = 63


# ==========================================================
# ===================== OBJECTIVES =========================
# ==========================================================

# Three realistic microscope objectives with different levels of aberration
OBJECTIVES = {

    "Great_Objective": {
        "NA": 1.40,
        "NI0": 1.518,        # Design immersion RI
        "NI": 1.518,         # Actual immersion RI
        "NG0": 1.518,
        "NG": 1.518,
        "TI0": 140.0,
        "DEPTH": 0.0,        # Imaging depth into sample
        "PINHOLE": 1.0,      # Ideal confocal pinhole
        "TG": 170.0,         # Actual coverslip thickness
    },

    "Moderate_Objective": {
        "NA": 1.40,
        "NI0": 1.518,
        "NI": 1.518,
        "NG0": 1.518,
        "NG": 1.518,
        "TI0": 140.0,
        "DEPTH": 0.5,        # Slight depth → minor aberration
        "PINHOLE": 1.0,
        "TG": 170.0,
    },

    "Poor_Objective": {
        "NA": 1.35,          # Lower NA
        "NI0": 1.518,
        "NI": 1.515,         # Immersion RI mismatch
        "NG0": 1.518,
        "NG": 1.518,
        "TI0": 140.0,
        "DEPTH": 3.0,        # Deeper imaging → strong spherical aberration
        "PINHOLE": 1.3,      # Enlarged pinhole (worse sectioning)
        "TG": 180.0,         # Wrong coverslip (+10 µm error)
    },
}


# ==========================================================
# ===================== PSF GENERATION =====================
# ==========================================================

def make_z_vector():
    """Create symmetric Z coordinate vector centered at 0."""
    half_range = (PSF_SIZE_Z_PX // 2) * COMMON_VOXEL_Z_UM
    return np.linspace(-half_range, half_range, PSF_SIZE_Z_PX, dtype=np.float32)


def make_scalar_psf(wvl_um, params):
    """Generate a single-wavelength (scalar) 3D PSF."""
    zvec = make_z_vector()

    psf = psfm.make_psf(
        zvec,
        nx=PSF_SIZE_XY_PX,
        dxy=COMMON_VOXEL_XY_UM,
        pz=params["DEPTH"],
        ti0=params["TI0"],
        ni0=params["NI0"],
        ni=params["NI"],
        tg0=TG0_DESIGN_COVERSLIP_UM,
        tg=params["TG"],
        ng0=params["NG0"],
        ng=params["NG"],
        ns=NS_SAMPLE,
        wvl=wvl_um,
        NA=params["NA"],
    ).astype(np.float32)

    psf /= psf.sum()          # Normalize to sum = 1
    return psf


def build_confocal_psf(params):
    """Build confocal PSF = excitation PSF × emission PSF + pinhole scaling."""
    psf_exc = make_scalar_psf(EXCITATION_WAVELENGTH_UM, params)
    psf_em  = make_scalar_psf(EMISSION_WAVELENGTH_UM,   params)

    psf_conf = psf_exc * psf_em
    psf_conf *= (1.0 / params["PINHOLE"])   # Larger pinhole = more light
    psf_conf /= psf_conf.sum()

    return psf_conf


# ==========================================================
# ===================== CONVOLUTION ========================
# ==========================================================

def apply_psf_to_image(image, psf):
    """
    Convolve a 3D image with a 3D PSF using FFT (very fast).

    Parameters
    ----------
    image : np.ndarray
        Shape (Z, Y, X) — normalized to [0, 1]
    psf   : np.ndarray
        3D PSF of shape (Z_psf, Y_psf, X_psf) normalized to sum = 1

    Returns
    -------
    blurred : np.ndarray
        Same shape as input, float32
    """
    # fftconvolve is the fastest and most accurate way for 3D microscopy data
    return fftconvolve(image, psf, mode='same')


def process_image_folder(psf_dict):
    """
    For each objective, create a subfolder and apply its PSF to every .tif
    in INPUT_IMAGE_FOLDER.
    
    NEW BEHAVIOR:
      1. Min-max normalize the WHOLE 3D image to [0, 1]
      2. Apply PSF convolution
      3. Scale result directly to uint16 (0–65535)
         → Peak intensity loss from blurring is preserved (realistic)
    """
    # Get all TIFF files in the source folder
    image_files = [f for f in os.listdir(INPUT_IMAGE_FOLDER) if f.lower().endswith(".tif")]

    print(f"Found {len(image_files)} images to process.")

    for obj_name, psf in psf_dict.items():
        print(f"\nApplying PSF → {obj_name}")

        # Create output subfolder for this objective
        obj_output_dir = os.path.join(OUTPUT_FOLDER, obj_name)
        os.makedirs(obj_output_dir, exist_ok=True)

        for fname in image_files:
            path = os.path.join(INPUT_IMAGE_FOLDER, fname)

            img = tifffile.imread(path).astype(np.float32)

            print(f"   → Processing {fname}  (shape: {img.shape})")

            # ===================== NEW NORMALIZATION (WHOLE 3D VOLUME) =====================
            img_min = img.min()
            img_max = img.max()
            if img_max > img_min + 1e-8:          # avoid division by zero
                img_norm = (img - img_min) / (img_max - img_min)
            else:
                img_norm = np.zeros_like(img, dtype=np.float32)

            # ===================== APPLY PSF AFTER NORMALIZATION =====================
            blurred = apply_psf_to_image(img_norm, psf)

            # Remove any tiny negative values from FFT numerical precision
            blurred = np.clip(blurred, 0.0, None)

            # Scale normalized blurred result directly to 16-bit
            # (peak reduction from PSF is preserved — no re-stretching)
            blurred = (blurred * 65535.0).astype(np.uint16)

            # Save blurred version with same filename
            save_path = os.path.join(obj_output_dir, fname)
            tifffile.imwrite(save_path, blurred)

    print("\nFinished applying all PSFs to the image folder!")


# ==========================================================
# ===================== VISUALIZATION ======================
# ==========================================================

def save_xy_subplot(psf_dict, output_folder):
    """Save a side-by-side central XY slice of all PSFs (publication-ready)."""
    fig, axes = plt.subplots(1, len(psf_dict), figsize=(5 * len(psf_dict), 5))

    if len(psf_dict) == 1:
        axes = [axes]

    for ax, (name, psf) in zip(axes, psf_dict.items()):
        zc = psf.shape[0] // 2
        xy = psf[zc]
        xy_norm = xy / xy.max()

        ax.imshow(xy_norm, cmap="hot")
        ax.set_title(name)
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, "PSF_subplot.png"), dpi=300)
    plt.close()


# ==========================================================
# ===================== MAIN ===============================
# ==========================================================

def main():
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    psf_results = {}

    # 1. Generate all PSFs
    for name, params in OBJECTIVES.items():
        print(f"Generating confocal PSF for → {name}")
        
        psf_conf = build_confocal_psf(params)
        psf_results[name] = psf_conf

        if SAVE_PSF:
            output_tif = os.path.join(OUTPUT_FOLDER, f"{name}_psf_3d.tif")
            scaled = (psf_conf / psf_conf.max() * 65535).astype(np.uint16)
            tifffile.imwrite(output_tif, scaled)
            print(f"   → Saved PSF: {output_tif}")

    # 2. Save PSF comparison image
    save_xy_subplot(psf_results, OUTPUT_FOLDER)

    # 3. Apply each PSF to the entire input image folder
    process_image_folder(psf_results)

    print("\n All PSFs generated and applied successfully!")


if __name__ == "__main__":
    main()