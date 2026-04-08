"""
Simulate and display the final applied confocal PSF
for four microscope objectives.

No image convolution — PSF only.

This script generates realistic 3D confocal PSFs for different
microscope objectives using the psfmodels library. It accounts
for excitation/emission wavelengths, refractive index mismatches,
coverslip thickness variations, and pinhole effects.

The resulting PSFs are:
- Normalized to sum to 1.0 (photon probability distribution)
- Saved as 16-bit TIFF stacks for downstream use
- Visualized as a side-by-side XY maximum-intensity slice comparison
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import tifffile
import psfmodels as psfm   # Gibson-Lanni / vectorial PSF model library


# ==========================================================
# ===================== USER SETTINGS ======================
# ==========================================================

# Toggle whether to display the PSF plot interactively
SHOW_PSF = False

# Whether to save the 3D PSF stacks and the summary subplot
SAVE_PSF = True

# Output directory (create if it doesn't exist)
OUTPUT_FOLDER = "S:/Lab Data/Python 3.14.2/Machine Learning and Imaging/Fake_Objective"

# Hoechst 33342 / 405 nm laser wavelengths (in microns)
EXCITATION_WAVELENGTH_UM = 0.405
EMISSION_WAVELENGTH_UM   = 0.460

# Sample refractive index (water-like mounting medium)
NS_SAMPLE = 1.33

# Design (nominal) coverslip thickness used by the manufacturer (microns)
TG0_DESIGN_COVERSLIP_UM = 170.0


# ==========================================================
# ===================== PSF GRID SETTINGS ==================
# ==========================================================

# Voxel size of the generated PSF (must be consistent across objectives)
COMMON_VOXEL_XY_UM = 0.08   # lateral sampling
COMMON_VOXEL_Z_UM  = 0.25   # axial sampling

# PSF volume dimensions (must be odd numbers for symmetric centering)
PSF_SIZE_XY_PX = 127        # X and Y pixels (127 × 127)
PSF_SIZE_Z_PX  = 63         # Z slices (63 slices centered at 0)


# ==========================================================
# ===================== OBJECTIVES =========================
# ==========================================================

# Dictionary of microscope objectives with their optical parameters.
# These parameters model real-world deviations from ideal conditions
# (e.g., spherical aberration from RI mismatch, coverslip thickness error).
OBJECTIVES = {

    "Great_Objective": {
        "NA": 1.40,          # Numerical aperture
        "NI0": 1.518,        # Design immersion oil RI
        "NI": 1.518,         # Actual immersion oil RI (perfect match)
        "NG0": 1.518,        # Design glass RI
        "NG": 1.518,         # Actual glass RI
        "TI0": 140.0,        # Design tube length (microns)
        "DEPTH": 0.0,        # Imaging depth into sample (microns)
        "PINHOLE": 1.0,      # Pinhole size factor (1.0 = ideal Airy disk)
        "TG": 170.0,         # Actual coverslip thickness (microns)
    },

    "Moderate_Objective": {
        "NA": 1.40,
        "NI0": 1.518,
        "NI": 1.518,
        "NG0": 1.518,
        "NG": 1.518,
        "TI0": 140.0,
        "DEPTH": 0.5,        # Slight defocus / depth
        "PINHOLE": 1.0,
        "TG": 170.0,
    },

    "Poor_Objective": {
        "NA": 1.35,          # Lower NA (worse resolution)
        "NI0": 1.518,
        "NI": 1.515,         # RI mismatch in immersion oil
        "NG0": 1.518,
        "NG": 1.518,
        "TI0": 140.0,
        "DEPTH": 3.0,        # Deeper imaging depth → more spherical aberration
        "PINHOLE": 1.3,      # Enlarged pinhole (reduced confocal sectioning)
        "TG": 180.0,         # Wrong coverslip thickness (+10 µm error)
    },

}


# ==========================================================
# ===================== PSF GENERATION =====================
# ==========================================================

def make_z_vector():
    """
    Create a symmetric Z coordinate vector centered at 0.
    Used for the axial sampling of the PSF.
    """
    half_range = (PSF_SIZE_Z_PX // 2) * COMMON_VOXEL_Z_UM
    return np.linspace(-half_range, half_range, PSF_SIZE_Z_PX, dtype=np.float32)


def make_scalar_psf(wvl_um, params):
    """
    Generate a single-wavelength (scalar) 3D PSF using the Gibson-Lanni model.

    Parameters
    ----------
    wvl_um : float
        Wavelength in microns (excitation or emission)
    params : dict
        Objective parameters (NA, immersion RI, etc.)

    Returns
    -------
    psf : np.ndarray
        3D PSF normalized so that sum(psf) = 1.0
    """
    zvec = make_z_vector()

    # psfm.make_psf returns the intensity PSF (already squared)
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

    # Normalize to total intensity = 1 (photon probability)
    psf /= psf.sum()
    return psf


def build_confocal_psf(params):
    """
    Build a confocal PSF by multiplying excitation and emission PSFs
    and applying the pinhole scaling factor.

    This simulates the confocal detection process (excitation * detection).

    Returns
    -------
    psf_conf : np.ndarray
        Final normalized 3D confocal PSF
    """
    psf_exc = make_scalar_psf(EXCITATION_WAVELENGTH_UM, params)
    psf_em  = make_scalar_psf(EMISSION_WAVELENGTH_UM,   params)

    # Confocal PSF = excitation PSF × emission PSF
    psf_conf = psf_exc * psf_em

    # Apply pinhole size scaling (larger pinhole = more light but worse sectioning)
    psf_conf *= (1.0 / params["PINHOLE"])

    # Final normalization
    psf_conf /= psf_conf.sum()

    return psf_conf


# ==========================================================
# ===================== VISUALIZATION ======================
# ==========================================================

def show_xy_comparison(psf_dict):
    """
    Interactive display of the central XY slice for each objective.
    Useful for quick visual inspection.
    """
    fig, axes = plt.subplots(1, len(psf_dict), figsize=(5 * len(psf_dict), 5))

    if len(psf_dict) == 1:
        axes = [axes]

    for ax, (name, psf) in zip(axes, psf_dict.items()):
        zc = psf.shape[0] // 2                    # central Z slice
        xy = psf[zc]
        xy_norm = xy / xy.max()                   # normalize for display

        ax.imshow(xy_norm, cmap="hot")
        ax.set_title(name, fontsize=12)
        ax.set_xlabel("X (pixels)")
        ax.set_ylabel("Y (pixels)")
        ax.axis("off")

    plt.tight_layout()
    plt.show()


def save_xy_subplot(psf_dict, output_folder):
    """
    Save a publication-ready side-by-side XY slice comparison as PNG.
    """
    fig, axes = plt.subplots(1, len(psf_dict), figsize=(5 * len(psf_dict), 5))

    if len(psf_dict) == 1:
        axes = [axes]

    for ax, (name, psf) in zip(axes, psf_dict.items()):
        zc = psf.shape[0] // 2
        xy = psf[zc]
        xy_norm = xy / xy.max()

        im = ax.imshow(xy_norm, cmap="hot")
        ax.set_title(name, fontsize=12)
        ax.set_xlabel("X (pixels)")
        ax.set_ylabel("Y (pixels)")
        ax.axis("off")

    plt.tight_layout()

    save_path = os.path.join(output_folder, "PSF_subplot.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved PSF subplot to: {save_path}")


# ==========================================================
# ===================== MAIN ===============================
# ==========================================================

def main():
    """
    Main execution routine:
    1. Create output folder
    2. Generate confocal PSF for every objective
    3. Save each PSF as a 16-bit TIFF stack
    4. Save a summary XY subplot
    """
    # Ensure output directory exists
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    psf_results = {}   # Store all generated PSFs for later visualization

    for name, params in OBJECTIVES.items():
        print(f"Generating confocal PSF for → {name}")
        
        psf_conf = build_confocal_psf(params)
        psf_results[name] = psf_conf

        if SAVE_PSF:
            # Save as 16-bit unsigned integer (0-65535) for compatibility
            # with ImageJ, Fiji, Python, etc.
            output_tif = os.path.join(OUTPUT_FOLDER, f"{name}_psf_3d.tif")
            # Scale to max = 65535 while preserving relative intensities
            scaled = (psf_conf / psf_conf.max() * 65535).astype(np.uint16)
            tifffile.imwrite(output_tif, scaled)
            print(f"   → Saved 3D PSF: {output_tif}")

    # Create and save the side-by-side XY comparison image
    save_xy_subplot(psf_results, OUTPUT_FOLDER)

    # Optional interactive display
    if SHOW_PSF:
        show_xy_comparison(psf_results)

    print("\nAll PSFs generated and saved successfully!")


if __name__ == "__main__":
    main()