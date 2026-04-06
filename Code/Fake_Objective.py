"""
Simulate and display the final applied confocal PSF
for four microscope objectives.

No image convolution — PSF only.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import tifffile
import psfmodels as psfm


# ==========================================================
# ===================== USER SETTINGS ======================
# ==========================================================

SHOW_PSF = False
SAVE_PSF = True
OUTPUT_FOLDER = "S:/Lab Data/Python 3.14.2/Machine Learning and Imaging/Fake_Objective"

# Hoechst wavelengths (microns)
EXCITATION_WAVELENGTH_UM = 0.405
EMISSION_WAVELENGTH_UM = 0.460

# Sample properties
NS_SAMPLE = 1.33
TG0_DESIGN_COVERSLIP_UM = 170.0

# Common PSF sampling grid (ensures aligned Z stacks)
COMMON_VOXEL_XY_UM = 0.08
COMMON_VOXEL_Z_UM  = 0.25

PSF_SIZE_XY_PX = 127  # must be odd
PSF_SIZE_Z_PX  = 63   # must be odd


# ==========================================================
# ===================== OBJECTIVES =========================
# ==========================================================

OBJECTIVES = {

    "Great_Objective": {
        "NA": 1.40,
        "NI0": 1.518,
        "NI": 1.518,
        "NG0": 1.518,
        "NG": 1.518,
        "TI0": 140.0,
        "DEPTH": 0.0,
        "PINHOLE": 1.0,
        "TG": 170.0,
    },

    "Moderate_Objective": {
        "NA": 1.40,
        "NI0": 1.518,
        "NI": 1.518,
        "NG0": 1.518,
        "NG": 1.518,
        "TI0": 140.0,
        "DEPTH": 0.5,
        "PINHOLE": 1.0,
        "TG": 170.0,
    },

    "Poor_Objective": {
        "NA": 1.35,
        "NI0": 1.518,
        "NI": 1.515,
        "NG0": 1.518,
        "NG": 1.518,
        "TI0": 140.0,
        "DEPTH": 3.0,
        "PINHOLE": 1.3,
        "TG": 180.0,
    },

}


# ==========================================================
# ===================== PSF GENERATION =====================
# ==========================================================

def make_z_vector():
    half_range = (PSF_SIZE_Z_PX // 2) * COMMON_VOXEL_Z_UM
    return np.linspace(-half_range, half_range, PSF_SIZE_Z_PX, dtype=np.float32)


def make_scalar_psf(wvl_um, params):

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

    psf /= psf.sum()
    return psf


def build_confocal_psf(params):

    psf_exc = make_scalar_psf(EXCITATION_WAVELENGTH_UM, params)
    psf_em  = make_scalar_psf(EMISSION_WAVELENGTH_UM, params)

    psf_conf = psf_exc * psf_em
    psf_conf *= (1.0 / params["PINHOLE"])
    psf_conf /= psf_conf.sum()

    return psf_conf


# ==========================================================
# ===================== VISUALIZATION ======================
# ==========================================================

def show_xy_comparison(psf_dict):

    fig, axes = plt.subplots(1, len(psf_dict), figsize=(5 * len(psf_dict), 5))

    if len(psf_dict) == 1:
        axes = [axes]

    for ax, (name, psf) in zip(axes, psf_dict.items()):
        zc = psf.shape[0] // 2
        xy = psf[zc]
        xy_norm = xy / xy.max()

        ax.imshow(xy_norm, cmap="hot")
        ax.set_title(name)
        ax.set_xlabel("X (pixels)")
        ax.set_ylabel("Y (pixels)")

    plt.tight_layout()
    plt.show()

def save_xy_subplot(psf_dict, output_folder):

    fig, axes = plt.subplots(1, len(psf_dict), figsize=(5 * len(psf_dict), 5))

    if len(psf_dict) == 1:
        axes = [axes]

    for ax, (name, psf) in zip(axes, psf_dict.items()):
        zc = psf.shape[0] // 2
        xy = psf[zc]
        xy_norm = xy / xy.max()

        im = ax.imshow(xy_norm, cmap="hot")
        ax.set_title(name)
        ax.set_xlabel("X (pixels)")
        ax.set_ylabel("Y (pixels)")

    plt.tight_layout()

    save_path = os.path.join(output_folder, "PSF_subplot.png")
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"Saved PSF subplot to: {save_path}")

# ==========================================================
# ===================== MAIN ===============================
# ==========================================================

def main():

    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    psf_results = {}

    for name, params in OBJECTIVES.items():
        print(f"Generating PSF for {name}")
        psf_conf = build_confocal_psf(params)
        psf_results[name] = psf_conf

        if SAVE_PSF:
            tifffile.imwrite(
                os.path.join(OUTPUT_FOLDER, f"{name}_psf_3d.tif"),
                (psf_conf / psf_conf.max() * 65535).astype(np.uint16)
            )
    save_xy_subplot(psf_results, OUTPUT_FOLDER)

    if SHOW_PSF:
        show_xy_comparison(psf_results)


if __name__ == "__main__":
    main()
    