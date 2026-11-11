"""
Modul rekomendasi cerdas untuk Image Enhancement Studio.
Menganalisis karakteristik statistik dan distribusi intensitas citra untuk
memberikan rekomendasi metode yang optimal secara deterministik.
"""

import cv2
import numpy as np

# ==============================================================================
# KONSTANTA AMBANG BATAS KARAKTERISTIK CITRA (SKALA LUMINANCE 0–255)
# ==============================================================================
THRESHOLD_DARK_MEAN = 85.0              # Di bawah 85: citra gelap / underexposed
THRESHOLD_BRIGHT_MEAN = 170.0           # Di atas 170: citra terang / overexposed
THRESHOLD_LOW_CONTRAST_STD = 40.0       # Di bawah 40: kontras rendah
THRESHOLD_NARROW_DYNAMIC_RANGE = 200    # Di bawah 200: rentang dinamis sempit
THRESHOLD_UNEVEN_LIGHTING_STD = 30.0    # Di atas 30: pencahayaan lokal tidak merata
THRESHOLD_HIGH_NOISE = 5.0              # Di atas 5.0: residu frekuensi tinggi mengindikasikan noise

METHOD_HE = "Histogram Equalization"
METHOD_CLAHE = "Contrast Limited Adaptive Histogram Equalization"
METHOD_MATCHING = "Histogram Matching"
METHOD_BILATERAL = "Histogram Equalization + Bilateral Filtering"


def _to_gray(image: np.ndarray) -> np.ndarray:
    """Normalisasi citra ke 2D grayscale uint8."""
    if not isinstance(image, np.ndarray):
        raise TypeError("Input harus berupa NumPy array.")

    if image.size == 0:
        raise ValueError("Citra tidak boleh kosong.")

    if image.ndim == 3 and image.shape[2] == 3:
        return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    elif image.ndim == 3 and image.shape[2] == 1:
        return image.squeeze(axis=2)
    elif image.ndim == 2:
        return np.clip(image, 0, 255).astype(np.uint8)
    else:
        raise ValueError("Format citra tidak valid.")


def estimate_noise_level(gray: np.ndarray) -> float:
    """
    Estimasi tingkat derau (noise) dari residu frekuensi tinggi
    menggunakan selisih antara citra asli dan citra Gaussian blur.
    """
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    residual = gray.astype(np.float64) - blur.astype(np.float64)
    return float(np.std(residual))


def estimate_lighting_uniformity(gray: np.ndarray) -> float:
    """
    Mengukur keseragaman pencahayaan dari simpangan baku rata-rata
    blok grid 4x4. Nilai > 30 menunjukkan pencahayaan lokal tidak merata.
    """
    h, w = gray.shape
    bh, bw = max(1, h // 4), max(1, w // 4)
    block_means = []
    for i in range(4):
        for j in range(4):
            y_end = (i + 1) * bh if i < 3 else h
            x_end = (j + 1) * bw if j < 3 else w
            block = gray[i * bh : y_end, j * bw : x_end]
            if block.size > 0:
                block_means.append(float(np.mean(block)))
    return float(np.std(block_means)) if block_means else 0.0


def recommend_methods(image: np.ndarray, has_reference: bool = False) -> dict:
    """
    Menganalisis karakteristik statistik citra dan merekomendasikan metode
    enhancement yang paling relevan secara deterministik berurutan:
    1. Noise tinggi dan kontras rendah -> HE + Bilateral Filter
    2. Pencahayaan tidak merata -> CLAHE
    3. Kontras rendah dan merata -> Histogram Equalization
    4. Kontras sudah memadai -> Saran: Opsional CLAHE ringan
    """
    gray = _to_gray(image)
    image_float = gray.astype(np.float64)

    mean = float(np.mean(image_float))
    std = float(np.std(image_float))
    min_val = int(np.min(gray))
    max_val = int(np.max(gray))
    dynamic_range = int(max_val - min_val)

    histogram = np.bincount(gray.ravel(), minlength=256)
    occupied_bins = int(np.count_nonzero(histogram))

    noise_level = estimate_noise_level(gray)
    lighting_std = estimate_lighting_uniformity(gray)

    # Diagnosa berbasis konstanta ambang
    if mean < THRESHOLD_DARK_MEAN:
        brightness_status = "Gelap"
    elif mean > THRESHOLD_BRIGHT_MEAN:
        brightness_status = "Terang"
    else:
        brightness_status = "Normal"

    is_low_contrast = (std < THRESHOLD_LOW_CONTRAST_STD) or (dynamic_range < THRESHOLD_NARROW_DYNAMIC_RANGE)
    contrast_status = "Rendah" if is_low_contrast else "Memadai"

    is_uneven_lighting = lighting_std > THRESHOLD_UNEVEN_LIGHTING_STD
    lighting_status = "Tidak Merata" if is_uneven_lighting else "Merata"

    is_high_noise = noise_level >= THRESHOLD_HIGH_NOISE
    noise_status = "Tinggi" if is_high_noise else "Rendah"

    # Evaluasi urutan aturan deterministik
    if is_high_noise and is_low_contrast:
        rec_type = "Rekomendasi Utama"
        rec_value = METHOD_BILATERAL
        rec_reason = (
            f"Noise tinggi ({noise_level:.1f}) dan kontras rendah (Std Dev {std:.1f}); "
            "kombinasi HE dan penapisan bilateral meratakan kontras sekaligus meredam noise."
        )
        target_params = {
            "method": METHOD_BILATERAL,
            "diameter": 9,
            "sigma_color": 75.0,
            "sigma_space": 75.0,
        }

    elif is_uneven_lighting:
        rec_type = "Rekomendasi Utama"
        rec_value = METHOD_CLAHE
        rec_reason = (
            f"Pencahayaan lokal tidak merata (deviasi blok {lighting_std:.1f} > {THRESHOLD_UNEVEN_LIGHTING_STD:.0f}); "
            "CLAHE mengadaptasi peningkatan kontras per ubin lokal tanpa amplifikasi noise ekstrem."
        )
        target_params = {
            "method": METHOD_CLAHE,
            "clip_limit": 2.0,
            "tile_size": 8,
        }

    elif is_low_contrast:
        rec_type = "Rekomendasi Utama"
        rec_value = METHOD_HE
        rec_reason = f"Std Dev {std:.1f} (di bawah {THRESHOLD_LOW_CONTRAST_STD:.0f}) dan pencahayaan merata."
        target_params = {
            "method": METHOD_HE,
        }

    else:
        rec_type = "Saran"
        rec_value = "Opsional: CLAHE ringan"
        rec_reason = "Kontras sudah memadai; CLAHE clip limit 1.0 dapat dicoba untuk penajaman detail lokal."
        target_params = {
            "method": METHOD_CLAHE,
            "clip_limit": 1.0,
            "tile_size": 8,
        }

    primary_recommendation = {
        "type": rec_type,
        "method": rec_value,
        "reason": rec_reason,
        "target_params": target_params,
    }

    recommendations = [primary_recommendation]

    # Opsi Tambahan: Citra referensi diunggah
    if has_reference:
        recommendations.append({
            "type": "Opsi Tambahan",
            "method": METHOD_MATCHING,
            "reason": "Citra referensi tersedia untuk penyesuaian distribusi intensitas.",
            "target_params": {"method": METHOD_MATCHING},
        })

    # Tambahkan alternatif metode pembanding
    all_methods = [METHOD_CLAHE, METHOD_HE, METHOD_BILATERAL]
    for alt in all_methods:
        if alt != primary_recommendation.get("target_params", {}).get("method"):
            alt_reason = "Metode pembanding peningkatan kualitas citra."
            if alt == METHOD_BILATERAL:
                alt_reason = "Opsi penapisan jika timbul butiran derau pada hasil peningkatan."
            elif alt == METHOD_CLAHE:
                alt_reason = "Alternatif peningkatan kontras secara lokal adaptif."
            elif alt == METHOD_HE:
                alt_reason = "Alternatif perataan kontras kumulatif secara global."
            recommendations.append({
                "type": "Alternatif",
                "method": alt,
                "reason": alt_reason,
                "target_params": {"method": alt},
            })

    return {
        "characteristics": {
            "mean": mean,
            "standard_deviation": std,
            "dynamic_range": dynamic_range,
            "occupied_histogram_bins": occupied_bins,
            "noise_level": noise_level,
            "lighting_std": lighting_std,
            "minimum": min_val,
            "maximum": max_val,
        },
        "diagnostics": {
            "brightness": brightness_status,
            "contrast": contrast_status,
            "distribution": f"{occupied_bins}/256 bin terisi",
            "lighting": lighting_status,
            "noise": noise_status,
        },
        "primary": primary_recommendation,
        "recommendations": recommendations,
    }
