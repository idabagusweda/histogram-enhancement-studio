import cv2
import numpy as np
from skimage import exposure


def _validate_grayscale(image: np.ndarray) -> np.ndarray:
    """Validasi dan normalisasi citra grayscale 8-bit."""
    if not isinstance(image, np.ndarray):
        raise TypeError("Input harus berupa NumPy array.")

    if image.size == 0:
        raise ValueError("Citra tidak boleh kosong.")

    if image.ndim == 3:
        if image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        elif image.shape[2] == 1:
            image = image.squeeze(axis=2)
        else:
            raise ValueError("Format channel citra tidak didukung.")

    if image.ndim != 2:
        raise ValueError("Input harus berupa citra 2 dimensi (grayscale).")

    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)

    return image


def _validate_rgb(image: np.ndarray) -> np.ndarray:
    """Validasi dan normalisasi citra RGB 8-bit."""
    if not isinstance(image, np.ndarray):
        raise TypeError("Input harus berupa NumPy array.")

    if image.size == 0:
        raise ValueError("Citra tidak boleh kosong.")

    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.ndim == 3 and image.shape[2] == 1:
        image = cv2.cvtColor(image.squeeze(axis=2), cv2.COLOR_GRAY2RGB)
    elif image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Input harus berupa citra RGB 3-channel.")

    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)

    return image


def _extract_reference_luminance(reference: np.ndarray) -> np.ndarray:
    """Ekstraksi kanal luminance dari citra referensi (bisa RGB atau grayscale)."""
    if not isinstance(reference, np.ndarray):
        raise TypeError("Citra referensi harus berupa NumPy array.")

    if reference.size == 0:
        raise ValueError("Citra referensi tidak boleh kosong.")

    if reference.ndim == 3 and reference.shape[2] == 3:
        ref_uint8 = np.clip(reference, 0, 255).astype(np.uint8)
        lab_ref = cv2.cvtColor(ref_uint8, cv2.COLOR_RGB2LAB)
        return lab_ref[:, :, 0]
    elif reference.ndim == 2:
        return _validate_grayscale(reference)
    elif reference.ndim == 3 and reference.shape[2] == 1:
        return _validate_grayscale(reference.squeeze(axis=2))
    else:
        raise ValueError("Format citra referensi tidak valid.")


# ==============================================================================
# 1. GRAYSCALE ENHANCEMENT METHODS
# ==============================================================================

def histogram_equalization(image: np.ndarray) -> np.ndarray:
    """
    Meningkatkan kontras global citra grayscale menggunakan Histogram Equalization.
    Meratakan fungsi distribusi kumulatif (CDF) intensitas piksel.
    """
    gray = _validate_grayscale(image)
    return cv2.equalizeHist(gray)


def clahe_enhancement(
    image: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: tuple[int, int] = (8, 8),
) -> np.ndarray:
    """
    Meningkatkan kontras lokal menggunakan Contrast Limited Adaptive Histogram Equalization.
    Mencegah amplifikasi noise berlebih dengan membatasi kontras per ubin (tile).
    """
    gray = _validate_grayscale(image)

    if clip_limit <= 0:
        raise ValueError("clip_limit harus lebih besar dari 0.")

    if (
        len(tile_grid_size) != 2
        or any(not isinstance(v, int) or v < 1 for v in tile_grid_size)
    ):
        raise ValueError("tile_grid_size harus berupa dua bilangan bulat positif.")

    clahe = cv2.createCLAHE(
        clipLimit=float(clip_limit),
        tileGridSize=tile_grid_size,
    )
    return clahe.apply(gray)


def histogram_matching(
    image: np.ndarray,
    reference: np.ndarray,
) -> np.ndarray:
    """
    Menyesuaikan distribusi intensitas citra dengan distribusi citra referensi
    (Histogram Specification / Matching).
    """
    gray = _validate_grayscale(image)
    ref_lum = _extract_reference_luminance(reference)

    matched = exposure.match_histograms(gray, ref_lum)
    return np.clip(matched, 0, 255).astype(np.uint8)


def equalization_bilateral(
    image: np.ndarray,
    diameter: int = 9,
    sigma_color: float = 75.0,
    sigma_space: float = 75.0,
) -> np.ndarray:
    """
    Histogram Equalization dilanjutkan dengan Bilateral Filtering
    untuk meredam noise di area datar sembari menjaga ketajaman tepi (edge-preserving).
    """
    gray = _validate_grayscale(image)

    if diameter < 1:
        raise ValueError("diameter harus minimal 1.")

    if sigma_color <= 0 or sigma_space <= 0:
        raise ValueError("Parameter sigma harus lebih besar dari 0.")

    equalized = cv2.equalizeHist(gray)

    result = cv2.bilateralFilter(
        equalized,
        d=int(diameter),
        sigmaColor=float(sigma_color),
        sigmaSpace=float(sigma_space),
    )
    return result


# ==============================================================================
# 2. COLOR ENHANCEMENT METHODS (LAB COLOR SPACE)
# ==============================================================================

def histogram_equalization_color(image_rgb: np.ndarray) -> np.ndarray:
    """
    Menerapkan Histogram Equalization pada kanal Luminance (L) di ruang warna LAB.
    Kanal krominansi (a dan b) dipertahankan untuk mencegah pergeseran warna.
    """
    rgb = _validate_rgb(image_rgb)
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    L, a, b = cv2.split(lab)

    L_enhanced = cv2.equalizeHist(L)

    lab_enhanced = cv2.merge([L_enhanced, a, b])
    rgb_enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2RGB)
    return np.clip(rgb_enhanced, 0, 255).astype(np.uint8)


def clahe_enhancement_color(
    image_rgb: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: tuple[int, int] = (8, 8),
) -> np.ndarray:
    """
    Menerapkan CLAHE pada kanal Luminance (L) di ruang warna LAB.
    Kanal krominansi (a dan b) dipertahankan tanpa perubahan.
    """
    rgb = _validate_rgb(image_rgb)

    if clip_limit <= 0:
        raise ValueError("clip_limit harus lebih besar dari 0.")
    if (
        len(tile_grid_size) != 2
        or any(not isinstance(v, int) or v < 1 for v in tile_grid_size)
    ):
        raise ValueError("tile_grid_size harus berupa dua bilangan bulat positif.")

    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    L, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(
        clipLimit=float(clip_limit),
        tileGridSize=tile_grid_size,
    )
    L_enhanced = clahe.apply(L)

    lab_enhanced = cv2.merge([L_enhanced, a, b])
    rgb_enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2RGB)
    return np.clip(rgb_enhanced, 0, 255).astype(np.uint8)


def histogram_matching_color(
    image_rgb: np.ndarray,
    reference: np.ndarray,
) -> np.ndarray:
    """
    Menyesuaikan distribusi kanal Luminance (L) citra input dengan kanal
    Luminance citra referensi di ruang warna LAB.
    Kanal krominansi (a dan b) input dipertahankan.
    """
    rgb = _validate_rgb(image_rgb)
    ref_lum = _extract_reference_luminance(reference)

    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    L, a, b = cv2.split(lab)

    matched_L = exposure.match_histograms(L, ref_lum)
    matched_L = np.clip(matched_L, 0, 255).astype(np.uint8)

    lab_enhanced = cv2.merge([matched_L, a, b])
    rgb_enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2RGB)
    return np.clip(rgb_enhanced, 0, 255).astype(np.uint8)


def equalization_bilateral_color(
    image_rgb: np.ndarray,
    diameter: int = 9,
    sigma_color: float = 75.0,
    sigma_space: float = 75.0,
) -> np.ndarray:
    """
    Menerapkan Histogram Equalization dan Bilateral Filtering pada kanal
    Luminance (L) di ruang warna LAB, lalu menggabungkannya kembali dengan kanal a dan b.
    """
    rgb = _validate_rgb(image_rgb)

    if diameter < 1:
        raise ValueError("diameter harus minimal 1.")
    if sigma_color <= 0 or sigma_space <= 0:
        raise ValueError("Parameter sigma harus lebih besar dari 0.")

    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    L, a, b = cv2.split(lab)

    L_eq = cv2.equalizeHist(L)
    L_filtered = cv2.bilateralFilter(
        L_eq,
        d=int(diameter),
        sigmaColor=float(sigma_color),
        sigmaSpace=float(sigma_space),
    )

    lab_enhanced = cv2.merge([L_filtered, a, b])
    rgb_enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2RGB)
    return np.clip(rgb_enhanced, 0, 255).astype(np.uint8)


# ==============================================================================
# 3. UNIFIED DISPATCHER & REGISTRIES
# ==============================================================================

METHODS = {
    "Histogram Equalization": histogram_equalization,
    "Contrast Limited Adaptive Histogram Equalization": clahe_enhancement,
    "Histogram Matching": histogram_matching,
    "Histogram Equalization + Bilateral Filtering": equalization_bilateral,
}

METHODS_COLOR = {
    "Histogram Equalization": histogram_equalization_color,
    "Contrast Limited Adaptive Histogram Equalization": clahe_enhancement_color,
    "Histogram Matching": histogram_matching_color,
    "Histogram Equalization + Bilateral Filtering": equalization_bilateral_color,
}


def process_enhancement(
    image: np.ndarray,
    method: str,
    color_mode: str = "Color Enhancement",
    reference: np.ndarray | None = None,
    clip_limit: float = 2.0,
    tile_size: int = 8,
    diameter: int = 9,
    sigma_color: float = 75.0,
    sigma_space: float = 75.0,
) -> np.ndarray:
    """
    Fungsi terpadu untuk memproses enhancement citra baik dalam mode Color maupun Grayscale.
    """
    is_color = color_mode == "Color Enhancement"

    if method == "Histogram Equalization":
        return histogram_equalization_color(image) if is_color else histogram_equalization(image)

    elif method == "Contrast Limited Adaptive Histogram Equalization":
        tile_grid = (int(tile_size), int(tile_size))
        if is_color:
            return clahe_enhancement_color(image, clip_limit=clip_limit, tile_grid_size=tile_grid)
        return clahe_enhancement(image, clip_limit=clip_limit, tile_grid_size=tile_grid)

    elif method == "Histogram Matching":
        if reference is None:
            raise ValueError("Histogram Matching memerlukan citra referensi.")
        if is_color:
            return histogram_matching_color(image, reference)
        return histogram_matching(image, reference)

    elif method == "Histogram Equalization + Bilateral Filtering":
        if is_color:
            return equalization_bilateral_color(
                image,
                diameter=diameter,
                sigma_color=sigma_color,
                sigma_space=sigma_space,
            )
        return equalization_bilateral(
            image,
            diameter=diameter,
            sigma_color=sigma_color,
            sigma_space=sigma_space,
        )

    else:
        raise ValueError(f"Metode '{method}' tidak dikenal.")