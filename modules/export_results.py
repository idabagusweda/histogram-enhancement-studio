import datetime
from io import BytesIO
import re
import numpy as np
import pandas as pd
from PIL import Image


def image_to_png_bytes(image: np.ndarray) -> bytes:
    """
    Mengubah array NumPy citra (RGB atau Grayscale) beresolusi penuh
    menjadi berkas biner PNG dalam memori tanpa kompresi lossy.
    """
    if image is None or image.size == 0:
        raise ValueError("Tidak ada citra untuk diekspor.")

    if not isinstance(image, np.ndarray):
        raise TypeError("Input citra harus berupa NumPy array.")

    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)

    buffer = BytesIO()
    pil_img = Image.fromarray(image)
    pil_img.save(buffer, format="PNG")
    return buffer.getvalue()


def build_export_csv(
    file_name: str,
    mode: str,
    method: str,
    parameters: str,
    dimensions: str,
    timestamp: str,
    comparison_df: pd.DataFrame,
    reference_file_name: str | None = None,
) -> bytes:
    """
    Menyusun berkas CSV evaluasi perbandingan metrik sesuai spesifikasi:
    nama berkas, mode, metode, semua parameter, dimensi, timestamp, citra referensi,
    metrik, metrik sebelum, metrik sesudah, delta, interpretasi.
    """
    df = comparison_df.copy()
    df.insert(0, "nama berkas", file_name)
    df.insert(1, "mode", mode)
    df.insert(2, "metode", method)
    df.insert(3, "semua parameter", parameters)
    df.insert(4, "dimensi", dimensions)
    df.insert(5, "timestamp", timestamp)
    df.insert(6, "citra referensi", reference_file_name if reference_file_name else "-")

    col_map = {
        "Metrik": "metrik",
        "Sebelum": "metrik sebelum",
        "Sesudah": "metrik sesudah",
        "Perubahan (Delta)": "delta",
        "Interpretasi": "interpretasi",
    }
    df = df.rename(columns=col_map)
    return df.to_csv(index=False).encode("utf-8-sig")


def build_all_methods_export_csv(
    file_name: str,
    mode: str,
    dimensions: str,
    timestamp: str,
    all_methods_df: pd.DataFrame,
    reference_file_name: str | None = None,
) -> bytes:
    """
    Menyusun berkas CSV evaluasi komparasi empat metode sekaligus.
    """
    df = all_methods_df.copy()
    df.insert(0, "nama berkas", file_name)
    df.insert(1, "mode", mode)
    df.insert(2, "dimensi", dimensions)
    df.insert(3, "timestamp", timestamp)
    df.insert(4, "citra referensi", reference_file_name if reference_file_name else "-")
    return df.to_csv(index=False).encode("utf-8-sig")


def sanitize_filename(name: str) -> str:
    """Membersihkan string untuk digunakan sebagai nama berkas yang aman."""
    s = name.lower().replace("+", "plus").strip()
    s = re.sub(r"[^\w\s-]", "", s)
    s = re.sub(r"[-\s]+", "_", s)
    return s


def generate_export_filename(
    original_filename: str,
    method: str,
    color_mode: str = "",
    extension: str = "png",
) -> str:
    """
    Membuat nama berkas unduhan ringkas: {nama_asli}_{metode_singkat}.{ext}
    Contoh: foto_clahe.png, foto_he.png, foto_hm.png, foto_he_bilateral.png
    """
    clean_base = original_filename.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
    clean_base = clean_base.rsplit(".", 1)[0]
    clean_base = sanitize_filename(clean_base) or "citra"

    method_short_map = {
        "Contrast Limited Adaptive Histogram Equalization": "clahe",
        "Histogram Equalization": "he",
        "Histogram Matching": "hm",
        "Histogram Equalization + Bilateral Filtering": "he_bilateral",
    }
    method_slug = method_short_map.get(method, sanitize_filename(method))
    ext_clean = extension.lstrip(".")
    return f"{clean_base}_{method_slug}.{ext_clean}"
