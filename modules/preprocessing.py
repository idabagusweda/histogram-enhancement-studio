import cv2
import numpy as np
from PIL import Image

SUPPORTED_FORMATS = ("png", "jpg", "jpeg", "bmp", "tif", "tiff")
MAX_DIMENSION = 4096


def load_image(uploaded_file):
    """
    Membaca berkas citra yang diunggah dan mengembalikan array RGB dan Grayscale.
    Menangani kasus khusus:
    - PNG dengan kanal alpha (RGBA/transparan): dikompositkan ke latar belakang putih.
    - Citra monokrom murni (L): dinormalisasi ke RGB dan grayscale 8-bit.
    - Validasi dimensi maksimum (4096 x 4096 px) untuk mencegah kehabisan memori.
    """
    if uploaded_file is None:
        raise ValueError("Silakan unggah berkas citra terlebih dahulu.")

    try:
        pil_img = Image.open(uploaded_file)

        # Penanganan kanal alpha (transparansi)
        if pil_img.mode in ("RGBA", "LA") or (pil_img.mode == "P" and "transparency" in pil_img.info):
            rgba = pil_img.convert("RGBA")
            background = Image.new("RGB", rgba.size, (255, 255, 255))
            background.paste(rgba, mask=rgba.split()[3])
            image_rgb = np.array(background)
        elif pil_img.mode == "L":
            gray_arr = np.array(pil_img)
            image_rgb = cv2.cvtColor(gray_arr, cv2.COLOR_GRAY2RGB)
            return image_rgb, gray_arr
        else:
            image_rgb = np.array(pil_img.convert("RGB"))

    except Exception as exc:
        raise ValueError(
            "Berkas citra rusak atau format tidak didukung. "
            "Pastikan berkas bertipe PNG, JPG, JPEG, BMP, atau TIFF yang valid."
        ) from exc

    if image_rgb.size == 0:
        raise ValueError("Citra kosong atau tidak memiliki data piksel.")

    height, width = image_rgb.shape[:2]
    if width > MAX_DIMENSION or height > MAX_DIMENSION:
        raise ValueError(
            f"Dimensi citra ({width} x {height} px) melebihi batas maksimum {MAX_DIMENSION} x {MAX_DIMENSION} px. "
            "Silakan gunakan citra dengan resolusi lebih rendah."
        )

    image_gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    return image_rgb, image_gray


def image_information(image_rgb: np.ndarray) -> dict[str, str | int]:
    """Mengambil informasi metadata dan dimensi dasar citra."""
    height, width = image_rgb.shape[:2]
    channels = image_rgb.shape[2] if image_rgb.ndim == 3 else 1

    return {
        "width": width,
        "height": height,
        "channels": channels,
        "pixels": width * height,
        "dtype": str(image_rgb.dtype),
        "min_intensity": int(np.min(image_rgb)),
        "max_intensity": int(np.max(image_rgb)),
        "aspect_ratio": f"{width}:{height}",
    }


def get_grayscale(image_rgb: np.ndarray) -> np.ndarray:
    """Mengonversi citra RGB menjadi grayscale 8-bit."""
    if not isinstance(image_rgb, np.ndarray):
        raise TypeError("Input harus berupa NumPy array.")

    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise ValueError("Input harus berupa citra RGB 3-channel.")

    return cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
