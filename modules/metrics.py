import cv2
import numpy as np
import pandas as pd
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

# ==============================================================================
# KONSTANTA ZONA TIDAK BERUBAH BERMAKNA (DEADBAND)
# ==============================================================================
DEADBAND_MEAN = 3.0          # Kurang dari 3 poin: relatif tidak berubah
DEADBAND_STD_PCT_LOW = 5.0   # Kurang dari 5%: relatif tetap; 5-15%: sedang; >15%: nyata
DEADBAND_STD_PCT_HIGH = 15.0
DEADBAND_ENTROPY = 0.05      # Kurang dari 0.05 bit: praktis sama
DEADBAND_LEVELS = 3          # Kurang dari 3 tingkat: praktis sama


# ==============================================================================
# PANDUAN DAN PENJELASAN EDUKATIF METRIK
# ==============================================================================
METRIC_EXPLANATIONS = {
    "Mean": {
        "title": "Mean (Rata-rata Intensitas)",
        "definition": (
            "Mean mengukur rata-rata intensitas piksel (0–255 pada 8-bit). "
            "Nilai rendah menunjukkan citra gelap, sedangkan nilai tinggi menunjukkan citra terang. "
            "Mean tinggi tidak otomatis berarti lebih baik karena dapat memicu overexposure atau clipping."
        ),
        "color_note": "Pada mode warna, dihitung dari representasi luminance L.",
        "unit": "skala 0–255",
    },
    "Std Dev": {
        "title": "Std Dev (Standar Deviasi / Kontras Global)",
        "definition": (
            "Std Dev mengukur sebaran intensitas piksel terhadap nilai rata-ratanya. "
            "Nilai lebih tinggi menunjukkan kontras lebih kuat, namun juga dapat meningkat akibat noise."
        ),
        "color_note": "Dihitung dari sebaran intensitas kanal luminance L.",
        "unit": "skala 0–127.5",
    },
    "Entropy": {
        "title": "Entropy (Entropi Informasi)",
        "definition": (
            "Entropy mengukur keragaman distribusi intensitas dalam satuan bit (maksimal 8 bit). "
            "Nilai lebih tinggi menandakan distribusi intensitas lebih bervariasi dan kaya informasi."
        ),
        "color_note": "Dihitung menggunakan formula Shannon berbasis probabilitas histogram.",
        "unit": "bit",
    },
    "Dynamic Range": {
        "title": "Dynamic Range (Rentang Dinamis Teramati)",
        "definition": (
            "Dynamic Range adalah selisih antara nilai piksel maksimum dan minimum (Max - Min). "
            "Menggambarkan bentangan spektrum intensitas yang dimanfaatkan oleh citra."
        ),
        "color_note": "Rentang intensitas teramati pada kanal luminance L.",
        "unit": "tingkat (0–255)",
    },
    "Bins": {
        "title": "Bins (Jumlah Bin Histogram Terisi)",
        "definition": (
            "Menunjukkan berapa banyak tingkat keabuan (dari 256 tingkat) yang memiliki setidaknya satu piksel. "
            "Penurunan bins sering terjadi pada Histogram Equalization akibat pengelompokan kuantisasi."
        ),
        "color_note": "Mengukur pemanfaatan spektrum kuantisasi intensitas.",
        "unit": "bin",
    },
    "Min": {
        "title": "Min (Intensitas Minimum)",
        "definition": "Nilai piksel tergelap dalam citra pada skala 0–255.",
        "color_note": "Batas bawah intensitas luminance.",
        "unit": "skala 0–255",
    },
    "Max": {
        "title": "Max (Intensitas Maksimum)",
        "definition": "Nilai piksel paling terang dalam citra pada skala 0–255.",
        "color_note": "Batas atas intensitas luminance.",
        "unit": "skala 0–255",
    },
    "Piksel jenuh": {
        "title": "Piksel Jenuh (Clipping)",
        "definition": (
            "Piksel jenuh mengukur persentase piksel yang berada pada batas ekstrem intensitas (0 atau 255). "
            "Persentase yang tinggi menandakan terjadinya kehilangan detail pada area bayangan gelap (underexposure) "
            "atau area sorotan terang (overexposure)."
        ),
        "color_note": "Pada mode warna, dihitung dari piksel yang minimal satu kanalnya bernilai 0 atau 255.",
        "unit": "persen (%)",
    },
    "PSNR": {
        "title": "Peak Signal-to-Noise Ratio (PSNR)",
        "definition": (
            "Mengukur rasio daya sinyal maksimum terhadap perbedaan sinyal antara hasil dan citra asli. "
            "Pada operasi enhancement, kontras sengaja diubah sehingga PSNR rendah adalah wajar."
        ),
        "color_note": "Dihitung terhadap kanal luminance citra masukan asli.",
        "unit": "dB",
    },
    "SSIM": {
        "title": "Structural Similarity Index (SSIM)",
        "definition": (
            "Mengukur kemiripan struktur, luminansi, dan kontras terhadap citra asli (skala 0 sampai 1). "
            "Skor yang lebih rendah mencerminkan besarnya perubahan struktur kontras yang diterapkan."
        ),
        "color_note": "Mengukur deviasi struktural terhadap citra acuan.",
        "unit": "indeks (0–1)",
    },
}


# ==============================================================================
# PERHITUNGAN METRIK DASAR
# ==============================================================================
def _to_luminance(image: np.ndarray) -> np.ndarray:
    """Konversi citra input ke array 2D uint8 luminance jika berupa RGB."""
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
        raise ValueError("Dimensi citra tidak valid.")


def calculate_saturated_pixels_pct(image: np.ndarray) -> float:
    """
    Menghitung persentase piksel jenuh:
    - Citra RGB: persentase piksel yang minimal satu kanalnya bernilai 0 atau 255.
    - Citra Grayscale: persentase piksel bernilai 0 atau 255.
    """
    if not isinstance(image, np.ndarray) or image.size == 0:
        return 0.0

    if image.ndim == 3 and image.shape[2] == 3:
        mask = (
            (image[:, :, 0] == 0) | (image[:, :, 0] == 255) |
            (image[:, :, 1] == 0) | (image[:, :, 1] == 255) |
            (image[:, :, 2] == 0) | (image[:, :, 2] == 255)
        )
        total_pixels = image.shape[0] * image.shape[1]
    else:
        gray = _to_luminance(image)
        mask = (gray == 0) | (gray == 255)
        total_pixels = gray.size

    if total_pixels == 0:
        return 0.0

    return float(np.count_nonzero(mask) / total_pixels * 100.0)


def normalize_metrics_keys(metrics: dict[str, float | int]) -> dict[str, float | int]:
    """
    Memastikan dictionary metrik memiliki semua kunci standar (baik nama pendek maupun alias)
    secara defensif untuk mencegah KeyError dari cache lama.
    """
    min_v = int(metrics.get("Min", metrics.get("Minimum Intensity", 0)))
    max_v = int(metrics.get("Max", metrics.get("Maximum Intensity", 255)))
    dr_v = int(metrics.get("Dynamic Range", max_v - min_v))
    std_v = float(metrics.get("Std Dev", metrics.get("Standard Deviation", 0.0)))
    bins_v = int(metrics.get("Bins", metrics.get("Occupied Bins", 0)))
    mean_v = float(metrics.get("Mean", 0.0))
    ent_v = float(metrics.get("Entropy", 0.0))
    sat_v = float(metrics.get("Piksel jenuh", 0.0))

    metrics["Min"] = min_v
    metrics["Max"] = max_v
    metrics["Minimum Intensity"] = min_v
    metrics["Maximum Intensity"] = max_v
    metrics["Dynamic Range"] = dr_v
    metrics["Std Dev"] = std_v
    metrics["Standard Deviation"] = std_v
    metrics["Bins"] = bins_v
    metrics["Occupied Bins"] = bins_v
    metrics["Mean"] = mean_v
    metrics["Entropy"] = ent_v
    metrics["Piksel jenuh"] = sat_v
    return metrics


def calculate_metrics(image: np.ndarray) -> dict[str, float | int]:
    """
    Menghitung kumpulan metrik kualitas dan distribusi intensitas citra.
    Menyediakan nama metrik ringkas standar (Mean, Std Dev, Entropy, Dynamic Range, Bins, Min, Max, Piksel jenuh)
    serta alias untuk kompatibilitas regresi.
    """
    sat_pct = calculate_saturated_pixels_pct(image)
    gray = _to_luminance(image)
    image_float = gray.astype(np.float64)

    mean_val = float(np.mean(image_float))
    std_val = float(np.std(image_float))

    min_val = int(np.min(gray))
    max_val = int(np.max(gray))
    dynamic_range = int(max_val - min_val)

    histogram = np.bincount(gray.ravel(), minlength=256).astype(np.float64)
    occupied_bins = int(np.count_nonzero(histogram))

    total_pixels = histogram.sum()
    if total_pixels > 0:
        probabilities = histogram / total_pixels
        probabilities = probabilities[probabilities > 0]
        entropy_val = float(-np.sum(probabilities * np.log2(probabilities)))
    else:
        entropy_val = 0.0

    res = {
        "Mean": mean_val,
        "Std Dev": std_val,
        "Entropy": entropy_val,
        "Dynamic Range": dynamic_range,
        "Bins": occupied_bins,
        "Min": min_val,
        "Max": max_val,
        "Piksel jenuh": sat_pct,
        # Alias kompatibilitas
        "Standard Deviation": std_val,
        "Occupied Bins": occupied_bins,
        "Minimum Intensity": min_val,
        "Maximum Intensity": max_val,
    }
    return normalize_metrics_keys(res)


def calculate_reference_metrics(
    image_enhanced: np.ndarray,
    image_reference: np.ndarray,
) -> dict[str, float]:
    """Menghitung PSNR dan SSIM terhadap citra asli."""
    gray_enh = _to_luminance(image_enhanced)
    gray_ref = _to_luminance(image_reference)

    if gray_enh.shape != gray_ref.shape:
        gray_ref = cv2.resize(
            gray_ref,
            (gray_enh.shape[1], gray_enh.shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )

    psnr_val = float(peak_signal_noise_ratio(gray_ref, gray_enh, data_range=255))
    ssim_val = float(structural_similarity(gray_ref, gray_enh, data_range=255))

    return {
        "PSNR": psnr_val,
        "SSIM": ssim_val,
    }


def calculate_histogram_distance(
    image_enhanced: np.ndarray,
    image_reference: np.ndarray,
) -> dict[str, float]:
    """Menghitung korelasi dan jarak Bhattacharyya terhadap citra referensi."""
    gray_enh = _to_luminance(image_enhanced)
    gray_ref = _to_luminance(image_reference)

    hist_enh = cv2.calcHist([gray_enh], [0], None, [256], [0, 256])
    hist_ref = cv2.calcHist([gray_ref], [0], None, [256], [0, 256])

    cv2.normalize(hist_enh, hist_enh, 1.0, 0.0, cv2.NORM_L1)
    cv2.normalize(hist_ref, hist_ref, 1.0, 0.0, cv2.NORM_L1)

    correlation = float(cv2.compareHist(hist_enh, hist_ref, cv2.HISTCMP_CORREL))
    bhattacharyya = float(cv2.compareHist(hist_enh, hist_ref, cv2.HISTCMP_BHATTACHARYYA))

    return {
        "Histogram Correlation": correlation,
        "Bhattacharyya Distance": bhattacharyya,
    }


# ==============================================================================
# ATURAN INTERPRETASI HASIL (DENGAN ZONA DEADBAND)
# ==============================================================================
def interpret_metric_change(metric_name: str, before: float, after: float) -> str:
    """
    Menghasilkan interpretasi perubahan metrik dengan deadband.
    Setiap kalimat dibatasi maksimal 70 karakter untuk mencegah teks terpotong.
    """
    delta = after - before

    if metric_name == "Mean":
        if abs(delta) < DEADBAND_MEAN:
            return "Relatif tidak berubah."
        elif delta > 0:
            return f"Naik (+{delta:.2f}), citra lebih terang."
        else:
            return f"Turun ({delta:.2f}), citra lebih gelap."

    elif metric_name in ("Std Dev", "Standard Deviation"):
        pct = (delta / (before + 1e-6)) * 100
        if abs(pct) < DEADBAND_STD_PCT_LOW:
            return "Relatif tetap, sebaran kontras stabil."
        elif pct >= DEADBAND_STD_PCT_HIGH:
            return f"Naik nyata (+{delta:.2f}), kontras melebar."
        elif pct > 0:
            return f"Naik sedang (+{delta:.2f}), kontras bertambah."
        elif pct <= -DEADBAND_STD_PCT_HIGH:
            return f"Turun nyata ({delta:.2f}), kontras menyempit."
        else:
            return f"Turun sedang ({delta:.2f}), kontras berkurang."

    elif metric_name == "Entropy":
        if abs(delta) < DEADBAND_ENTROPY:
            return "Praktis sama, variasi informasi stabil."
        elif delta > 0:
            return f"Naik (+{delta:.3f} bit), informasi bertambah."
        else:
            return f"Turun ({delta:.3f} bit), informasi berkurang."

    elif metric_name == "Dynamic Range":
        d_int = int(round(after)) - int(round(before))
        if abs(d_int) < DEADBAND_LEVELS:
            return "Praktis sama, rentang intensitas tetap."
        elif d_int > 0:
            return f"Melebar (+{d_int} tingkat)."
        else:
            return f"Menyempit ({d_int} tingkat)."

    elif metric_name in ("Bins", "Occupied Bins"):
        d_int = int(round(after)) - int(round(before))
        if abs(d_int) < DEADBAND_LEVELS:
            return "Praktis sama, jumlah bin aktif stabil."
        elif d_int > 0:
            return f"Bertambah (+{d_int} bin terisi)."
        else:
            return f"Berkurang ({d_int} bin terisi)."

    elif metric_name in ("Min", "Minimum Intensity"):
        d_int = int(round(after)) - int(round(before))
        if abs(d_int) < DEADBAND_LEVELS:
            return "Praktis sama, batas bawah tetap."
        elif d_int > 0:
            return f"Naik (+{d_int} tingkat), shadow terangkat."
        else:
            return f"Turun ({d_int} tingkat), shadow lebih pekat."

    elif metric_name in ("Max", "Maximum Intensity"):
        d_int = int(round(after)) - int(round(before))
        if abs(d_int) < DEADBAND_LEVELS:
            return "Praktis sama, batas atas tetap."
        elif d_int > 0:
            return f"Naik (+{d_int} tingkat), highlight melebar."
        else:
            return f"Turun ({d_int} tingkat), highlight berkurang."

    elif metric_name == "Piksel jenuh":
        delta = after - before
        if abs(delta) < 0.1:
            return "Praktis sama."
        elif delta > 0:
            return f"Naik (+{delta:.2f}%)."
        else:
            return f"Turun ({delta:.2f}%)."

    return "Nilai metrik berubah sesuai pemetaan."


def get_metric_interpretations(
    metrics_before: dict[str, float | int],
    metrics_after: dict[str, float | int],
) -> list[str]:
    """Ringkasan interpretasi perubahan metrik utama (Mean, Std Dev, Entropy)."""
    interp_mean = interpret_metric_change("Mean", float(metrics_before["Mean"]), float(metrics_after["Mean"]))
    interp_std = interpret_metric_change("Std Dev", float(metrics_before.get("Std Dev", metrics_before.get("Standard Deviation"))), float(metrics_after.get("Std Dev", metrics_after.get("Standard Deviation"))))
    interp_ent = interpret_metric_change("Entropy", float(metrics_before["Entropy"]), float(metrics_after["Entropy"]))

    return [
        f"Mean: {interp_mean}",
        f"Std Dev: {interp_std}",
        f"Entropy: {interp_ent}",
    ]


def build_metrics_comparison_table(
    metrics_before: dict[str, float | int],
    metrics_after: dict[str, float | int],
) -> pd.DataFrame:
    """
    Menyusun tabel perbandingan metrik dengan format numerik yang rapi:
    Bilangan bulat untuk Dynamic Range, Bins, Min, Max (tanpa desimal .0).
    """
    key_order = [
        "Mean",
        "Std Dev",
        "Entropy",
        "Dynamic Range",
        "Bins",
        "Min",
        "Max",
        "Piksel jenuh",
    ]

    key_aliases = {
        "Std Dev": "Standard Deviation",
        "Bins": "Occupied Bins",
        "Min": "Minimum Intensity",
        "Max": "Maximum Intensity",
    }

    rows = []
    for key in key_order:
        alt_key = key_aliases.get(key, key)
        val_b = metrics_before.get(key, metrics_before.get(alt_key))
        val_a = metrics_after.get(key, metrics_after.get(alt_key))

        if val_b is not None and val_a is not None:
            delta = val_a - val_b

            if key in ("Mean", "Std Dev"):
                b_str = f"{float(val_b):.2f}"
                a_str = f"{float(val_a):.2f}"
                d_str = f"{float(delta):+.2f}"
            elif key == "Entropy":
                b_str = f"{float(val_b):.3f}"
                a_str = f"{float(val_a):.3f}"
                d_str = f"{float(delta):+.3f}"
            elif key == "Piksel jenuh":
                b_str = f"{float(val_b):.2f}%"
                a_str = f"{float(val_a):.2f}%"
                d_str = f"{float(delta):+.2f}%"
            else:
                b_str = f"{int(round(float(val_b)))}"
                a_str = f"{int(round(float(val_a)))}"
                d_str = f"{int(round(float(delta))):+d}"

            interpretation = interpret_metric_change(key, float(val_b), float(val_a))

            rows.append({
                "Metrik": key,
                "Sebelum": b_str,
                "Sesudah": a_str,
                "Perubahan (Delta)": d_str,
                "Interpretasi": interpretation,
            })

    return pd.DataFrame(rows)


def generate_quality_summary(
    metrics_before: dict[str, float | int],
    metrics_after: dict[str, float | int],
    method_name: str,
    color_mode: str = "Color Enhancement",
) -> tuple[str, str]:
    """
    Menyusun sintesis naratif dalam dua paragraf pendek terpisah:
    1. Paragraf hasil metrik objektif (tanpa bold di tengah kalimat).
    2. Paragraf catatan evaluasi inspeksi visual.
    """
    mean_b = float(metrics_before.get("Mean", 0))
    mean_a = float(metrics_after.get("Mean", 0))
    delta_mean = mean_a - mean_b

    std_b = float(metrics_before.get("Std Dev", metrics_before.get("Standard Deviation", 0)))
    std_a = float(metrics_after.get("Std Dev", metrics_after.get("Standard Deviation", 0)))
    delta_std = std_a - std_b
    pct_std = (delta_std / (std_b + 1e-6)) * 100

    dr_b = int(round(float(metrics_before.get("Dynamic Range", 0))))
    dr_a = int(round(float(metrics_after.get("Dynamic Range", 0))))
    delta_dr = dr_a - dr_b

    parts = []

    # 1. Kontras
    if abs(pct_std) < DEADBAND_STD_PCT_LOW:
        parts.append(f"Metode {method_name} mempertahankan sebaran kontras relatif tetap (Std Dev {std_a:.1f} vs {std_b:.1f}).")
    elif pct_std >= DEADBAND_STD_PCT_HIGH:
        parts.append(f"Metode {method_name} meningkatkan sebaran kontras global secara nyata (Std Dev naik dari {std_b:.1f} ke {std_a:.1f}).")
    elif pct_std > 0:
        parts.append(f"Metode {method_name} meningkatkan sebaran kontras sedang (Std Dev naik dari {std_b:.1f} ke {std_a:.1f}).")
    elif pct_std <= -DEADBAND_STD_PCT_HIGH:
        parts.append(f"Metode {method_name} menghasilkan penurunan sebaran kontras secara nyata (Std Dev turun dari {std_b:.1f} ke {std_a:.1f}).")
    else:
        parts.append(f"Metode {method_name} menghasilkan penurunan sebaran kontras sedang (Std Dev turun dari {std_b:.1f} ke {std_a:.1f}).")

    # 2. Kecerahan
    if abs(delta_mean) < DEADBAND_MEAN:
        parts.append(f"Keseimbangan pencahayaan rata-rata relatif tidak berubah (mean {mean_a:.1f} vs {mean_b:.1f}).")
    elif delta_mean > 0:
        parts.append(f"Pencahayaan rata-rata bergeser lebih terang (mean naik dari {mean_b:.1f} ke {mean_a:.1f}).")
    else:
        parts.append(f"Pencahayaan rata-rata bergeser lebih gelap (mean turun dari {mean_b:.1f} ke {mean_a:.1f}).")

    # 3. Dynamic Range
    if abs(delta_dr) < DEADBAND_LEVELS:
        parts.append("Rentang dinamis praktis sama.")
    elif delta_dr > 0:
        parts.append(f"Pemanfaatan dynamic range meluas dari {dr_b} menjadi {dr_a} tingkat.")
    else:
        parts.append(f"Pemanfaatan dynamic range menyempit dari {dr_b} menjadi {dr_a} tingkat.")

    # 4. Mode Warna
    if color_mode == "Color Enhancement":
        parts.append("Pemrosesan hanya pada kanal L; kanal a dan b tidak diubah.")
    else:
        parts.append("Pemrosesan dilakukan pada citra grayscale 8-bit secara langsung.")

    par1 = " ".join(parts)
    par2 = (
        "Catatan evaluasi: Angka statistik di atas memberikan panduan kuantitatif, "
        "namun penilaian kualitas citra tetap harus dikonfirmasi melalui inspeksi visual subjektif "
        "untuk memastikan tidak ada artefak yang tidak diinginkan seperti saturasi berlebih atau peningkatan noise."
    )

    return par1, par2
