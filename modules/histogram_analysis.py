"""
Modul analisis dan visualisasi histogram untuk Image Enhancement Studio.
Menyediakan kalkulasi histogram terpadu sebagai sumber nilai tunggal (single source of truth)
dan generator visualisasi interaktif Altair serta grafik Matplotlib untuk ekspor.
"""

import altair as alt
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from modules.metrics import calculate_saturated_pixels_pct

# Design system color tokens
COLOR_BEFORE = "#64748B"      # Slate Gray (Kategori Sebelum)
COLOR_AFTER = "#1D4ED8"       # Deep Royal Blue (Kategori Sesudah)
COLOR_REF = "#334155"         # Dark Slate Netral (Citra Referensi)
COLOR_PRIMARY = "#1D4ED8"     # Deep Royal Blue
COLOR_SECONDARY = "#0F766E"   # Deep Teal
COLOR_BORDER = "#E2E8F0"      # Light Border
COLOR_BG_CARD = "#FFFFFF"     # White Card
COLOR_TEXT_MAIN = "#0F172A"   # Slate 900
COLOR_TEXT_MUTED = "#64748B"  # Muted Slate


def _ensure_2d_luminance(image: np.ndarray) -> np.ndarray:
    """Memastikan citra berdimensi 2 (grayscale / luminance) uint8."""
    if not isinstance(image, np.ndarray):
        raise TypeError("Input harus berupa NumPy array.")

    if image.size == 0:
        raise ValueError("Citra tidak boleh kosong.")

    if image.ndim == 3 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    elif image.ndim == 3 and image.shape[2] == 1:
        image = image.squeeze(axis=2)
    elif image.ndim != 2:
        raise ValueError("Dimensi citra tidak valid.")

    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)

    return image


def calculate_histogram(image: np.ndarray) -> np.ndarray:
    """
    Menghitung histogram intensitas 256-bin untuk citra grayscale atau kanal luminance.
    """
    gray = _ensure_2d_luminance(image)
    histogram = cv2.calcHist([gray], [0], None, [256], [0, 256])
    return histogram.flatten().astype(np.float64)


def calculate_rgb_histograms(image_rgb: np.ndarray) -> dict[str, np.ndarray]:
    """
    Menghitung histogram untuk masing-masing kanal R, G, dan B.
    """
    if not isinstance(image_rgb, np.ndarray) or image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise ValueError("Input harus berupa citra RGB 3-channel.")

    hist_r = cv2.calcHist([image_rgb], [0], None, [256], [0, 256]).flatten().astype(np.float64)
    hist_g = cv2.calcHist([image_rgb], [1], None, [256], [0, 256]).flatten().astype(np.float64)
    hist_b = cv2.calcHist([image_rgb], [2], None, [256], [0, 256]).flatten().astype(np.float64)

    return {"R": hist_r, "G": hist_g, "B": hist_b}


def format_pct(pct: float, count: int) -> str:
    """Format persentase dua desimal, atau 'kurang dari 0.01%' jika non-nol tapi < 0.01%."""
    if count == 0 or pct == 0.0:
        return "0.00%"
    elif pct < 0.01:
        return "kurang dari 0.01%"
    else:
        return f"{pct:.2f}%"


def format_saturation_caption(
    b0_cnt: int, b0_pct: float,
    a0_cnt: int, a0_pct: float,
    b255_cnt: int, b255_pct: float,
    a255_cnt: int, a255_pct: float,
    min_val: int = 0, max_val: int = 255,
    is_luminance: bool = False,
) -> str:
    """
    Format caption piksel jenuh satu baris yang cerdas:
    - Jika kedua sisi 0: tampilkan rentang (luminance) atau 'Tidak ada piksel jenuh.' (RGB).
    - Jika ada sisi bernilai > 0: hanya tampilkan sisi yang memiliki isi.
    - Persentase ditampilkan dua desimal (atau 'kurang dari 0.01%' jika < 0.01%).
    """
    has_0 = (b0_cnt > 0 or a0_cnt > 0)
    has_255 = (b255_cnt > 0 or a255_cnt > 0)

    if not has_0 and not has_255:
        if is_luminance:
            return f"Tidak ada piksel jenuh pada luminance (rentang {min_val} sampai {max_val})."
        else:
            return "Tidak ada piksel jenuh."

    parts = []
    if has_0:
        p0_b = format_pct(b0_pct, b0_cnt)
        p0_a = format_pct(a0_pct, a0_cnt)
        parts.append(f"Piksel pada intensitas 0: {p0_b} menjadi {p0_a} ({b0_cnt:,} menjadi {a0_cnt:,} piksel)")

    if has_255:
        p255_b = format_pct(b255_pct, b255_cnt)
        p255_a = format_pct(a255_pct, a255_cnt)
        parts.append(f"Piksel pada intensitas 255: {p255_b} menjadi {p255_a} ({b255_cnt:,} menjadi {a255_cnt:,} piksel)")

    return ". ".join(parts) + "."


def compute_tab2_histogram_data(
    disp_before: np.ndarray,
    enhanced_image: np.ndarray,
    reference_image: np.ndarray | None = None,
    is_color: bool = True,
) -> dict:
    """
    Satu sumber angka (single source of truth) untuk seluruh data di Tab 2.
    Menghasilkan array histogram dasar dan menurunkan seluruh metrik (Mean, rentang,
    persentase piksel jenuh di 0 dan 255) langsung dari array histogram yang digambar.
    """
    # 1. Luminance Histograms
    lum_b = calculate_histogram(disp_before)
    lum_a = calculate_histogram(enhanced_image)
    lum_ref = calculate_histogram(reference_image) if reference_image is not None else None

    total_pixels = int(np.sum(lum_b))
    bins = np.arange(256)

    # 2. Statistik Luminance langsung dari array histogram
    mean_b = float(np.sum(bins * lum_b) / (total_pixels + 1e-9))
    mean_a = float(np.sum(bins * lum_a) / (total_pixels + 1e-9))
    mean_ref = float(np.sum(bins * lum_ref) / (np.sum(lum_ref) + 1e-9)) if lum_ref is not None else None

    nonzero_b = np.where(lum_b > 0)[0]
    min_b = int(nonzero_b[0]) if len(nonzero_b) > 0 else 0
    max_b = int(nonzero_b[-1]) if len(nonzero_b) > 0 else 255

    nonzero_a = np.where(lum_a > 0)[0]
    min_a = int(nonzero_a[0]) if len(nonzero_a) > 0 else 0
    max_a = int(nonzero_a[-1]) if len(nonzero_a) > 0 else 255

    # Saturasi Luminance
    lum_sat = {
        "b0_count": int(lum_b[0]),
        "b0_pct": float(lum_b[0] / total_pixels * 100.0) if total_pixels > 0 else 0.0,
        "a0_count": int(lum_a[0]),
        "a0_pct": float(lum_a[0] / total_pixels * 100.0) if total_pixels > 0 else 0.0,
        "b255_count": int(lum_b[255]),
        "b255_pct": float(lum_b[255] / total_pixels * 100.0) if total_pixels > 0 else 0.0,
        "a255_count": int(lum_a[255]),
        "a255_pct": float(lum_a[255] / total_pixels * 100.0) if total_pixels > 0 else 0.0,
    }

    # Persentase piksel jenuh gabungan (C.4)
    sat_overall_b = calculate_saturated_pixels_pct(disp_before)
    sat_overall_a = calculate_saturated_pixels_pct(enhanced_image)

    # 3. Kanal RGB (jika citra berwarna)
    rgb_data = None
    has_rgb = is_color and (disp_before.ndim == 3 and disp_before.shape[2] == 3)

    if has_rgb:
        hists_rgb_b = calculate_rgb_histograms(disp_before)
        hists_rgb_a = calculate_rgb_histograms(enhanced_image)
        hists_rgb_ref = None
        if reference_image is not None:
            if reference_image.ndim == 3 and reference_image.shape[2] == 3:
                hists_rgb_ref = calculate_rgb_histograms(reference_image)
            else:
                ref_gray_h = calculate_histogram(reference_image)
                hists_rgb_ref = {"R": ref_gray_h, "G": ref_gray_h, "B": ref_gray_h}

        # Hitung saturasi per kanal
        sat_channels = {}
        for ch in ["R", "G", "B"]:
            b_cnt0 = int(hists_rgb_b[ch][0])
            a_cnt0 = int(hists_rgb_a[ch][0])
            b_cnt255 = int(hists_rgb_b[ch][255])
            a_cnt255 = int(hists_rgb_a[ch][255])

            sat_channels[ch] = {
                "b0_count": b_cnt0,
                "b0_pct": (b_cnt0 / total_pixels * 100.0) if total_pixels > 0 else 0.0,
                "a0_count": a_cnt0,
                "a0_pct": (a_cnt0 / total_pixels * 100.0) if total_pixels > 0 else 0.0,
                "b255_count": b_cnt255,
                "b255_pct": (b_cnt255 / total_pixels * 100.0) if total_pixels > 0 else 0.0,
                "a255_count": a_cnt255,
                "a255_pct": (a_cnt255 / total_pixels * 100.0) if total_pixels > 0 else 0.0,
            }

        all_zero_saturation = all(
            sat_channels[ch]["b0_count"] == 0
            and sat_channels[ch]["a0_count"] == 0
            and sat_channels[ch]["b255_count"] == 0
            and sat_channels[ch]["a255_count"] == 0
            for ch in ["R", "G", "B"]
        )

        rgb_data = {
            "before": hists_rgb_b,
            "after": hists_rgb_a,
            "ref": hists_rgb_ref,
            "saturation": sat_channels,
            "all_zero_saturation": all_zero_saturation,
        }

    return {
        "total_pixels": total_pixels,
        "saturated_pixels_pct_before": sat_overall_b,
        "saturated_pixels_pct_after": sat_overall_a,
        "luminance": {
            "before": lum_b,
            "after": lum_a,
            "ref": lum_ref,
            "mean_before": mean_b,
            "mean_after": mean_a,
            "mean_ref": mean_ref,
            "min_before": min_b,
            "max_before": max_b,
            "min_after": min_a,
            "max_after": max_a,
            "saturation": lum_sat,
        },
        "rgb": rgb_data,
    }


def create_altair_histogram_chart(
    images_dict: dict[str, np.ndarray],
    mean_before: float,
    mean_after: float,
    title: str = "",
) -> alt.Chart:
    """
    Membuat visualisasi histogram luminance interaktif berbasis Altair:
    - Sebelum: Abu slate (#64748B) + Area unstacked (stack=None)
    - Sesudah: Biru aksen (#1D4ED8) + Area unstacked (stack=None)
    - Referensi (jika ada): Garis putus-putus abu gelap (#334155) tanpa area, dengan legenda 'Referensi'
    - Penanda garis vertikal tipis untuk Mean sebelum (abu) dan Mean sesudah (biru) dengan label nilainya
    - Tinggi grafik: minimum 320 px (memenuhi syarat minimum 300 px)
    """
    bins = np.arange(256)
    records = []
    has_ref = "Citra Referensi" in images_dict or "Referensi" in images_dict
    ref_key = "Citra Referensi" if "Citra Referensi" in images_dict else "Referensi"

    # Data sebelum dan sesudah
    for label in ["Sebelum Enhancement", "Sesudah Enhancement"]:
        if label in images_dict:
            hist = calculate_histogram(images_dict[label])
            for b, count in zip(bins, hist):
                records.append({
                    "Intensitas": int(b),
                    "Jumlah Piksel": int(count),
                    "Kategori": "Sebelum" if "Sebelum" in label else "Sesudah",
                })

    df_main = pd.DataFrame(records)

    color_domain = ["Sebelum", "Sesudah"]
    color_range = [COLOR_BEFORE, COLOR_AFTER]

    df_ref = None
    if has_ref:
        hist_ref = calculate_histogram(images_dict[ref_key])
        records_ref = []
        for b, count in zip(bins, hist_ref):
            records_ref.append({
                "Intensitas": int(b),
                "Jumlah Piksel": int(count),
                "Kategori": "Referensi",
            })
        df_ref = pd.DataFrame(records_ref)
        color_domain.append("Referensi")
        color_range.append(COLOR_REF)

    color_scale = alt.Scale(domain=color_domain, range=color_range)

    # 1. Base Layer untuk Area (stack=None agar tidak saling menumpuk)
    x_axis_lum = alt.Axis(
        values=[0, 50, 100, 150, 200, 250],
        grid=False,
        title="Intensitas piksel (0 sampai 255)",
        titleFontSize=9.5,
        titleColor=COLOR_TEXT_MUTED,
        labelFontSize=9,
        labelColor=COLOR_TEXT_MUTED,
    )
    y_axis_lum = alt.Axis(
        grid=True,
        gridColor="#94A3B8",
        gridOpacity=0.2,
        gridWidth=0.8,
        title="Jumlah piksel",
        titleFontSize=9.5,
        titleColor=COLOR_TEXT_MUTED,
        labelFontSize=9,
        labelColor=COLOR_TEXT_MUTED,
        format=",d",
    )

    base_area = (
        alt.Chart(df_main)
        .encode(
            x=alt.X("Intensitas:Q", scale=alt.Scale(domain=[0, 255], clamp=True), axis=x_axis_lum),
            y=alt.Y("Jumlah Piksel:Q", stack=None, axis=y_axis_lum),
            color=alt.Color(
                "Kategori:N",
                scale=color_scale,
                legend=alt.Legend(title=None, orient="top", direction="horizontal"),
            ),
        )
    )
    area_layer = base_area.mark_area(opacity=0.35)

    # 2. Base Layer untuk Line Sebelum & Sesudah
    base_line = (
        alt.Chart(df_main)
        .encode(
            x=alt.X("Intensitas:Q", scale=alt.Scale(domain=[0, 255], clamp=True), axis=x_axis_lum),
            y=alt.Y("Jumlah Piksel:Q", stack=None, axis=y_axis_lum),
            color=alt.Color("Kategori:N", scale=color_scale, legend=None),
            tooltip=[
                alt.Tooltip("Kategori:N", title="Kategori"),
                alt.Tooltip("Intensitas:Q", title="Tingkat Intensitas"),
                alt.Tooltip("Jumlah Piksel:Q", title="Jumlah Piksel", format=","),
            ],
        )
    )
    line_layer = base_line.mark_line(strokeWidth=1.8)

    layers = [area_layer, line_layer]

    # 3. Layer Referensi (garis putus-putus tipis dengan strokeDash=[5, 4])
    if df_ref is not None:
        ref_line = (
            alt.Chart(df_ref)
            .encode(
                x=alt.X("Intensitas:Q", scale=alt.Scale(domain=[0, 255], clamp=True), axis=x_axis_lum),
                y=alt.Y("Jumlah Piksel:Q", stack=None, axis=y_axis_lum),
                color=alt.Color("Kategori:N", scale=color_scale, legend=alt.Legend(title=None, orient="top", direction="horizontal")),
                tooltip=[
                    alt.Tooltip("Kategori:N", title="Kategori"),
                    alt.Tooltip("Intensitas:Q", title="Tingkat Intensitas"),
                    alt.Tooltip("Jumlah Piksel:Q", title="Jumlah Piksel", format=","),
                ],
            )
            .mark_line(strokeWidth=1.6, strokeDash=[5, 4])
        )
        layers.append(ref_line)

    # 4. Penanda Garis Vertikal Mean (B.4 & B.2)
    df_mean = pd.DataFrame([
        {"Mean": mean_before, "Label": f"Mean Sebelum: {mean_before:.1f}", "Color": COLOR_BEFORE},
        {"Mean": mean_after, "Label": f"Mean Sesudah: {mean_after:.1f}", "Color": COLOR_AFTER},
    ])

    rule_mean = (
        alt.Chart(df_mean)
        .mark_rule(strokeDash=[4, 4], strokeWidth=1.4)
        .encode(
            x=alt.X("Mean:Q"),
            color=alt.Color("Color:N", scale=None),
            tooltip=[alt.Tooltip("Label:N", title="Statistik")],
        )
    )

    df_mean_b = pd.DataFrame([{"Mean": mean_before, "Label": f"Mean Sebelum: {mean_before:.1f}"}])
    text_mean_b = (
        alt.Chart(df_mean_b)
        .mark_text(align="right", dx=-5, fontSize=10.5, fontWeight="bold", color=COLOR_BEFORE)
        .encode(
            x=alt.X("Mean:Q"),
            y=alt.value(16),
            text="Label:N",
        )
    )

    df_mean_a = pd.DataFrame([{"Mean": mean_after, "Label": f"Mean Sesudah: {mean_after:.1f}"}])
    text_mean_a = (
        alt.Chart(df_mean_a)
        .mark_text(align="left", dx=5, fontSize=10.5, fontWeight="bold", color=COLOR_AFTER)
        .encode(
            x=alt.X("Mean:Q"),
            y=alt.value(34),
            text="Label:N",
        )
    )
    layers.extend([rule_mean, text_mean_b, text_mean_a])

    # Gabungkan seluruh layer
    props = {"width": "container", "height": 320}
    if title:
        props["title"] = title

    chart = (
        alt.layer(*layers)
        .properties(**props)
        .interactive()
    )

    return chart


def create_altair_channel_comparison_chart(
    hist_before: np.ndarray,
    hist_after: np.ndarray,
    channel_name: str,
    channel_color: str,
    hist_ref: np.ndarray | None = None,
    y_scale_mode: str = "Potong lonjakan",
    y_max_limit: float = 1000.0,
) -> alt.Chart:
    """
    Membuat grafik perbandingan per kanal RGB (R, G, atau B) berdampingan:
    - Sebelum: Abu slate (#64748B) + Area unstacked (stack=None)
    - Sesudah: Warna kanal spesifik (Merah/Hijau/Biru) + Area unstacked (stack=None)
    - Referensi (opsional): Garis tipis (#334155)
    - Tanpa judul sumbu X per grafik (disatukan di bawah 3 kolom)
    - Tanpa legenda per grafik (disatukan di atas 3 kolom)
    - Skala sumbu Y rapi: minimal 4 tick pada mode potong lonjakan, tick rapi pada log
    """
    records = []
    for b in range(256):
        records.append({
            "Intensitas": b,
            "Jumlah Piksel": int(hist_before[b]),
            "Kategori": "Sebelum",
        })
        records.append({
            "Intensitas": b,
            "Jumlah Piksel": int(hist_after[b]),
            "Kategori": "Sesudah",
        })

    df_main = pd.DataFrame(records)

    color_domain = ["Sebelum", "Sesudah"]
    color_range = [COLOR_BEFORE, channel_color]

    df_ref = None
    if hist_ref is not None:
        records_ref = []
        for b in range(256):
            records_ref.append({
                "Intensitas": b,
                "Jumlah Piksel": int(hist_ref[b]),
                "Kategori": "Referensi",
            })
        df_ref = pd.DataFrame(records_ref)
        color_domain.append("Referensi")
        color_range.append(COLOR_REF)

    color_scale = alt.Scale(domain=color_domain, range=color_range)

    # Konfigurasi Sumbu X dan Y (A.1, B.1, D, E)
    x_axis_rgb = alt.Axis(
        values=[0, 50, 100, 150, 200, 250],
        grid=False,
        title="Intensitas piksel (0 sampai 255)",
        titleFontSize=9.5,
        titleColor=COLOR_TEXT_MUTED,
        labelFontSize=9,
        labelColor=COLOR_TEXT_MUTED,
    )
    x_enc = alt.X("Intensitas:Q", scale=alt.Scale(domain=[0, 255], clamp=True), axis=x_axis_rgb)

    if y_scale_mode == "Log":
        df_main["Jumlah Piksel"] = df_main["Jumlah Piksel"].clip(lower=1)
        if df_ref is not None:
            df_ref["Jumlah Piksel"] = df_ref["Jumlah Piksel"].clip(lower=1)
        y_scale = alt.Scale(type="log", domain=[1, max(10, y_max_limit)])
        y_axis = alt.Axis(
            values=[1, 10, 100, 1000, 10000, 100000],
            grid=True,
            gridColor="#94A3B8",
            gridOpacity=0.2,
            gridWidth=0.8,
            format=",d",
            title="Jumlah piksel",
            titleFontSize=9.5,
            titleColor=COLOR_TEXT_MUTED,
            labelFontSize=9,
            labelColor=COLOR_TEXT_MUTED,
        )
    elif y_scale_mode == "Potong lonjakan":
        y_scale = alt.Scale(domain=[0, max(10, y_max_limit)], clamp=True)
        y_axis = alt.Axis(
            tickCount=5,
            grid=True,
            gridColor="#94A3B8",
            gridOpacity=0.2,
            gridWidth=0.8,
            format=",d",
            title="Jumlah piksel",
            titleFontSize=9.5,
            titleColor=COLOR_TEXT_MUTED,
            labelFontSize=9,
            labelColor=COLOR_TEXT_MUTED,
        )
    else:  # Linear
        y_scale = alt.Scale(domain=[0, max(10, y_max_limit)])
        y_axis = alt.Axis(
            tickCount=5,
            grid=True,
            gridColor="#94A3B8",
            gridOpacity=0.2,
            gridWidth=0.8,
            format=",d",
            title="Jumlah piksel",
            titleFontSize=9.5,
            titleColor=COLOR_TEXT_MUTED,
            labelFontSize=9,
            labelColor=COLOR_TEXT_MUTED,
        )

    layers = []

    # E: Mode Log tanpa isian area!
    if y_scale_mode == "Log":
        line_before = (
            alt.Chart(df_main)
            .transform_filter(alt.datum.Kategori == "Sebelum")
            .encode(
                x=x_enc,
                y=alt.Y("Jumlah Piksel:Q", scale=y_scale, stack=None, axis=y_axis),
                color=alt.Color("Kategori:N", scale=color_scale, legend=None),
                tooltip=[
                    alt.Tooltip("Kategori:N", title="Kategori"),
                    alt.Tooltip("Intensitas:Q", title="Intensitas"),
                    alt.Tooltip("Jumlah Piksel:Q", title="Jumlah Piksel", format=","),
                ],
            )
            .mark_line(strokeWidth=1.2)
        )
        line_after = (
            alt.Chart(df_main)
            .transform_filter(alt.datum.Kategori == "Sesudah")
            .encode(
                x=x_enc,
                y=alt.Y("Jumlah Piksel:Q", scale=y_scale, stack=None, axis=y_axis),
                color=alt.Color("Kategori:N", scale=color_scale, legend=None),
                tooltip=[
                    alt.Tooltip("Kategori:N", title="Kategori"),
                    alt.Tooltip("Intensitas:Q", title="Intensitas"),
                    alt.Tooltip("Jumlah Piksel:Q", title="Jumlah Piksel", format=","),
                ],
            )
            .mark_line(strokeWidth=2.2)
        )
        layers.extend([line_before, line_after])
    else:
        # 1. Area Layer Sebelum & Sesudah (stack=None)
        base_area = (
            alt.Chart(df_main)
            .encode(
                x=x_enc,
                y=alt.Y("Jumlah Piksel:Q", scale=y_scale, stack=None, axis=y_axis),
                color=alt.Color("Kategori:N", scale=color_scale, legend=None),
            )
        )
        area_layer = base_area.mark_area(opacity=0.35)

        # 2. Line Layer Sebelum & Sesudah
        base_line = (
            alt.Chart(df_main)
            .encode(
                x=x_enc,
                y=alt.Y("Jumlah Piksel:Q", scale=y_scale, stack=None, axis=y_axis),
                color=alt.Color("Kategori:N", scale=color_scale, legend=None),
                tooltip=[
                    alt.Tooltip("Kategori:N", title="Kategori"),
                    alt.Tooltip("Intensitas:Q", title="Intensitas"),
                    alt.Tooltip("Jumlah Piksel:Q", title="Jumlah Piksel", format=","),
                ],
            )
        )
        line_layer = base_line.mark_line(strokeWidth=1.8)
        layers.extend([area_layer, line_layer])

    # 3. Line Layer Referensi (garis tipis, D)
    if df_ref is not None:
        ref_line = (
            alt.Chart(df_ref)
            .encode(
                x=x_enc,
                y=alt.Y("Jumlah Piksel:Q", scale=y_scale, stack=None, axis=y_axis),
                color=alt.Color("Kategori:N", scale=color_scale, legend=None),
                tooltip=[
                    alt.Tooltip("Kategori:N", title="Kategori"),
                    alt.Tooltip("Intensitas:Q", title="Intensitas"),
                    alt.Tooltip("Jumlah Piksel:Q", title="Jumlah Piksel", format=","),
                ],
            )
            .mark_line(strokeWidth=1.2, strokeDash=[4, 3])
        )
        layers.append(ref_line)

    chart = (
        alt.layer(*layers)
        .properties(
            title=f"Kanal {channel_name}",
            width="container",
            height=270,
        )
        .interactive()
    )

    return chart


def plot_histograms(
    images: list[np.ndarray],
    labels: list[str],
    colors: list[str] | None = None,
    title: str = "Perbandingan Distribusi Histogram",
) -> plt.Figure:
    """
    Membuat gambar figure Matplotlib untuk berkas ekspor/arsip jika diperlukan.
    """
    if len(images) == 0 or len(images) != len(labels):
        raise ValueError("Jumlah citra dan label harus sama dan tidak kosong.")

    default_colors = [COLOR_BEFORE, COLOR_AFTER, COLOR_REF, "#7C3AED"]
    if colors is None:
        colors = default_colors[: len(images)]

    fig, ax = plt.subplots(figsize=(9, 3.8), dpi=100)
    fig.patch.set_facecolor(COLOR_BG_CARD)
    ax.set_facecolor(COLOR_BG_CARD)

    bins = np.arange(256)

    for image, label, col in zip(images, labels, colors):
        hist = calculate_histogram(image)
        ax.plot(bins, hist, label=label, color=col, linewidth=1.6, alpha=0.9)
        ax.fill_between(bins, hist, color=col, alpha=0.08)

    ax.set_title(title, fontsize=12, fontweight="bold", color=COLOR_TEXT_MAIN, pad=12)
    ax.set_xlabel("Tingkat Intensitas (0-255)", fontsize=10, color=COLOR_TEXT_MUTED)
    ax.set_ylabel("Frekuensi Piksel", fontsize=10, color=COLOR_TEXT_MUTED)
    ax.set_xlim(0, 255)
    ax.set_ylim(bottom=0)

    ax.tick_params(colors=COLOR_TEXT_MUTED, labelsize=9)
    for spine in ax.spines.values():
        spine.set_color(COLOR_BORDER)
        spine.set_linewidth(1.0)

    ax.grid(True, linestyle="--", alpha=0.5, color=COLOR_BORDER)
    ax.legend(frameon=True, facecolor=COLOR_BG_CARD, edgecolor=COLOR_BORDER, fontsize=9)
    fig.tight_layout()

    return fig


def plot_rgb_histograms(image_rgb: np.ndarray, title: str = "Distribusi Kanal Warna (RGB)") -> plt.Figure:
    """
    Membuat figure Matplotlib untuk histogram 3 kanal warna RGB.
    """
    hists = calculate_rgb_histograms(image_rgb)
    bins = np.arange(256)

    fig, ax = plt.subplots(figsize=(9, 3.2), dpi=100)
    fig.patch.set_facecolor(COLOR_BG_CARD)
    ax.set_facecolor(COLOR_BG_CARD)

    ax.plot(bins, hists["R"], color="#DC2626", label="Kanal Merah (R)", linewidth=1.4, alpha=0.85)
    ax.plot(bins, hists["G"], color="#16A34A", label="Kanal Hijau (G)", linewidth=1.4, alpha=0.85)
    ax.plot(bins, hists["B"], color="#2563EB", label="Kanal Biru (B)", linewidth=1.4, alpha=0.85)

    ax.set_title(title, fontsize=11, fontweight="bold", color=COLOR_TEXT_MAIN, pad=10)
    ax.set_xlabel("Intensitas (0-255)", fontsize=9, color=COLOR_TEXT_MUTED)
    ax.set_ylabel("Jumlah Piksel", fontsize=9, color=COLOR_TEXT_MUTED)
    ax.set_xlim(0, 255)
    ax.set_ylim(bottom=0)

    ax.tick_params(colors=COLOR_TEXT_MUTED, labelsize=9)
    for spine in ax.spines.values():
        spine.set_color(COLOR_BORDER)

    ax.grid(True, linestyle="--", alpha=0.5, color=COLOR_BORDER)
    ax.legend(frameon=True, facecolor=COLOR_BG_CARD, edgecolor=COLOR_BORDER, fontsize=8.5)
    fig.tight_layout()

    return fig


def plot_metrics_comparison_chart(
    metrics_before: dict[str, float | int],
    metrics_after: dict[str, float | int],
) -> plt.Figure:
    """
    Membuat diagram batang horizontal untuk membandingkan metrik kunci sebelum dan sesudah.
    """
    keys = ["Mean", "Std Dev", "Entropy", "Dynamic Range"]
    labels = ["Mean (Kecerahan)", "Std Dev (Kontras)", "Entropy (bit)", "Dynamic Range"]

    vals_b = [float(metrics_before.get(k, metrics_before.get("Standard Deviation", 0))) for k in keys]
    vals_a = [float(metrics_after.get(k, metrics_after.get("Standard Deviation", 0))) for k in keys]

    y = np.arange(len(keys))
    height = 0.35

    fig, ax = plt.subplots(figsize=(8, 3.4), dpi=100)
    fig.patch.set_facecolor(COLOR_BG_CARD)
    ax.set_facecolor(COLOR_BG_CARD)

    bars_b = ax.barh(y + height / 2, vals_b, height, label="Sebelum", color=COLOR_BEFORE, alpha=0.9)
    bars_a = ax.barh(y - height / 2, vals_a, height, label="Sesudah", color=COLOR_PRIMARY, alpha=0.95)

    for bar in bars_b:
        width = bar.get_width()
        ax.annotate(f"{width:.1f}",
                    xy=(width, bar.get_y() + bar.get_height() / 2),
                    xytext=(4, 0), textcoords="offset points",
                    ha="left", va="center", fontsize=8, color=COLOR_TEXT_MUTED)

    for bar in bars_a:
        width = bar.get_width()
        ax.annotate(f"{width:.1f}",
                    xy=(width, bar.get_y() + bar.get_height() / 2),
                    xytext=(4, 0), textcoords="offset points",
                    ha="left", va="center", fontsize=8, fontweight="bold", color=COLOR_PRIMARY)

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9.5, color=COLOR_TEXT_MAIN, fontweight="medium")
    ax.set_title("Perbandingan Metrik Utama Sebelum vs Sesudah", fontsize=11, fontweight="bold", color=COLOR_TEXT_MAIN, pad=10)
    ax.tick_params(colors=COLOR_TEXT_MUTED, labelsize=9)

    for spine in ax.spines.values():
        spine.set_color(COLOR_BORDER)

    ax.grid(True, axis="x", linestyle="--", alpha=0.5, color=COLOR_BORDER)
    ax.legend(frameon=True, facecolor=COLOR_BG_CARD, edgecolor=COLOR_BORDER, fontsize=9)
    fig.tight_layout()

    return fig
