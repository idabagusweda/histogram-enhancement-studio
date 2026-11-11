import datetime
import importlib
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

import modules.enhancement
import modules.export_results
import modules.histogram_analysis
import modules.metrics
import modules.preprocessing
import modules.processing
import modules.recommendation

importlib.reload(modules.preprocessing)
importlib.reload(modules.enhancement)
importlib.reload(modules.processing)
importlib.reload(modules.histogram_analysis)
importlib.reload(modules.metrics)
importlib.reload(modules.recommendation)
importlib.reload(modules.export_results)

from modules.export_results import (
    build_all_methods_export_csv,
    build_export_csv,
    generate_export_filename,
    image_to_png_bytes,
)
from modules.histogram_analysis import (
    calculate_histogram,
    calculate_rgb_histograms,
    compute_tab2_histogram_data,
    create_altair_channel_comparison_chart,
    create_altair_histogram_chart,
    format_saturation_caption,
)
from modules.metrics import (
    METRIC_EXPLANATIONS,
    build_metrics_comparison_table,
    calculate_histogram_distance,
    calculate_metrics,
    calculate_reference_metrics,
    generate_quality_summary,
    normalize_metrics_keys,
)
from modules.preprocessing import SUPPORTED_FORMATS, image_information, load_image
from modules.processing import (
    METHODS,
    METHODS_COLOR,
    process_enhancement,
)
from modules.recommendation import recommend_methods


# ==============================================================================
# 1. KONFIGURASI HALAMAN
# ==============================================================================

st.set_page_config(
    page_title="Image Enhancement Studio",
    layout="wide",
    initial_sidebar_state="expanded",
)

METHOD_NAMES = [
    "Contrast Limited Adaptive Histogram Equalization",
    "Histogram Equalization",
    "Histogram Matching",
    "Histogram Equalization + Bilateral Filtering",
]

METHOD_LABELS = {
    "Contrast Limited Adaptive Histogram Equalization": "CLAHE",
    "Histogram Equalization": "Histogram Equalization",
    "Histogram Matching": "Histogram Matching",
    "Histogram Equalization + Bilateral Filtering": "HE + Bilateral Filter",
}


# ==============================================================================
# 2. INISIALISASI SESSION STATE DAN CACHING
# ==============================================================================

@st.cache_data(show_spinner=False)
def cached_process_enhancement(
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
    """Fungsi pemrosesan citra dengan cache untuk mencegah komputasi ulang yang tidak perlu."""
    return process_enhancement(
        image=image,
        method=method,
        color_mode=color_mode,
        reference=reference,
        clip_limit=clip_limit,
        tile_size=tile_size,
        diameter=diameter,
        sigma_color=sigma_color,
        sigma_space=sigma_space,
    )


@st.cache_data(show_spinner=False)
def cached_calculate_metrics(image: np.ndarray) -> dict[str, float | int]:
    """Kalkulasi metrik kualitas dengan cache berbasis array citra."""
    return normalize_metrics_keys(calculate_metrics(image))


@st.cache_data(show_spinner=False)
def cached_calculate_reference_metrics(
    image_enhanced: np.ndarray,
    image_reference: np.ndarray,
) -> dict[str, float]:
    """Kalkulasi PSNR dan SSIM dengan cache berbasis array citra."""
    return calculate_reference_metrics(image_enhanced, image_reference)


# Inisialisasi state parameter
if "method_selection" not in st.session_state:
    st.session_state.method_selection = METHOD_NAMES[0]
if "clahe_clip_limit" not in st.session_state:
    st.session_state.clahe_clip_limit = 2.0
if "clahe_tile_size" not in st.session_state:
    st.session_state.clahe_tile_size = 8
if "bilateral_diameter" not in st.session_state:
    st.session_state.bilateral_diameter = 9
if "bilateral_sigma_color" not in st.session_state:
    st.session_state.bilateral_sigma_color = 75.0
if "bilateral_sigma_space" not in st.session_state:
    st.session_state.bilateral_sigma_space = 75.0
if "crop_pos_x" not in st.session_state:
    st.session_state.crop_pos_x = 0
if "crop_pos_y" not in st.session_state:
    st.session_state.crop_pos_y = 0


def reset_parameters():
    st.session_state.clahe_clip_limit = 2.0
    st.session_state.clahe_tile_size = 8
    st.session_state.bilateral_diameter = 9
    st.session_state.bilateral_sigma_color = 75.0
    st.session_state.bilateral_sigma_space = 75.0


def apply_recommendation_callback(target_params: dict):
    """Callback on_click untuk tombol Terapkan saran."""
    if "method" in target_params:
        st.session_state.method_selection = target_params["method"]
    if "clip_limit" in target_params:
        st.session_state.clahe_clip_limit = float(target_params["clip_limit"])
    if "tile_size" in target_params:
        st.session_state.clahe_tile_size = int(target_params["tile_size"])
    if "diameter" in target_params:
        st.session_state.bilateral_diameter = int(target_params["diameter"])
    if "sigma_color" in target_params:
        st.session_state.bilateral_sigma_color = float(target_params["sigma_color"])
    if "sigma_space" in target_params:
        st.session_state.bilateral_sigma_space = float(target_params["sigma_space"])


def set_crop_shortcut(target_type: str, w: int, h: int, win: int, gray_orig: np.ndarray, gray_enh: np.ndarray):
    """Callback on_click untuk tombol pintas lokasi crop."""
    max_x = max(0, w - win)
    max_y = max(0, h - win)

    if target_type == "center":
        target_x = max_x // 2
        target_y = max_y // 2
    elif target_type == "darkest":
        if max_x > 0 and max_y > 0 and win < min(w, h):
            box_dark = cv2.boxFilter(gray_orig.astype(np.float32), -1, (win, win), normalize=True)
            valid_dark = box_dark[win // 2 : h - win // 2, win // 2 : w - win // 2]
            min_val, _, min_loc, _ = cv2.minMaxLoc(valid_dark)
            target_x = min(max_x, max(0, min_loc[0]))
            target_y = min(max_y, max(0, min_loc[1]))
        else:
            target_x = 0
            target_y = 0
    elif target_type == "diff":
        if max_x > 0 and max_y > 0 and win < min(w, h):
            diff = cv2.absdiff(gray_orig, gray_enh)
            box_diff = cv2.boxFilter(diff.astype(np.float32), -1, (win, win), normalize=True)
            valid_diff = box_diff[win // 2 : h - win // 2, win // 2 : w - win // 2]
            _, _, _, max_loc = cv2.minMaxLoc(valid_diff)
            target_x = min(max_x, max(0, max_loc[0]))
            target_y = min(max_y, max(0, max_loc[1]))
        else:
            target_x = 0
            target_y = 0

    st.session_state.crop_pos_x = target_x
    st.session_state.crop_pos_y = target_y
    st.session_state.slider_crop_x = target_x
    st.session_state.slider_crop_y = target_y


# ==============================================================================
# 3. SIDEBAR: WORKSPACE DAN INPUT PARAMETER (C.5)
# ==============================================================================

with st.sidebar:
    st.subheader("Berkas Masukan")

    uploaded_file = st.file_uploader(
        "Unggah Citra Uji",
        type=list(SUPPORTED_FORMATS),
        help="Mendukung format PNG, JPG, JPEG, BMP, TIF, TIFF.",
    )

    has_uploaded = uploaded_file is not None
    use_sample = st.checkbox(
        "Gunakan Citra Contoh",
        value=False,
        disabled=has_uploaded,
        help="Gunakan citra uji bawaan jika belum mengunggah berkas sendiri.",
    )

    reference_file = st.file_uploader(
        "Citra Referensi (Opsional)",
        type=list(SUPPORTED_FORMATS),
        key="ref_uploader",
        help="Dipakai hanya untuk Histogram Matching.",
    )
    st.caption("Dipakai hanya untuk Histogram Matching")

    active_file = None
    orig_name = ""
    if has_uploaded:
        active_file = uploaded_file
        orig_name = getattr(uploaded_file, "name", "citra_uji")
    elif use_sample:
        active_file = "assets/sample_test.png"
        orig_name = "sample_test.png"
        if reference_file is None:
            reference_file = "assets/sample_reference.png"

    if active_file is not None:
        st.caption(f"Sumber aktif: {orig_name}")

    st.subheader("Mode Pemrosesan Citra")
    processing_mode = st.segmented_control(
        "Pilih Mode Pemrosesan:",
        options=["Color Enhancement", "Grayscale Enhancement"],
        default="Color Enhancement",
        label_visibility="collapsed",
    )

    st.subheader("Metode")
    if "method_selection" not in st.session_state or st.session_state.method_selection not in METHOD_NAMES:
        st.session_state.method_selection = METHOD_NAMES[0]

    selected_method = st.segmented_control(
        "Pilih Metode:",
        options=METHOD_NAMES,
        format_func=lambda m: METHOD_LABELS.get(m, m),
        key="method_selection",
        label_visibility="collapsed",
    )
    if selected_method is None:
        selected_method = st.session_state.method_selection

    st.subheader("Parameter")
    ref_display_name = None
    if reference_file is not None:
        ref_display_name = getattr(reference_file, "name", str(reference_file))

    clip_limit = float(st.session_state.clahe_clip_limit)
    tile_size = int(st.session_state.clahe_tile_size)
    diameter = int(st.session_state.bilateral_diameter)
    sigma_color = float(st.session_state.bilateral_sigma_color)
    sigma_space = float(st.session_state.bilateral_sigma_space)

    if selected_method == "Contrast Limited Adaptive Histogram Equalization":
        clip_limit = st.slider(
            "Clip Limit",
            min_value=0.5,
            max_value=5.0,
            step=0.5,
            key="clahe_clip_limit",
            help="Batas pemotongan kontras lokal untuk meredam noise.",
        )
        tile_size = st.select_slider(
            "Ukuran Grid Lokal (px)",
            options=[4, 6, 8, 10, 12, 16],
            key="clahe_tile_size",
            help="Ukuran jendela pembagian blok histogram adaptif.",
        )
        st.button("Reset Parameter", on_click=reset_parameters)

    elif selected_method == "Histogram Equalization + Bilateral Filtering":
        diameter = st.slider(
            "Diameter Lingkungan (d)",
            min_value=3,
            max_value=15,
            step=2,
            key="bilateral_diameter",
            help="Diameter piksel tetangga yang disertakan dalam filter.",
        )
        sigma_color = st.slider(
            "Sigma Color",
            min_value=10.0,
            max_value=150.0,
            step=5.0,
            key="bilateral_sigma_color",
            help="Toleransi perbedaan intensitas pada penapisan tepi.",
        )
        sigma_space = st.slider(
            "Sigma Space",
            min_value=10.0,
            max_value=150.0,
            step=5.0,
            key="bilateral_sigma_space",
            help="Toleransi jarak spasial piksel tetangga.",
        )
        st.button("Reset Parameter", on_click=reset_parameters)

    elif selected_method == "Histogram Matching":
        if ref_display_name:
            st.info(f"Referensi: {ref_display_name}")
        else:
            st.warning("Perlu citra referensi.")

    else:
        st.caption("Histogram Equalization tidak memerlukan parameter tambahan.")


# ==============================================================================
# 4. HEADER UTAMA (C.8)
# ==============================================================================

status_label = "Mode: Color LAB" if processing_mode == "Color Enhancement" else "Mode: Grayscale"
status_color = "primary" if processing_mode == "Color Enhancement" else "gray"

col_head_title, col_head_badge = st.columns([0.84, 0.16])
with col_head_title:
    st.title("Image Enhancement Studio", anchor=False)
with col_head_badge:
    st.badge(status_label, color=status_color)

st.caption("Laboratorium Pengolahan Citra Digital: Operasi Titik dan Modifikasi Histogram")


# ==============================================================================
# 5. EMPTY STATE (BELUM ADA CITRA)
# ==============================================================================

if active_file is None:
    with st.container(border=True):
        st.subheader("Panduan Penggunaan Laboratorium")
        st.write(
            "Aplikasi ini menyediakan laboratorium interaktif untuk menganalisis karakteristik "
            "citra dan menerapkan metode peningkatan kualitas citra berbasis transformasi titik "
            "dan modifikasi histogram."
        )

        step_col1, step_col2, step_col3 = st.columns(3)

        with step_col1:
            with st.container(border=True):
                st.markdown("**Langkah 1: Unggah Citra**")
                st.write(
                    "Unggah citra uji melalui bilah samping (sidebar) dalam format PNG, JPG, "
                    "BMP, atau TIFF, atau aktifkan opsi Gunakan Citra Contoh."
                )

        with step_col2:
            with st.container(border=True):
                st.markdown("**Langkah 2: Pilih Metode dan Parameter**")
                st.write(
                    "Tentukan mode pemrosesan (Color LAB atau Grayscale) serta pilih algoritma "
                    "enhancement dan sesuaikan parameter operasi sesuai kebutuhan."
                )

        with step_col3:
            with st.container(border=True):
                st.markdown("**Langkah 3: Analisis dan Ekspor**")
                st.write(
                    "Eksplorasi perbandingan visual sebelum dan sesudah, grafik distribusi "
                    "histogram, evaluasi metrik kuantitatif, lalu unduh berkas hasil."
                )
    st.stop()


# ==============================================================================
# 6. PEMBACAAN DAN PREPROCESSING CITRA
# ==============================================================================

try:
    image_rgb, image_gray = load_image(active_file)
    info = image_information(image_rgb)

    reference_rgb = None
    reference_gray = None
    if reference_file is not None:
        try:
            reference_rgb, reference_gray = load_image(reference_file)
        except Exception as ref_err:
            st.sidebar.error(f"Gagal membaca referensi: {ref_err}")
            reference_rgb, reference_gray = None, None

except Exception as err:
    st.error(f"Gagal membaca citra: {err}")
    st.stop()


# ==============================================================================
# 7. DIAGNOSTIC STRIP (B.1, B.8, C.1)
# ==============================================================================

input_for_proc = image_rgb if processing_mode == "Color Enhancement" else image_gray
reference_for_proc = reference_rgb if processing_mode == "Color Enhancement" else reference_gray
disp_before = image_rgb if processing_mode == "Color Enhancement" else image_gray

# Sumber nilai tunggal untuk metrik citra asli (B.1)
metrics_before = normalize_metrics_keys(cached_calculate_metrics(disp_before))
min_intensity = int(metrics_before.get("Min", metrics_before.get("Minimum Intensity", 0)))
max_intensity = int(metrics_before.get("Max", metrics_before.get("Maximum Intensity", 255)))
dynamic_range = int(metrics_before.get("Dynamic Range", max_intensity - min_intensity))

rec_data = recommend_methods(image_gray, has_reference=(reference_for_proc is not None))
primary_rec = rec_data["primary"]
diagnostics = rec_data["diagnostics"]

with st.container(border=True):
    diag_c1, diag_c2, diag_c3, diag_c4 = st.columns([1.1, 1.2, 1.1, 2.6])

    with diag_c1:
        st.caption("Dimensi")
        st.markdown(f"**{info['width']} × {info['height']}**")
        st.caption(f"{info['pixels']:,} px ({'RGB' if info['channels'] == 3 else 'Gray'})")

    with diag_c2:
        st.caption("Rentang Intensitas")
        st.markdown(f"**{min_intensity} – {max_intensity}**")
        st.caption(f"Dynamic Range: {dynamic_range}")

    with diag_c3:
        st.caption("Kontras")
        st.markdown(f"**{diagnostics['contrast']}**")
        st.caption(f"Kecerahan: {diagnostics['brightness']}")

    with diag_c4:
        st.caption(primary_rec["type"])
        st.markdown(f"**{primary_rec['method']}**")
        st.caption(primary_rec["reason"])
        st.button(
            "Terapkan saran",
            key="apply_suggestion_btn",
            on_click=apply_recommendation_callback,
            args=(primary_rec["target_params"],),
        )


# ==============================================================================
# 8. EKSEKUSI PEMROSESAN CITRA DENGAN CACHING
# ==============================================================================

if selected_method == "Histogram Matching" and reference_for_proc is None:
    st.warning("Metode Histogram Matching membutuhkan citra referensi. Silakan unggah citra referensi pada bilah samping.")
    st.stop()

# Cache state checking to prevent redundant enhancement when sliders move
cache_key = (
    hash(disp_before.tobytes()),
    selected_method,
    processing_mode,
    clip_limit,
    tile_size,
    diameter,
    sigma_color,
    sigma_space,
    hash(reference_for_proc.tobytes()) if reference_for_proc is not None else 0,
)

if (
    "last_enhancement_key" not in st.session_state
    or st.session_state.last_enhancement_key != cache_key
    or "current_enhanced_image" not in st.session_state
):
    try:
        enhanced_image = cached_process_enhancement(
            image=input_for_proc,
            method=selected_method,
            color_mode=processing_mode,
            reference=reference_for_proc,
            clip_limit=clip_limit,
            tile_size=tile_size,
            diameter=diameter,
            sigma_color=sigma_color,
            sigma_space=sigma_space,
        )
        st.session_state.current_enhanced_image = enhanced_image
        st.session_state.last_enhancement_key = cache_key
    except Exception as exc:
        st.error(f"Gagal memproses transformasi citra: {exc}")
        st.stop()
else:
    enhanced_image = st.session_state.current_enhanced_image

# Ringkasan parameter teknis aktif (C.3)
if selected_method == "Contrast Limited Adaptive Histogram Equalization":
    param_text = f"CLAHE, clip limit {clip_limit}, grid {tile_size} × {tile_size}"
elif selected_method == "Histogram Equalization + Bilateral Filtering":
    param_text = f"HE + Bilateral, d {diameter}, σ color {sigma_color}, σ space {sigma_space}"
elif selected_method == "Histogram Matching":
    param_text = f"Histogram Matching, referensi: {ref_display_name if ref_display_name else '-'}"
else:
    param_text = "Histogram Equalization, global transform"


# ==============================================================================
# 9. NAVIGASI EMPAT TAB
# ==============================================================================

tab_enhancement, tab_histogram, tab_metrics, tab_export = st.tabs([
    "Enhancement Studio",
    "Histogram dan Distribusi",
    "Analisis Kualitas dan Metrik",
    "Ekspor dan Laporan",
])


# ------------------------------------------------------------------------------
# TAB 1: ENHANCEMENT STUDIO (G.1 - G.6)
# ------------------------------------------------------------------------------
with tab_enhancement:
    st.caption(f"{param_text} | Mode: {processing_mode}")

    view_col1, view_col2 = st.columns([1, 3])
    with view_col1:
        view_mode = st.segmented_control(
            "Mode Tampilan:",
            options=["Muat layar", "Detail area"],
            default="Muat layar",
            help="Muat layar menampilkan citra penuh. Detail area menampilkan perbesaran piksel asli 1:1 pada jendela crop.",
        )

    disp_view_before = disp_before
    disp_view_after = enhanced_image
    detail_caption = ""

    if view_mode == "Detail area":
        h_img, w_img = disp_before.shape[:2]
        min_side = min(h_img, w_img)
        max_win = max(100, min_side)

        col_crop_ctrl, col_crop_map = st.columns([3, 1])

        with col_crop_ctrl:
            if max_win > 100:
                win_val = min(200, max_win)
                window_size = st.slider(
                    "Ukuran Jendela Crop (px)",
                    min_value=100,
                    max_value=max_win,
                    value=win_val,
                    step=50,
                    key="crop_window_slider",
                )
            else:
                window_size = max_win

            max_x = max(0, w_img - window_size)
            max_y = max(0, h_img - window_size)

            # Clamp state jika ukuran jendela membesar (G.2)
            if "slider_crop_x" in st.session_state and st.session_state.slider_crop_x > max_x:
                st.session_state.slider_crop_x = max_x
            if "slider_crop_y" in st.session_state and st.session_state.slider_crop_y > max_y:
                st.session_state.slider_crop_y = max_y

            st.session_state.crop_pos_x = min(st.session_state.get("crop_pos_x", 0), max_x)
            st.session_state.crop_pos_y = min(st.session_state.get("crop_pos_y", 0), max_y)

            cx1, cx2 = st.columns(2)
            with cx1:
                if max_x > 0:
                    crop_x = st.slider(
                        "Posisi X (px)",
                        min_value=0,
                        max_value=max_x,
                        key="slider_crop_x",
                    )
                    st.session_state.crop_pos_x = crop_x
                else:
                    crop_x = 0
                    st.session_state.crop_pos_x = 0
                    st.caption("Posisi X: 0 px (lebar penuh)")

            with cx2:
                if max_y > 0:
                    crop_y = st.slider(
                        "Posisi Y (px)",
                        min_value=0,
                        max_value=max_y,
                        key="slider_crop_y",
                    )
                    st.session_state.crop_pos_y = crop_y
                else:
                    crop_y = 0
                    st.session_state.crop_pos_y = 0
                    st.caption("Posisi Y: 0 px (tinggi penuh)")

            # Tombol pintas lokasi crop (G.4)
            st.caption("Pintas Lokasi:")
            btn_c1, btn_c2, btn_c3 = st.columns(3)
            gray_orig_full = image_gray
            gray_enh_full = cv2.cvtColor(enhanced_image, cv2.COLOR_RGB2GRAY) if enhanced_image.ndim == 3 else enhanced_image

            with btn_c1:
                st.button(
                    "Tengah",
                    key="btn_shortcut_center",
                    on_click=set_crop_shortcut,
                    args=("center", w_img, h_img, window_size, gray_orig_full, gray_enh_full),
                )
            with btn_c2:
                st.button(
                    "Area perubahan terbesar",
                    key="btn_shortcut_diff",
                    on_click=set_crop_shortcut,
                    args=("diff", w_img, h_img, window_size, gray_orig_full, gray_enh_full),
                )
            with btn_c3:
                st.button(
                    "Area paling gelap",
                    key="btn_shortcut_dark",
                    on_click=set_crop_shortcut,
                    args=("darkest", w_img, h_img, window_size, gray_orig_full, gray_enh_full),
                )

        with col_crop_map:
            # Minimap penanda area crop (G.3)
            st.caption("Penanda Area:")
            thumb_w = 160
            thumb_h = max(1, int(h_img * (thumb_w / w_img)))
            thumb = cv2.resize(image_rgb, (thumb_w, thumb_h), interpolation=cv2.INTER_AREA)

            scale_x = thumb_w / w_img
            scale_y = thumb_h / h_img
            x1_rect = int(crop_x * scale_x)
            y1_rect = int(crop_y * scale_y)
            x2_rect = int((crop_x + window_size) * scale_x)
            y2_rect = int((crop_y + window_size) * scale_y)

            thumb_marked = thumb.copy()
            cv2.rectangle(thumb_marked, (x1_rect, y1_rect), (x2_rect, y2_rect), (29, 78, 216), 2)
            st.image(thumb_marked, width=thumb_w)

        # Potong area citra
        crop_raw_before = disp_before[crop_y : crop_y + window_size, crop_x : crop_x + window_size]
        crop_raw_after = enhanced_image[crop_y : crop_y + window_size, crop_x : crop_x + window_size]

        # Perbesar dengan INTER_NEAREST untuk menjaga piksel asli (G.1)
        scale_factor = max(1, round(450 / window_size))
        target_w = window_size * scale_factor
        target_h = window_size * scale_factor

        disp_view_before = cv2.resize(crop_raw_before, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
        disp_view_after = cv2.resize(crop_raw_after, (target_w, target_h), interpolation=cv2.INTER_NEAREST)

        detail_caption = f"Area X {crop_x} sampai {crop_x + window_size}, Y {crop_y} sampai {crop_y + window_size}, perbesaran {scale_factor} kali"

    col_view_before, col_view_after = st.columns(2)

    with col_view_before:
        with st.container(border=True):
            st.markdown("**Sebelum Enhancement**")
            st.image(disp_view_before, use_container_width=True)
            if view_mode == "Muat layar":
                st.caption(f"Ditampilkan diperkecil dari {info['width']} × {info['height']} px")
            else:
                st.caption(detail_caption)

    with col_view_after:
        with st.container(border=True):
            st.markdown(f"**Sesudah Enhancement ({METHOD_LABELS.get(selected_method, selected_method)})**")
            st.image(disp_view_after, use_container_width=True)
            if view_mode == "Muat layar":
                st.caption(f"Ditampilkan diperkecil dari {info['width']} × {info['height']} px")
            else:
                st.caption(detail_caption)

    # Tampilan opsional: Peta Selisih (G.6)
    with st.expander("Peta Selisih (Difference Map)"):
        gray_b = cv2.cvtColor(disp_before, cv2.COLOR_RGB2GRAY) if disp_before.ndim == 3 else disp_before
        gray_a = cv2.cvtColor(enhanced_image, cv2.COLOR_RGB2GRAY) if enhanced_image.ndim == 3 else enhanced_image
        diff_map = cv2.absdiff(gray_b, gray_a)
        diff_norm = cv2.normalize(diff_map, None, 0, 255, cv2.NORM_MINMAX)
        st.image(diff_norm, caption="Peta selisih absolut luminance dinormalisasi (0 = tanpa perubahan, 255 = perubahan maksimum).", use_container_width=True)


# ------------------------------------------------------------------------------
# TAB 2: HISTOGRAM DAN DISTRIBUSI (A, B, C, D)
# ------------------------------------------------------------------------------
with tab_histogram:
    # Penanganan D: Jika Histogram Matching dipilih tapi referensi belum diunggah
    if selected_method == "Histogram Matching" and reference_for_proc is None:
        st.warning("Unggah citra referensi di sidebar")
    else:
        # A: Satu sumber angka terpadu untuk seluruh data Tab 2
        is_color_proc = (processing_mode == "Color Enhancement")
        tab2_data = compute_tab2_histogram_data(
            disp_before=disp_before,
            enhanced_image=enhanced_image,
            reference_image=reference_for_proc,
            is_color=is_color_proc,
        )

        sub_tab_lum, sub_tab_rgb = st.tabs(["Luminance", "RGB"])

        # ======================================================================
        # SUB TAB LUMINANCE (B.1 - B.6)
        # ======================================================================
        with sub_tab_lum:
            # B.2: Judul tunggal tanpa redundansi
            st.subheader(f"Histogram Luminance, {METHOD_LABELS.get(selected_method, selected_method)}")

            lum_info = tab2_data["luminance"]

            # B.3: Keterangan otomatis satu baris dari angka sebenarnya tanpa kata "lebih baik/buruk"
            st.caption(
                f"Rentang intensitas berubah dari {lum_info['min_before']} sampai {lum_info['max_before']} "
                f"menjadi {lum_info['min_after']} sampai {lum_info['max_after']}; "
                f"Mean berubah dari {lum_info['mean_before']:.1f} menjadi {lum_info['mean_after']:.1f}."
            )

            # B.1: Gambar grafik histogram Altair unstacked, B.4 garis mean, D garis referensi
            images_hist = {
                "Sebelum Enhancement": disp_before,
                "Sesudah Enhancement": enhanced_image,
            }
            if selected_method == "Histogram Matching" and reference_for_proc is not None:
                images_hist["Citra Referensi"] = reference_for_proc

            chart_hist = create_altair_histogram_chart(
                images_dict=images_hist,
                mean_before=lum_info["mean_before"],
                mean_after=lum_info["mean_after"],
            )
            st.altair_chart(chart_hist, use_container_width=True)

            # C.1: Keterangan saturasi luminance satu baris cerdas (tanpa kartu besar)
            lum_sat = lum_info["saturation"]
            cap_lum = format_saturation_caption(
                b0_cnt=lum_sat["b0_count"],
                b0_pct=lum_sat["b0_pct"],
                a0_cnt=lum_sat["a0_count"],
                a0_pct=lum_sat["a0_pct"],
                b255_cnt=lum_sat["b255_count"],
                b255_pct=lum_sat["b255_pct"],
                a255_cnt=lum_sat["a255_count"],
                a255_pct=lum_sat["a255_pct"],
                min_val=lum_info["min_before"],
                max_val=lum_info["max_before"],
                is_luminance=True,
            )
            st.caption(cap_lum)

            # D: Jarak histogram hasil ke referensi (korelasi & Bhattacharyya)
            if selected_method == "Histogram Matching" and reference_for_proc is not None:
                dist_b = calculate_histogram_distance(disp_before, reference_for_proc)
                dist_a = calculate_histogram_distance(enhanced_image, reference_for_proc)
                st.caption("Jarak Histogram Luminance terhadap Citra Referensi:")
                d_c1, d_c2 = st.columns(2)
                with d_c1:
                    st.metric(
                        label="Histogram Correlation",
                        value=f"{dist_b['Histogram Correlation']:.4f} → {dist_a['Histogram Correlation']:.4f}",
                    )
                with d_c2:
                    st.metric(
                        label="Bhattacharyya Distance",
                        value=f"{dist_b['Bhattacharyya Distance']:.4f} → {dist_a['Bhattacharyya Distance']:.4f}",
                    )

        # ======================================================================
        # SUB TAB RGB (C.1 - C.6)
        # ======================================================================
        with sub_tab_rgb:
            rgb_info = tab2_data["rgb"]
            if rgb_info is not None:
                st.subheader("Distribusi Kanal Warna (RGB)")

                # C.1: Jika tidak ada piksel jenuh di ketiga kanal, tulis di atas grafik
                if rgb_info["all_zero_saturation"]:
                    st.caption("Tidak ada piksel jenuh pada ketiga kanal.")

                # Kontrol Skala Sumbu Y
                scale_mode = st.segmented_control(
                    "Skala Sumbu Y:",
                    options=["Potong lonjakan", "Linear", "Log"],
                    default="Potong lonjakan",
                )

                # Tentukan batas maksimum Y bersama (C.6)
                all_counts = []
                for ch in ["R", "G", "B"]:
                    all_counts.extend(rgb_info["before"][ch])
                    all_counts.extend(rgb_info["after"][ch])
                    if rgb_info["ref"] is not None:
                        all_counts.extend(rgb_info["ref"][ch])

                if scale_mode == "Potong lonjakan":
                    inner_counts = [c for c in all_counts if c > 0]
                    y_max = float(np.percentile(inner_counts, 99)) * 1.15 if inner_counts else 1000.0
                    y_max = max(10.0, y_max)

                    # C.2: Catatan pemotongan dinamis berbasis data riil
                    clipped_bins = sorted(list(set([
                        b for b in range(256)
                        if any(
                            rgb_info[k][ch][b] > y_max
                            for k in ["before", "after"]
                            for ch in ["R", "G", "B"]
                        ) or (rgb_info["ref"] is not None and any(rgb_info["ref"][ch][b] > y_max for ch in ["R", "G", "B"]))
                    ])))

                    if clipped_bins:
                        bin_str = ", ".join(map(str, clipped_bins[:12]))
                        if len(clipped_bins) > 12:
                            bin_str += "..."
                        st.caption(f"Sumbu Y dibatasi pada {int(y_max):,} piksel; {len(clipped_bins)} bin melebihi batas, yaitu intensitas {bin_str}.")
                    else:
                        st.caption(f"Sumbu Y dibatasi pada {int(y_max):,} piksel; tidak ada bin yang melebihi batas.")

                elif scale_mode == "Log":
                    y_max = float(np.max(all_counts)) if all_counts else 1000.0
                    # C.5: Keterangan bin kosong pada skala log
                    st.caption("Bin kosong ditampilkan sebagai 1.")
                else:  # Linear
                    y_max = float(np.max(all_counts)) if all_counts else 1000.0

                # C.3: Satu legenda bersama di atas tiga grafik
                ref_legend_suffix = " | Garis tipis: Referensi" if rgb_info["ref"] is not None else ""
                st.caption(f"Garis abu: Sebelum | Garis warna: Sesudah{ref_legend_suffix}")

                rgb_col1, rgb_col2, rgb_col3 = st.columns(3)
                ch_configs = [
                    (rgb_col1, "R", "Merah (R)", "#DC2626"),
                    (rgb_col2, "G", "Hijau (G)", "#16A34A"),
                    (rgb_col3, "B", "Biru (B)", "#2563EB"),
                ]

                for col_obj, ch_key, ch_title, ch_color in ch_configs:
                    with col_obj:
                        ref_ch_hist = rgb_info["ref"][ch_key] if rgb_info["ref"] is not None else None
                        chart_ch = create_altair_channel_comparison_chart(
                            hist_before=rgb_info["before"][ch_key],
                            hist_after=rgb_info["after"][ch_key],
                            channel_name=ch_title,
                            channel_color=ch_color,
                            hist_ref=ref_ch_hist,
                            y_scale_mode=scale_mode,
                            y_max_limit=y_max,
                        )
                        st.altair_chart(chart_ch, use_container_width=True)

                        # C.2 & C.3: Informasi saturasi per kanal cerdas
                        sat_ch = rgb_info["saturation"][ch_key]
                        cap_ch = format_saturation_caption(
                            b0_cnt=sat_ch["b0_count"],
                            b0_pct=sat_ch["b0_pct"],
                            a0_cnt=sat_ch["a0_count"],
                            a0_pct=sat_ch["a0_pct"],
                            b255_cnt=sat_ch["b255_count"],
                            b255_pct=sat_ch["b255_pct"],
                            a255_cnt=sat_ch["a255_count"],
                            a255_pct=sat_ch["a255_pct"],
                            is_luminance=False,
                        )
                        st.caption(cap_ch)
            else:
                st.info("Histogram RGB tersedia saat menggunakan mode Color Enhancement pada citra berwarna.")


# ------------------------------------------------------------------------------
# TAB 3: ANALISIS KUALITAS DAN METRIK (E.1 - E.5)
# ------------------------------------------------------------------------------
with tab_metrics:
    metrics_after = normalize_metrics_keys(cached_calculate_metrics(enhanced_image))

    delta_mean = float(metrics_after["Mean"]) - float(metrics_before["Mean"])
    delta_std = float(metrics_after["Std Dev"]) - float(metrics_before["Std Dev"])
    delta_entropy = float(metrics_after["Entropy"]) - float(metrics_before["Entropy"])
    delta_dr = int(metrics_after["Dynamic Range"]) - int(metrics_before["Dynamic Range"])
    delta_bins = int(metrics_after["Bins"]) - int(metrics_before["Bins"])

    # Kartu metrik dengan nilai sebelum sebagai caption (E.3)
    qc1, qc2, qc3, qc4, qc5 = st.columns(5)
    with qc1:
        st.metric("Mean", f"{float(metrics_after['Mean']):.2f}", f"{delta_mean:+.2f}", delta_color="off")
        st.caption(f"Sebelum: {float(metrics_before['Mean']):.2f}")
    with qc2:
        st.metric("Std Dev", f"{float(metrics_after['Std Dev']):.2f}", f"{delta_std:+.2f}", delta_color="off")
        st.caption(f"Sebelum: {float(metrics_before['Std Dev']):.2f}")
    with qc3:
        st.metric("Entropy", f"{float(metrics_after['Entropy']):.3f}", f"{delta_entropy:+.3f}", delta_color="off")
        st.caption(f"Sebelum: {float(metrics_before['Entropy']):.3f}")
    with qc4:
        st.metric("Dynamic Range", f"{int(metrics_after['Dynamic Range'])}", f"{delta_dr:+d}", delta_color="off")
        st.caption(f"Sebelum: {int(metrics_before['Dynamic Range'])}")
    with qc5:
        st.metric("Bins", f"{int(metrics_after['Bins'])}", f"{delta_bins:+d}", delta_color="off")
        st.caption(f"Sebelum: {int(metrics_before['Bins'])}")

    st.subheader("Evaluasi Kuantitatif")
    comparison_df = build_metrics_comparison_table(metrics_before, metrics_after)
    st.dataframe(
        comparison_df,
        column_config={
            "Interpretasi": st.column_config.TextColumn("Interpretasi", width="large")
        },
        use_container_width=True,
        hide_index=True,
    )

    # Tabel Perbandingan Empat Metode Sekaligus (E.2)
    st.subheader("Perbandingan Empat Metode pada Citra yang Sama")
    all_methods_records = [
        {
            "Aktif": "-",
            "Metode": "Citra Asli",
            "Parameter": "Kondisi awal tanpa enhancement",
            "Mean": f"{float(metrics_before['Mean']):.2f}",
            "Std Dev": f"{float(metrics_before['Std Dev']):.2f}",
            "Entropy": f"{float(metrics_before['Entropy']):.3f}",
            "Dynamic Range": str(int(metrics_before["Dynamic Range"])),
            "Bins": str(int(metrics_before["Bins"])),
            "Piksel jenuh": f"{float(metrics_before['Piksel jenuh']):.2f}%",
        }
    ]

    for m_name in METHOD_NAMES:
        is_active = "Aktif" if m_name == selected_method else "-"
        m_short = METHOD_LABELS.get(m_name, m_name)

        if m_name == "Contrast Limited Adaptive Histogram Equalization":
            m_params = f"Clip {clip_limit}, grid {tile_size}×{tile_size}"
        elif m_name == "Histogram Equalization + Bilateral Filtering":
            m_params = f"d {diameter}, σ color {sigma_color}, σ space {sigma_space}"
        elif m_name == "Histogram Matching":
            m_params = f"Referensi: {ref_display_name if ref_display_name else '-'}"
        else:
            m_params = "Global transform"

        if m_name == "Histogram Matching" and reference_for_proc is None:
            all_methods_records.append({
                "Aktif": is_active,
                "Metode": m_short,
                "Parameter": "Unggah citra referensi di sidebar",
                "Mean": "-",
                "Std Dev": "-",
                "Entropy": "-",
                "Dynamic Range": "-",
                "Bins": "-",
                "Piksel jenuh": "-",
            })
            continue

        try:
            m_enh = cached_process_enhancement(
                image=input_for_proc,
                method=m_name,
                color_mode=processing_mode,
                reference=reference_for_proc,
                clip_limit=clip_limit,
                tile_size=tile_size,
                diameter=diameter,
                sigma_color=sigma_color,
                sigma_space=sigma_space,
            )
            m_met = cached_calculate_metrics(m_enh)
            all_methods_records.append({
                "Aktif": is_active,
                "Metode": m_short,
                "Parameter": m_params,
                "Mean": f"{m_met['Mean']:.2f}",
                "Std Dev": f"{m_met['Std Dev']:.2f}",
                "Entropy": f"{m_met['Entropy']:.3f}",
                "Dynamic Range": str(m_met["Dynamic Range"]),
                "Bins": str(m_met["Bins"]),
                "Piksel jenuh": f"{float(m_met['Piksel jenuh']):.2f}%",
            })
        except Exception:
            all_methods_records.append({
                "Aktif": is_active,
                "Metode": m_short,
                "Parameter": m_params,
                "Mean": "-",
                "Std Dev": "-",
                "Entropy": "-",
                "Dynamic Range": "-",
                "Bins": "-",
                "Piksel jenuh": "-",
            })

    all_methods_df = pd.DataFrame(all_methods_records)
    st.dataframe(all_methods_df, use_container_width=True, hide_index=True)

    # Sintesis Kualitas (dua paragraf pendek, E.4)
    par1, par2 = generate_quality_summary(
        metrics_before=metrics_before,
        metrics_after=metrics_after,
        method_name=METHOD_LABELS.get(selected_method, selected_method),
        color_mode=processing_mode,
    )
    with st.container(border=True):
        st.write(par1)
        st.caption(par2)

    # Dua expander bersih (E.5)
    with st.expander("Panduan Akademis: Memahami Arti Setiap Metrik"):
        for key, item in METRIC_EXPLANATIONS.items():
            if key in ["PSNR", "SSIM"]:
                continue
            st.markdown(f"**{item['title']}** ({item['unit']})")
            st.write(item["definition"])
            st.caption(f"Catatan mode: {item['color_note']}")
            st.markdown("---")

    with st.expander("Kemiripan terhadap citra asli (PSNR dan SSIM)"):
        st.info(
            "Peringatan Metodologis: PSNR dan SSIM mengukur seberapa jauh citra berubah dari aslinya, "
            "bukan seberapa baik kualitasnya. Operasi enhancement memang sengaja mengubah kontras, "
            "sehingga PSNR yang lebih tinggi hanya berarti perubahannya lebih kecil."
        )

        # Tabel perbandingan PSNR dan SSIM antar metode (E.2)
        sim_records = []
        for m_name in METHOD_NAMES:
            if m_name == "Histogram Matching" and reference_for_proc is None:
                sim_records.append({
                    "Metode": METHOD_LABELS.get(m_name, m_name),
                    "PSNR (dB)": "-",
                    "SSIM": "-",
                    "Keterangan": "Perlu citra referensi",
                })
                continue
            try:
                m_enh = cached_process_enhancement(
                    image=input_for_proc,
                    method=m_name,
                    color_mode=processing_mode,
                    reference=reference_for_proc,
                    clip_limit=clip_limit,
                    tile_size=tile_size,
                    diameter=diameter,
                    sigma_color=sigma_color,
                    sigma_space=sigma_space,
                )
                m_ref = cached_calculate_reference_metrics(m_enh, disp_before)
                sim_records.append({
                    "Metode": METHOD_LABELS.get(m_name, m_name),
                    "PSNR (dB)": f"{m_ref['PSNR']:.2f}",
                    "SSIM": f"{m_ref['SSIM']:.4f}",
                    "Keterangan": "Terhadap citra masukan asli",
                })
            except Exception:
                pass

        st.dataframe(pd.DataFrame(sim_records), use_container_width=True, hide_index=True)

        if selected_method == "Histogram Matching" and reference_for_proc is not None:
            dist_metrics = calculate_histogram_distance(enhanced_image, reference_for_proc)
            dcol1, dcol2 = st.columns(2)
            with dcol1:
                st.metric("Histogram Correlation (vs Citra Referensi)", f"{dist_metrics['Histogram Correlation']:.4f}")
            with dcol2:
                st.metric("Bhattacharyya Distance (vs Citra Referensi)", f"{dist_metrics['Bhattacharyya Distance']:.4f}")


# ------------------------------------------------------------------------------
# TAB 4: EKSPOR DAN LAPORAN (F.1 - F.3)
# ------------------------------------------------------------------------------
with tab_export:
    export_c1, export_c2 = st.columns(2)

    png_bytes = image_to_png_bytes(enhanced_image)
    image_filename = generate_export_filename(
        original_filename=orig_name,
        method=selected_method,
        extension="png",
    )
    png_size_kb = len(png_bytes) / 1024.0

    with export_c1:
        with st.container(border=True):
            st.subheader("Citra Hasil Enhancement")
            # Thumbnail kecil hasil (F.2)
            st.image(enhanced_image, width=160)
            st.caption(f"PNG | {info['width']} × {info['height']} px | Ukuran berkas: ~{png_size_kb:.1f} KB")
            st.download_button(
                label="Unduh PNG",
                data=png_bytes,
                file_name=image_filename,
                mime="image/png",
                use_container_width=True,
            )
            st.caption(image_filename)

    timestamp_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    dimensions_str = f"{info['width']} × {info['height']} px"

    csv_bytes = build_export_csv(
        file_name=orig_name,
        mode=processing_mode,
        method=selected_method,
        parameters=param_text,
        dimensions=dimensions_str,
        timestamp=timestamp_str,
        comparison_df=comparison_df,
        reference_file_name=ref_display_name,
    )
    csv_filename = generate_export_filename(
        original_filename=orig_name,
        method=selected_method,
        extension="csv",
    )
    csv_size_kb = len(csv_bytes) / 1024.0

    csv_all_methods_bytes = build_all_methods_export_csv(
        file_name=orig_name,
        mode=processing_mode,
        dimensions=dimensions_str,
        timestamp=timestamp_str,
        all_methods_df=all_methods_df,
        reference_file_name=ref_display_name,
    )
    csv_all_filename = generate_export_filename(
        original_filename=orig_name,
        method="komparasi_semua",
        extension="csv",
    )

    with export_c2:
        with st.container(border=True):
            st.subheader("Laporan Evaluasi Metrik")
            st.caption(f"CSV (UTF-8-SIG) | 12 kolom evaluasi | Ukuran berkas: ~{csv_size_kb:.1f} KB")
            st.download_button(
                label="Unduh CSV",
                data=csv_bytes,
                file_name=csv_filename,
                mime="text/csv",
                use_container_width=True,
            )
            st.caption(csv_filename)

            st.download_button(
                label="Unduh CSV Komparasi 4 Metode",
                data=csv_all_methods_bytes,
                file_name=csv_all_filename,
                mime="text/csv",
                use_container_width=True,
            )
            st.caption(csv_all_filename)
