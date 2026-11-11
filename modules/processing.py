"""
Modul pemrosesan citra untuk Image Enhancement Studio.
Menyediakan algoritma peningkatan kualitas citra (HE, CLAHE, Histogram Matching, HE + Bilateral)
baik untuk citra Grayscale 8-bit maupun citra warna (CIE LAB L-channel).
"""

from modules.enhancement import (
    histogram_equalization,
    clahe_enhancement,
    histogram_matching,
    equalization_bilateral,
    histogram_equalization_color,
    clahe_enhancement_color,
    histogram_matching_color,
    equalization_bilateral_color,
    process_enhancement,
    METHODS,
    METHODS_COLOR,
)

__all__ = [
    "histogram_equalization",
    "clahe_enhancement",
    "histogram_matching",
    "equalization_bilateral",
    "histogram_equalization_color",
    "clahe_enhancement_color",
    "histogram_matching_color",
    "equalization_bilateral_color",
    "process_enhancement",
    "METHODS",
    "METHODS_COLOR",
]
