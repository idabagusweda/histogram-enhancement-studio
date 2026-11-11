<p align="center">
  <img src="./assets/sample_test.png" alt="Histogram Enhancement Studio" width="180" />
</p>

<h1 align="center">Histogram Enhancement Studio</h1>

<p align="center">
  An interactive web-based digital image processing studio for pixel intensity transformation and histogram enhancement.
</p>

<p align="center">
  <a href="https://histogram-enhancement-studio.streamlit.app">Live Demo</a>
  &nbsp;·&nbsp;
  <a href="https://github.com/idabagusweda/historgram-enhancement-studio/issues">Report an Issue</a>
</p>

<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white" />
  <img alt="Streamlit" src="https://img.shields.io/badge/Streamlit-1.35%2B-FF4B4B?logo=streamlit&logoColor=white" />
  <img alt="OpenCV" src="https://img.shields.io/badge/OpenCV-Headless-5C3EE8?logo=opencv&logoColor=white" />
  <img alt="NumPy" src="https://img.shields.io/badge/NumPy-Scientific-013243?logo=numpy&logoColor=white" />
  <a href="./LICENSE"><img alt="License: MIT" src="https://img.shields.io/badge/License-MIT-yellow.svg" /></a>
</p>

---

## Overview

Histogram Enhancement Studio is an interactive digital image processing application designed to visualize, analyze, and enhance digital image quality through intensity distribution manipulation. The studio implements modular image enhancement workflows, separating user interface orchestration, image preprocessing, enhancement algorithms, histogram analytics, objective quality metrics, heuristic recommendation, and export services.

The application features CIE LAB color space enhancement to prevent hue shifts during contrast expansion, as well as classic grayscale point operations. Users can interactively compare Before/After states with crop minimaps, inspect distribution shifts via Altair histograms and CDF graphs, examine objective metrics, and obtain algorithmic recommendations based on image statistical properties.

A live deployment is available at [histogram-enhancement-studio.streamlit.app](https://histogram-enhancement-studio.streamlit.app).

## Features

### Image Enhancement and Processing

- **Color enhancement via CIE LAB:** Transforms images into the CIE LAB color space and applies contrast enhancement exclusively to the Luminance ($L$) channel, keeping chrominance ($a$ and $b$) intact to prevent hue distortions.
- **Grayscale enhancement:** Monochromatic 8-bit mapping for classical grayscale intensity distribution experiments.
- **Histogram Equalization (HE):** Flattens the Cumulative Distribution Function (CDF) globally to maximize dynamic contrast across $[0, 255]$.
- **Contrast Limited Adaptive Histogram Equalization (CLAHE):** Tile-based adaptive local enhancement with configurable clip limit and grid size to avoid over-amplifying noise in uniform regions.
- **Histogram Matching (Specification):** Adjusts the intensity distribution of an input image to align with the distribution profile of a chosen reference image.
- **HE + Bilateral Filtering:** Combines global histogram equalization with edge-preserving bilateral filtering to suppress background noise while maintaining sharp object boundaries.

### Analysis, Diagnostics, and Reporting

- **Interactive visual comparison:** Dual-panel inspection featuring interactive crop coordinates, magnification minimaps, and localized zoom.
- **Distribution analytics:** Interactive Altair charts for intensity histograms, CDF curves, multi-channel comparisons, and pixel saturation warnings.
- **Quantitative quality evaluation:** Computes statistical metrics including Mean Intensity, Standard Deviation, Shannon Entropy (bits), Dynamic Range, Occupied Histogram Bins, and optional full-reference metrics (PSNR and SSIM).
- **Automated method recommendation:** Heuristic diagnostics assessing brightness, contrast, and histogram dispersion to suggest suitable algorithms with quantitative explanations.
- **Full-resolution export:** Lossless PNG image downloads at original source dimensions alongside comprehensive evaluation tables exported as UTF-8-SIG CSV files for spreadsheet compatibility.

## Technology Stack

| Area | Technology |
| --- | --- |
| Application Framework | Streamlit (Python) |
| Image Processing | OpenCV (`opencv-python-headless`), scikit-image, Pillow (PIL) |
| Scientific Computing | NumPy, SciPy, Pandas |
| Interactive Visualizations | Altair, Matplotlib |
| Theming and Design | Custom Streamlit Design System (`.streamlit/config.toml`) |
| Data Export | UTF-8-SIG CSV, Lossless PNG (`io.BytesIO`) |

## Getting Started

### Prerequisites

- Python 3.10 or later
- pip package manager or virtual environment tool (`venv`, `uv`, `conda`)
- Modern web browser (Chrome, Edge, Firefox, Safari)

### 1. Clone the repository

```bash
git clone https://github.com/idabagusweda/historgram-enhancement-studio.git
cd historgram-enhancement-studio
```

### 2. Install dependencies

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install required packages:

```bash
pip install -r requirements.txt
```

### 3. Configure environment variables

Create a local environment file from the example (optional):

```bash
cp .env.example .env
```

Histogram Enhancement Studio runs entirely client-side and in-memory. No private API keys or database credentials are required. Optional Streamlit server parameters (e.g., port or headless flags) can be customized if needed.

### 4. Configure Supabase

This project operates as a self-contained Python application and does not require Supabase or any external database backend. All image transformations and analytics are computed locally in-memory.

### 5. Run the development server

Start the Streamlit application server:

```bash
python -m streamlit run app.py
```

The application will launch and become available at `http://localhost:8501`.

## Environment Variables

### Frontend (`.env`)

| Variable | Description |
| --- | --- |
| `STREAMLIT_SERVER_PORT` | Optional custom port for Streamlit server (default: `8501`). |
| `STREAMLIT_SERVER_ADDRESS` | Optional bind address for the server (default: `localhost`). |
| `STREAMLIT_SERVER_HEADLESS` | Optional headless mode flag (`true` or `false`). |

### Supabase Edge Functions

Not applicable. The application does not rely on serverless edge functions or remote backend microservices.

## Project Structure

```text
historgram-enhancement-studio/
├── assets/                     # Sample images for testing and demonstration
│   ├── sample_reference.png    # Target reference image for histogram matching
│   └── sample_test.png         # Input image for quick preview and verification
├── modules/                    # Modular image processing and analysis backend
│   ├── enhancement.py          # Core enhancement algorithms (HE, CLAHE, Matching, Bilateral)
│   ├── export_results.py       # Full-resolution PNG and CSV report exporters
│   ├── histogram_analysis.py   # Histogram computations and Altair chart visualizers
│   ├── metrics.py              # Statistical quality metrics (Mean, Entropy, PSNR, SSIM)
│   ├── preprocessing.py        # Image decoding, validation, and metadata extraction
│   ├── processing.py           # Orchestration router between color modes and algorithms
│   └── recommendation.py       # Automated diagnostic and method recommendation heuristics
├── .streamlit/                 # Streamlit configuration and custom theme tokens
│   └── config.toml             # Custom color palette and viewer configurations
├── .env.example                # Optional environment variables template
├── .gitignore                  # Git ignore rules for Python, cache, and IDE files
├── app.py                      # Main Streamlit application entry point and UI layout
├── CONTRIBUTING.md             # Contribution guidelines and local development instructions
├── LICENSE                     # MIT License file
├── README.md                   # Project documentation
└── requirements.txt            # Locked and bounded Python dependencies
```

## Testing

The project does not currently include an automated unit test suite. The following core flows should be checked manually when validating a local or deployed instance:

- Upload a test image or enable the built-in sample test image.
- Switch between Color Enhancement (CIE LAB) and Grayscale Enhancement modes to verify color preservation.
- Execute each of the four enhancement algorithms (HE, CLAHE, Histogram Matching, and HE + Bilateral Filtering).
- Verify interactive Altair histogram graphs and CDF curves across tabs.
- Review calculated quantitative metrics (Mean, Std Dev, Shannon Entropy, Dynamic Range, Bins).
- Test image cropping, localized minimap zoom, and parameter adjustment sliders.
- Export processed images as lossless PNG and download evaluation tables as CSV.

## Build and Deployment

To deploy Histogram Enhancement Studio on Streamlit Community Cloud:

1. Push the repository to GitHub: `https://github.com/idabagusweda/historgram-enhancement-studio`.
2. Navigate to [Streamlit Community Cloud](https://share.streamlit.io/) and sign in with your GitHub account.
3. Select **New app**, choose repository `idabagusweda/historgram-enhancement-studio`, branch `main`, and main file path `app.py`.
4. Click **Deploy**.

## Contributing

Contributions and suggestions are welcome. To propose a change:

1. Fork the repository.
2. Create a branch for your change (`feat/your-feature-name`).
3. Make and test your updates locally.
4. Open a pull request with a clear description of the changes.

Please follow the existing code style and update the documentation when behavior or configuration changes. See [`CONTRIBUTING.md`](./CONTRIBUTING.md) for details.

## License

This project is distributed under the MIT License. See the [`LICENSE`](./LICENSE) file for details.

## Acknowledgements

This project uses open-source tools and libraries including [Streamlit](https://streamlit.io/), [OpenCV](https://opencv.org/), [scikit-image](https://scikit-image.org/), [NumPy](https://numpy.org/), [Pandas](https://pandas.pydata.org/), [Matplotlib](https://matplotlib.org/), [Altair](https://altair-viz.github.io/), and [Pillow](https://python-pillow.org/).

---

<p align="center">
  <a href="https://github.com/idabagusweda">GitHub: idabagusweda</a>
</p>
