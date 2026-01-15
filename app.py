import os
from typing import List, Tuple, Optional

import streamlit as st
from PIL import Image


# ============================================================
# CONFIGURATION (adjust if needed)
# ============================================================
WIND10_DIR = "wind"
PRESSURE_DIR = "pressure"
WIND850_DIR = "wind_850"
PMIN_PLOT_PATH = os.path.join("pmin_evolution", "pmin_timeseries.png")

# Display width (pixels). Reduce if needed: 800, 750, etc.
IMAGE_WIDTH = 900

# To center images, we use a three-column layout.
CENTER_COLS = [1, 4, 1]  # [left, center, right]


# ============================================================
# STREAMLIT PAGE
# ============================================================
st.set_page_config(
    page_title="Hurricane Dorian (2019) — Demonstrator",
    layout="wide",
)


# ============================================================
# IMAGE HELPERS
# ============================================================
def list_png_paths(folder: str) -> Tuple[List[str], List[str]]:
    """Return (paths, filenames), sorted by filename."""
    if not os.path.isdir(folder):
        raise FileNotFoundError(f"Folder '{folder}' does not exist.")
    files = sorted([f for f in os.listdir(folder) if f.lower().endswith(".png")])
    if not files:
        raise RuntimeError(f"No .png images found in '{folder}'.")
    paths = [os.path.join(folder, f) for f in files]
    return paths, files


@st.cache_data(show_spinner=False)
def cached_list_png_paths(folder: str) -> Tuple[List[str], List[str]]:
    # Cache the file list only (images themselves are not loaded into memory)
    return list_png_paths(folder)


def open_image(path: str) -> Image.Image:
    # Images are loaded at their native resolution;
    # only the display size is controlled in Streamlit.
    return Image.open(path)


def render_slider_image_section(
    *,
    slider_label: str,
    folder: str,
    state_key: str,
    caption_prefix: str,
):
    """
    Slider + image display for a PNG folder.
    Session state warnings are avoided, and the image is centered
    with a controlled display size.
    """
    try:
        paths, files = cached_list_png_paths(folder)
    except Exception as e:
        st.error(str(e))
        return

    # IMPORTANT:
    # We let Streamlit fully manage the slider value via `key=state_key`.
    # No manual session_state initialization, no explicit `value=...`,
    # which avoids Streamlit warnings.
    idx = st.slider(
        slider_label,
        min_value=0,
        max_value=len(paths) - 1,
        step=1,
        key=state_key,
    )

    st.caption(f"{caption_prefix} — frame {idx} — file: {files[idx]}")

    col_l, col_c, col_r = st.columns(CENTER_COLS)
    with col_c:
        st.image(open_image(paths[idx]), width=IMAGE_WIDTH)


def render_single_image(
    *,
    image_path: str,
    caption: Optional[str] = None,
):
    """Display a single image (e.g. a time series), centered and resized."""
    if not os.path.isfile(image_path):
        st.error(f"Image not found: {image_path}")
        return

    if caption:
        st.caption(caption)

    col_l, col_c, col_r = st.columns(CENTER_COLS)
    with col_c:
        st.image(open_image(image_path), width=IMAGE_WIDTH)


# ============================================================
# TEXT CONTENT
# ============================================================

INTRO_MD = """
# Hurricane Dorian (2019) — Demonstrator

This notebook represents an initial prototype of a future interactive application
designed to analyze extreme weather events and to compare the quality of forecasts
produced by:

- traditional numerical weather prediction models used at the time of the events,
- and more recent, AI-based models.

The long-term idea is to provide a public-facing tool allowing users to select a major
event (hurricane, heatwave, atmospheric river, etc.), visualize its observed evolution
using ERA5 data, and visually compare forecasts produced by different models.

This notebook focuses on a first case study: Hurricane Dorian (2019), an exceptionally
intense event that was poorly anticipated in some critical aspects, making it an ideal
starting point for exploring differences between forecasting approaches.

Two complementary models are considered conceptually: **HRES**, the operational reference
model in 2019 (available through archived forecasts but not re-runnable), and **GraphCast**,
a modern, open-source AI model that can be executed efficiently. This combination allows
a direct comparison between a historical operational model and a contemporary AI-based
approach.
"""

VARIABLES_MD = """
## Variables used in the analysis

The variables are grouped into three complementary categories:

- **Visual variables**, because clear and intuitive representations are essential for
  understanding a cyclone — an aspect that is often neglected in existing tools;
- **Intensity variables**, which directly describe the strength of the cyclone and are
  a frequent source of divergence between models;
- **Trajectory variables**, because storm steering and track deviations lie at the heart
  of many historical forecast errors, including those from HRES.

For the visual analysis, we primarily use **10 m wind speed**, which immediately reveals
the spiral structure and the cyclone eye, and **surface pressure**, whose isobars clearly
highlight the pressure depression. Together, these fields produce intuitive and readable maps.

Cyclone intensity is described using **10 m wind speed** and **minimum sea-level pressure**,
defined as the lowest pressure value within the cyclone area. These two indicators determine
the cyclone category. **850 hPa wind speed** further complements the analysis by providing
information on low-level structure and rapid intensification phases.

Finally, trajectory analysis relies on the **center of minimum pressure**, and, more broadly,
on **500 hPa geopotential height**, which represents the large-scale atmospheric “relief”
responsible for storm steering. **500 hPa winds** and **850 hPa winds** help characterize
upper-level and low-level steering flows, which are essential for understanding track errors.
"""

WIND10_TEXT_MD = """
### 10 m wind speed

The 10 m wind speed provides a very direct view of cyclone structure: spiral rainbands,
circulation wrapping, and sometimes the eye become immediately visible when the system
is well organized. It is a simple variable, yet highly informative, as it reflects the
near-surface circulation where the most destructive winds occur.

From a dynamical perspective, surface winds respond directly to pressure drops at the
cyclone center: as central pressure decreases, air accelerates toward the low-pressure
core. This is why increasing wind speed is often one of the earliest signs of cyclone
intensification.

Finally, 10 m wind speed is the reference variable used for cyclone classification
(Saffir–Simpson scale), making it a common benchmark between observations and models.
Tracking its evolution helps characterize structure, intensity, and life-cycle phases
of the system.
"""

PRESSURE_TEXT_MD = """
## Surface pressure

Surface pressure is one of the most reliable indicators of cyclone structure. When
isobars are plotted, a closed depression immediately appears, marking the signature
of the cyclonic system. The lower the central pressure, the stronger the surrounding
pressure gradient and the more organized the circulation.

This field is particularly useful from a visual standpoint: it highlights the spatial
extent of the system, vortex symmetry, and the evolution of the pressure minimum over
time. It is also dynamically meaningful, as surface pressure reflects the depth of the
air column and the coupling between the cyclone core and the lower atmosphere.

In short, surface pressure reveals the “framework” of the cyclone — the shape of the
depression, its spatial footprint, and the evolution of its minimum — and naturally
complements the surface wind analysis.

On the maps shown here, the **minimum surface pressure (Pmin)** is explicitly identified
and annotated. This provides a natural bridge between a spatial view (where is the
minimum located?) and a more synthetic intensity metric (how does Pmin evolve over time?).
"""

PMIN_TEXT_MD = """
## Intensity: evolution of minimum surface pressure (Pmin(t))

Pressure maps allow the minimum to be identified visually, but comparing intensity
over time is clearer when the information is summarized into a single indicator:
**Pmin(t)**.

At each time step, the lowest pressure associated with the cyclone is retained (in this
approach: the minimum searched in the vicinity of the tracked center). This time series
provides a robust measure of cyclone deepening: when **Pmin decreases**, the cyclone
is intensifying.

This plot therefore corresponds to the **intensity** dimension. It does not describe
the cyclone trajectory, but rather how its depth evolves through time. It directly
complements pressure maps and facilitates comparisons between models or between
different phases of the event.
"""

WIND850_TEXT_MD = """
## 850 hPa wind

The 850 hPa wind (approximately 1–1.5 km above the surface) complements the intensity
analysis by providing insight into the low-level structure of the cyclone, less directly
affected by surface processes.

This field is useful for identifying phases of organization or rapid intensification:
a cyclone may strengthen in the lower troposphere before this signal becomes fully
visible at the surface. Conversely, asymmetries linked to environmental interactions
or vertical wind shear may be more clearly expressed at 850 hPa.

The goal is not to replace 10 m wind speed — which remains the reference for impacts —
but to complement it: if 10 m wind describes “what happens at the surface”, 850 hPa wind
helps understand “how the system is organized” in the lower atmosphere.
"""

TRACK_TEXT_MD = """
## Trajectory (coming next)

The next logical step of the analysis concerns the **trajectory**, i.e. the spatial
displacement of the cyclone. This relies on tracking the **center of minimum pressure**
and on analyzing the large-scale fields responsible for storm steering, in particular
500 hPa geopotential height and winds.

This dimension is essential for understanding track errors and, ultimately, impact
misplacement. It will be integrated in a later stage of the demonstrator through
dedicated figures and maps focusing on center displacement and synoptic-scale context.
"""


# ============================================================
# APP CONTENT (STRICT ORDER)
# ============================================================
st.markdown(INTRO_MD)
st.markdown(VARIABLES_MD)

st.markdown("---")
st.markdown(WIND10_TEXT_MD)
render_slider_image_section(
    slider_label="10 m wind",
    folder=WIND10_DIR,
    state_key="wind10_idx",
    caption_prefix="10 m wind",
)

st.markdown("---")
st.markdown(PRESSURE_TEXT_MD)
render_slider_image_section(
    slider_label="Mean sea-level pressure (MSLP)",
    folder=PRESSURE_DIR,
    state_key="pressure_idx",
    caption_prefix="Surface pressure (MSLP)",
)

st.markdown("---")
st.markdown(PMIN_TEXT_MD)
render_single_image(
    image_path=PMIN_PLOT_PATH,
    caption="Minimum pressure time series (Pmin)",
)

st.markdown("---")
st.markdown(WIND850_TEXT_MD)
render_slider_image_section(
    slider_label="850 hPa wind",
    folder=WIND850_DIR,
    state_key="wind850_idx",
    caption_prefix="850 hPa wind",
)

st.markdown("---")
st.markdown(TRACK_TEXT_MD)
