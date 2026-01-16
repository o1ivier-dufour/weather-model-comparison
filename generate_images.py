# ============================================================
# IMPORTS
# ============================================================
import os
import time
import numpy as np
import xarray as xr
import plotly.graph_objects as go
import cartopy.feature as cfeature
from shapely.geometry import Polygon, MultiPolygon


# ============================================================
# 1) GLOBAL SETTINGS
# ============================================================

ERA5_ZARR_URL = "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"

# Spatial domain (Atlantic / Bahamas region)
LAT_MIN, LAT_MAX = 5, 40
LON_MIN, LON_MAX = -90, -50

# Time window (can be wide; we still restrict to N_FRAMES)
START_TIME = "2019-08-24T00:00"
END_TIME   = "2019-09-07T18:00"

# ERA5 is hourly; keep one every TIME_STEP_HOURS
TIME_STEP_HOURS = 6

# Generate exactly N_FRAMES frames after subsampling (5 frames => 30 hours coverage)
N_FRAMES = 5

# Filaments look great but are expensive; keep this False for fast iterations
ENABLE_FILAMENTS = True

# If True, regenerate PNGs even if they already exist
FORCE_REGEN = False

# Output folders
WIND10_DIR = "wind"
WIND850_DIR = "wind_850"
PRESSURE_DIR = "pressure"
PMIN_DIR = "pmin_evolution"
PMIN_PLOT_PATH = os.path.join(PMIN_DIR, "pmin_timeseries.png")

# Plot size for PNG export (smaller = faster)
MAP_WIDTH = 850
MAP_HEIGHT = 650

# Wind colormap ("storm-like")
WINDY_COLORSCALE = [
    [0.0,  "rgb(40, 20, 120)"],
    [0.15, "rgb(30, 60, 170)"],
    [0.3,  "rgb(30, 140, 200)"],
    [0.45, "rgb(60, 200, 120)"],
    [0.6,  "rgb(220, 220, 60)"],
    [0.75, "rgb(230, 120, 30)"],
    [0.9,  "rgb(200, 40, 80)"],
    [1.0,  "rgb(230, 0, 160)"],
]

# Pressure colormap (blue = low pressure, brown = high pressure)
PRESSURE_COLORSCALE = [
    [0.00, "rgb(40, 70, 140)"],
    [0.35, "rgb(60, 130, 170)"],
    [0.55, "rgb(150, 190, 170)"],
    [0.75, "rgb(210, 185, 140)"],
    [1.00, "rgb(190, 120, 80)"],
]

# Tracking settings
TRACK_BOX_DEG   = 8.0   # +/- around previous center to search Vmax
PMIN_RADIUS_DEG = 3.0   # radius around center to search Pmin


# ============================================================
# 2) SIMPLE LOGGER
# ============================================================

T0 = time.time()

def log(msg: str) -> None:
    elapsed = time.time() - T0
    print(f"[{elapsed:7.1f}s] {msg}", flush=True)


# ============================================================
# 3) UTILS: FILAMENTS + LAND
# ============================================================

def bilinear_interp(x, y, x_grid, y_grid, field):
    if x < x_grid[0] or x > x_grid[-1] or y < y_grid[0] or y > y_grid[-1]:
        return 0.0
    ix = np.searchsorted(x_grid, x) - 1
    iy = np.searchsorted(y_grid, y) - 1
    if ix < 0 or ix >= len(x_grid) - 1 or iy < 0 or iy >= len(y_grid) - 1:
        return 0.0

    x1, x2 = x_grid[ix], x_grid[ix + 1]
    y1, y2 = y_grid[iy], y_grid[iy + 1]
    tx = (x - x1) / (x2 - x1)
    ty = (y - y1) / (y2 - y1)

    f11 = field[iy, ix]
    f21 = field[iy, ix + 1]
    f12 = field[iy + 1, ix]
    f22 = field[iy + 1, ix + 1]

    return (
        (1 - tx) * (1 - ty) * f11
        + tx * (1 - ty) * f21
        + (1 - tx) * ty * f12
        + tx * ty * f22
    )


def compute_filament_traces(u2d, v2d, lon_full, lat_full,
                            lon_min, lon_max, lat_min, lat_max):
    S_full = np.hypot(u2d, v2d)
    S_max = np.percentile(S_full, 99)
    if S_max <= 0:
        S_max = 1
    S_norm = np.clip(S_full / S_max, 0, 1)

    step_seed = 4
    n_steps = 10
    L_max_deg = 1.0
    L_min_deg = 0.15
    expo = 0.7

    lon_seeds = lon_full[::step_seed]
    lat_seeds = lat_full[::step_seed]

    filaments = []
    for iy, lat0 in enumerate(lat_seeds):
        for ix, lon0 in enumerate(lon_seeds):
            j = iy * step_seed
            i = ix * step_seed
            if j >= S_norm.shape[0] or i >= S_norm.shape[1]:
                continue
            speed0 = S_norm[j, i]
            if speed0 < 0.02:
                continue

            L_tot = L_min_deg + (L_max_deg - L_min_deg) * (speed0 ** expo)
            ds = L_tot / n_steps

            x, y = lon0, lat0
            xs, ys = [x], [y]

            for _ in range(n_steps):
                u_loc = bilinear_interp(x, y, lon_full, lat_full, u2d)
                v_loc = bilinear_interp(x, y, lon_full, lat_full, v2d)
                speed = np.hypot(u_loc, v_loc)
                if speed < 1e-3:
                    break
                theta = np.arctan2(v_loc, u_loc)
                x += ds * np.cos(theta)
                y += ds * np.sin(theta)

                if not (lon_min - 1 <= x <= lon_max + 1 and lat_min - 1 <= y <= lat_max + 1):
                    break
                xs.append(x)
                ys.append(y)

            if len(xs) > 1:
                filaments.append((np.array(xs), np.array(ys)))

    layer_styles = [
        (1.8, "rgba(255,255,255,1.0)"),
        (1.2, "rgba(255,255,255,0.7)"),
        (0.7, "rgba(255,255,255,0.4)"),
    ]
    n_layers = 3

    traces = []
    for layer_idx, (lw, color) in enumerate(layer_styles):
        xs_layer = []
        ys_layer = []
        for xs, ys in filaments:
            n_pts = len(xs)
            if n_pts < 2:
                continue
            for k in range(n_pts - 1):
                t = k / (n_pts - 2) if n_pts > 2 else 0
                band = int(t * n_layers)
                if band == layer_idx:
                    xs_layer.extend([xs[k], xs[k + 1], None])
                    ys_layer.extend([ys[k], ys[k + 1], None])

        if xs_layer:
            traces.append(
                go.Scatter(
                    x=xs_layer, y=ys_layer,
                    mode="lines",
                    line=dict(color=color, width=lw),
                    hoverinfo="skip",
                    showlegend=False,
                )
            )
    return traces


def compute_land_polygons(lon_min, lon_max, lat_min, lat_max):
    land = cfeature.NaturalEarthFeature("physical", "land", "50m")
    polys = []
    for geom in land.geometries():
        if isinstance(geom, Polygon):
            x, y = geom.exterior.xy
            polys.append((np.asarray(x), np.asarray(y)))
        elif isinstance(geom, MultiPolygon):
            for p in geom.geoms:
                x, y = p.exterior.xy
                polys.append((np.asarray(x), np.asarray(y)))
    return polys


# ============================================================
# 4) TRACKING: CENTER BY VMAX, THEN PMIN NEAR CENTER
# ============================================================

def find_center_by_vmax(ws2d, lon, lat, prev_center=None, box_deg=8.0):
    if prev_center is None:
        j, i = np.unravel_index(np.nanargmax(ws2d), ws2d.shape)
        return float(lon[i]), float(lat[j]), float(ws2d[j, i])

    clon0, clat0 = prev_center
    lon_mask = (lon >= clon0 - box_deg) & (lon <= clon0 + box_deg)
    lat_mask = (lat >= clat0 - box_deg) & (lat <= clat0 + box_deg)

    if not lon_mask.any() or not lat_mask.any():
        j, i = np.unravel_index(np.nanargmax(ws2d), ws2d.shape)
        return float(lon[i]), float(lat[j]), float(ws2d[j, i])

    sub = ws2d[np.ix_(lat_mask, lon_mask)]
    jj, ii = np.unravel_index(np.nanargmax(sub), sub.shape)
    lat_idx = np.where(lat_mask)[0][jj]
    lon_idx = np.where(lon_mask)[0][ii]
    return float(lon[lon_idx]), float(lat[lat_idx]), float(ws2d[lat_idx, lon_idx])


def find_pmin_near_center(mslp2d, lon, lat, center, radius_deg=3.0):
    clon, clat = center
    Lon, Lat = np.meshgrid(lon, lat)
    dist = np.sqrt((Lon - clon) ** 2 + (Lat - clat) ** 2)
    mask = dist <= radius_deg

    if not np.any(mask):
        j, i = np.unravel_index(np.nanargmin(mslp2d), mslp2d.shape)
        return float(lon[i]), float(lat[j]), float(mslp2d[j, i])

    sub = np.where(mask, mslp2d, np.nan)
    j, i = np.unravel_index(np.nanargmin(sub), sub.shape)
    return float(lon[i]), float(lat[j]), float(mslp2d[j, i])


# ============================================================
# 5) HELPERS: FILE SKIP + MAP RENDERING
# ============================================================

def should_render(path: str) -> bool:
    return FORCE_REGEN or (not os.path.isfile(path))


def render_wind_map(u2d, v2d, speed2d, t_str, out_path, title, vmin, vmax,
                    lon_full, lat_full, land_polys):
    heatmap = go.Heatmap(
        x=lon_full, y=lat_full, z=speed2d,
        colorscale=WINDY_COLORSCALE,
        zmin=vmin, zmax=vmax,
        zsmooth="best",
        colorbar=dict(title=title),
        hoverinfo="skip",
    )

    coast_traces = [
        go.Scatter(
            x=x, y=y,
            mode="lines",
            line=dict(color="rgba(0,0,0,0.9)", width=1.0),
            hoverinfo="skip",
            showlegend=False,
        )
        for (x, y) in land_polys
    ]

    filaments = []
    if ENABLE_FILAMENTS:
        filaments = compute_filament_traces(
            u2d, v2d, lon_full, lat_full,
            LON_MIN, LON_MAX, LAT_MIN, LAT_MAX,
        )

    fig = go.Figure([heatmap] + coast_traces + filaments)
    fig.update_xaxes(range=[LON_MIN, LON_MAX], showgrid=False)
    fig.update_yaxes(range=[LAT_MIN, LAT_MAX], scaleanchor="x", scaleratio=1, showgrid=False)
    fig.update_layout(
        width=MAP_WIDTH, height=MAP_HEIGHT,
        title=f"{title} — {t_str}",
        margin=dict(l=50, r=50, t=60, b=50),
        plot_bgcolor="rgb(230,230,230)",
        paper_bgcolor="rgb(230,230,230)",
    )

    img_bytes = fig.to_image(format="png")
    with open(out_path, "wb") as f:
        f.write(img_bytes)


# ============================================================
# 6) LOAD ERA5 + SUBSAMPLE + LIMIT TO 5 FRAMES
# ============================================================

log("Opening ERA5 Zarr (remote)...")
ds_full = xr.open_zarr(ERA5_ZARR_URL, storage_options={"token": "anon"})

log("Selecting variables...")
ds = ds_full[
    [
        "10m_u_component_of_wind",
        "10m_v_component_of_wind",
        "mean_sea_level_pressure",
        "u_component_of_wind",
        "v_component_of_wind",
    ]
]

log("Normalizing longitude to [-180, 180]...")
lons180 = ((ds.longitude + 180) % 360) - 180
ds = ds.assign_coords(longitude=lons180).sortby("longitude")

log("Applying spatial and temporal selection...")
ds = ds.sel(
    time=slice(START_TIME, END_TIME),
    latitude=slice(LAT_MAX, LAT_MIN),
    longitude=slice(LON_MIN, LON_MAX),
)

log(f"Subsampling time: 1 frame every {TIME_STEP_HOURS} hours...")
ds = ds.isel(time=slice(0, None, TIME_STEP_HOURS))

n_frames = int(ds.sizes["time"])
log(f"Final number of frames: {n_frames}")

# Extract fields
log("Extracting fields...")
u10  = ds["10m_u_component_of_wind"].sortby("latitude")
v10  = ds["10m_v_component_of_wind"].sortby("latitude")
u850 = ds["u_component_of_wind"].sel(level=850, method="nearest").sortby("latitude")
v850 = ds["v_component_of_wind"].sel(level=850, method="nearest").sortby("latitude")
mslp = ds["mean_sea_level_pressure"].sortby("latitude") / 100.0  # hPa

# Pressure filling (keeps plotting stable)
mslp_filled = (
    mslp
    .interpolate_na("latitude",  method="nearest", fill_value="extrapolate")
    .interpolate_na("longitude", method="nearest", fill_value="extrapolate")
)

time_labels = ds.time.dt.strftime("%Y-%m-%d %H:%M").values.tolist()

lat_full = u10.latitude.values
lon_full = u10.longitude.values

assert np.array_equal(lat_full, u850.latitude.values), "Latitude mismatch between 10m and 850hPa wind"
assert np.array_equal(lon_full, u850.longitude.values), "Longitude mismatch between 10m and 850hPa wind"

log("Building coastline polygons (Natural Earth 50m)...")
land_polys = compute_land_polygons(LON_MIN, LON_MAX, LAT_MIN, LAT_MAX)

# Derived fields (km/h)
wind10_speed  = np.hypot(u10,  v10)  * 3.6
wind850_speed = np.hypot(u850, v850) * 3.6

# Global bounds (computed on the restricted time window)
vmin10 = 0.0
vmax10 = float(wind10_speed.quantile(0.99))

vmin850 = 0.0
vmax850 = float(wind850_speed.quantile(0.99))

pmin_bg = float(mslp_filled.quantile(0.01))
pmax_bg = float(mslp_filled.quantile(0.99))


# ============================================================
# 7) GENERATE PNGs: 10m WIND + 850hPa WIND
# ============================================================

os.makedirs(WIND10_DIR, exist_ok=True)
os.makedirs(WIND850_DIR, exist_ok=True)

log(f"Generating PNGs (filaments={'ON' if ENABLE_FILAMENTS else 'OFF'}, force_regen={'YES' if FORCE_REGEN else 'NO'})...")

for t_idx, t_str in enumerate(time_labels):
    out_path_10 = os.path.join(WIND10_DIR, f"wind_{t_idx:02d}.png")
    out_path_850 = os.path.join(WIND850_DIR, f"wind_850_{t_idx:02d}.png")

    # 10m wind
    if should_render(out_path_10):
        log(f"WIND10  {t_idx+1}/{n_frames}  {t_str}  -> {out_path_10}")
        u = u10.isel(time=t_idx).values
        v = v10.isel(time=t_idx).values
        ws = wind10_speed.isel(time=t_idx).values
        render_wind_map(
            u2d=u, v2d=v, speed2d=ws,
            t_str=t_str, out_path=out_path_10,
            title="10m wind (km/h)", vmin=vmin10, vmax=vmax10,
            lon_full=lon_full, lat_full=lat_full, land_polys=land_polys,
        )
    else:
        log(f"WIND10  {t_idx+1}/{n_frames}  {t_str}  -> (skip)")

    # 850hPa wind
    if should_render(out_path_850):
        log(f"WIND850 {t_idx+1}/{n_frames}  {t_str}  -> {out_path_850}")
        u = u850.isel(time=t_idx).values
        v = v850.isel(time=t_idx).values
        ws = wind850_speed.isel(time=t_idx).values
        render_wind_map(
            u2d=u, v2d=v, speed2d=ws,
            t_str=t_str, out_path=out_path_850,
            title="850 hPa wind (km/h)", vmin=vmin850, vmax=vmax850,
            lon_full=lon_full, lat_full=lat_full, land_polys=land_polys,
        )
    else:
        log(f"WIND850 {t_idx+1}/{n_frames}  {t_str}  -> (skip)")


# ============================================================
# 8) GENERATE PNGs: MSLP + CENTER + PMIN, THEN PMIN(t)
# ============================================================

os.makedirs(PRESSURE_DIR, exist_ok=True)
os.makedirs(PMIN_DIR, exist_ok=True)

prev_center = None
pmin_series = []
pmin_times = []

for t_idx, t_str in enumerate(time_labels):
    out_path_p = os.path.join(PRESSURE_DIR, f"pressure_{t_idx:02d}.png")

    ws_t = wind10_speed.isel(time=t_idx).values
    mslp_t = mslp_filled.isel(time=t_idx).values

    c_lon, c_lat, _ = find_center_by_vmax(
        ws_t, lon_full, lat_full,
        prev_center=prev_center,
        box_deg=TRACK_BOX_DEG,
    )
    prev_center = (c_lon, c_lat)

    _, _, pmin_here = find_pmin_near_center(
        mslp_t, lon_full, lat_full,
        center=(c_lon, c_lat),
        radius_deg=PMIN_RADIUS_DEG,
    )

    pmin_series.append(float(pmin_here))
    pmin_times.append(t_str)

    if not should_render(out_path_p):
        log(f"MSLP    {t_idx+1}/{n_frames}  {t_str}  -> (skip)")
        continue

    log(f"MSLP    {t_idx+1}/{n_frames}  {t_str}  -> {out_path_p}")

    pressure_bg = go.Contour(
        x=lon_full, y=lat_full, z=mslp_t,
        colorscale=PRESSURE_COLORSCALE,
        zmin=pmin_bg, zmax=pmax_bg,
        contours=dict(coloring="heatmap"),
        line=dict(width=0),
        colorbar=dict(title="MSLP (hPa)"),
        hoverinfo="skip",
        showscale=True,
    )

    coast_traces = [
        go.Scatter(
            x=x, y=y,
            mode="lines",
            line=dict(color="rgba(0,0,0,1)", width=0.5),
            hoverinfo="skip",
            showlegend=False,
        )
        for (x, y) in land_polys
    ]

    isobars = go.Contour(
        x=lon_full, y=lat_full, z=mslp_t,
        ncontours=25,
        contours=dict(coloring="none"),
        line=dict(color="rgba(255,255,255,1)", width=1.3),
        showscale=False,
        hoverinfo="skip",
    )

    center_marker = go.Scatter(
        x=[c_lon], y=[c_lat],
        mode="markers",
        marker=dict(size=10, color="white", line=dict(color="black", width=2)),
        hoverinfo="skip",
        showlegend=False,
    )

    label = go.Scatter(
        x=[c_lon], y=[c_lat],
        mode="text",
        text=[f"Pmin ≈ {pmin_here:.1f} hPa"],
        textposition="top right",
        textfont=dict(size=16, color="white"),
        hoverinfo="skip",
        showlegend=False,
    )

    fig = go.Figure([pressure_bg] + coast_traces + [isobars, center_marker, label])
    fig.update_xaxes(range=[LON_MIN, LON_MAX], showgrid=False)
    fig.update_yaxes(range=[LAT_MIN, LAT_MAX], scaleanchor="x", scaleratio=1, showgrid=False)
    fig.update_layout(
        width=MAP_WIDTH, height=MAP_HEIGHT,
        title=f"Mean sea-level pressure (hPa) — {t_str}",
        margin=dict(l=50, r=50, t=60, b=50),
        plot_bgcolor="rgb(235,235,235)",
        paper_bgcolor="rgb(235,235,235)",
    )

    img_bytes = fig.to_image(format="png")
    with open(out_path_p, "wb") as f:
        f.write(img_bytes)

log("Generating Pmin(t) timeseries plot...")
fig_pmin = go.Figure()
fig_pmin.add_trace(
    go.Scatter(
        x=pmin_times,
        y=pmin_series,
        mode="lines+markers",
        line=dict(width=3),
        marker=dict(size=8),
        name="Pmin",
    )
)

fig_pmin.update_layout(
    title="Cyclone minimum pressure evolution (Pmin)",
    xaxis_title="Time (UTC)",
    yaxis_title="Minimum pressure (hPa)",
    yaxis=dict(autorange="reversed"),
    width=900,
    height=450,
    margin=dict(l=60, r=40, t=60, b=60),
    plot_bgcolor="white",
    paper_bgcolor="white",
)

fig_pmin.write_image(PMIN_PLOT_PATH)
log(f"Saved: {PMIN_PLOT_PATH}")

log("All done.")