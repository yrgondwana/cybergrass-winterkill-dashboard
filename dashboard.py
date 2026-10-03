# ══════════════════════════════════════════════════════════════════════════
# CyberGrass 2.0 — Winter Kill Dashboard  (v2 — all 6 issues fixed)
# Run: streamlit run dashboard.py
# ══════════════════════════════════════════════════════════════════════════
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import geopandas as gpd
import numpy as np
from pathlib import Path
from PIL import Image
import json, math

# ── Config ─────────────────────────────────────────────────────────────────
ASSET_DIR = Path(__file__).parent / "13_dashboard_assets"

st.set_page_config(
    page_title="CyberGrass 2.0: Potential Winter Kill Dashboard",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded")

# ── Cached loaders ─────────────────────────────────────────────────────────
@st.cache_data
def load_manifest():
    p = ASSET_DIR / "manifest.json"
    if not p.exists():
        return {"rgb": [], "indices": [], "overlays": []}
    with open(p) as f:
        return json.load(f)

@st.cache_data
def load_meta(name):
    p = ASSET_DIR / "metadata" / name
    return pd.read_csv(p) if p.exists() else pd.DataFrame()

@st.cache_data
def load_wk():
    df = load_meta("wk_all_fields.csv")
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"])
    return df

@st.cache_data
def load_missions():
    """All 62 missions for timeline - from missions_all.csv."""
    df = load_meta("missions_all.csv")
    if df.empty:
        return pd.DataFrame()
    df["date"] = pd.to_datetime(df["date"])
    df["field"] = df["field"].str.upper()
    return df

@st.cache_data
def load_idx_stats():
    df = load_meta("index_stats.csv")
    if not df.empty and "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"].astype(str), format="%Y%m%d")
    return df


MASK_DIR = Path(__file__).parent / "13_dashboard_assets" / "3_Field_Masks"

@st.cache_data
def load_field_polygons():
    """Load exact field boundary polygons from shapefiles, reprojected to WGS84."""
    masks = {
        "EXP1": ["Mask_exp1_HA", "Mask_exp1_LA"],
        "EXP2": ["Mask_exp2_HA", "Mask_exp2_LA"],
        "F12":  ["Mask_F12_HA",  "Mask_F12_LA"],
        "F17":  ["Mask_F17_HA",  "Mask_F17_LA"],
        "F17B": ["Mask_F17b_LA"],   # only LA exists
        "F21":  ["Mask_F21_HA",  "Mask_F21_LA"],
    }
    polys = []  # list of dicts: field, altitude, lats, lons
    for field, fnames in masks.items():
        for fname in fnames:
            shp = MASK_DIR / f"{fname}.shp"
            if not shp.exists():
                continue
            alt = "LA" if fname.endswith("_LA") or "la" in fname.lower() else "HA"
            try:
                gdf = gpd.read_file(shp)
                if gdf.crs is None:
                    gdf = gdf.set_crs("EPSG:3006")
                gdf = gdf.to_crs("EPSG:4326")
                
                for geom in gdf.geometry:
                    if geom is None:
                        continue
                    if geom.geom_type == "Polygon":
                        coords = list(geom.exterior.coords)
                    elif geom.geom_type == "MultiPolygon":
                        coords = list(list(geom.geoms)[0].exterior.coords)
                    else:
                        continue
                    lons_p = [c[0] for c in coords]
                    lats_p = [c[1] for c in coords]
                    polys.append({"field": field, "altitude": alt,
                                  "lats": lats_p, "lons": lons_p})
            except Exception as e:
                st.warning(f"Could not load {fname}: {e}")
    return polys


manifest    = load_manifest()
df_wk       = load_wk()
df_rv       = load_meta("ring_validation_metrics.csv")
df_cent     = load_meta("field_centroids.csv")
df_missions = load_missions()
df_idx      = load_idx_stats()

# ── Constants ──────────────────────────────────────────────────────────────
FIELD_COLORS = {
    "F12":"#1f77b4","EXP2":"#ff7f0e","F17":"#2ca02c",
    "F17B":"#98df8a","F21":"#d62728","EXP1":"#9467bd"}
VALID_FIELDS = {"F12","EXP2","F17","F17B","F21","EXP1"}
INDEX_LONG = {
    "NDVI" :"Normalised Difference VI (NIR)",
    "GNDVI":"Green NDVI (NIR + Green)",
    "NDRE" :"Red-Edge NDVI (RedEdge band)",
    "OSAVI":"Optimised Soil-Adjusted VI (NIR)",
    "VARI" :"Visible Atmospherically Resistant Index (RGB)",
    "ExG"  :"Excess Green Index (RGB)"}
INDEX_SENSOR = {
    "NDVI":"Multispectral","GNDVI":"Multispectral",
    "NDRE":"Multispectral","OSAVI":"Multispectral",
    "VARI":"RGB","ExG":"RGB"}

# GSD at LA in cm/px — used for approximate area calculation
GSD_LA_M  = 0.0093   # metres
PIXEL_AREA = GSD_LA_M ** 2  # m² per LA pixel

# Approximate pixel counts per field (LA, for area calculation)
FIELD_PX = {
    "F12":"235130512","EXP2":"14861508","F17":"208520929",
    "F17B":"188750554","F21":"173272969","EXP1":"13239151"}

def field_radius_deg(field):
    """Approximate field radius in lat/lon degrees from pixel count."""
    px  = int(FIELD_PX.get(field, 1000000))
    area_m2 = px * PIXEL_AREA
    r_m = math.sqrt(area_m2 / math.pi)
    r_lat = r_m / 111000
    return r_lat

def show_img(rel, caption=""):
    # rel may be an absolute path (written on original machine) or relative to ASSET_DIR
    candidate = Path(rel)
    if candidate.is_absolute() and candidate.exists():
        p = candidate
    else:
        # Try as relative path (forward or back slashes)
        p = ASSET_DIR / Path(rel.replace("\\", "/"))
        if not p.exists():
            # Last resort: just use the filename in ASSET_DIR recursively
            fname = Path(rel).name
            matches = list(ASSET_DIR.rglob(fname))
            p = matches[0] if matches else p
    if p.exists():
        st.image(Image.open(p), caption=caption, use_container_width=True)
    else:
        st.warning(f"Image not found: {Path(rel).name}")

# ── Sidebar ────────────────────────────────────────────────────────────────
try:
    _logo = ASSET_DIR / "drone_pic.jpeg"
    if _logo.exists():
        st.sidebar.image(Image.open(_logo), width=220)
    else:
        st.sidebar.markdown("**SLU**")
except Exception:
    st.sidebar.markdown("**SLU**")

st.sidebar.markdown("### CyberGrass 2.0 (Interreg Northern Periphery & Arctic)")
st.sidebar.markdown("Work Package 2: Complementing forage scouting with low-cost drone technology  \n"
                    "  \n"
                    "Activity 2.2: Forage Winter Kill Assessment  \n"
                    "  \n")
st.sidebar.markdown("Erasmus+ Traineeship-cum-Internship  \n"
                    "  \n"
                    "Author: \n" 
                    "Yatharth Ratan Gondwana \n"
                    "Supervisor: \n" 
                    "Dr. Julianne de Castro Oliveira \n") 


st.sidebar.divider()

page = st.sidebar.radio("Navigate", [
    "Overview",
    "Field Viewer",
    "DL Pipeline",
    "Results",
    "Findings & Discussion"])

# ══════════════════════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════════════════
if page == "Overview":
    st.title("Quantifying Potential Winter Kill in Nordic Leys")
    st.markdown(
        "Towards automated winter kill detection in forage grasslands  \n"
        "  \n"
        "Missions' Timeline: 23 April- 11 June 2025"
        "  \n")

    # ── What is this? ────────────────────────────────────────────────────
    with st.container(border=True):
        st.markdown("""
##### Background

Forage leys in the Nordic region often emerge from winter with patches of dead or
damaged grass due to **winter kill,** caused by ice encasement, prolonged snow cover, or fungal
snow moulds. Currently it is assessed generally by walking fields and estimating damage by eye: slow,
subjective, and hard to scale across many hectares.

This activity under **CyberGrass 2.0 project** tests whether a **low-cost consumer drone**
plus a **deep learning model** can do this automatically instead. A UAV flies standard RGB
(colour imagery) missions over a field, and a trained U-Net model classifies every pixel as
*green (healthy)* or *non-green (potential winter kill)*, producing a map and a percentage
figure, repeatable across the whole spring growing season.

**Important caveat:** "non-green" from RGB imagery alone may mean winter-killed grass, but
can also mean bare soil, standing water, or shadow. That is why we call it *potential* winter
kill throughout as it is a validated proxy (see **Results → Ring Validation**).

Use the sidebar to explore: **Field Viewer** (raw imagery), **DL Pipeline** (how the model
works), **Results** (how well it performs, and where recovery is worst).
""")

    # ── Metrics row ────────────────────────────────────────────────────────
    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("Fields monitored", "6",
              help="5 natural fields (F12, EXP2, F17, F17b, F21) + 1 experimental (EXP1: herbicide strip design)")
    c2.metric("Total UAV missions", "62")
    c3.metric("Best R²: ring val.", "0.825",
              help="F17/LA ring plot validation (4-class Equal Interval, n=14, p<0.001)")
    c4.metric("LA model mIoU", "0.887",
              help="LA_final_v2 training mIoU on LA annotation dataset")
    c5.metric("HA model mIoU", "0.883",
              help="HA_final_v2 training mIoU on HA annotation dataset")
    st.divider()

    # ── Study area map ─────────────────────────────────────────────────────
    st.subheader("📍 Study Area: Röbäcksdalen Field Station, Umeå, Sweden")

    field_polys = load_field_polygons()

    if field_polys:
        fig_map = go.Figure()

        # Distinct color per field+altitude combination
        POLY_COLORS = {
            ("EXP1", "HA"): "#9467bd",  # purple
            ("EXP1", "LA"): "#9467bd",  # same (HA=LA area)
            ("EXP2", "HA"): "#ff7f0e",  # orange
            ("EXP2", "LA"): "#ff7f0e",  # same (HA=LA area)
            ("F12",  "HA"): "#d62728",  # dark red
            ("F12",  "LA"): "#ffbb78",  # light orange/gold
            ("F17",  "HA"): "#3748e5",  # navy blue
            ("F17",  "LA"): "#98df8a",  # light green
            ("F17B", "LA"): "#17becf",  # teal/cyan
            ("F21",  "HA"): "#e377c2",  # pink/magenta
            ("F21",  "LA"): "#f7b6d2",  # light pink
        }

        # Label offset direction per field+altitude (lat_offset, lon_offset in degrees)
        # Points the leader line outward from the polygon into clear map space
        LABEL_OFFSET = {
            ("EXP1", "HA"): ( 0.0003,  0.0003),
            ("EXP2", "HA"): ( 0.0003,  0.0003),
            ("F12",  "HA"): ( 0.0006,  0.0006),
            ("F12",  "LA"): ( 0.0003,  0.0005),
            ("F17",  "HA"): ( 0.0008,  0.0008),
            ("F17",  "LA"): ( 0.0004,  0.0004),
            ("F17B", "LA"): ( 0.0004,  0.0004),
            ("F21",  "HA"): ( 0.0005,  0.0005),
            ("F21",  "LA"): ( 0.0005,  0.0004),
        }

        # Draw order: HA first (bottom), LA on top, F17B last
        draw_order = [("HA", 0.25), ("LA", 0.40), ("F17B_LA", 0.55)]
        already_labeled = set()

        for alt_group, alpha in draw_order:
            for poly in field_polys:
                field = poly["field"]
                alt   = poly["altitude"]

                if alt_group == "F17B_LA":
                    if not (field == "F17B" and alt == "LA"):
                        continue
                else:
                    if field == "F17B":
                        continue
                    if alt != alt_group:
                        continue

                color     = POLY_COLORS.get((field, alt), "#888888")
                label_key = f"{field}/{alt}"
                show_leg  = label_key not in already_labeled
                if show_leg:
                    already_labeled.add(label_key)

                r_hex = int(color[1:3], 16)
                g_hex = int(color[3:5], 16)
                b_hex = int(color[5:7], 16)
                fill_color = f"rgba({r_hex},{g_hex},{b_hex},{alpha})"

                # Draw filled polygon — no hover
                fig_map.add_trace(go.Scattermapbox(
                    lat=poly["lats"], lon=poly["lons"],
                    mode="lines",
                    fill="toself",
                    fillcolor=fill_color,
                    line=dict(color=color, width=2),
                    name=label_key,
                    hoverinfo="none",
                    showlegend=show_leg))

                # ── Leader line + label ────────────────────────────────────
                # Skip EXP1/EXP2 LA (same polygon as HA, labeled once)
                if field in ("EXP1", "EXP2") and alt == "LA":
                    continue

                # Anchor: vertex with max lat (topmost point of polygon)
                max_lat_idx = poly["lats"].index(max(poly["lats"]))
                anc_lat = poly["lats"][max_lat_idx]
                anc_lon = poly["lons"][max_lat_idx]

                # Label position: anchor + offset
                d_lat, d_lon = LABEL_OFFSET.get((field, alt), (0.002, 0.002))
                lbl_lat = anc_lat + d_lat
                lbl_lon = anc_lon + d_lon

                label_text = f"{field}_{alt}"

                # Leader line (anchor → label)
                fig_map.add_trace(go.Scattermapbox(
                    lat=[anc_lat, lbl_lat],
                    lon=[anc_lon, lbl_lon],
                    mode="lines",
                    line=dict(color=color, width=1.5),
                    hoverinfo="none",
                    showlegend=False))

                # Label dot + text at end of leader line
                fig_map.add_trace(go.Scattermapbox(
                    lat=[lbl_lat], lon=[lbl_lon],
                    mode="markers+text",
                    marker=dict(size=6, color=color),
                    text=[label_text],
                    textposition="top right",
                    textfont=dict(size=10, color="white"),
                    hoverinfo="none",
                    showlegend=False))

        fig_map.update_layout(
            mapbox=dict(
                style="white-bg",
                center=dict(lat=63.808, lon=20.228),
                zoom=14,
                layers=[{
                    "below": "traces",
                    "sourcetype": "raster",
                    "source": ["https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}"],
                }]
            ),
            height=500,
            margin={"r":0,"t":20,"l":0,"b":0},
            legend=dict(orientation="h", y=-0.06),
            dragmode="zoom")
        st.plotly_chart(fig_map, use_container_width=True,
                config={"scrollZoom": True})
        st.caption(
            "Exact field boundaries from field mask shapefiles (SWEREF99/TM, EPSG:3006 → WGS84). "
            "HA = larger outer boundary · LA = inner sub-area · F17b overlaid within F17.")

    elif not df_cent.empty:
        # Fallback: approximate circles if shapefiles not found
        fig_map = go.Figure()
        for _, row in df_cent.iterrows():
            field  = row["field"].upper()
            lat_c  = row["lat"]
            lon_c  = row["lon"]
            ftype  = row["type"]
            color  = FIELD_COLORS.get(field, "#888888")
            r_lat  = field_radius_deg(field)
            r_lon  = r_lat / math.cos(math.radians(lat_c)) if lat_c else r_lat
            n_pts  = 32
            angles = [2*math.pi*i/n_pts for i in range(n_pts+1)]
            lats   = [lat_c + r_lat*math.sin(a) for a in angles]
            lons   = [lon_c + r_lon*math.cos(a) for a in angles]
            r_hex  = int(color[1:3], 16)
            g_hex  = int(color[3:5], 16)
            b_hex  = int(color[5:7], 16)
            fig_map.add_trace(go.Scattermapbox(
                lat=lats, lon=lons, mode="lines",
                fill="toself",
                fillcolor=f"rgba({r_hex},{g_hex},{b_hex},0.25)",
                line=dict(color=color, width=2),
                name=field, hoverinfo="name", showlegend=True))
            area_ha = round(PIXEL_AREA * int(FIELD_PX.get(field,1000000)) / 10000, 2)
            fig_map.add_trace(go.Scattermapbox(
                lat=[lat_c], lon=[lon_c], mode="markers+text",
                marker=dict(size=8, color=color),
                text=[field], textposition="top center",
                textfont=dict(size=12, color="white"),
                hovertemplate=(
                    f"<b>{field}</b><br>Type: {ftype}<br>"
                    f"Area ≈ {area_ha} ha (approx)<extra></extra>"),
                showlegend=False))
        fig_map.update_layout(
            mapbox=dict(
                style="white-bg",
                center=dict(lat=63.808, lon=20.228),
                zoom=14,
                layers=[{
                    "below": "traces",
                    "sourcetype": "raster",
                    "source": ["https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}"],
                }]
            ),
            height=480,
            margin={"r":0,"t":20,"l":0,"b":0},
            legend=dict(orientation="h", y=-0.05))
        st.plotly_chart(fig_map, use_container_width=True)
        st.caption(
            "⚠ Field mask shapefiles not found- showing approximate field footprints "
            "(computed from LA pixel counts × GSD).")

# ── Mission timeline (all 62 flights) ──────────────────────────────────
    st.subheader("📅 Mission Timeline: 62 UAV Flights between Late April and Early June 2025")
    st.markdown("Double click on legend to isolate missions based on type")

    if df_missions.empty:
        if not df_wk.empty:
            df_tl = df_wk.drop_duplicates(
                subset=["field","altitude","date"]).copy()
            df_tl["has_inference"] = True
            st.caption(
                "⚠ missions_all.csv not found — showing 56 inference missions.")
        else:
            df_tl = pd.DataFrame()
    else:
        df_tl = df_missions.copy()

        # ── Build inference key set from df_wk ──────────────────────────────
        # Key: (field_upper, altitude, date_str YYYYMMDD)
        if not df_wk.empty:
            wk_keys = set(
                df_wk["field"].str.upper() + "_" +
                df_wk["altitude"] + "_" +
                df_wk["date"].dt.strftime("%Y%m%d"))
        else:
            wk_keys = set()

        # Build the same key format from missions CSV
        # date column in missions_all.csv is stored as YYYY-MM-DD string
        df_tl["date_parsed"] = pd.to_datetime(
            df_tl["date"].astype(str), format="mixed",
            dayfirst=False)
        df_tl["mission_key"] = (
            df_tl["field"].str.upper() + "_" +
            df_tl["altitude"] + "_" +
            df_tl["date_parsed"].dt.strftime("%Y%m%d"))
        df_tl["has_inference"] = df_tl["mission_key"].isin(wk_keys)
        df_tl["date"] = df_tl["date_parsed"]   # use datetime for x-axis

    if not df_tl.empty:
        fig_tl = px.scatter(
            df_tl,
            x="date", y="field",
            color="altitude",
            symbol="has_inference",
            symbol_map={True: "circle", False: "x"},
            color_discrete_map={"LA": "#1f77b4", "HA": "#d62728"},
            hover_name="field",
            hover_data={
                "altitude": True,
                "has_inference": True,
                "date": "|%d %b %Y"},
            height=400,
            labels={"date": "Date", "field": "Field",
                    "altitude": "Altitude",
                    "has_inference": "Inference run"})
        fig_tl.update_traces(marker=dict(size=13))
        fig_tl.update_layout(
            legend=dict(orientation="h", y=1.12),
            xaxis=dict(tickformat="%d %b", dtick="M0.5"))
        st.plotly_chart(fig_tl, use_container_width=True)
        st.caption(
            "● Circle = inference prediction generated  "
            "✕ = flight recorded but inference not processed  "
            "Blue = LA (0.93 cm/px) · Red = HA (3.67 cm/px)")    
    

    # ── Pipeline summary ───────────────────────────────────────────────────
    st.subheader("🔄 Pipeline Overview")
    st.caption(
        "From raw drone flight to a validated damage map, in seven steps- "
        "detailed further under **DL Pipeline** and **Results**.")
    st.markdown("""
| Step | Description | Output |
|---|---|---|
| 1 · Data Prep | UAV RGB orthomosaic tiling (128×128 px) | 1,863 tiles from F12 + EXP2 |
| 2 · Annotation | Manual SAM-2 annotation- DigitalSreeni | 336 annotated tiles (LA: 144, HA: 192) |
| 3 · Training | VGG16-BN encoder U-Net · CE+Dice loss · 100 epochs | LA mIoU=0.887 · HA mIoU=0.883 |
| 4 · Inference | Sliding window over full orthomosaics | 56 prediction GeoTIFFs across 5 fields |
| 5 · Validation | Ring plot comparison: 5 fields & 4-class scheme | R²=0.331-0.825 · all p<0.05 except F21/LA |
| 6 · LA vs HA | Altitude effect · Mixed pixel quantification | HA near-zero bias confirmed 4/5 fields |
| 7 · Maps | Persistent damage → reseeding priority GeoTIFFs | 7 classified GeoTIFFs (SWEREF99/TM) |
""")
    st.caption(
        "**LA** = Low Altitude flight (~0.93 cm/px, sharper detail, smaller area/flight) · "
        "**HA** = High Altitude flight (~3.67 cm/px, coarser detail, larger area/flight). "
        "Both are tested to see whether a faster, coarser high-altitude flight is "
        "'good enough' for practical monitoring, or whether the extra detail from "
        "low-altitude flights is necessary.")





# ══════════════════════════════════════════════════════════════════════════
# PAGE 2 — FIELD VIEWER
# ══════════════════════════════════════════════════════════════════════════
elif page == "Field Viewer":
    st.title("Field Viewer: RGB Imagery & Spectral Indices")
    st.markdown(
        "Select field, altitude and date to view the RGB mosaic and "
        "any spectral index map via Viewer Filters on the left sidebar.  \n"
        "NDVI, GNDVI, NDRE, OSAVI = **multispectral sensor** bands.  "
        "VARI & ExG = **RGB-derived** indices.")

    with st.expander("What does this page convey?"):
        st.markdown("""
- **RGB Mosaic**: The raw drone photo of the field, stitched into one image and cropped to
the field boundary. This is what the model sees as input.
- **Spectral Index map**: A colour-coded "vegetation health" layer computed from the
imagery. Indices marked *RGB* (VARI, ExG) come from the same consumer drone photo used
for the U-Net model; indices marked *Multispectral* (NDVI, GNDVI, NDRE, OSAVI) come from
a **separate, supplementary multispectral sensor** used for cross-checking, not part of the
main low-cost pipeline (see WP3 exploratory findings under **DL Pipeline**).
- **Green % / Non-Green %**: The U-Net model's classification for the selected date, 
is the core project output. Non-green is reported as *potential* winter kill, since bare
soil, water and shadow can also appear non-green in RGB.
- **Temporal Trend** (below): Tracks the selected index over the whole season, so you can
see whether a field is greening up (recovering) or staying damaged.
""")

    # Build lookups from manifest
    rgb_lookup = {(r["field"],r["alt"],r["date"]): r["path"]
                  for r in manifest.get("rgb",[])}
    idx_lookup = {(r["field"],r["alt"],r["date"],r["index"]): r["path"]
                  for r in manifest.get("indices",[])}

    all_fields = sorted({r["field"] for r in manifest.get("rgb",[])})
    if not all_fields:
        st.warning("No assets found. Run StepDB1_PrepareAssets.ipynb first.")
        st.stop()

    st.sidebar.subheader("🔍 Viewer Filters")
    sel_field = st.sidebar.selectbox("Field", all_fields)
    avail_alts = sorted({r["alt"] for r in manifest["rgb"]
                         if r["field"]==sel_field})
    sel_alt   = st.sidebar.selectbox("Altitude", avail_alts)
    avail_dates = sorted({r["date"] for r in manifest["rgb"]
                          if r["field"]==sel_field and r["alt"]==sel_alt})
    sel_date  = st.sidebar.selectbox(
        "Date", avail_dates,
        format_func=lambda d: f"{d[:4]}-{d[4:6]}-{d[6:]}")
    sel_index = st.sidebar.selectbox(
        "Spectral Index",
        list(INDEX_LONG.keys()),
        format_func=lambda i: f"{i}  [{INDEX_SENSOR.get(i,'')}]")

    # ── RGB + Index side by side ───────────────────────────────────────────
    st.subheader(
        f"Field {sel_field}  ·  {sel_alt}  ·  "
        f"{sel_date[:4]}-{sel_date[4:6]}-{sel_date[6:]}")

    if sel_field == "EXP1":
        st.warning(
            "⚠️ **EXP1 is an experimental herbicide-strip field, not a natural winter kill "
            "site.** Its non-green zones were deliberately created via herbicide application "
            "to test the pipeline, not caused by winter damage. As a result, EXP1 does **not** "
            "show the typical monotonic greening/recovery pattern seen in the natural fields "
            "(F12, EXP2, F17, F17B, F21) as non-green areas here may persist by design rather "
            "than reflect real damage or recovery. Treat EXP1 imagery and indices as a "
            "pipeline-validation case, not a potential winter kill case.")

    col_rgb, col_idx = st.columns(2)
    rgb_path = rgb_lookup.get((sel_field, sel_alt, sel_date))
    with col_rgb:
        st.markdown("**RGB Mosaic (field-masked)**")
        if rgb_path:
            show_img(rgb_path)
        else:
            st.info("RGB thumbnail not available for this selection.")

    idx_path = idx_lookup.get((sel_field, sel_alt, sel_date, sel_index))
    with col_idx:
        st.markdown(
            f"**{sel_index}** - {INDEX_LONG.get(sel_index,'')}  "
            f"*(sensor: {INDEX_SENSOR.get(sel_index,'')})*")
        if idx_path:
            show_img(idx_path)
        else:
            st.info(f"{sel_index} not available for this date/altitude. "
                    f"(3 missions have no index files.)")

    # ── WK metrics for selected date ───────────────────────────────────────
    if not df_wk.empty:
        mask = ((df_wk["field"]==sel_field) &
                (df_wk["altitude"]==sel_alt) &
                (df_wk["date"].dt.strftime("%Y%m%d")==sel_date))
        row = df_wk[mask]
        if not row.empty:
            r = row.iloc[0]
            st.divider()
            m1,m2,m3 = st.columns(3)
            m1.metric("Green %",      f"{r['green_pct']:.1f}%")
            m2.metric("Non-Green/ Potential Winter Kill %",f"{r['wk_pct']:.1f}%")
            m3.metric("Field type",   str(r.get("field_type","—")))
            st.caption(
                "ℹ️ Potential Winter Kill (WK) metrics shown here as contextual reference for the selected date. "
                "Full temporal analysis available in the **Results** page.") 

    # ── Index trend section ────────────────────────────────────────────────
    st.divider()
    st.subheader(f"📈 {sel_index} Temporal Trend: All Fields")

    if df_idx.empty:
        st.info(
            "Index statistics not computed yet.  \n"
            "Run **Cell 6** of StepDB1_PrepareAssets.ipynb to generate "
            "`metadata/index_stats.csv`, then restart the dashboard.")
    else:
        df_ix_sel = df_idx[df_idx["index_name"]==sel_index].copy()
        # Exclude F21/LA for multispectral indices — known Pix4D calibration artefact
        MULTISPECTRAL_INDICES = {"NDVI", "GNDVI", "NDRE", "OSAVI"}
        if sel_index in MULTISPECTRAL_INDICES:
            df_ix_sel = df_ix_sel[
                ~((df_ix_sel["field"].str.upper() == "F21") &
                (df_ix_sel["altitude"].str.upper() == "LA"))
            ]
        if df_ix_sel.empty:
            st.info(f"No data for {sel_index}.")
        else:
            c1, c2 = st.columns([3,1])
            with c2:
                compare_alt = st.selectbox(
                    "Altitude for trend",
                    ["LA","HA","Both"], index=0,
                    key="idx_alt")
                show_band = st.checkbox("Show IQR band", value=True)

            df_plot = df_ix_sel.copy()
            if compare_alt != "Both":
                df_plot = df_plot[df_plot["altitude"]==compare_alt]

            fig_ix = go.Figure()
            for field in sorted(df_plot["field"].unique()):
                color = FIELD_COLORS.get(field.upper(),"#888888")
                df_f  = df_plot[df_plot["field"]==field]\
                            .sort_values("date")
                if len(df_f)==0:
                    continue
                fig_ix.add_trace(go.Scatter(
                    x=df_f["date"], y=df_f["mean"],
                    mode="lines+markers",
                    name=field,
                    line=dict(color=color, width=2),
                    marker=dict(size=8),
                    hovertemplate=
                        f"<b>{field}</b><br>%{{x|%d %b}}<br>"
                        f"{sel_index} mean=%{{y:.3f}}<extra></extra>"))
                if show_band and "p25" in df_f.columns:
                    hex_c = color.lstrip("#")
                    r = int(hex_c[0:2], 16)
                    g = int(hex_c[2:4], 16)
                    b = int(hex_c[4:6], 16)
                    fill_rgba = f"rgba({r},{g},{b},0.15)"
                    fig_ix.add_trace(go.Scatter(
                        x=pd.concat([df_f["date"],
                                     df_f["date"].iloc[::-1]]),
                        y=pd.concat([df_f["p75"],
                                     df_f["p25"].iloc[::-1]]),
                        fill="toself",
                        fillcolor=fill_rgba,
                        line=dict(color="rgba(0,0,0,0)"),
                        showlegend=False,
                        hoverinfo="skip"))

            fig_ix.update_layout(
                height=400,
                xaxis_title="Date",
                yaxis_title=f"{sel_index} mean value",
                legend=dict(orientation="h", y=-0.18),
                hovermode="x unified")
            with c1:
                st.plotly_chart(fig_ix, use_container_width=True)
                st.caption(
                    f"Field-masked mean **{sel_index}** over time.  "
                    f"Source: `metadata/index_stats.csv`  "
                    + ("  ⚠ F21/LA excluded from multispectral indices- Pix4D calibration artefact on 2025-05-18."
                        if sel_index in MULTISPECTRAL_INDICES else ""))

# ══════════════════════════════════════════════════════════════════════════
# PAGE 3 — DL PIPELINE
# ══════════════════════════════════════════════════════════════════════════
elif page == "DL Pipeline":
    st.title("Deep Learning Pipeline")
    st.markdown(
        "How a drone photo becomes a green / potential-winter-kill map. "
        "This page documents the model itself and for results and accuracy, see the "
        "**Results** page.")

    tab1, tab2, tab3 = st.tabs([
        "🏗️ Architecture & Training",
        "📈 Model Performance",
        "🗺️ Prediction Viewer"])

    with tab1:
        st.markdown("""
In plain terms, U-Net is a type of neural network built for image segmentation. It
labels *every pixel* of an image, rather than just tagging the whole photo. Here it looks at
a 128×128-pixel patch (tile) of drone imagery and decides, pixel by pixel, whether that
patch of grass is green (healthy) or non-green (potential winter kill). It was trained on
**336 tiles that was manually annotated** (144 LA + 192 HA) using SAM-2-assisted
annotation software, drawn from the two training fields (F12 and EXP2). The trained model
then generalises to classify entire orthomosaics it has never seen labelled examples from-
applied via sliding-window inference across **all 5 monitored fields** (F12, EXP2, F17,
F17B, F21).

Two separate models were trained, one for each flight altitude (**LA** and **HA**)
since image detail differs enough between them that a single model underperforms on both.
""")
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Model Architecture: VGG16-BN U-Net")
            st.markdown("""
| Component | Detail |
|---|---|
| Encoder | VGG16-BN (pretrained ImageNet) |
| Decoder | 4 transposed-conv upsampling stages |
| Skip connections | Encoder → decoder at each level |
| Output | 2-class softmax (green / non-green) |
| Input | 128 × 128 px RGB tile, normalised [0,1] |
| Parameters | ~20.4 million |
""")
        with c2:
            st.subheader("Training Configuration")
            st.markdown("""
| Setting | Value |
|---|---|
| Loss | Combined CE + Dice (50 / 50) |
| Optimiser | Adam,  lr = 1 × 10⁻⁴ |
| Epochs | 100, best mIoU checkpoint saved |
| Batch size | 10 |
| Geometric aug. | 4 rotations × H-flip = 8 combos |
| Spectral aug. | Brightness ±30 %, contrast 0.8–1.1× |
| Hardware | Google Colab T4 GPU |

**Two models: one per altitude:**
- `best_model_LA_final_v2.pth` - LA tiles (144 annotated)
- `best_model_HA_final_v2.pth` - HA tiles (192 annotated)
""")
        st.subheader("Annotation Strategy")
        st.markdown("""
| Detail | Value |
|---|---|
| Tool | DigitalSreeni Image Annotator + SAM-2 Base |
| Training fields | F12 and EXP2 only |
| LA annotated tiles | 144 |
| HA annotated tiles | 192 |
| Classes | `green` (1)  ·  `non_green` (0)  ·  unlabelled (255) |
| Key finding | 77 consistent manual tiles outperformed 800 auto-thresholded tiles (mIoU 0.897 vs 0.889) |
""")

    with tab2:
        st.subheader("Training Performance: LA vs HA Final Models")
        st.caption(
            "**mIoU** (mean Intersection-over-Union) measures how closely the model's "
            "pixel labels match the human-annotated ground truth, from 0 (no overlap) to "
            "1 (perfect match). Values above ~0.85 are considered strong for this kind of "
            "segmentation task.")
        df_perf = pd.DataFrame({
            "Model"         : ["LA_final_v2","HA_final_v2"],
            "GSD (cm/px)"   : [0.93, 3.67],
            "Training tiles": [144, 192],
            "mIoU"          : [0.8867, 0.8834],
            "IoU green"     : [0.891, 0.886],
            "IoU non-green" : [0.882, 0.880],
        })
        st.dataframe(
            df_perf.style.highlight_max(
                subset=["mIoU","IoU green","IoU non-green"],
                color="#0be436"),
            hide_index=True, use_container_width=True)
        st.caption("mIoU = mean Intersection-over-Union on held-out validation tiles.")

        st.divider()
        st.subheader("Ring Plot Validation: All 5 Fields · 4-Class Equal Interval")
        if not df_rv.empty:
            styled = (df_rv.style
                .background_gradient(
                    subset=["R2"], cmap="YlGn", vmin=0.3, vmax=0.9)
                .background_gradient(
                    subset=["RMSE"], cmap="RdYlGn_r", vmin=18, vmax=35)
                .background_gradient(
                    subset=["Adj"], cmap="YlGn", vmin=65, vmax=100)
                .map(lambda v: "color:red;font-style:italic"
                     if str(v)=="p=0.082" else "", subset=["Sig"])
                .format({"R2":"{:.3f}","RMSE":"{:.1f}%",
                         "Bias":"{:+.1f}%","Adj":"{:.1f}%"}))
            st.dataframe(styled, hide_index=True, use_container_width=True)
            st.caption(
                "Adj = adjacent class agreement (prediction within one class of observed). "
                "⚠ F21/LA p = 0.082- not statistically significant (n=10, 2 dates only).")

    with tab3:
        st.subheader("Prediction Overlays: RGB vs U-Net Output")

        # Filter manifest overlays to valid fields only (removes old ModelA entries)
        ovl_records = [r for r in manifest.get("overlays",[])
                       if r.get("field","") in VALID_FIELDS
                       and r.get("alt","") in ("LA","HA")]

        if not ovl_records:
            st.info("No prediction overlays found.  \n"
                    "Run Cell 4 (and Cell 6) of StepDB1_PrepareAssets.ipynb, "
                    "then restart the dashboard.")
        else:
            st.sidebar.subheader("🔍 Prediction Filters")
            ovl_fields = sorted({r["field"] for r in ovl_records})
            sel_of = st.sidebar.selectbox("Field ", ovl_fields, key="ovl_f")
            ovl_alts = sorted({r["alt"] for r in ovl_records
                               if r["field"]==sel_of})
            sel_oa = st.sidebar.selectbox("Altitude ", ovl_alts, key="ovl_a")
            ovl_dates = sorted({r["date"] for r in ovl_records
                                if r["field"]==sel_of and r["alt"]==sel_oa})
            sel_od = st.sidebar.selectbox(
                "Date ", ovl_dates,
                format_func=lambda d: f"{d[:4]}-{d[4:6]}-{d[6:]}",
                key="ovl_d")

            match = [r for r in ovl_records
                     if r["field"]==sel_of and r["alt"]==sel_oa
                     and r["date"]==sel_od]
            if match:
                show_img(match[0]["path"])
                st.caption(
                    "Left: field-masked RGB mosaic  ·  "
                    "Right: U-Net prediction  "
                    "(green = living vegetation · red = potential winter kill / non-green)")
            else:
                st.info("No overlay for this selection.")

            # Show WK% for context
            if not df_wk.empty:
                m = df_wk[(df_wk["field"]==sel_of) &
                           (df_wk["altitude"]==sel_oa) &
                           (df_wk["date"].dt.strftime("%Y%m%d")==sel_od)]
                if not m.empty:
                    r = m.iloc[0]
                    ca, cb = st.columns(2)
                    ca.metric("Green %",      f"{r['green_pct']:.1f}%")
                    cb.metric("Potential Winter Kill %",f"{r['wk_pct']:.1f}%")

# ══════════════════════════════════════════════════════════════════════════
# PAGE 4 — RESULTS
# ══════════════════════════════════════════════════════════════════════════
elif page == "Results":
    st.title("Results")

    tab1, tab2, tab3 = st.tabs([
        "📉 Potential Winter Kill Trajectories",
        "✅ Ring Validation",
        "🗺️ Reseeding Maps"])

    # ── Tab 1: Temporal WK ─────────────────────────────────────────────────
    with tab1:
        st.subheader("Winter Kill Fraction Over Time: LA vs HA")
        if df_wk.empty:
            st.warning("wk_all_fields.csv not found in metadata folder.")
        else:
            c1, c2 = st.columns([2,1])
            with c1:
                all_f = sorted(df_wk["field"].unique())
                sel_fields = st.multiselect(
                    "Select fields", all_f,
                    default=[f for f in ["F12","EXP2","F17","F21"] if f in all_f])
            with c2:
                sel_alts = st.multiselect(
                    "Altitudes",["LA","HA"], default=["LA","HA"])

            df_plot = df_wk[
                df_wk["field"].isin(sel_fields) &
                df_wk["altitude"].isin(sel_alts)].copy()

            if df_plot.empty:
                st.info("No data for current selection.")
            else:
                df_plot["label"] = (df_plot["field"] + " / " +
                                    df_plot["altitude"])
                fig = px.line(
                    df_plot, x="date", y="wk_pct",
                    color="label", line_dash="altitude",
                    markers=True, height=500,
                    color_discrete_sequence=list(FIELD_COLORS.values()),
                    labels={"wk_pct":"Winter Kill (%)",
                            "date":"Date","label":"Field / Altitude"})
                fig.update_traces(marker=dict(size=8))
                fig.update_layout(
                    yaxis_range=[-2,102],
                    legend=dict(orientation="h", y=-0.2))
                st.plotly_chart(fig, use_container_width=True)
                st.caption(
                    "Solid = LA (0.93 cm/px) · Dashed = HA (3.67 cm/px)  |  "
                    "EXP1 shows no monotonic recovery: herbicide strip design, "
                    "not natural winter kill.")

    # ── Tab 2: Ring Validation ─────────────────────────────────────────────
    with tab2:
        st.subheader("Ring Plot Validation: All 5 Fields")
        with st.expander("What is a ring plot and why validate this way?"):
            st.markdown("""
A **ring plot** is a physical ground-truth check: a hoop of 70 cm is placed at multiple
location in the field and GPS location recorded, and a person visually estimates what % of the grass inside the hoop is
non-green/damaged, bucketed into one of 4 classes (0-25%, 26-50%, 51-75%, 76-100%).
That person's visual estimate is then compared against what the drone + U-Net model predicted for
the same GPS location on the same date.

**R²** tells us how well the model's predictions track the human ring estimates across many
ring locations (closer to 1.0 = better agreement). **Adj** (adjacent-class agreement) is a more
forgiving check: did the model land within one damage class of the human estimate, even if
not an exact match. This is the main evidence that "potential winter kill" from the model
corresponds to real, human-observable field damage and not just an image artefact.
""")
        st.markdown(
            "**4-class Equal Interval** scheme: 0-25% · 26-50% · 51-75% · 76-100%  \n"
            "Ring radius: 0.33 m (70 cm hoop - 2 cm buffer) · CRS: SWEREF99/TM")

        if not df_rv.empty:
            styled = (df_rv.style
                .background_gradient(
                    subset=["R2"], cmap="YlGn", vmin=0.3, vmax=0.9)
                .background_gradient(
                    subset=["RMSE"], cmap="RdYlGn_r", vmin=18, vmax=35)
                .background_gradient(
                    subset=["Adj"], cmap="YlGn", vmin=65, vmax=100)
                .map(lambda v: "color:red;font-style:italic"
                     if str(v)=="p=0.082" else "", subset=["Sig"])
                .format({"R2":"{:.3f}","RMSE":"{:.1f}%",
                         "Bias":"{:+.1f}%","Adj":"{:.1f}%"}))
            st.dataframe(styled, hide_index=True, use_container_width=True)

        # ── Curated validation plots only ─────────────────────────────────
        st.divider()
        st.subheader("Validation Plots")

        PLOT_CATALOG = {
            # Training fields (F12 + EXP2)
            "Overall assessment: F12 & EXP2 (scatter · residuals · confusion)":
                "validation_plots/02_overall_assessment.png",
            "Phase-based scatter: Phase 1 (Apr-May 8) vs Phase 2 (May 13-Jun 11)":
                "validation_plots/03a_phase_scatter.png",
            "Isolated field × altitude: scatter & residuals":
                "validation_plots/04a_isolated_scatter_residuals.png",
            "Summary dashboard: R², RMSE, Bias, Adj across all analyses":
                "validation_plots/05_summary_dashboard.png",
            # New fields
            "New fields overall: EXP1 · F17 · F21 (scatter & confusion)":
                "validation_plots/01_overall_assessment_newfields.png",
            "New fields per-date scatter":
                "validation_plots/02_per_date_scatter_newfields.png",
        }

        # Resolve actual filenames (new_fields plots are copied with different names)
        # Try direct path first, then alternative locations
        def find_plot(rel_path: str):
            direct = ASSET_DIR / rel_path.replace("/","\\")
            if direct.exists():
                return direct
            # Try results_new_fields subfolder variants
            fn = rel_path.split("/")[-1]
            for alt in [
                ASSET_DIR / "validation_plots" / fn,
                ASSET_DIR / "validation_plots" / fn.replace("_newfields",""),
            ]:
                if alt.exists():
                    return alt
            return None

        sel_plot = st.selectbox("Select plot", list(PLOT_CATALOG.keys()))
        found = find_plot(PLOT_CATALOG[sel_plot])
        if found:
            st.image(Image.open(found), use_container_width=True)
        else:
            # Fallback: let user browse all available plots in the folder
            val_dir = ASSET_DIR / "validation_plots"
            avail = sorted(val_dir.glob("*.png")) if val_dir.exists() else []
            if avail:
                st.info(f"Named plot not found. Choose from available files:")
                fb = st.selectbox("Available plots", [p.name for p in avail],
                                  key="fb_plot")
                st.image(Image.open(val_dir/fb), use_container_width=True)
            else:
                st.info("No validation plot PNGs found in validation_plots/.")

    # ── Tab 3: Reseeding Maps ──────────────────────────────────────────────
    with tab3:
        st.subheader("Reseeding Priority Maps")
        st.markdown(
            "This is the practical, farmer-facing output of the whole pipeline: instead of a "
            "single snapshot, each zone of a field is classified by how *persistently* it "
            "stayed non-green across the whole spring, turning repeat drone flights into "
            "an actionable reseeding decision map.")
        st.markdown(
            "🔴 **High Priority**: non-green on ≥ 2/3 of assessment dates  \n"
            "🟠 **Monitor**: non-green on 1/3 - 2/3 of dates  \n"
            "🟢 **Recovered**: non-green on < 1/3 of dates")
        st.caption(
            "A patch that is only briefly non-green early in spring likely just recovered "
            "naturally; a patch that stays non-green across most flights is a stronger "
            "candidate for actual winter kill and reseeding.")

        res_dir = ASSET_DIR / "reseeding_plots"
        MAP_LABELS = {
            "01_maps_F12_EXP2.png": "F12 and EXP2 (training fields)",
            "02_maps_F17_F21.png" : "F17 and F21 (natural recovery: new fields)",
        }
        if res_dir.exists():
            res_imgs = sorted(res_dir.glob("0*.png"))
            if res_imgs:
                sel_rm = st.selectbox(
                    "Select map figure",
                    [p.name for p in res_imgs],
                    format_func=lambda n: MAP_LABELS.get(n, n))
                st.image(Image.open(res_dir/sel_rm), use_container_width=True)
                st.caption(
                    "Green = Recovered · Orange = Monitor · Red = High Priority.  "
                    "Full-resolution GeoTIFFs (SWEREF99/TM) in reseeding_maps\\ folder.")
            else:
                st.info("No reseeding map PNGs found.")
        else:
            st.info("reseeding_plots folder not found in dashboard_assets.")

        st.divider()
        st.subheader("Zone Statistics")
        res_csv = (res_dir / "reseeding_summary.csv"
                   if res_dir and res_dir.exists() else None)
        if res_csv and res_csv.exists():
            df_rs = pd.read_csv(res_csv)
            ca, cb = st.columns([1,2])
            with ca:
                keep_cols = [c for c in
                    ["Field","Altitude","High Priority%",
                     "Monitor%","Recovered%","Ring R2"]
                    if c in df_rs.columns]
                fmt = {}
                if "High Priority%" in keep_cols:
                    fmt["High Priority%"] = "{:.1f}"
                if "Monitor%" in keep_cols:
                    fmt["Monitor%"] = "{:.1f}"
                if "Recovered%" in keep_cols:
                    fmt["Recovered%"] = "{:.1f}"
                if "Ring R2" in keep_cols:
                    fmt["Ring R2"] = "{:.3f}"
                st.dataframe(
                    df_rs[keep_cols].style
                        .background_gradient(
                            subset=[c for c in ["High Priority%"] if c in keep_cols],
                            cmap="Reds", vmin=0, vmax=80)
                        .background_gradient(
                            subset=[c for c in ["Recovered%"] if c in keep_cols],
                            cmap="Greens", vmin=0, vmax=100)
                        .format(fmt),
                    hide_index=True)
            with cb:
                melt_cols = [c for c in
                    ["High Priority%","Monitor%","Recovered%"]
                    if c in df_rs.columns]
                id_cols   = [c for c in ["Field","Altitude"] if c in df_rs.columns]
                df_melt   = df_rs.melt(
                    id_vars=id_cols, value_vars=melt_cols,
                    var_name="Zone", value_name="Area %")
                fig_bar = px.bar(
                    df_melt, x="Area %", y="Field",
                    color="Zone", barmode="stack",
                    facet_col="Altitude" if "Altitude" in id_cols else None,
                    color_discrete_map={
                        "High Priority%":"#d62728",
                        "Monitor%"      :"#ff7f0e",
                        "Recovered%"    :"#2ca02c"},
                    height=360,
                    labels={"Area %":"Field area (%)"},
                    title="Reseeding zone breakdown per field")
                fig_bar.update_layout(legend=dict(orientation="h",y=-0.25))
                st.plotly_chart(fig_bar, use_container_width=True)
        else:
            st.info("reseeding_summary.csv not found.")


# ══════════════════════════════════════════════════════════════════════════
# PAGE 5 — FINDINGS & DISCUSSION
# ══════════════════════════════════════════════════════════════════════════
elif page == "Findings & Discussion":
    st.title("Findings, Discussion & Conclusion")
    st.markdown(
        "What the results actually mean, where the pipeline can be trusted, "
        "where it can't yet, and what comes next. The **Results** page shows "
        "the evidence; this page interprets it.")

    # ── Key Findings ─────────────────────────────────────────────────────
    st.subheader("🔑 Key Findings")

    with st.container(border=True):
        st.markdown("""
**1. The model generalises - it didn't just memorise the training fields.**
Trained only on F12 and EXP2, the model was then applied to three fields it had
never seen labelled examples from F17, F21, & EXP1 fields. All three produced 
statistically significant results on atleast one altitude, and F17/LA reached 
**R² = 0.825**, higher than either training field, and EXP1 - a structurally different,
non-natural damage pattern (herbicide strip) - was still classified significantly
at both altitudes. This is the strongest evidence that the model learned a
genuine, field-invariant definition of "green" rather than overfitting to two
specific fields.

**2. HA is a reliable, lower-cost substitute for LA - but only above a field-size
threshold.**
Across four of five fields, the high-altitude (HA) model's ring-validation bias
stayed within **±4%** of ground truth, consistently close to zero. The one
exception, EXP2 (+12.3%), is explained by field size: at under 1 million HA
pixels, boundary-pixel effects dominate the estimate. For fields above roughly
49 million HA pixels, HA is essentially as trustworthy as LA, at a fraction of
the flight time and data volume.

**3. LA vs HA is not an either/or choice - it depends on season and field size.**
| Season / field | Recommended altitude | Why |
|---|---|---|
| Early season (April – early May) | **LA** | HA cannot resolve sub-tiller greening; overestimates potential WK by 10-20 pp |
| Mid-to-late season (mid-May onward) | **HA** | Near-zero bias; LA and HA converge; HA is cheaper and simpler to fly |
| Small fields (≲ 5 ha) | **LA** | HA boundary-pixel effects become dominant, producing irregular estimates |

**4. "Potential" winter kill is a validated proxy, not a direct measurement -
and that qualifier matters.**
Ring-plot validation against human ground-truth observations gives R² from
0.331 to 0.825 across fields, with 4 of 5 field/altitude combinations
statistically significant at p < 0.05. This confirms the RGB + U-Net pipeline
tracks real, human-observable field damage but "non-green" can also mean
bare soil, standing water, or shadow, which is why the dashboard never claims
to measure winter kill directly.
""")

    # ── Discussion ──────────────────────────────────────────────────────────
    st.subheader("💬 Discussion")

    disc_tab1, disc_tab2 = st.tabs(["Interpreting the Results", "Limitations"])

    with disc_tab1:
        st.markdown("""
**Why cross-field generalization matters here.** A model trained on two fields
that then performs *at or above* its training-field accuracy on three unseen
fields is a meaningfully stronger result than high accuracy on held-out tiles
from the same fields it was trained on. It suggests the model is responding to
the actual visual signal of green vs. non-green canopy, rather than to
field-specific artefacts (soil colour, lighting conditions, sensor quirks)
that a model can otherwise latch onto.

**The spring plateau is a genuine ceiling on the current RGB-only approach,
not a tuning problem.** Across all fields, the green-fraction time series
shows a clear plateau through roughly mid-April to mid-May: recovering canopy
and fully healthy canopy are spectrally difficult to tell apart in visible
light alone. This isn't something more training data or a better-tuned U-Net
would fix on its own - it's a missing-information problem. The WP3 exploratory
work (see **DL Pipeline**) identifies the Red Edge and near-infrared bands as
the most direct way to close this gap, since both are physically linked to
chlorophyll content and canopy structure in ways visible-band imagery cannot
capture.

**What the reseeding maps are, and aren't.** The reseeding priority
classification (Results → Reseeding Maps) turns repeat flights into a
persistence-based decision aid - a zone flagged High Priority stayed
non-green across most of the season, not just on one unlucky flight date.
That said, maps built on only three assessment dates (e.g. F21/HA) are
flagged with caution in this dashboard for a reason: with few dates, an
early-season patch that recovers can still get counted toward "High Priority"
threshold. More assessment dates, not a different model, is what resolves
this.
""")

    with disc_tab2:
        st.markdown("""
Several limitations affect how confidently these results can be generalised,
and are stated here directly rather than left implicit:

- **Ring validation sample sizes are modest** (n = 10-25 per field/altitude
  combination). EXP1 and the new-field results in particular would benefit
  from additional assessment dates before being treated as fully settled.
- **F21/LA is not statistically significant** (p = 0.082, n = 10) and is
  **excluded** from validated LA outputs for that field - a result this
  dashboard does not report as confirmed.
- **Visual ground-truth estimates carry their own uncertainty**, roughly
  10–20% between observers. LA MAE values of 15-25% are only marginally above
  that threshold, meaning some of the apparent model error may actually be
  human measurement noise.
- **Training covers two fields and one growing season.** The cross-field
  generalisation result (Finding 1, above) is genuinely encouraging, but
  validating across a full spring–summer season and additional geographic
  regions would make that claim considerably stronger.
- **F21 HA reseeding maps require caution**: with only 3 assessment dates and
  a 2/3 High-Priority threshold, two heavily-damaged early dates can
  structurally inflate the High-Priority zone even where later recovery
  occurred.

None of these limitations undermine the core result - that a low-cost RGB
drone and a lightweight U-Net can produce a validated, field-invariant
proxy for winter kill - but they do mark where the evidence is strong versus
where it is still preliminary.
""")

    # ── Conclusion ────────────────────────────────────────────────────────
    st.subheader("✅ Conclusion")
    with st.container(border=True):
        st.markdown("""
This project demonstrates that a **consumer-grade RGB drone paired with a
lightweight U-Net segmentation model** is a viable, validated proxy for
assessing potential winter kill in Nordic forage leys - at a fraction of the
cost and labour of manual field walking. The model generalises to fields it
was never trained on, offers a clear, data-backed answer on when a cheaper
high-altitude flight is "good enough" versus when low-altitude detail is
needed, and produces a reseeding priority map that translates pixel-level
predictions into an actionable, farmer-facing recommendation.

The main constraint on the current RGB-only pipeline is the early-spring
plateau, where recovering and healthy canopy are hard to separate in visible
light. The natural next step, explored preliminarily under **WP3** (see
**DL Pipeline**), is extending the input pipeline with Red Edge and
near-infrared bands - the spectral regions most directly tied to chlorophyll
recovery - to resolve this gap without discarding the validated RGB baseline
established here.
""")
