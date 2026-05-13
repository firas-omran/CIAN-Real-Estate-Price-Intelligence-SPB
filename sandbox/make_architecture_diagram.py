"""Generate BPMN-style swimlane architecture diagrams for CIAN project defense."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patheffects as pe
from pathlib import Path

OUT_DIR = Path(__file__).parent.parent / "docs" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── color palette ────────────────────────────────────────────────────────────
BG        = "#0d1117"
GRID_LINE = "#21262d"

LANE_COLORS = {
    "ext":      {"bg": "#161b22", "hdr": "#1f3a5f", "hdr_txt": "#79c0ff"},
    "collect":  {"bg": "#0e2218", "hdr": "#1a4731", "hdr_txt": "#56d364"},
    "clean":    {"bg": "#1c1208", "hdr": "#4a2800", "hdr_txt": "#ffa657"},
    "features": {"bg": "#17102b", "hdr": "#3d1f7a", "hdr_txt": "#d2a8ff"},
    "ml":       {"bg": "#0e1f1f", "hdr": "#0e4f4f", "hdr_txt": "#39d353"},
}

BOX_COLORS = {
    "ext":      {"face": "#1f3d6b", "edge": "#58a6ff", "txt": "#cae8ff"},
    "collect":  {"face": "#1a4731", "edge": "#2ea043", "txt": "#aff5b4"},
    "clean":    {"face": "#5a2e00", "edge": "#f0883e", "txt": "#ffdfb6"},
    "features": {"face": "#3a1f6b", "edge": "#8b5cf6", "txt": "#e2ccff"},
    "ml":       {"face": "#0d3d3d", "edge": "#00bcd4", "txt": "#b3ecec"},
}

ARROW_CLR = "#58a6ff"
TEXT_CLR  = "#f0f6fc"
SUB_CLR   = "#8b949e"
WHITE     = "#ffffff"


def rounded_box(ax, x, y, w, h, style, label, sublabel="", fontsize=9):
    """Draw a BPMN-style rounded-rectangle task box."""
    c = BOX_COLORS[style]
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.04",
        facecolor=c["face"], edgecolor=c["edge"], linewidth=1.8,
        zorder=3,
    )
    ax.add_patch(box)
    cx, cy = x + w / 2, y + h / 2
    offset = 0.08 if sublabel else 0
    ax.text(cx, cy + offset, label, ha="center", va="center",
            fontsize=fontsize, fontweight="bold", color=c["txt"],
            zorder=4, wrap=True)
    if sublabel:
        ax.text(cx, cy - 0.18, sublabel, ha="center", va="center",
                fontsize=fontsize - 1.5, color=SUB_CLR, zorder=4)


def diamond(ax, cx, cy, size, style, label, fontsize=7.5):
    """Draw a BPMN gateway diamond."""
    c = BOX_COLORS[style]
    d = size
    xs = [cx, cx + d, cx, cx - d, cx]
    ys = [cy + d, cy, cy - d, cy, cy + d]
    ax.fill(xs, ys, facecolor=c["face"], edgecolor=c["edge"], linewidth=1.5, zorder=3)
    ax.text(cx, cy, label, ha="center", va="center", fontsize=fontsize,
            fontweight="bold", color=c["txt"], zorder=4)


def arrow(ax, x0, y0, x1, y1, label="", color=ARROW_CLR, style="-|>", lw=1.4):
    """Draw a labeled arrow."""
    ax.annotate(
        "", xy=(x1, y1), xytext=(x0, y0),
        arrowprops=dict(arrowstyle=style, color=color, lw=lw),
        zorder=5,
    )
    if label:
        mx, my = (x0 + x1) / 2, (y0 + y1) / 2
        ax.text(mx + 0.05, my + 0.05, label, fontsize=7, color=SUB_CLR, zorder=6)


def draw_lane(ax, y_bot, height, key, label, x_start=0, x_end=28):
    """Draw a horizontal swimlane with header stripe."""
    c = LANE_COLORS[key]
    lane_bg = mpatches.FancyBboxPatch(
        (x_start, y_bot), x_end - x_start, height,
        boxstyle="square,pad=0", facecolor=c["bg"], edgecolor=GRID_LINE,
        linewidth=0.8, zorder=1,
    )
    ax.add_patch(lane_bg)
    hdr = mpatches.FancyBboxPatch(
        (x_start, y_bot), 2.1, height,
        boxstyle="square,pad=0", facecolor=c["hdr"], edgecolor=GRID_LINE,
        linewidth=0.8, zorder=2,
    )
    ax.add_patch(hdr)
    ax.text(
        x_start + 1.05, y_bot + height / 2, label,
        ha="center", va="center", fontsize=8.5, fontweight="bold",
        color=c["hdr_txt"], rotation=90, zorder=3,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Diagram 1 — DATA PIPELINE (BPMN swimlane, landscape)
# ─────────────────────────────────────────────────────────────────────────────
def make_pipeline_diagram():
    fig, ax = plt.subplots(figsize=(28, 14))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.set_xlim(0, 28)
    ax.set_ylim(0, 14)
    ax.axis("off")

    # title
    ax.text(14, 13.5, "CIAN Price Intelligence — Data Pipeline",
            ha="center", va="center", fontsize=16, fontweight="bold",
            color=WHITE, zorder=10)
    ax.text(14, 13.1,
            "Checkpoint 2  •  ETL: Extract → Normalize → Validate → Clean → Geocode → Features",
            ha="center", va="center", fontsize=9, color=SUB_CLR, zorder=10)

    # ── lanes ─────────────────────────────────────────────────────────────
    # Lane heights (bottom → top)
    # y: 0-2.4  → ML & Serving
    # y: 2.4-5  → Feature Engineering
    # y: 5-7.6  → Cleaning & Geocoding
    # y: 7.6-10.2 → Collection & Validation
    # y: 10.2-12.8 → External Systems

    draw_lane(ax, 0,    2.4,  "ml",       "ML &\nServing")
    draw_lane(ax, 2.4,  2.6,  "features", "Feature\nEngineering")
    draw_lane(ax, 5,    2.6,  "clean",    "Cleaning &\nGeoкодинг")
    draw_lane(ax, 7.6,  2.6,  "collect",  "Collection &\nValidation")
    draw_lane(ax, 10.2, 2.6,  "ext",      "External\nSystems")

    # ── EXTERNAL SYSTEMS (y: 10.2 - 12.8) ────────────────────────────────
    # CIAN
    rounded_box(ax, 2.5, 10.7, 2.8, 1.6, "ext",
                "CIAN\ncian.ru", "listing pages", fontsize=9)
    # Nominatim
    rounded_box(ax, 13.2, 10.7, 2.8, 1.6, "ext",
                "Nominatim\nOSM API", "rate: 1 req/s\ncache JSON", fontsize=8.5)

    # ── COLLECTION & VALIDATION (y: 7.6 - 10.2) ──────────────────────────
    bw, bh = 2.8, 1.6
    by = 7.9

    # Extract
    rounded_box(ax, 2.5, by, bw, bh, "collect",
                "Extract Listings", "collect_cian_spb.py\n1 400 raw rows", fontsize=8.5)
    # Normalize
    rounded_box(ax, 6.2, by, bw, bh, "collect",
                "Normalize Schema", "NORMALIZED_COLUMNS\n1 359 rows × 24 cols", fontsize=8.5)
    # Validate Contract
    rounded_box(ax, 9.9, by, bw, bh, "collect",
                "Validate Contract", "contract_cian.py\nMAX_NULL / RANGE / ENUM", fontsize=8.5)

    # ── CLEANING & GEOCODING (y: 5 - 7.6) ────────────────────────────────
    cy_ = 5.3

    # Clean
    rounded_box(ax, 9.9, cy_, bw, bh, "clean",
                "Clean Data", "clean_cian.py\n1 304 → 1 300 rows\nVALID_SPB_DISTRICTS", fontsize=8)
    # Geocode
    rounded_box(ax, 13.6, cy_, 3.2, bh, "clean",
                "Geocode Listings", "geocoder.py  3-tier Nominatim\nlat, lon, geo_precision\ndist_center, dist_metro, metro_known",
                fontsize=7.8)

    # ── FEATURE ENGINEERING (y: 2.4 - 5) ─────────────────────────────────
    fy = 2.7

    # Build offline features
    rounded_box(ax, 13.6, fy, 3.2, 1.7, "features",
                "Build Offline Features", "build_features.py\ntarget_price_per_sqm\nlog_target_price_per_sqm\n1 300 × 47 cols", fontsize=8)
    # Build aggregates
    rounded_box(ax, 17.5, fy, 3.2, 1.7, "features",
                "Build Market Aggregates", "Median price/sqm\nby district / rooms\nmétro / room_segment", fontsize=8)

    # Feature Registry
    rounded_box(ax, 21.4, fy, 2.8, 1.7, "features",
                "Feature Registry", "feature_registry.json\n17 features\nleakage audit", fontsize=8)

    # ── ML & SERVING (y: 0 - 2.4) ─────────────────────────────────────────
    my = 0.3

    # Train (future)
    rounded_box(ax, 13.6, my, 3.2, 1.6, "ml",
                "Train Models (CP3)", "CatBoost / Linear\nQuantile regression\nlog-MAE loss", fontsize=8)

    # Serving
    rounded_box(ax, 17.5, my, 3.2, 1.6, "ml",
                "Prediction API", "FastAPI  /predict\n< 500 ms  10 RPS\nrecon: expm1(pred)×sqm", fontsize=8)

    # Baselines (current)
    rounded_box(ax, 21.4, my, 2.8, 1.6, "ml",
                "Baselines B0 – B3\n(current)", "B2 R²_price=0.686\nB2 R²_sqm=0.314\nB3 KNN: 0.681/0.301", fontsize=7.8)

    # ── ARROWS ────────────────────────────────────────────────────────────
    # CIAN → Extract (cross-lane, down)
    arrow(ax, 3.9, 10.7, 3.9, 9.5, color=BOX_COLORS["ext"]["edge"])
    # Extract → Normalize
    arrow(ax, 5.3, 8.7, 6.2, 8.7, color=BOX_COLORS["collect"]["edge"])
    # Normalize → Validate
    arrow(ax, 9.0, 8.7, 9.9, 8.7, color=BOX_COLORS["collect"]["edge"])
    # Validate → Clean (same x, cross lane)
    arrow(ax, 11.3, 7.9, 11.3, 6.9, color=BOX_COLORS["collect"]["edge"])
    # Nominatim → Geocode (cross lane, down)
    arrow(ax, 14.6, 10.7, 15.2, 7.2, color=BOX_COLORS["ext"]["edge"])
    # Clean → Geocode
    arrow(ax, 12.7, 6.1, 13.6, 6.1, color=BOX_COLORS["clean"]["edge"])
    # Geocode → Build Features (cross lane, down)
    arrow(ax, 15.2, 5.3, 15.2, 4.4, color=BOX_COLORS["clean"]["edge"])
    # Build Features → Build Aggregates
    arrow(ax, 16.8, 3.55, 17.5, 3.55, color=BOX_COLORS["features"]["edge"])
    # Build Aggregates → Feature Registry
    arrow(ax, 20.7, 3.55, 21.4, 3.55, color=BOX_COLORS["features"]["edge"])
    # Build Features → Train (cross lane, down)
    arrow(ax, 15.2, 2.7, 15.2, 1.9, color=BOX_COLORS["features"]["edge"])
    # Build Aggregates → Serving
    arrow(ax, 19.1, 2.7, 19.1, 1.9, color=BOX_COLORS["features"]["edge"])
    # Train → Serving
    arrow(ax, 16.8, 1.1, 17.5, 1.1, color=BOX_COLORS["ml"]["edge"])

    # ── legend / artifact labels ──────────────────────────────────────────
    def artifact(x, y, txt, clr):
        ax.text(x, y, txt, fontsize=6.8, color=clr,
                ha="left", va="center", style="italic", zorder=6)

    artifact(2.6, 7.75, "data/raw/cian_spb_raw_*.csv", SUB_CLR)
    artifact(6.3, 7.75, "data/raw/cian_spb_normalized_*.csv", SUB_CLR)
    artifact(9.5, 7.75, "validation report", SUB_CLR)
    artifact(10.1, 4.97, "data/processed/cian_spb_clean.csv", SUB_CLR)
    artifact(13.7, 4.97, "data/processed/cian_spb_clean_geo.csv", SUB_CLR)
    artifact(13.7, 2.67, "data/features/cian_spb_offline_features.csv", SUB_CLR)
    artifact(17.6, 2.67, "data/features/*_market_aggregates.csv", SUB_CLR)

    plt.tight_layout(pad=0)
    out = OUT_DIR / "bpmn_data_pipeline.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"Saved: {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Diagram 2 — INFERENCE SEQUENCE (UML swimlane, portrait)
# ─────────────────────────────────────────────────────────────────────────────
def make_sequence_diagram():
    fig, ax = plt.subplots(figsize=(18, 22))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)

    # participants
    PARTS = [
        ("Пользователь",  "ext",      2.5),
        ("FastAPI",        "collect",  6.0),
        ("Geocoder",       "clean",    9.5),
        ("Nominatim\n(OSM)","ext",    13.0),
        ("Feature Store",  "features",16.5),
        ("ML Model\n(CP3)","ml",      20.0),
    ]
    X_MAX = 22.5
    Y_TOP = 20.5
    Y_BOT = 1.0

    ax.set_xlim(0, X_MAX)
    ax.set_ylim(Y_BOT, Y_TOP + 2)
    ax.axis("off")

    # title
    ax.text(X_MAX / 2, Y_TOP + 1.4,
            "CIAN Price Intelligence — Inference Flow",
            ha="center", va="center", fontsize=15, fontweight="bold", color=WHITE)
    ax.text(X_MAX / 2, Y_TOP + 0.85,
            "Sequence diagram: user request → price estimate",
            ha="center", va="center", fontsize=9, color=SUB_CLR)

    # participant boxes + lifelines
    for name, style, px in PARTS:
        c = BOX_COLORS[style]
        w, h = 2.8, 0.9
        box = FancyBboxPatch(
            (px - w / 2, Y_TOP - h), w, h,
            boxstyle="round,pad=0.05",
            facecolor=c["face"], edgecolor=c["edge"], linewidth=2, zorder=4,
        )
        ax.add_patch(box)
        ax.text(px, Y_TOP - h / 2, name,
                ha="center", va="center", fontsize=8.5, fontweight="bold",
                color=c["txt"], zorder=5)
        # lifeline
        ax.plot([px, px], [Y_TOP - h, Y_BOT],
                color=c["edge"], lw=0.8, linestyle="--", alpha=0.5, zorder=2)

    # ── messages ──────────────────────────────────────────────────────────
    # Each message: (y, x_from, x_to, label, style_key, note)
    def msg(y, xf, xt, label, style, note="", return_arrow=False):
        c_col = BOX_COLORS[style]["edge"]
        lw = 1.5
        ax.annotate("", xy=(xt, y), xytext=(xf, y),
                    arrowprops=dict(
                        arrowstyle="->" if not return_arrow else "->",
                        color=c_col, lw=lw,
                        linestyle="--" if return_arrow else "-",
                    ), zorder=5)
        # label above
        mx = (xf + xt) / 2
        ax.text(mx, y + 0.1, label, ha="center", va="bottom",
                fontsize=8, color=c_col, fontweight="bold", zorder=6)
        if note:
            ax.text(mx, y - 0.18, note, ha="center", va="top",
                    fontsize=7, color=SUB_CLR, style="italic", zorder=6)

    def activation_box(ax, px, y_top, y_bot, style):
        c = BOX_COLORS[style]
        w = 0.25
        box = mpatches.Rectangle(
            (px - w / 2, y_bot), w, y_top - y_bot,
            facecolor=c["face"], edgecolor=c["edge"], linewidth=1.2, zorder=3, alpha=0.9,
        )
        ax.add_patch(box)

    def loop_box(ax, y_top, y_bot, label):
        x0, x1 = 7.5, 14.8
        rect = mpatches.FancyBboxPatch(
            (x0, y_bot), x1 - x0, y_top - y_bot,
            boxstyle="square,pad=0",
            facecolor="none", edgecolor="#f0883e", linewidth=1.4,
            linestyle=":", zorder=3,
        )
        ax.add_patch(rect)
        ax.text(x0 + 0.1, y_top - 0.05, f"loop  [{label}]",
                ha="left", va="top", fontsize=7.5, color="#f0883e",
                fontweight="bold", zorder=7)

    def note_box(ax, x, y, txt, style):
        c = BOX_COLORS[style]
        w, h = 3.5, 0.6
        box = FancyBboxPatch(
            (x - w / 2, y - h / 2), w, h,
            boxstyle="round,pad=0.04",
            facecolor=c["face"], edgecolor=c["edge"],
            linewidth=1, alpha=0.8, zorder=6,
        )
        ax.add_patch(box)
        ax.text(x, y, txt, ha="center", va="center",
                fontsize=7.5, color=c["txt"], zorder=7)

    USER_X = 2.5
    API_X  = 6.0
    GEO_X  = 9.5
    NOM_X  = 13.0
    FS_X   = 16.5
    ML_X   = 20.0

    y = Y_TOP - 1.6

    # 1. User → API
    msg(y, USER_X, API_X, "POST /predict", "collect",
        "{ district, street, house, rooms,\n  total_meters, floor, underground }")
    activation_box(ax, API_X, y, y - 11.5, "collect")
    y -= 0.9

    note_box(ax, API_X, y, "Validate input params", "collect")
    y -= 0.8

    # 2. API → Geocoder
    msg(y, API_X, GEO_X, "enrich_listing(address)", "clean")
    activation_box(ax, GEO_X, y, y - 4.6, "clean")
    y -= 0.8

    # loop: cache check
    loop_y_top = y + 0.2
    y -= 0.3
    note_box(ax, (GEO_X + NOM_X) / 2, y, "check geocode_cache.json", "clean")
    y -= 0.75

    # 3. Geocoder → Nominatim (if cache miss)
    msg(y, GEO_X, NOM_X, "structured query { street, city }", "ext")
    y -= 0.6
    msg(y, NOM_X, GEO_X, "lat, lon  (house / street precision)", "ext",
        return_arrow=True)
    note_box(ax, NOM_X, y - 0.4, "rate-limit 1 req/s\ncache hit → skip", "ext")
    y -= 1.0
    loop_box(ax, loop_y_top, y + 0.1, "cache miss — call Nominatim")
    y -= 0.3

    # 4. Geocoder computes distances
    note_box(ax, GEO_X, y, "haversine → dist_center, dist_metro\ngeo_precision tier", "clean")
    y -= 0.9

    # 5. Geocoder → API
    msg(y, GEO_X, API_X, "{ lat, lon, geo_precision,\n  dist_center, dist_metro, metro_known }",
        "clean", return_arrow=True)
    y -= 1.0

    # 6. API → Feature Store
    msg(y, API_X, FS_X, "lookup(district, rooms_count)", "features")
    activation_box(ax, FS_X, y, y - 0.6, "features")
    y -= 0.5
    msg(y, FS_X, API_X, "market aggregate features", "features", return_arrow=True)
    y -= 0.8

    note_box(ax, API_X, y, "Assemble feature vector (47 dims)", "collect")
    y -= 0.9

    # 7. API → ML Model
    msg(y, API_X, ML_X, "predict(feature_vector)", "ml")
    activation_box(ax, ML_X, y, y - 0.7, "ml")
    y -= 0.6
    msg(y, ML_X, API_X, "log_price_per_sqm_pred", "ml", return_arrow=True)
    y -= 0.9

    note_box(ax, API_X, y, "price = expm1(pred) × total_meters", "collect")
    y -= 0.9

    # 8. API → User
    msg(y, API_X, USER_X, "200 OK", "collect", return_arrow=True)
    y -= 0.4
    note_box(ax, (USER_X + API_X) / 2, y,
             "{ price_estimate, price_per_sqm,\n  confidence_interval, market_stats }",
             "collect")

    plt.tight_layout(pad=0)
    out = OUT_DIR / "sequence_inference.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"Saved: {out}")


if __name__ == "__main__":
    make_pipeline_diagram()
    make_sequence_diagram()
    print("Done.")
