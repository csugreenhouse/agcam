"""
Copyright         : 2026 Luke Portugal
File Name         : luke_plant_finder.py
Description       : Plant finder 2026. Color and edgebased detection.
                  : pipeline: 0: bring in original image 1: find green blobs, 2: remove blobs with odd aspect ratio & discard blobs far from the largest anchor or too small, 3: grow regions from seed mask (final output)
 
Revision History  :
Date        Author          Comments
------------------------------------------------------------------

"""
# IMPORTS
import cv2
import numpy as np
from collections import deque
from pathlib import Path

# ============================================================================
# VARIABLES (Control Panel)
# ============================================================================

# --- Strict green bounds (bright foliage) ------------------------------------
GREEN_LOWER  = np.array([ 25,  30,  20], dtype="uint8")
GREEN_UPPER  = np.array([100, 255, 255], dtype="uint8")
GREEN_BOUNDS = (GREEN_LOWER, GREEN_UPPER)

# --- Pale-green bounds (leaf undersides) -------------------------------------
GREEN_PALE_LOWER = np.array([ 28,  22, 110], dtype="uint8")
GREEN_PALE_UPPER = np.array([105, 150, 255], dtype="uint8")

# --- Dark-green bounds -------------------------------------------------------
GREEN_DARK_LOWER = np.array([ 25,  22,  10], dtype="uint8")
GREEN_DARK_UPPER = np.array([100, 255, 100], dtype="uint8")

# --- Very-dark-green bounds --------------------------------------------------
GREEN_VERYDARK_LOWER = np.array([ 25,  22,   5], dtype="uint8")
GREEN_VERYDARK_UPPER = np.array([100, 255,  40], dtype="uint8")

# --- Controlled Cool-glare & shadow bounds -----------------------------------
GREEN_COOL_LOWER = np.array([ 30,  20,  15], dtype="uint8")
GREEN_COOL_UPPER = np.array([105, 220, 200], dtype="uint8")

# --- Red bract bounds --------------------------------------------------------
RED_LOWER_1 = np.array([  0,  90,  70], dtype="uint8")
RED_UPPER_1 = np.array([ 10, 255, 255], dtype="uint8")
RED_LOWER_2 = np.array([170,  90,  70], dtype="uint8")
RED_UPPER_2 = np.array([179, 255, 255], dtype="uint8")

RED_DARK_LOWER_1 = np.array([  0,  22,  10], dtype="uint8")
RED_DARK_UPPER_1 = np.array([ 10, 255,  90], dtype="uint8")
RED_DARK_LOWER_2 = np.array([170,  22,  10], dtype="uint8")
RED_DARK_UPPER_2 = np.array([179, 255,  90], dtype="uint8")

# --- ExG & GRD thresholds -----------------------------------------------------
EXG_THRESHOLD      = 0.06
EXG_THRESHOLD_DARK = 0.02
EXG_MIN_DARK       = 0.02
EXG_MIN_RESCUE     = 0.05
RED_MIN_RESCUE     = 0.10
DARK_V_CUTOFF      = 80

# --- Shape-filter defaults ---------------------------------------------------
DEFAULT_MIN_ASPECT_RATIO = 0.05
DEFAULT_MAX_ASPECT_RATIO = 15.0

# --- Outliner defaults ------------------------------------------------------
OUTLINE_BRIDGE_PX          = 25
OUTLINE_MIN_COMPONENT_AREA = 300
OUTLINE_MIN_SOLIDITY       = 0.20
OUTLINE_DILATE_BUFFER_PX   = 10

# --- Outlier-filter defaults ------------------------------------------------
ANCHOR_AREA_FRACTION    = 0.25
ANCHOR_MAX_PROXIMITY_PX = 50

# --- Self-anchor gates ------------------------------------------------------
SELF_ANCHOR_MIN_PX         = 1500
SELF_ANCHOR_MIN_PLANT_FRAC = 0.40
SELF_ANCHOR_MAX_BBOX_FILL  = 0.90

# --- Seed purification ------------------------------------------------------
SEED_PURIFY_MIN_EXG           = 0.03
SEED_PURIFY_MIN_RED           = 0.06
SEED_PURIFY_MIN_SAT           = 20
SEED_PURIFY_FALLBACK_FRACTION = 0.30
SEED_PURIFY_MIN_COMPONENT_PX  = 100

# --- Strict-plant seed extension -------------------------------------------
SEED_EXTEND_ITERATIONS = 35

# --- Rescue pass ------------------------------------------------------------
RESCUE_MIN_ENCLOSED_FRAC = 0.55
RESCUE_MIN_SAT           = 18

# --- Local colour extension -------------------------------------------------
LOCAL_EXTEND_ITERATIONS = 25
LOCAL_EXTEND_MIN_EXG    = 0.02
LOCAL_EXTEND_MIN_RED    = 0.04
LOCAL_EXTEND_MIN_SAT    = 18

# --- Final plant-purity filter (DENSITY-BASED) -----------------------------
FINAL_PURITY_DENSITY_WIN  = 9
FINAL_PURITY_DENSITY_MIN  = 0.25
FINAL_PURITY_EDGE_SLOP_PX = 2
FINAL_PURITY_MIN_EXG      = 0.04
FINAL_PURITY_MIN_RED      = 0.05
FINAL_PURITY_MIN_SAT      = 22

# --- Region-grow & hole fill ------------------------------------------------
REGION_GROW_TOLERANCE  = 25
MAX_ENCLOSED_HOLE_AREA = 10000


# ============================================================================
# FAST CHANNEL CACHE
# ============================================================================

def _get_color_cache(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    H_ch, S, V = cv2.split(hsv)
    img_f = image.astype(np.float32)
    B, G, R = cv2.split(img_f)
    s = R + G + B + 1e-6
    exg = (2.0 * G - R - B) / s
    grd = (G - R) / s
    red_excess = (R - np.maximum(G, B)) / s
    return {
        "hsv": hsv, "H": H_ch, "S": S, "V": V,
        "B": B, "G": G, "R": R, "s": s,
        "exg": exg, "grd": grd, "red_excess": red_excess
    }


# ============================================================================
# PUBLIC PIPELINE
# ============================================================================

def find_plants(image_file_path,
                color_bounds=GREEN_BOUNDS,
                use_grabcut=False,
                outline_bridge_px=OUTLINE_BRIDGE_PX,
                outline_min_component_area=OUTLINE_MIN_COMPONENT_AREA,
                outline_min_solidity=OUTLINE_MIN_SOLIDITY,
                color_tolerance=REGION_GROW_TOLERANCE):
    if isinstance(image_file_path, (str, Path)):
        image = cv2.imread(str(image_file_path))
    else:
        image = image_file_path

    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_file_path}")

    H, W = image.shape[:2]
    cache = _get_color_cache(image)

    # ---- Stage 1 -----------------------------------------------------------
    plant_blob_list, veg_mask, red_mask = find_green_blobs(
        image, color_bounds, _cache=cache, _return_masks=True
    )
    print(f"[1] blobs detected as plants: {len(plant_blob_list)}")

    # ---- Stage 2 -----------------------------------------------------------
    kept, discarded = filter_suspect_blobs(plant_blob_list)
    print(f"[2] aspect-ratio kept={len(kept)} discarded={len(discarded)}")

    # ---- Stage 3 -----------------------------------------------------------
    if kept:
        largest_area = max(b["area"] for b in kept)
        min_anchor_area = int(largest_area * ANCHOR_AREA_FRACTION)
    else:
        min_anchor_area = 0

    kept, outliers = filter_outlier_blobs(
        kept,
        min_anchor_area=min_anchor_area,
        max_proximity_px=ANCHOR_MAX_PROXIMITY_PX,
        self_anchor_min_px=SELF_ANCHOR_MIN_PX,
        self_anchor_min_plant_frac=SELF_ANCHOR_MIN_PLANT_FRAC,
        self_anchor_max_bbox_fill=SELF_ANCHOR_MAX_BBOX_FILL,
        image_shape=(H, W),
    )
    print(f"[3] outlier filter kept={len(kept)} discarded={len(outliers)}")

    if discarded is None:
        discarded = []
    discarded += outliers

    seed_mask = _assemble_blobs_mask(kept, (H, W))
    pre_area = cv2.countNonZero(seed_mask)
    if pre_area == 0:
        print("[!] empty seed mask")
        return np.zeros((H, W), dtype="uint8")

    # ---- Stage 4: seed purification ---------------------------------------
    plant_gate_seed = _plant_colour_gate(image,
                                         min_exg=SEED_PURIFY_MIN_EXG,
                                         min_red=SEED_PURIFY_MIN_RED,
                                         min_sat=SEED_PURIFY_MIN_SAT,
                                         _cache=cache)
    purified = cv2.bitwise_and(seed_mask, plant_gate_seed)
    pur_area = cv2.countNonZero(purified)

    if pur_area >= SEED_PURIFY_FALLBACK_FRACTION * pre_area and pur_area > 0:
        n, labels, stats, _ = cv2.connectedComponentsWithStats(
            purified, connectivity=8)
        if n > 1:
            areas = stats[:, cv2.CC_STAT_AREA]
            valid_lut = (areas >= SEED_PURIFY_MIN_COMPONENT_PX)
            valid_lut[0] = False
            kept_seed = np.where(valid_lut[labels], 255, 0).astype(np.uint8)
            seed_mask = kept_seed if cv2.countNonZero(kept_seed) > 0 else purified
        else:
            seed_mask = purified
        print(f"    seed purification: {pre_area} -> "
              f"{cv2.countNonZero(seed_mask)} (kept)")
    else:
        print(f"    seed purification would drop {pre_area} -> {pur_area}; "
              f"keeping original seeds")

    print(f"    final seed mask area = {cv2.countNonZero(seed_mask)}")
    if cv2.countNonZero(seed_mask) == 0:
        return np.zeros((H, W), dtype="uint8")

    # ---- Stage 5a: strict-plant seed extension ---------------------------
    strict_plant = cv2.bitwise_or(veg_mask, red_mask)
    extended_seed = _strict_plant_extension(seed_mask, strict_plant,
                                            iterations=SEED_EXTEND_ITERATIONS)
    print(f"[5a] seed extension: {cv2.countNonZero(seed_mask)} -> "
          f"{cv2.countNonZero(extended_seed)}")

    # ---- Stage 5b: component-based outline silhouette --------------------
    outline = build_outline_from_seeds(
        cv2.bitwise_or(seed_mask, extended_seed),
        bridge_px=outline_bridge_px,
        min_component_area=outline_min_component_area,
        min_solidity=outline_min_solidity,
    )
    print(f"[5b] outline area = {cv2.countNonZero(outline)}")

    # ---- Stage 6: global-mean region grow ---------------------------------
    grown_mask = region_grow(image, seed_mask, color_tolerance=color_tolerance)
    grown_mask = cv2.bitwise_or(grown_mask, extended_seed)
    grown_mask = cv2.bitwise_and(grown_mask, outline)
    grown_mask = cv2.bitwise_or(grown_mask, seed_mask)

    # ---- Stage 7: seed-overlap gate ---------------------------------------
    grown_mask = keep_seed_components(grown_mask, seed_mask)

    # ---- Stage 8: bounded local colour extension --------------------------
    grown_mask = cv2.bitwise_and(grown_mask, outline)
    grown_mask = local_color_extension(grown_mask, image,
                                       iterations=LOCAL_EXTEND_ITERATIONS,
                                       _cache=cache)
    grown_mask = cv2.bitwise_and(grown_mask, outline)
    print(f"[8] after local extension = {cv2.countNonZero(grown_mask)}")

    # ---- Stage 9: DENSITY-BASED plant-purity filter ----------------------
    grown_mask = apply_plant_purity_filter(
        grown_mask, seed_mask, strict_plant,
        image=image, _cache=cache
    )
    grown_mask = keep_seed_components(grown_mask, seed_mask)
    print(f"[9] after density purity filter = {cv2.countNonZero(grown_mask)}")

    # ---- Stage 10: optional GrabCut ---------------------------------------
    if use_grabcut and cv2.countNonZero(grown_mask) > 0:
        refined = refine_with_grabcut(image, seed_mask, grown_mask)
        before  = cv2.countNonZero(grown_mask)
        after   = cv2.countNonZero(refined)
        if after > 0 and after >= 0.7 * before:
            grown_mask = refined
        else:
            print(f"[10] GrabCut rejected ({before} -> {after})")

    # ---- Stage 11 ---------------------------------------------------------
    grown_mask = _fill_enclosed_holes(grown_mask, max_hole_area=MAX_ENCLOSED_HOLE_AREA)
    print(f"[11] final mask area = {cv2.countNonZero(grown_mask)}")
    return grown_mask


# ============================================================================
# FAST GEODESIC EXTENSION
# ============================================================================

def _fast_geodesic_dilate(seed_mask, allowed_mask, iterations):
    if iterations <= 0 or cv2.countNonZero(seed_mask) == 0:
        return seed_mask.copy()

    extended = seed_mask.copy()
    step = 5
    k_step = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * step + 1, 2 * step + 1))
    full_steps = int(iterations) // step
    rem = int(iterations) % step

    for _ in range(full_steps):
        dilated = cv2.dilate(extended, k_step)
        next_ext = cv2.bitwise_or(extended, cv2.bitwise_and(dilated, allowed_mask))
        if np.array_equal(next_ext, extended):
            return extended
        extended = next_ext

    if rem > 0:
        k_rem = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * rem + 1, 2 * rem + 1))
        dilated = cv2.dilate(extended, k_rem)
        extended = cv2.bitwise_or(extended, cv2.bitwise_and(dilated, allowed_mask))

    return keep_seed_components(extended, seed_mask)


def _strict_plant_extension(seed_mask, strict_plant_mask, iterations=35):
    return _fast_geodesic_dilate(seed_mask, strict_plant_mask, iterations)


# ============================================================================
# DENSITY-BASED PLANT-PURITY FILTER
# ============================================================================

def apply_plant_purity_filter(mask, seed_mask, strict_plant_mask,
                              density_win=FINAL_PURITY_DENSITY_WIN,
                              density_min=FINAL_PURITY_DENSITY_MIN,
                              edge_slop_px=FINAL_PURITY_EDGE_SLOP_PX,
                              min_exg=FINAL_PURITY_MIN_EXG,
                              min_red=FINAL_PURITY_MIN_RED,
                              min_sat=FINAL_PURITY_MIN_SAT,
                              image=None,
                              _cache=None):
    plant_f = (strict_plant_mask > 0).astype(np.float32)
    density = cv2.boxFilter(plant_f, -1, (density_win, density_win),
                            normalize=True,
                            borderType=cv2.BORDER_REPLICATE)
    dense = (density >= float(density_min)).astype(np.uint8) * 255

    allowed = cv2.bitwise_or(seed_mask, dense)

    if image is not None or _cache is not None:
        k_slop = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (2 * edge_slop_px + 1, 2 * edge_slop_px + 1))
        near_dense = cv2.dilate(dense, k_slop)
        loose = _plant_colour_gate(image, min_exg=min_exg,
                                   min_red=min_red, min_sat=min_sat,
                                   _cache=_cache)
        allowed = cv2.bitwise_or(
            allowed, cv2.bitwise_and(loose, near_dense))

    return cv2.bitwise_and(mask, allowed)


# ============================================================================
# BLOB DETECTION
# ============================================================================

def _assemble_blobs_mask(blobs, shape):
    mask = np.zeros(shape, dtype=np.uint8)
    for b in blobs:
        if "bbox" in b and "roi_mask" in b:
            x, y, w, h = b["bbox"]
            mask[y:y+h, x:x+w] = cv2.bitwise_or(mask[y:y+h, x:x+w], b["roi_mask"])
        elif "mask" in b:
            mask = cv2.bitwise_or(mask, b["mask"])
    return mask


def find_green_blobs(image,
                     color_bounds=GREEN_BOUNDS,
                     open_kernel_size=(5, 5),
                     close_kernel_size=(5, 5),
                     minimum_area_pixels=300,
                     maximum_area_pixels=None,
                     _cache=None,
                     _return_masks=False):
    if _cache is None:
        _cache = _get_color_cache(image)

    veg_mask = _build_vegetation_mask(image, color_bounds, _cache=_cache)
    red_mask = _build_red_mask(image, _cache=_cache)
    combined = cv2.bitwise_or(veg_mask, red_mask)

    k_open  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, open_kernel_size)
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, close_kernel_size)
    combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN,  k_open)
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, k_close)

    hsv_dark = _build_dark_mask_for_rescue(image, _cache=_cache)
    combined = _rescue_enclosed_dark_pixels(combined, hsv_dark)

    numb_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        combined, connectivity=8
    )

    flat_labels = labels.ravel()
    green_counts = np.bincount(flat_labels, weights=(veg_mask.ravel() > 0), minlength=numb_labels)
    red_counts   = np.bincount(flat_labels, weights=(red_mask.ravel() > 0), minlength=numb_labels)

    plant_blob_list = []
    for label in range(1, numb_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < minimum_area_pixels:
            continue
        if maximum_area_pixels is not None and area > maximum_area_pixels:
            continue

        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        w = int(stats[label, cv2.CC_STAT_WIDTH])
        h = int(stats[label, cv2.CC_STAT_HEIGHT])

        roi_mask = (labels[y:y+h, x:x+w] == label).astype(np.uint8) * 255
        green_px = int(green_counts[label])
        red_px   = int(red_counts[label])

        plant_blob_list.append({
            "label":        label,
            "centroid":     (float(centroids[label][0]),
                             float(centroids[label][1])),
            "area":         area,
            "bbox":         (x, y, w, h),
            "roi_mask":     roi_mask,
            "green_px":     green_px,
            "red_px":       red_px,
            "is_flowering": red_px > green_px,
        })

    if _return_masks:
        return plant_blob_list, veg_mask, red_mask
    return plant_blob_list


# ----------------------------------------------------------------------------
# Colour helpers
# ----------------------------------------------------------------------------

def _build_vegetation_mask(image, color_bounds=GREEN_BOUNDS, _cache=None):
    if _cache is None:
        _cache = _get_color_cache(image)
    hsv = _cache["hsv"]
    B, G, R = _cache["B"], _cache["G"], _cache["R"]
    exg_norm = _cache["exg"]
    grd_norm = _cache["grd"]
    v = _cache["V"]
    S = _cache["S"]

    hsv_mask      = cv2.inRange(hsv, color_bounds[0],      color_bounds[1])
    hsv_pale_mask = cv2.inRange(hsv, GREEN_PALE_LOWER,     GREEN_PALE_UPPER)
    hsv_dark_mask = cv2.inRange(hsv, GREEN_DARK_LOWER,     GREEN_DARK_UPPER)
    hsv_vdark     = cv2.inRange(hsv, GREEN_VERYDARK_LOWER, GREEN_VERYDARK_UPPER)
    hsv_cool_mask = cv2.inRange(hsv, GREEN_COOL_LOWER,     GREEN_COOL_UPPER)

    threshold    = np.where(v < DARK_V_CUTOFF, EXG_THRESHOLD_DARK,
                                               EXG_THRESHOLD)
    exg_adaptive = (exg_norm > threshold).astype("uint8") * 255
    exg_darkgate = (exg_norm > EXG_MIN_DARK).astype("uint8") * 255

    strict = cv2.bitwise_and(hsv_mask, exg_adaptive)
    pale_underside = ((hsv_pale_mask > 0) & (G >= R) & (G >= B)).astype("uint8") * 255
    dark_green     = ((hsv_dark_mask > 0) & (exg_darkgate > 0) & (G >= R) & (G >= B)).astype("uint8") * 255
    verydark       = ((hsv_vdark > 0) & (exg_darkgate > 0) & (G >= R) & (G >= B)).astype("uint8") * 255

    cool_or_shadow = (
        (hsv_cool_mask > 0)
        & (grd_norm > 0.04)
        & (exg_norm > -0.05)
        & (G >= R + 4)
        & (G * 1.15 >= B)
    ).astype("uint8") * 255

    mask = strict
    mask = cv2.bitwise_or(mask, pale_underside)
    mask = cv2.bitwise_or(mask, dark_green)
    mask = cv2.bitwise_or(mask, verydark)
    mask = cv2.bitwise_or(mask, cool_or_shadow)
    return mask


def _build_red_mask(image, _cache=None):
    if _cache is None:
        _cache = _get_color_cache(image)
    hsv = _cache["hsv"]
    red_excess = _cache["red_excess"]

    m1 = cv2.inRange(hsv, RED_LOWER_1, RED_UPPER_1)
    m2 = cv2.inRange(hsv, RED_LOWER_2, RED_UPPER_2)
    hsv_red = cv2.bitwise_or(m1, m2)

    red_mask = np.where(red_excess > 0.12, 255, 0).astype("uint8")
    return cv2.bitwise_and(hsv_red, red_mask)


def _plant_colour_gate(image,
                       min_exg=SEED_PURIFY_MIN_EXG,
                       min_red=SEED_PURIFY_MIN_RED,
                       min_sat=25,
                       _cache=None):
    if _cache is None:
        _cache = _get_color_cache(image)
    H_ch, S, V = _cache["H"], _cache["S"], _cache["V"]
    B, G, R = _cache["B"], _cache["G"], _cache["R"]
    exg, grd, red_excess = _cache["exg"], _cache["grd"], _cache["red_excess"]

    colour_pass = (exg > float(min_exg)) | (red_excess > float(min_red))

    cool_green_pass = (
        (H_ch >= 28) & (H_ch <= 110)
        & (grd > max(0.035, float(min_exg) * 0.9))
        & (exg > -0.04)
        & (G >= R + 3)
        & (G * 1.15 >= B)
        & (V >= 12) & (V <= 215)
    )

    sat_pass = S >= int(min_sat)
    return ((colour_pass | cool_green_pass) & sat_pass).astype(np.uint8) * 255


def _build_dark_mask_for_rescue(image, _cache=None):
    if _cache is None:
        _cache = _get_color_cache(image)
    hsv = _cache["hsv"]
    S = _cache["S"]
    exg, grd, red_excess = _cache["exg"], _cache["grd"], _cache["red_excess"]

    dark = cv2.inRange(hsv, GREEN_DARK_LOWER, GREEN_DARK_UPPER)
    dark = cv2.bitwise_or(dark, cv2.inRange(hsv,
        GREEN_VERYDARK_LOWER, GREEN_VERYDARK_UPPER))
    dark = cv2.bitwise_or(dark, cv2.inRange(hsv,
        GREEN_COOL_LOWER, GREEN_COOL_UPPER))
    dark = cv2.bitwise_or(dark, cv2.inRange(hsv,
        RED_DARK_LOWER_1, RED_DARK_UPPER_1))
    dark = cv2.bitwise_or(dark, cv2.inRange(hsv,
        RED_DARK_LOWER_2, RED_DARK_UPPER_2))

    color_gate = (
        ((exg > EXG_MIN_RESCUE) | (grd > 0.04) | (red_excess > RED_MIN_RESCUE))
        & (S >= RESCUE_MIN_SAT)
    ).astype(np.uint8) * 255
    return cv2.bitwise_and(dark, color_gate)


def _rescue_enclosed_dark_pixels(combined_mask, hsv_dark_mask,
                                 dilation_iters=2,
                                 min_enclosed_frac=RESCUE_MIN_ENCLOSED_FRAC):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    enclosed_zone = cv2.dilate(combined_mask, kernel, iterations=dilation_iters)
    candidates = cv2.bitwise_and(hsv_dark_mask, enclosed_zone)
    candidates = cv2.bitwise_and(candidates, cv2.bitwise_not(combined_mask))
    if cv2.countNonZero(candidates) == 0:
        return combined_mask

    n, labels, stats, _ = cv2.connectedComponentsWithStats(candidates,
                                                           connectivity=8)
    if n <= 1:
        return combined_mask

    plant_f = (combined_mask > 0).astype(np.float32)
    local_density = cv2.boxFilter(plant_f, -1, (5, 5),
                                  normalize=True,
                                  borderType=cv2.BORDER_REPLICATE)

    flat_labels = labels.ravel()
    density_sums = np.bincount(flat_labels, weights=local_density.ravel(), minlength=n)
    areas = stats[:, cv2.CC_STAT_AREA].astype(np.float32)
    mean_densities = density_sums / np.maximum(areas, 1.0)

    valid_lut = (mean_densities >= float(min_enclosed_frac))
    valid_lut[0] = False
    rescue = np.where(valid_lut[labels], 255, 0).astype(np.uint8)
    return cv2.bitwise_or(combined_mask, rescue)


# ============================================================================
# LOCAL COLOUR EXTENSION
# ============================================================================

def local_color_extension(grown_mask, image,
                          iterations=LOCAL_EXTEND_ITERATIONS,
                          min_exg=LOCAL_EXTEND_MIN_EXG,
                          min_red=LOCAL_EXTEND_MIN_RED,
                          min_sat=LOCAL_EXTEND_MIN_SAT,
                          _cache=None):
    plant_gate = _plant_colour_gate(image, min_exg, min_red, min_sat, _cache=_cache)
    return _fast_geodesic_dilate(grown_mask, plant_gate, iterations)


# ============================================================================
# OUTLINER (Component-Based)
# ============================================================================

def build_outline_from_seeds(seed_mask,
                             bridge_px=OUTLINE_BRIDGE_PX,
                             min_component_area=OUTLINE_MIN_COMPONENT_AREA,
                             min_solidity=OUTLINE_MIN_SOLIDITY):
    H, W = seed_mask.shape[:2]
    if cv2.countNonZero(seed_mask) == 0:
        return np.zeros_like(seed_mask)

    k_bridge = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                         (bridge_px, bridge_px))
    bridged = cv2.morphologyEx(seed_mask, cv2.MORPH_CLOSE, k_bridge)

    n, labels, stats, _ = cv2.connectedComponentsWithStats(bridged, connectivity=8)
    out = np.zeros_like(seed_mask)

    for lbl in range(1, n):
        if stats[lbl, cv2.CC_STAT_AREA] < min_component_area:
            continue
        x, y, w, h, _ = stats[lbl]
        
        pad = 20
        x0 = max(0, x - pad)
        y0 = max(0, y - pad)
        x1 = min(W, x + w + pad)
        y1 = min(H, y + h + pad)
        
        roi_bridged = (labels[y0:y1, x0:x1] == lbl).astype(np.uint8) * 255
        
        ff = roi_bridged.copy()
        mask_flood = np.zeros((y1 - y0 + 2, x1 - x0 + 2), np.uint8)
        cv2.floodFill(ff, mask_flood, (0, 0), 255)
        interior = cv2.bitwise_not(ff)
        filled = cv2.bitwise_or(roi_bridged, interior)
        
        conts, _ = cv2.findContours(filled, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
        if conts:
            c = max(conts, key=cv2.contourArea)
            area = cv2.contourArea(c)
            hull = cv2.contourArea(cv2.convexHull(c))
            if hull > 0 and (area / hull) >= (min_solidity - 0.05):
                cv2.drawContours(out[y0:y1, x0:x1], [c], -1, 255, -1)

    out = cv2.bitwise_or(out, seed_mask)
    if OUTLINE_DILATE_BUFFER_PX > 0:
        k_buf = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (2 * OUTLINE_DILATE_BUFFER_PX + 1, 2 * OUTLINE_DILATE_BUFFER_PX + 1)
        )
        out = cv2.dilate(out, k_buf)
    return out


# ============================================================================
# GLOBAL-MEAN REGION GROW
# ============================================================================

def region_grow(image, seed_mask, color_tolerance=REGION_GROW_TOLERANCE):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)

    seed_pixels = lab[seed_mask > 0]
    if len(seed_pixels) == 0:
        return seed_mask.copy()
    seed_mean = seed_pixels.mean(axis=0)

    diff  = lab - seed_mean
    dist2 = diff[..., 0] ** 2 + diff[..., 1] ** 2 + diff[..., 2] ** 2
    close_enough = (dist2 <= float(color_tolerance) ** 2).astype(np.uint8)

    n, labels = cv2.connectedComponents(close_enough, connectivity=8)
    if n <= 1:
        return seed_mask.copy()

    seed_labels = np.unique(labels[seed_mask > 0])
    seed_labels = seed_labels[seed_labels > 0]
    if len(seed_labels) == 0:
        return seed_mask.copy()

    keep = np.zeros(n, dtype=bool)
    keep[seed_labels] = True
    keep[0] = False
    grown = np.where(keep[labels], 255, 0).astype(np.uint8)

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    grown = cv2.morphologyEx(grown, cv2.MORPH_CLOSE, k)
    return grown


def keep_seed_components(mask, seed_mask):
    if cv2.countNonZero(mask) == 0:
        return mask
    n, labels = cv2.connectedComponents(mask, connectivity=8)
    if n <= 1:
        return mask
    seed_labels = np.unique(labels[seed_mask > 0])
    seed_labels = seed_labels[seed_labels > 0]
    if len(seed_labels) == 0:
        return np.zeros_like(mask)
    keep = np.zeros(n, dtype=bool)
    keep[seed_labels] = True
    keep[0] = False
    return np.where(keep[labels], 255, 0).astype(np.uint8)


def refine_with_grabcut(image, seed_mask, outline, iterations=2):
    if cv2.countNonZero(outline) == 0:
        return outline.copy()
    gc = np.full(image.shape[:2], cv2.GC_BGD, dtype=np.uint8)
    gc[outline > 0] = cv2.GC_PR_FGD
    k_sure = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    sure = cv2.erode(seed_mask, k_sure)
    if cv2.countNonZero(sure) == 0:
        sure = seed_mask
    gc[sure > 0] = cv2.GC_FGD
    bgd = np.zeros((1, 65), np.float64)
    fgd = np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(image, gc, None, bgd, fgd, iterations,
                    cv2.GC_INIT_WITH_MASK)
    except cv2.error:
        return outline.copy()
    return np.where((gc == cv2.GC_FGD) | (gc == cv2.GC_PR_FGD),
                    255, 0).astype(np.uint8)


# ============================================================================
# SHAPE FILTERING
# ============================================================================

def compute_blob_shape(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour    = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(contour)
    return (float(w) / h) if h > 0 else 0.0


def _blob_bbox_fill(blob):
    if "bbox" in blob:
        _, _, w, h = blob["bbox"]
        return float(blob["area"]) / float(w * h) if (w * h) > 0 else 0.0
    mask = blob["mask"]
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return 0.0
    w = int(xs.max() - xs.min() + 1)
    h = int(ys.max() - ys.min() + 1)
    if w * h == 0:
        return 0.0
    return float(blob["area"]) / float(w * h)


def filter_suspect_blobs(plant_blob_list,
                         min_aspect_ratio=DEFAULT_MIN_ASPECT_RATIO,
                         max_aspect_ratio=DEFAULT_MAX_ASPECT_RATIO):
    kept, discarded = [], []
    for blob in plant_blob_list:
        if "bbox" in blob:
            _, _, w, h = blob["bbox"]
            ar = (float(w) / h) if h > 0 else None
        else:
            ar = compute_blob_shape(blob["mask"])

        if ar is None:
            discarded.append({**blob, "shape_metrics": None,
                              "discard_reason": "no contour"})
            continue
        blob = {**blob, "shape_metrics": ar}
        reason = None
        if ar < min_aspect_ratio:
            reason = f"ar {ar:.2f} < {min_aspect_ratio}"
        elif ar > max_aspect_ratio:
            reason = f"ar {ar:.2f} > {max_aspect_ratio}"
        if reason:
            discarded.append({**blob, "discard_reason": reason})
        else:
            kept.append(blob)
    return kept, discarded


def filter_outlier_blobs(plant_blob_list,
                         min_anchor_area=500,
                         max_proximity_px=50,
                         self_anchor_min_px=SELF_ANCHOR_MIN_PX,
                         self_anchor_min_plant_frac=SELF_ANCHOR_MIN_PLANT_FRAC,
                         self_anchor_max_bbox_fill=SELF_ANCHOR_MAX_BBOX_FILL,
                         image_shape=None):
    if not plant_blob_list:
        return [], []

    primary_anchors = [b for b in plant_blob_list
                       if b["area"] >= min_anchor_area]
    remaining       = [b for b in plant_blob_list
                       if b["area"] <  min_anchor_area]

    self_anchors = []
    for b in remaining:
        if b["area"] < self_anchor_min_px:
            continue
        plant_px = b["green_px"] + b["red_px"]
        if plant_px < self_anchor_min_plant_frac * b["area"]:
            continue
        if _blob_bbox_fill(b) > self_anchor_max_bbox_fill:
            continue
        self_anchors.append(b)

    self_anchor_ids = set(id(a) for a in self_anchors)
    candidates = [b for b in remaining if id(b) not in self_anchor_ids]

    anchors = primary_anchors + self_anchors
    if not anchors:
        return plant_blob_list, []

    if image_shape is not None:
        H, W = image_shape
    elif "mask" in plant_blob_list[0]:
        H, W = plant_blob_list[0]["mask"].shape[:2]
    else:
        H = max(b["bbox"][1] + b["bbox"][3] for b in plant_blob_list)
        W = max(b["bbox"][0] + b["bbox"][2] for b in plant_blob_list)

    anchor_union = _assemble_blobs_mask(anchors, (H, W))

    dist = cv2.distanceTransform(cv2.bitwise_not(anchor_union),
                                 cv2.DIST_L2, cv2.DIST_MASK_5)

    kept = list(anchors)
    discarded = []
    for blob in candidates:
        if "bbox" in blob and "roi_mask" in blob:
            x, y, w, h = blob["bbox"]
            d = dist[y:y+h, x:x+w][blob["roi_mask"] > 0]
        else:
            d = dist[blob["mask"] == 255]

        min_dist = float(d.min()) if len(d) else float("inf")
        if min_dist <= max_proximity_px:
            kept.append({**blob, "min_dist_to_anchor": round(min_dist, 1)})
        else:
            discarded.append({
                **blob,
                "min_dist_to_anchor": round(min_dist, 1),
                "discard_reason": (f"isolated: {min_dist:.1f}px > "
                                   f"{max_proximity_px}px from anchor"),
            })
    return kept, discarded


# ============================================================================
# POST-PROCESS
# ============================================================================

def _fill_enclosed_holes(mask, max_hole_area=MAX_ENCLOSED_HOLE_AREA):
    H, W = mask.shape[:2]
    inv = cv2.bitwise_not(mask)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(inv, connectivity=8)
    if n <= 1:
        return mask

    x = stats[:, cv2.CC_STAT_LEFT]
    y = stats[:, cv2.CC_STAT_TOP]
    w = stats[:, cv2.CC_STAT_WIDTH]
    h = stats[:, cv2.CC_STAT_HEIGHT]
    areas = stats[:, cv2.CC_STAT_AREA]

    touches_border = (x == 0) | (y == 0) | (x + w >= W) | (y + h >= H)
    fill_lut = (~touches_border) & (areas <= max_hole_area)
    fill_lut[0] = False

    if not np.any(fill_lut):
        return mask

    filled = mask.copy()
    filled[fill_lut[labels]] = 255
    return filled