import cv2
import numpy as np
from collections import deque

# --- Direct (strict) green bounds --------------------------------------------
GREEN_LOWER = np.array([ 25,  35,  20], dtype="uint8")
GREEN_UPPER = np.array([ 95, 255, 255], dtype="uint8")
GREEN_BOUNDS = ((25, 35, 20), (95, 255, 255))

# --- Direct pale-green bounds (leaf undersides) ------------------------------
# Bright, slightly desaturated, still green-tinted. S floor at 25 rejects
# bright metal specular highlights (which sit at S < 20).
GREEN_PALE_LOWER = np.array([ 28,  25, 130], dtype="uint8")
GREEN_PALE_UPPER = np.array([100, 110, 255], dtype="uint8")

# --- Direct dark-green bounds ------------------------------------------------
# S floor raised to 25 so dark GREY metal (S < 20) stays out, but dark green
# leaves (S ≈ 25–60) come in. An ExG gate is also applied (see below) which
# is a very clean separator at low luminance: dark green has 2G-R-B > 0,
# dark metal shadow has 2G-R-B ≈ 0.
GREEN_DARK_LOWER = np.array([ 25,  25,  10], dtype="uint8")
GREEN_DARK_UPPER = np.array([ 95, 255,  80], dtype="uint8")

# --- Direct very-dark-green bounds -------------------------------------------
# For near-black foliage in deep canopy shadow. S floor still 25. ExG gate
# still applied. V floor of 5 keeps true black (which has unstable hue) out.
GREEN_VERYDARK_LOWER = np.array([ 25,  25,   5], dtype="uint8")
GREEN_VERYDARK_UPPER = np.array([ 95, 255,  25], dtype="uint8")

# Red bracts (poinsettia flowers). Red wraps around hue 0 in OpenCV (0-179).
RED_LOWER_1 = np.array([  0, 100,  80], dtype="uint8")
RED_UPPER_1 = np.array([ 10, 255, 255], dtype="uint8")
RED_LOWER_2 = np.array([170, 100,  80], dtype="uint8")
RED_UPPER_2 = np.array([179, 255, 255], dtype="uint8")

RED_DARK_LOWER_1 = np.array([  0,  25,  10], dtype="uint8")
RED_DARK_UPPER_1 = np.array([ 10, 255,  80], dtype="uint8")
RED_DARK_LOWER_2 = np.array([170,  25,  10], dtype="uint8")
RED_DARK_UPPER_2 = np.array([179, 255,  80], dtype="uint8")

EXG_THRESHOLD      = 0.08
EXG_THRESHOLD_DARK = 0.02
DARK_V_CUTOFF      = 80

# Minimum ExG on the dark tiers. Dark green leaves clear this easily;
# dark metal shadow does not.
EXG_MIN_DARK       = 0.02


def _build_vegetation_mask(image):
    """
    Direct mask with four tiers, all gated by G >= R and G >= B:
      1. Strict:    HSV-green AND adaptive ExG (normal foliage)
      2. Pale:      bright, slightly desaturated leaf undersides
      3. Dark:      dark green leaves, gated by ExG_MIN_DARK
      4. VeryDark:  near-black foliage, gated by ExG_MIN_DARK
    The ExG gate on the dark tiers is what keeps dark metal shadow out while
    admitting the specific dark-green shade that was previously being lost.
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv_mask      = cv2.inRange(hsv, GREEN_LOWER,        GREEN_UPPER)
    hsv_pale_mask = cv2.inRange(hsv, GREEN_PALE_LOWER,   GREEN_PALE_UPPER)
    hsv_dark_mask = cv2.inRange(hsv, GREEN_DARK_LOWER,   GREEN_DARK_UPPER)
    hsv_vdark     = cv2.inRange(hsv, GREEN_VERYDARK_LOWER, GREEN_VERYDARK_UPPER)

    img_f = image.astype(np.float32)
    B, G, R = cv2.split(img_f)
    s = R + G + B + 1e-6
    exg_norm = (2.0 * G - R - B) / s

    v = np.maximum.reduce([B, G, R])
    threshold = np.where(v < DARK_V_CUTOFF, EXG_THRESHOLD_DARK, EXG_THRESHOLD)
    exg_adaptive = (exg_norm > threshold).astype("uint8") * 255
    exg_darkgate = (exg_norm > EXG_MIN_DARK).astype("uint8") * 255

    # Tier 1: strict HSV-green AND adaptive ExG
    strict = cv2.bitwise_and(hsv_mask, exg_adaptive)

    # Tier 2: pale undersides. G >= R,B rejects grey metal (R≈G≈B).
    pale_underside = (
        (hsv_pale_mask > 0)
        & (G >= R)
        & (G >= B)
    ).astype("uint8") * 255

    # Tier 3: dark green, HSV gate AND ExG dark gate AND G >= R,B.
    dark_green = (
        (hsv_dark_mask > 0)
        & (exg_darkgate > 0)
        & (G >= R)
        & (G >= B)
    ).astype("uint8") * 255

    # Tier 4: very dark green, same gates.
    verydark = (
        (hsv_vdark > 0)
        & (exg_darkgate > 0)
        & (G >= R)
        & (G >= B)
    ).astype("uint8") * 255

    mask = strict
    mask = cv2.bitwise_or(mask, pale_underside)
    mask = cv2.bitwise_or(mask, dark_green)
    mask = cv2.bitwise_or(mask, verydark)
    return mask


def _build_red_mask(image):
    """Red bract mask handling hue wrap-around, AND'd with a red-excess test."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    m1 = cv2.inRange(hsv, RED_LOWER_1, RED_UPPER_1)
    m2 = cv2.inRange(hsv, RED_LOWER_2, RED_UPPER_2)
    hsv_red = cv2.bitwise_or(m1, m2)

    img_f = image.astype(np.float32)
    B, G, R = cv2.split(img_f)
    s = R + G + B + 1e-6
    red_excess = (R - np.maximum(G, B)) / s
    red_mask = np.where(red_excess > 0.12, 255, 0).astype("uint8")
    return cv2.bitwise_and(hsv_red, red_mask)


def _rescue_enclosed_dark_pixels(combined_mask, hsv_dark_mask,
                                 dilation_iters=2, min_enclosed_frac=0.55):
    """
    Recover dark pockets ENCLOSED by confident plant (canopy interior
    shadows). Non-enclosed dark leaves are already handled by the direct
    dark tiers, so this pass only needs to catch the enclosed ones.
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    enclosed_zone = cv2.dilate(combined_mask, kernel, iterations=dilation_iters)

    candidates = cv2.bitwise_and(hsv_dark_mask, enclosed_zone)
    candidates = cv2.bitwise_and(candidates, cv2.bitwise_not(combined_mask))

    if not np.any(candidates):
        return combined_mask

    n, labels, stats, _ = cv2.connectedComponentsWithStats(candidates, connectivity=8)
    plant_f = (combined_mask > 0).astype(np.float32)
    neigh_k = np.ones((5, 5), np.float32)
    local_density = cv2.filter2D(plant_f, -1, neigh_k, borderType=cv2.BORDER_REPLICATE)
    local_density /= 25.0

    rescue = np.zeros_like(combined_mask)
    for lbl in range(1, n):
        region = (labels == lbl)
        if not np.any(region):
            continue
        mean_density = float(local_density[region].mean())
        if mean_density >= min_enclosed_frac:
            rescue[region] = 255

    return cv2.bitwise_or(combined_mask, rescue)


def _build_dark_mask_for_rescue(image):
    """Dark-pixel candidates for the enclosed-dark rescue pass."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    dark = cv2.inRange(hsv, GREEN_DARK_LOWER,      GREEN_DARK_UPPER)
    dark = cv2.bitwise_or(dark, cv2.inRange(hsv, GREEN_VERYDARK_LOWER, GREEN_VERYDARK_UPPER))
    dark = cv2.bitwise_or(dark, cv2.inRange(hsv, RED_DARK_LOWER_1, RED_DARK_UPPER_1))
    dark = cv2.bitwise_or(dark, cv2.inRange(hsv, RED_DARK_LOWER_2, RED_DARK_UPPER_2))
    return dark


def find_poinsettia_blobs(image,
                          open_kernel_size=(3, 3),
                          close_kernel_size=(5, 5),
                          minimum_area_pixels=100,
                          maximum_area_pixels=500000):
    veg_mask = _build_vegetation_mask(image)
    red_mask = _build_red_mask(image)

    combined = cv2.bitwise_or(veg_mask, red_mask)

    k_open  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, open_kernel_size)
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, close_kernel_size)
    combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN,  k_open)
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, k_close)

    hsv_dark_mask = _build_dark_mask_for_rescue(image)
    combined = _rescue_enclosed_dark_pixels(combined, hsv_dark_mask)

    numb_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        combined, connectivity=8
    )
    blobs = []
    for label in range(1, numb_labels):
        area = stats[label, cv2.CC_STAT_AREA]
        if minimum_area_pixels <= area <= maximum_area_pixels:
            blob_mask = (labels == label).astype("uint8") * 255
            green_px = int(np.count_nonzero(cv2.bitwise_and(blob_mask, veg_mask)))
            red_px   = int(np.count_nonzero(cv2.bitwise_and(blob_mask, red_mask)))
            blobs.append({
                "label":     label,
                "centroid":  (float(centroids[label][0]), float(centroids[label][1])),
                "area":      int(area),
                "mask":      blob_mask,
                "green_px":  green_px,
                "red_px":    red_px,
                "is_flowering": red_px > green_px,
            })
    return blobs


def compute_blob_shape(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour      = max(contours, key=cv2.contourArea)
    x, y, w, h   = cv2.boundingRect(contour)
    aspect_ratio = (float(w) / h) if h > 0 else 0.0
    area         = cv2.contourArea(contour)
    hull         = cv2.convexHull(contour)
    hull_area    = cv2.contourArea(hull) + 1e-6
    solidity     = area / hull_area
    return {"aspect_ratio": aspect_ratio, "solidity": solidity}


def annotate_blob_shapes(plant_blob_list,
                         min_aspect_ratio=0.1,
                         max_aspect_ratio=10.0,
                         min_solidity=0.35):
    annotated  = []
    suspicious = []
    for blob in plant_blob_list:
        metrics = compute_blob_shape(blob["mask"])
        if metrics is None:
            b = {**blob, "shape_metrics": None, "shape_flag": "no_contour"}
            annotated.append(b)
            suspicious.append(b)
            continue
        ar  = metrics["aspect_ratio"]
        sol = metrics["solidity"]
        if ar < min_aspect_ratio or ar > max_aspect_ratio:
            flag = f"aspect_ratio {ar:.2f}"
        elif sol < min_solidity:
            flag = f"solidity {sol:.2f}"
        else:
            flag = None
        b = {**blob, "shape_metrics": metrics, "shape_flag": flag}
        annotated.append(b)
        if flag:
            suspicious.append(b)
    return annotated, suspicious


def filter_suspect_blobs(plant_blob_list,
                         min_aspect_ratio=0.1,
                         max_aspect_ratio=10.0,
                         min_solidity=0.35):
    kept, discarded = [], []
    for blob in plant_blob_list:
        metrics = compute_blob_shape(blob["mask"])
        if metrics is None:
            discarded.append({**blob, "shape_metrics": None, "discard_reason": "no contour"})
            continue
        b = {**blob, "shape_metrics": metrics}
        ar  = metrics["aspect_ratio"]
        sol = metrics["solidity"]
        if ar < min_aspect_ratio or ar > max_aspect_ratio:
            discarded.append({**b, "discard_reason": f"aspect_ratio {ar:.2f} out of range"})
        elif sol < min_solidity:
            discarded.append({**b, "discard_reason": f"solidity {sol:.2f} < {min_solidity}"})
        else:
            kept.append(b)
    return kept, discarded


def filter_outlier_blobs(plant_blob_list,
                         min_anchor_area=500,
                         max_proximity_px=50):
    if not plant_blob_list:
        return [], []

    anchors    = [b for b in plant_blob_list if b["area"] >= min_anchor_area]
    candidates = [b for b in plant_blob_list if b["area"] <  min_anchor_area]

    if not anchors:
        return plant_blob_list, []

    H, W = plant_blob_list[0]["mask"].shape[:2]
    anchor_union = np.zeros((H, W), dtype="uint8")
    for blob in anchors:
        anchor_union = cv2.bitwise_or(anchor_union, blob["mask"])

    inverted       = cv2.bitwise_not(anchor_union)
    dist_to_anchor = cv2.distanceTransform(inverted, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)

    kept      = list(anchors)
    discarded = []
    for blob in candidates:
        blob_pixels_dist = dist_to_anchor[blob["mask"] == 255]
        min_dist = float(blob_pixels_dist.min()) if len(blob_pixels_dist) > 0 else float("inf")
        if min_dist <= max_proximity_px:
            kept.append({**blob, "min_dist_to_anchor": round(min_dist, 1)})
        else:
            discarded.append({
                **blob,
                "min_dist_to_anchor": round(min_dist, 1),
                "discard_reason": f"isolated: {min_dist:.1f}px > {max_proximity_px}px"
            })
    return kept, discarded


def region_grow(image, seed_mask,
                hue_tolerance=10,
                s_tolerance=70,
                v_tolerance=90):
    """
    Grow seeds by HUE similarity with tight S/V gates. Conservative on
    purpose: the direct mask already covers bright tops, pale undersides,
    and dark leaves, so region_grow only needs to fill small gaps between
    seed blobs.
    """
    H, W = image.shape[:2]
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.int16)

    seed_pixels = hsv[seed_mask == 255]
    if len(seed_pixels) == 0:
        return seed_mask.copy()
    seed_h, seed_s, seed_v = np.median(seed_pixels, axis=0)

    visited = (seed_mask > 0).copy()
    result  = (seed_mask > 0).copy()
    queue   = deque()

    def hue_dist(h):
        d = abs(int(h) - int(seed_h))
        return min(d, 180 - d)

    for (py, px) in np.argwhere(seed_mask == 255):
        for dy, dx in [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]:
            ny, nx = py + dy, px + dx
            if 0 <= ny < H and 0 <= nx < W and not visited[ny, nx]:
                queue.append((ny, nx))
                visited[ny, nx] = True

    while queue:
        py, px = queue.popleft()
        h, s, v = hsv[py, px]
        if (hue_dist(h) <= hue_tolerance
                and abs(int(s) - int(seed_s)) <= s_tolerance
                and abs(int(v) - int(seed_v)) <= v_tolerance):
            result[py, px] = True
            for dy, dx in [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]:
                ny, nx = py + dy, px + dx
                if 0 <= ny < H and 0 <= nx < W and not visited[ny, nx]:
                    queue.append((ny, nx))
                    visited[ny, nx] = True

    grown_mask = result.astype("uint8") * 255
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    grown_mask = cv2.morphologyEx(grown_mask, cv2.MORPH_CLOSE, k)
    return grown_mask


def _fill_enclosed_holes(mask, max_hole_area=4000):
    H, W = mask.shape[:2]
    inv = cv2.bitwise_not(mask)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(inv, connectivity=8)
    filled = mask.copy()
    for lbl in range(1, n):
        x, y, w, h, area = stats[lbl]
        touches_border = (x == 0 or y == 0 or x + w >= W or y + h >= H)
        if not touches_border and area <= max_hole_area:
            filled[labels == lbl] = 255
    return filled


def find_plants(image,
                hue_tolerance=10,
                s_tolerance=70,
                v_tolerance=90,
                strict_filter=False,
                anchor_area_fraction=0.75,
                anchor_max_proximity_px=25):
    """
    Full pipeline:
      1. Direct mask: strict + pale + dark + verydark tiers.
      2. Red bract mask (bright + dark).
      3. Rescue pass for enclosed dark pockets.
      4. Annotate blob shapes (nothing discarded by default).
      5. Seed mask from all kept blobs.
      6. Conservative region grow.
      7. Fill enclosed holes.
    """
    H, W = image.shape[:2]

    blobs = find_poinsettia_blobs(image)

    if strict_filter:
        kept, suspicious = filter_suspect_blobs(blobs)
    else:
        kept, suspicious = annotate_blob_shapes(blobs)

    if kept:
        largest_area = max(b["area"] for b in kept)
        min_anchor_area = int(largest_area * anchor_area_fraction)
    else:
        min_anchor_area = 0

    seed_mask = np.zeros((H, W), dtype="uint8")
    for blob in kept:
        seed_mask = cv2.bitwise_or(seed_mask, blob["mask"])

    grown_mask = region_grow(
        image,
        seed_mask,
        hue_tolerance=hue_tolerance,
        s_tolerance=s_tolerance,
        v_tolerance=v_tolerance,
    )

    grown_mask = _fill_enclosed_holes(grown_mask, max_hole_area=4000)

    return {
        "seed_mask":       seed_mask,
        "grown_mask":      grown_mask,
        "filtered_mask":   grown_mask,
        "kept_blobs":      kept,
        "discarded_blobs": suspicious,
    }