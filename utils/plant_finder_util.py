import cv2
import numpy as np
from collections import deque

GREEN_LOWER  = np.array([ 45, 120, 120], dtype="uint8")
GREEN_UPPER  = np.array([ 75, 255, 220], dtype="uint8")
#GREEN_BOUNDS = (GREEN_LOWER, GREEN_UPPER)
GREEN_BOUNDS = ((31, 50, 50), (75, 255, 200))

def find_green_blobs(image,
                     color_bounds=GREEN_BOUNDS,
                     open_kernel_size=(5, 5),
                     close_kernel_size=(5, 5),
                     minimum_area_pixels=100,
                     maximum_area_pixels=100000):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv_mask = cv2.inRange(hsv, color_bounds[0], color_bounds[1])
    

    img_f = image.astype(np.float32)
    B, G, R = cv2.split(img_f)# below I put a further categorization to remove stuff that is not super green
    exg = 2.0 * G - R - B
    exg_mask = np.where(exg > 10, 255, 0).astype("uint8")

    combined = cv2.bitwise_and(hsv_mask, exg_mask) #and the two (HSV and green) to be strict

    k_open  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, open_kernel_size)
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, close_kernel_size)
    combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN,  k_open)
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, k_close)

    numb_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(combined, connectivity=8)
    plant_blob_list = []
    for label in range(1, numb_labels):
        area = stats[label, cv2.CC_STAT_AREA]
        if minimum_area_pixels <= area <= maximum_area_pixels:
            plant_blob_list.append({
                "label":    label,
                "centroid": (float(centroids[label][0]), float(centroids[label][1])),
                "area":     int(area),
                "mask":     (labels == label).astype("uint8") * 255
            })
    return plant_blob_list


def compute_blob_shape(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour      = max(contours, key=cv2.contourArea)
    x, y, w, h   = cv2.boundingRect(contour)
    aspect_ratio = (float(w) / h) if h > 0 else 0.0
    return {"aspect_ratio": aspect_ratio} #if the blobs are skinny they are likely to be ceiling cracks or stems etc


def filter_suspect_blobs(plant_blob_list,
                          min_aspect_ratio=0.15,
                          max_aspect_ratio=6.0):
    kept      = []
    discarded = []

    for blob in plant_blob_list:
        metrics = compute_blob_shape(blob["mask"])

        if metrics is None:
            discarded.append({**blob, "shape_metrics": None, "discard_reason": "no contour"})
            continue

        blob_with_metrics = {**blob, "shape_metrics": metrics}

        reason = None
        if metrics["aspect_ratio"] < min_aspect_ratio:
            reason = f"aspect_ratio {metrics['aspect_ratio']:.2f} < {min_aspect_ratio}"
        elif metrics["aspect_ratio"] > max_aspect_ratio:
            reason = f"aspect_ratio {metrics['aspect_ratio']:.2f} > {max_aspect_ratio}"

        if reason:
            discarded.append({**blob_with_metrics, "discard_reason": reason})
        else:
            kept.append(blob_with_metrics)

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

    inverted     = cv2.bitwise_not(anchor_union) #needed to invert to have the non plant area defined
    dist_to_anchor = cv2.distanceTransform(inverted, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)

    kept      = list(anchors)  # anchors kept
    discarded = []

    for blob in candidates: #each blob below threshold needs to pass inquisition
        blob_pixels_dist = dist_to_anchor[blob["mask"] == 255]
        min_dist = float(blob_pixels_dist.min()) if len(blob_pixels_dist) > 0 else float("inf")

        if min_dist <= max_proximity_px:
            kept.append({**blob, "min_dist_to_anchor": round(min_dist, 1)})
        else:
            discarded.append({
                **blob,
                "min_dist_to_anchor": round(min_dist, 1),
                "discard_reason": f"isolated: min_dist {min_dist:.1f}px > {max_proximity_px}px from anchor"
            })

    return kept, discarded

def region_grow(image, seed_mask, color_tolerance=22):
    H, W = image.shape[:2]
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)

    seed_pixels = lab[seed_mask == 255]
    if len(seed_pixels) == 0:
        return seed_mask.copy()
    seed_mean = seed_pixels.mean(axis=0)

    visited = (seed_mask > 0).copy()
    result  = (seed_mask > 0).copy()

    queue = deque()
    for (py, px) in np.argwhere(seed_mask == 255):
        for dy, dx in [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]:
            ny, nx = py + dy, px + dx
            if 0 <= ny < H and 0 <= nx < W and not visited[ny, nx]:
                queue.append((ny, nx))
                visited[ny, nx] = True

    while queue:
        py, px = queue.popleft()
        if np.linalg.norm(lab[py, px] - seed_mean) <= color_tolerance:
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

#fully assembled pipeline method for plant detection
# 0: bring in original image 1: find green blobs, 2: remove blobs with odd aspect ratio & discard blobs far from the largest anchor or too small, 3: grow regions from seed mask (final output)
def find_plants(image, color_bounds=GREEN_BOUNDS,
                    color_tolerance=22,
                    anchor_area_fraction=0.75,
                    anchor_max_proximity_px=25):
    H, W = image.shape[:2]

    plant_blob_list = find_green_blobs(image, color_bounds)                         #pipeline stage 1: find green blobs

    kept, discarded = filter_suspect_blobs(plant_blob_list)                         #pipeline stage 2: remove blobs with odd aspect ratio

    if kept:
        largest_area   = max(b["area"] for b in kept)
        min_anchor_area = int(largest_area * anchor_area_fraction)
    else:
        min_anchor_area = 0

    '''kept, outliers = filter_outlier_blobs(kept,                                     #pipeline stage 3: discard blobs far from the largest anchor or too small
                                          min_anchor_area=min_anchor_area,
                                          max_proximity_px=anchor_max_proximity_px)
    discarded += outliers'''

    if discarded is None or len(discarded) == 0:
        discarded = np.zeros((H, W), dtype=np.uint8)

    seed_mask = np.zeros((H, W), dtype="uint8")
    for blob in kept:
        seed_mask = cv2.bitwise_or(seed_mask, blob["mask"])

    grown_mask    = region_grow(image, seed_mask, color_tolerance=color_tolerance)  #pipeline stage 4: grow regions from seed mask

    return {
        "seed_mask":       seed_mask,
        "grown_mask":      grown_mask,
        "filtered_mask":   grown_mask,
        "kept_blobs":      kept,
        "discarded_blobs": discarded
    }