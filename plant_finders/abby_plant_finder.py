"""
Copyright         : 2026 Abby Robertson
File Name         : abby_plant_finder.py
Description       : Plant finder 2026. Color based detection.
                  : pipeline: 0: bring in original image 1: find green blobs, 2: remove blobs with odd aspect ratio & discard blobs far from the largest anchor or too small, 3: grow regions from seed mask (final output)
 
Revision History  :
Date		Author 			Comments
------------------------------------------------------------------

"""

# IMPORTS
import cv2
import numpy as np
from collections import deque

# VARIABLES (Control Panel)
GREEN_LOWER  = np.array([ 31, 50, 50], dtype="uint8")
GREEN_UPPER  = np.array([ 75, 255, 220], dtype="uint8")
RED_LOWER = np.array([39,24,28], dtype="uint8")
RED_UPPER = np.array([60,35,46])
GREEN_BOUNDS = (GREEN_LOWER, GREEN_UPPER)

def find_plants(image_file_path, color_bounds=GREEN_BOUNDS):
    """
    fully assembled pipeline method for plant detection
    0: bring in original image 1: find green blobs, 2: remove blobs with odd aspect ratio & discard blobs far from the largest anchor or too small, 3: grow regions from seed mask (final output)

    Args:
        image: the image to detect plants in

    Returns:
        grown_mask: OpenCV mask of detected plants
    """

    image = cv2.imread(str(image_file_path))

    color_tolerance=22
    anchor_area_fraction=0.75
    anchor_max_proximity_px=25

    H, W = image.shape[:2]

    #pipeline stage 1: find green blobs
    plant_blob_list = find_green_blobs(image, color_bounds)
    print(f"Length of list of blobs detected as plants: {len(plant_blob_list)}")

    #pipeline stage 2: remove blobs with odd aspect ratio
    kept, discarded = filter_suspect_blobs(plant_blob_list)
    print(f"After sorting by aspect ratio, number blobs kept: {len(kept)}; number blobs discarded: {len(discarded)}")

    if kept:
        largest_area   = max(b["area"] for b in kept)
        min_anchor_area = int(largest_area * anchor_area_fraction)
    else:
        min_anchor_area = 0

    #pipeline stage 3: (currently in disuse) discard blobs far from the largest anchor or too small
    """kept, outliers = filter_outlier_blobs(kept,
                                          min_anchor_area=min_anchor_area,
                                          max_proximity_px=anchor_max_proximity_px)
    discarded += outliers"""

    if discarded is None or len(discarded) == 0:
        discarded = np.zeros((H, W), dtype=np.uint8)

    seed_mask = np.zeros((H, W), dtype="uint8")
    for blob in kept:
        seed_mask = cv2.bitwise_or(seed_mask, blob["mask"])

    print(f"Seed Mask area: {cv2.countNonZero(seed_mask)}")

    grown_mask    = region_grow(image, seed_mask, color_tolerance=color_tolerance)  #pipeline stage 4: grow regions from seed mask
    print(f"Grown Mask area: {cv2.countNonZero(grown_mask)}")

    output_dict = {
        "plant_blob_list": plant_blob_list,
        "kept_blobs":      kept,
        "discarded_blobs": discarded,
        "seed_mask":       seed_mask,
        "grown_mask":      grown_mask,
    }

    return output_dict["grown_mask"]

def find_green_blobs(image,
                     color_bounds=GREEN_BOUNDS,
                     open_kernel_size=(5, 5),
                     close_kernel_size=(5, 5),
                     minimum_area_pixels=100,
                     maximum_area_pixels=100000):
    """
    detect `green blobs' according to HSV bounds, with size based limits

    Args:
        image: the image to detect plants in
        color_bounds: HSV color bounds to consider as `green'
        open_kernel_size: 
        close_kernel_size: 
        minimum_area_pixels: largest possible blob
        maximum_area_pixels: smallest thing that may be detected as a blob

    Returns:
        plant_blob_list: Array containing dicts for every blob with keys label, centroid, area, and mask
    """

    #apply hsv mask
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv_mask = cv2.inRange(hsv, color_bounds[0], color_bounds[1])

    # use exg as a further categorization to remove stuff that is not super green
    img_f = image.astype(np.float32)
    B, G, R = cv2.split(img_f)
    exg = 2.0 * G - R - B
    exg_mask = np.where(exg > 10, 255, 0).astype("uint8")

    #and the two (HSV and green) to be strict
    combined = cv2.bitwise_and(hsv_mask, exg_mask)

    #clean the mask?
    k_open  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, open_kernel_size)
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, close_kernel_size)
    combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN,  k_open)
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, k_close)

    #get details on the blobs
    numb_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(combined, connectivity=8)

    #format output for each blob and add to array. then return.
    plant_blob_list = []
    for label in range(1, numb_labels):
        area = stats[label, cv2.CC_STAT_AREA]
        if minimum_area_pixels <= area: #<= maximum_area_pixels:
            plant_blob_list.append({
                "label":    label,
                "centroid": (float(centroids[label][0]), float(centroids[label][1])),
                "area":     int(area),
                "mask":     (labels == label).astype("uint8") * 255
            })
    return plant_blob_list

def compute_blob_shape(mask):
    """
    find the aspect ratio of a mask

    Args:
        mask: the mask to find the aspect ratio of

    Returns:
        aspect_ratio: float of aspect ratio
    """

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour      = max(contours, key=cv2.contourArea)
    x, y, w, h   = cv2.boundingRect(contour)
    aspect_ratio = (float(w) / h) if h > 0 else 0.0
    return aspect_ratio

def filter_suspect_blobs(plant_blob_list,
                          min_aspect_ratio=0.15,
                          max_aspect_ratio=6.0):
    """
    filter out blobs by aspect ratio.
    if the blobs are skinny they are likely to be ceiling cracks or stems etc

    Args:
        plant_blob_list: Array containing dicts for every blob with keys label, centroid, area, and mask
        min_aspect_ratio: control for the minimum acceptable aspect ratio for a plant
        max_aspect_ratio: control for the maximum acceptable aspect ratio for a plant

    Returns:
        kept: Array of dictionaries of masks and their associated aspect ratios
        discarded: Array of dictionaries of masks, their aspect ratios, and their discard reasons
    """
    
    kept      = []
    discarded = []

    for blob in plant_blob_list:
        aspect_ratio = compute_blob_shape(blob["mask"])

        if aspect_ratio is None:
            discarded.append({**blob, "shape_metrics": None, "discard_reason": "no contour"})
            continue

        blob_with_metrics = {**blob, "shape_metrics": aspect_ratio}

        reason = None
        if aspect_ratio < min_aspect_ratio:
            reason = f"aspect ratio {aspect_ratio:.2f} < {min_aspect_ratio}"
        elif aspect_ratio > max_aspect_ratio:
            reason = f"aspect ratio {aspect_ratio:.2f} > {max_aspect_ratio}"

        if reason:
            discarded.append({**blob_with_metrics, "discard_reason": reason})
        else:
            kept.append(blob_with_metrics)

    return kept, discarded

def filter_outlier_blobs(plant_blob_list,
                          min_anchor_area=500,
                          max_proximity_px=50):
    """
    Assume plants are the biggest detected blobs in the photo. Define some blobs as anchors then delete small blobs that are far away from these.

    Args:
        plant_blob_list: Array containing dicts for every blob with keys label, centroid, area, and mask
        min_anchor_area: Minimum size for a blob to be defined as an `anchor' (safely plant)
        max_proximity: max distance from an anchor that is considered to be `safe area' for tiny things to still be plant

    Returns:
        kept: Array of dictionaries of masks and their distance to the closest anchor
        discarded: Array of dictionaries of masks and their discard reason (distance to closest anchor too large)
    """
    
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
    """
    Apply openCV 

    Args:
        image: Original image to be passed into the function.
        seed_mask: mask of blobs that are supposed to be plants, at this point it should be quite accurate
        color_tolerance: maximum color change allowance for pushing the region grow out from each blob

    Returns:
        grown_mask: mask after region grow.
    """

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

def test_pipeline_on_image(src_path):
    src_path = Path(src_path)
    dst_path = Path(str(src_path).replace(".jpg", "_pipeline_out.jpg"))

    img = cv2.imread(str(src_path))

    if img is None:
        raise FileNotFoundError(f"Missing {src_path}.")

    img_rgb   = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    result = plant_finder.find_plants(img)

    # END PIPELINE, MAKE DISPLAYS #

    """#display pipeline functionality by stage (commented out because this one is not useable with the refactored plantfinders. if you want tests like this, you have to define them in your plantfinder.)
    fig, axes = plt.subplots(1, 4, figsize=(25, 6))
    fig.suptitle(f"Plant Pipeline Stage Outputs")

    axes[0].imshow(img_rgb)
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    blob_overlay = img_rgb.copy()
    blob_overlay[result == 255] = (
        blob_overlay[result == 255] * 0.5 + np.array([0, 0, 200]) * 0.5
    ).astype("uint8")
    axes[1].imshow(blob_overlay)
    axes[1].set_title("Blob Detection")
    axes[1].axis("off")

    discard_overlay = img_rgb.copy()
    discard_overlay[result["discarded_blobs"] == 255] = (
        discard_overlay[result["discarded_blobs"] == 255] * 0.5 + np.array([200, 0, 0]) * 0.5
    ).astype("uint8")
    axes[2].imshow(discard_overlay)
    axes[2].set_title("Discarded Blobs")
    axes[2].axis("off")

    output_overlay = img_rgb.copy()
    output_overlay[result["filtered_mask"] == 255] = (
        output_overlay[result["filtered_mask"] == 255] * 0.5 + np.array([0, 0, 200]) * 0.5
    ).astype("uint8")
    axes[3].imshow(output_overlay)
    axes[3].set_title("Region Grow (Final Plant Detection Output)")
    axes[3].axis("off")

    stage_path = str(src_path).replace(".jpg", "_pipeline_stages_out.jpg")
    plt.tight_layout()
    plt.savefig(stage_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"intermediate stages display saved to: {stage_path}")"""

    #final simplified output image, shows original pic and final plant mask.
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f"{Path(src_path).name}  Plant Detection Results")

    axes[0].imshow(img_rgb)
    axes[0].set_title("Original")
    axes[0].axis("off")

    output_overlay = img_rgb.copy()
    output_overlay[result == 255] = (
        output_overlay[result == 255] * 0.5 + np.array([0, 0, 200]) * 0.5
    ).astype("uint8")

    axes[1].imshow(output_overlay)
    axes[1].set_title("Plant Detection Overlay")
    axes[1].axis("off")

    plt.tight_layout()
    plt.savefig(str(dst_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"final plant detection output saved to: {dst_path}")