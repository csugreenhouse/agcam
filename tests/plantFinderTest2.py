import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import importlib

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

pf = importlib.import_module("utils.plantFinderUtil2")
graph_util = importlib.import_module("utils.graph_util")
reference_util = importlib.import_module("utils.reference_tag_util")


def test_blobber(src_path):
    src_path = Path(src_path)
    dst_path = Path(str(src_path).replace(".jpg", "_blobber_out.jpg"))

    img = cv2.imread(str(src_path))
    if img is None:
        raise FileNotFoundError(f"Src path missing {src_path}.")

    # find_poinsettia_blobs replaces find_green_blobs — it detects green
    # foliage AND red bracts, so it works before and after flowering.
    blob_list = pf.find_poinsettia_blobs(img)
    assert len(blob_list) != 0, "No blobs found"

    # Log flowering status per blob so we can see which ones are red.
    flowering = sum(1 for b in blob_list if b.get("is_flowering"))
    print(f"  {len(blob_list)} blob(s) found, {flowering} flowering")

    graph_util.plot_blobs(img, dst_path, blob_list)
    print(f"blob test saved to: {dst_path}")


def test_pipeline_on_image(src_path):
    src_path = Path(src_path)
    dst_path = Path(str(src_path).replace(".jpg", "_pipeline_out.jpg"))

    img = cv2.imread(str(src_path))
    if img is None:
        raise FileNotFoundError(f"Missing {src_path}.")

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # find_plants now takes no color_bounds kwarg (green + red are baked in)
    result = pf.find_plants(img)

    # --- Discarded blobs may be a list OR a zero mask -------------------------
    # The new find_plants() returns discarded_blobs as a list of dicts when
    # any were rejected, but falls back to a zero mask if none were discarded.
    discarded_mask = np.zeros(img.shape[:2], dtype="uint8")
    discarded = result.get("discarded_blobs", [])
    if isinstance(discarded, list):
        for blob in discarded:
            mask = blob.get("mask")
            if mask is not None:
                discarded_mask = cv2.bitwise_or(discarded_mask, mask)
    else:
        # Legacy fallback: it's already an image
        discarded_mask = discarded

    # --- Build overlays -------------------------------------------------------
    seed_overlay = img_rgb.copy()
    seed_overlay[result["seed_mask"] == 255] = (
        seed_overlay[result["seed_mask"] == 255] * 0.5
        + np.array([0, 0, 200]) * 0.5
    ).astype("uint8")

    discard_overlay = img_rgb.copy()
    discard_overlay[discarded_mask == 255] = (
        discard_overlay[discarded_mask == 255] * 0.5
        + np.array([200, 0, 0]) * 0.5
    ).astype("uint8")

    output_overlay = img_rgb.copy()
    output_overlay[result["filtered_mask"] == 255] = (
        output_overlay[result["filtered_mask"] == 255] * 0.5
        + np.array([0, 0, 200]) * 0.5
    ).astype("uint8")

    # --- Stage-by-stage visualization ----------------------------------------
    fig, axes = plt.subplots(1, 4, figsize=(25, 6))
    fig.suptitle("Plant Pipeline Stage Outputs")

    axes[0].imshow(img_rgb)
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    axes[1].imshow(seed_overlay)
    axes[1].set_title("Blob Detection (Seed Mask)")
    axes[1].axis("off")

    axes[2].imshow(discard_overlay)
    axes[2].set_title("Discarded Blobs")
    axes[2].axis("off")

    axes[3].imshow(output_overlay)
    axes[3].set_title("Region Grow (Final Output)")
    axes[3].axis("off")

    stage_path = str(src_path).replace(".jpg", "_pipeline_stages_out.jpg")
    plt.tight_layout()
    plt.savefig(stage_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"intermediate stages display saved to: {stage_path}")

    # --- Final simplified output ---------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f"{Path(src_path).name}  Plant Detection Results")

    axes[0].imshow(img_rgb)
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(output_overlay)
    axes[1].set_title("Plant Detection Overlay")
    axes[1].axis("off")

    plt.tight_layout()
    plt.savefig(str(dst_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"final plant detection output saved to: {dst_path}")


if __name__ == "__main__":
    test_blobber("tests/testimages/IMG_3387.jpg")
    test_pipeline_on_image("tests/testimages/IMG_3387.jpg")

    test_blobber("tests/testimages/IMG_3398.jpg")
    test_pipeline_on_image("tests/testimages/IMG_3398.jpg")