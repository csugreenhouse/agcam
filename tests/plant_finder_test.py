import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import importlib

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

pf = importlib.import_module("utils.plant_finder_util")
graph_util = importlib.import_module("utils.graph_util")
reference_util = importlib.import_module("utils.reference_tag_util")


def test_blobber():
    src_path = "tests/testimages/IMG_3387.jpg"
    dst_path = "tests/testimages/IMG_3387_1blobber_out.png"

    img = cv2.imread(str(src_path))
    if img is None:
        raise FileNotFoundError(f"Src path missing {src_path}.")

    plastic_color_bounds = ((31, 50, 50), (75, 255, 200))
    plant_bounds = (.35, .6, 0, 1)
    plant_id = 1
    bias = .01

    blob_list = pf.find_green_blobs(img, plastic_color_bounds)
    assert len(blob_list) != 0, "No blobs found"
    graph_util.plot_blobs(img, dst_path, blob_list)


def test_region_grow(): #testing whole image pipeline.
    src_path = "tests/testimages/IMG_3387.jpg"
    dst_path = "tests/testimages/IMG_3387_2regiongrow_out.png"

    img = cv2.imread(str(src_path))
    if img is None:
        raise FileNotFoundError(f"Missing {src_path}.")

    hsv_detection_bounds = ((31, 50, 50), (75, 255, 200)) #manual definition here

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

#blob detection from hsv as in earlier versions
    blob_list = pf.find_green_blobs(img, hsv_detection_bounds)
    assert len(blob_list) != 0, "No blobs found"

#grown results
    result = pf.grow_plant_mask(img, blob_list, color_tolerance=22)

#visual for discarded blobs
    if result["discarded_blobs"]:
        discard_vis = img_rgb.copy()
        for blob in result["discarded_blobs"]:
            contours, _ = cv2.findContours(blob["mask"], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(discard_vis, contours, -1, (255, 50, 50), 2)
            cx, cy = int(blob["centroid"][0]), int(blob["centroid"][1])
            cv2.putText(discard_vis, blob.get("discard_reason", "?"),
                        (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 50, 50), 1)

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.imshow(discard_vis)
        ax.set_title(f"({len(result['discarded_blobs'])}) Discarded Blobs")
        ax.axis("off")
        discard_path = dst_path.replace(".png", "_discarded.png")
        plt.savefig(discard_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        print("no blobs discarded")

# Overlay display
    overlay = img_rgb.copy()
    overlay[result["filtered_mask"] == 255] = (
        overlay[result["filtered_mask"] == 255] * 0.5 + np.array([0, 200, 80]) * 0.5
    ).astype("uint8")

#merged all for display
    fig, axes = plt.subplots(1, 4, figsize=(22, 6))
    fig.suptitle(f"Plant Bounding Results")

    axes[0].imshow(img_rgb)
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(result["seed_mask"], cmap="gray")
    axes[1].set_title("Simple Blob Detection")
    axes[1].axis("off")

    axes[2].imshow(result["grown_mask"], cmap="gray")
    axes[2].set_title("After Region Grow")
    axes[2].axis("off")

    axes[3].imshow(overlay)
    axes[3].set_title("After Filtering")
    axes[3].axis("off")

    plt.tight_layout()
    plt.savefig(dst_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved to: {dst_path}")

IMG_DIR = Path("tests/testimages")

def run_pipeline_on_image(src_path, dst_path, color_bounds, color_tolerance=22):
    #Runs the full pipeline on an image
    img = cv2.imread(str(src_path))
    if img is None:
        raise FileNotFoundError(f"Missing {src_path}.")

    img_rgb   = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    blob_list = pf.find_green_blobs(img, color_bounds)

    if len(blob_list) == 0:
        print(f"no blobs found")
        return

    result = pf.grow_plant_mask(img, blob_list, color_tolerance=color_tolerance)

    overlay = img_rgb.copy()
    overlay[result["filtered_mask"] == 255] = (
        overlay[result["filtered_mask"] == 255] * 0.5 + np.array([0, 200, 80]) * 0.5
    ).astype("uint8")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f"{Path(src_path).name}  Plant Detection Results")

    axes[0].imshow(img_rgb)
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(overlay)
    axes[1].set_title("Filtered mask overlay")
    axes[1].axis("off")

    plt.tight_layout()
    plt.savefig(str(dst_path), dpi=150, bbox_inches="tight")
    plt.close(fig)

'''def test_more():
    test_cases = [
        (
            "badmint.jpg",
            ((31, 50, 50), (75, 255, 200)),  # green bounds
            22,
        ),
        (
            "lettuce_8.jpg",
            ((31, 50, 50), (75, 255, 200)), 
            25,
        ),
    ]

    for filename, color_bounds, tolerance in test_cases:
        src = IMG_DIR / filename
        dst = IMG_DIR / filename.replace(".jpg", "_grown_out.png")
        run_pipeline_on_image(src, dst, color_bounds, color_tolerance=tolerance)'''

if __name__ == "__main__":
    test_blobber()
    test_region_grow()
    '''test_more()'''