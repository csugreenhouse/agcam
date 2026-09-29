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

def test_blobber(src_path):
    src_path = Path(src_path)
    dst_path = Path(str(src_path).replace(".jpg", "_blobber_out.jpg"))

    img = cv2.imread(str(src_path))
    if img is None:
        raise FileNotFoundError(f"Src path missing {src_path}.")

    blob_list = pf.find_green_blobs(img)
    assert len(blob_list) != 0, "No blobs found"
    graph_util.plot_blobs(img, dst_path, blob_list)
    print(f"blob test saved to: {dst_path}")

def test_pipeline_on_image(src_path):
    src_path = Path(src_path)
    dst_path = Path(str(src_path).replace(".jpg", "_pipeline_out.jpg"))

    img = cv2.imread(str(src_path))

    if img is None:
        raise FileNotFoundError(f"Missing {src_path}.")

    img_rgb   = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    result = pf.find_plants(img)

    # END PIPELINE, MAKE DISPLAYS #

    #display pipeline functionality by stage
    fig, axes = plt.subplots(1, 4, figsize=(25, 6))
    fig.suptitle(f"Plant Pipeline Stage Outputs")

    axes[0].imshow(img_rgb)
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    blob_overlay = img_rgb.copy()
    blob_overlay[result["seed_mask"] == 255] = (
        blob_overlay[result["seed_mask"] == 255] * 0.5 + np.array([0, 0, 200]) * 0.5
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
    print(f"intermediate stages display saved to: {stage_path}")

    #final simplified output image, shows original pic and final plant mask.
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
    test_blobber("tests/testimages/IMG_3387.jpg")
    test_pipeline_on_image("tests/testimages/IMG_3387.jpg")

    test_blobber("tests/testimages/IMG_3398.jpg")
    test_pipeline_on_image("tests/testimages/IMG_3398.jpg")