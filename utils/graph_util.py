import cv2
import numpy as np
import matplotlib.pyplot as plt
import warnings
import numpy as np
import matplotlib.patches as patches
import importlib
import sys
from pathlib import Path

sys.path.append('/mnt/db/agcam')

from matplotlib.lines import Line2D

line_util = importlib.import_module("utils.line_util")
scanner_util = importlib.import_module("utils.reference_tag_util")
plant_finder = importlib.import_module("plant_finders.legacyHSV_plant_finder")
height_request = importlib.import_module("processing.height_request")
openCV_util = importlib.import_module("utils.openCV_util")

color_pallet = ['cyan', 'yellow', 'magenta', 'orange', 'pink', 'lime', 'aqua']

#####################################
# REFERENCE TAG GRAPHING UTILITIES
#####################################

def add_tag(ax, tag, color='cyan', center_size=20):
    try:
        corners = tag["corners"]
        tl = corners["top_left"]
        tr = corners["top_right"]
        br = corners["bottom_right"]
        bl = corners["bottom_left"]
    except KeyError:
        warnings.warn("Tag corners not found, cannot plot tag.")
        return
    ax.add_patch(plt.Polygon([tl, tr, br, bl], fill=None, edgecolor=color, linewidth=2))
    cx, cy = tag["center"]
    ax.add_patch(plt.Circle((cx, cy), center_size, color=color, fill=True))
    ax.text(tag["center"][0], tag["center"][1], str(tag["data"]), fontsize=8, ha='center', va='center')
    ax.text(tl[0], tl[1], "TL", fontsize=8, ha='center', va='center')
    ax.text(tr[0], tr[1], "TR", fontsize=8, ha='center', va='center')
    ax.text(br[0], br[1], "BR", fontsize=8, ha='center', va='center')
    ax.text(bl[0], bl[1], "BL", fontsize=8, ha='center', va='center')
    #ax.plot([], [], color=color, label=f'Tag ID: {tag["data"]}')

def plot_reference_tag(image,out_path, reference_tag):
    graph_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    W,H = graph_rgb.shape[1], graph_rgb.shape[0]
    fig, ax = plt.subplots()
    ax.imshow(graph_rgb)
    ax.axis('on')

    add_tag(ax,reference_tag)

    ax.set_title("REFERENCE TAG")

    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),ncol=2)
    plt.savefig(str(out_path), bbox_inches="tight")
    plt.close(fig)

#####################################
# GREEN BLOB GRAPHING UTILITIES
#####################################

def plot_contour_and_mask(src_path, contours, mask, show_mask = True):
    print("plotting contour and mask")
    src_path = Path(src_path)
    out_path = Path(str(src_path).replace(".jpg", "_blobplot_out.jpg"))

    #read image & get sizes
    image = cv2.imread(str(src_path))    
    graph_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    W,H = graph_rgb.shape[1], graph_rgb.shape[0]

    #setup graph with img bg
    fig, ax = plt.subplots()
    ax.imshow(graph_rgb)
    ax.axis('on')
    color = 'cyan'

    try:
        if(show_mask):
            add_mask_to_plot(ax,mask)
        #add_contours_to_plot(ax,contours,color)
        
        plt.savefig(str(out_path), bbox_inches="tight")
        plt.close(fig)
        print(f"Contour and mask successfully plotted, saved to: {out_path}")
    except Exception as e:
        plt.savefig(str(out_path), bbox_inches="tight")
        plt.close(fig)
        print(f"No blobs plotted, returning blank image at {out_path}. Exception: {e}")

def add_mask_to_plot(ax, mask, cmap='bwr'):
    invertedmask = cv2.bitwise_not(mask)
    ax.imshow(invertedmask, cmap = cmap, alpha = 0.25)

#this method has a known error for parsing answer keys, however it is non fatal.
# this method is not being used now in a way that breaks anything so it is not being fixed urgently
def add_contours_to_plot(ax, contours, color='cyan'):
    for contour in contours:
        contour = contour.squeeze()
        ax.plot(contour[:, 0], contour[:, 1], color=color, linewidth=2)

######################################
# BASIC GRAPHING UTILITIES
###################################

def add_point(ax, point, color='blue', size=10):
    if point == None:
        return
    x, y = point
    ax.add_patch(plt.Circle((x, y), size, color=color, fill=True))

def add_line(ax, equation, color='yellow', linestyle='--', label=None):
    slope, intercept = equation
    x_vals = np.array(ax.get_xlim())
    if slope == float('inf'):
        x_line = np.full_like(x_vals, intercept)
        y_line = np.array(ax.get_ylim())
    else:
        y_vals = slope * x_vals + intercept
        x_line = x_vals
        y_line = y_vals
    ax.plot(x_line, y_line, color=color, linestyle=linestyle, label=label)

#####################################
# FULL GRAPHING UTILITIES
#####################################

def plot_height(image, out_path):
    #graph setup.
    graph_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    W,H = graph_rgb.shape[1], graph_rgb.shape[0]

    fig = plt.figure(figsize=(W/500,H/500))
    ax = fig.add_axes([0, 0, 1, 1])

    color = color_pallet[0]

    #graph add details.
    try:
        reference_tags = scanner_util.scan_reference_tags(image) #find tags in image
        result = plant_finder.find_plants(image) #find plant matter

        contours = openCV_util.get_contours_from_mask(result)

        if len(reference_tags) != 0 and len(result) != 0: #common case-tag and plant present
            reference_tag = reference_tags[0]
            tag_id = int(reference_tag['data'])
            add_tag(ax,reference_tag)
            add_contours_to_plot(ax, contours, color)

            view_response = height_request.estimate_heights_reference_tag(image, reference_tag)
            estimated_height = float(view_response[0]["estimated_height"])

        if len(reference_tags) != 0 and len(result) == 0:
            reference_tag = reference_tags[0]
            tag_id = int(reference_tag['data'])
            add_tag(ax, reference_tag)
            estimated_height = None
        
        if len(reference_tags) == 0 and len(result) != 0:
            tag_id = None
            add_contours_to_plot(ax, contours, color)
            estimated_height = None
        
        if len(reference_tags) == 0 and len(result) == 0: #if there is nothing, we will return the input image with nothing plotted
            tag_id = None
            estimated_height = None

    except Exception as e:
        print(f"\033[91mError occurred while plotting height: {e}\033[00m")
        tag_id = None
        estimated_height = None
        
        #graph styling and output
    plt.axis('off')
    props = dict(boxstyle='round', facecolor='white', alpha=0.75)
    if estimated_height is not None:
        ax.text(0.05, 0.95, f"Estimated Height: {round(estimated_height*100,1)}cm", transform=ax.transAxes, fontsize=14, verticalalignment='top', bbox =props)

    #ax.plot([], [], color=color, label=
     #       f"Estimated Height: {round(estimated_height*100,2)}cm \n")
    ax.imshow(graph_rgb)
    #ax.axis('on')
    #ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),ncol=1)
    
    plt.savefig(str(out_path), bbox_inches="tight", pad_inches=0)
    plt.close(fig)

    #returns for database entries.
    return (tag_id, estimated_height, out_path)