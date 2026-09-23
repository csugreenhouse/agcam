import cv2
import numpy as np
import matplotlib.pyplot as plt
import warnings
import numpy as np
import matplotlib.patches as patches
from matplotlib.lines import Line2D
from utils.line_util import get_equation_of_line, get_vertical_line, fractional_height_between_lines, get_intercept_of_lines

color_pallet = ['cyan', 'yellow', 'magenta', 'orange', 'pink', 'lime', 'aqua']

# PLOT THE REFERENCE TAG

'''def plot_reference_tag(image,out_path, reference_tag):
    graph_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    W,H = graph_rgb.shape[1], graph_rgb.shape[0]
    fig, ax = plt.subplots()
    ax.imshow(graph_rgb)
    ax.axis('on')

    add_tag(ax,reference_tag)
    views = reference_tag["views"]

    for i, view in enumerate(views):
        add_view(ax,W,H,view,facecolor=color_pallet[i])
 
    ax.set_title("REFERENCE TAG")

    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),ncol=2)
    plt.savefig(str(out_path), bbox_inches="tight")
    plt.close(fig)'''
    
#PLOT THE estimate_heights_reference_tags response from height_request
    
'''def plot_height_request_response(image, out_path, response):
    graph_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    W,H = graph_rgb.shape[1], graph_rgb.shape[0]
    fig, ax = plt.subplots()
    
    for i, view_response in enumerate(response):
        add_tag(ax,view_response["reference_tag"], color=color_pallet[i%len(color_pallet)])
        color = color_pallet[i%len(color_pallet)]
        plant_id = view_response["plant_id"]
        estimated_height = view_response["estimated_height"]
        green_blob_list = view_response["green_blob_list"]
        tag_bias = view_response["bias_units_m"]
        add_point(ax,view_response["heighest_green_pixel"],color="green")
        add_plant_bounds(ax,W,H,view_response["plant_bounds"],color=color)
        add_green_blobs(ax,green_blob_list,color)
        ax.plot([], [], color=color, label=
                f"plant {plant_id}: {round(estimated_height*100,2)}cm FH {round(view_response['fractional_height']*100,2)}cm \n"
                  f"bias: {round(tag_bias*100,2)}cm color bounds: {view_response['color_bounds'][0]},{view_response['color_bounds'][1]}\n")
    
    ax.imshow(graph_rgb)
    ax.axis('on')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),ncol=1)
    plt.savefig(str(out_path), bbox_inches="tight")
    plt.close(fig)
    '''
'''
def plot_widths_request_response(image, out_path, response):
    graph_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    W,H = graph_rgb.shape[1], graph_rgb.shape[0]
    fig, ax = plt.subplots()
    ax.imshow(graph_rgb)
    ax.axis('on')
    
    for i, view_response in enumerate(response):
        add_tag(ax,view_response["reference_tag"], color=color_pallet[i%len(color_pallet)])#outline tag
        color = color_pallet[i%len(color_pallet)]
        plant_id = view_response["plant_id"]
        estimated_width = view_response["estimated_width"]
        green_blob_list = view_response["green_blob_list"]
        add_point(ax, view_response["leftmost_green_pixel"], color="blue")
        add_point(ax, view_response["rightmost_green_pixel"], color="red")
        add_plant_bounds(ax,W,H,view_response["plant_bounds"],color=color)
        add_green_blobs(ax,green_blob_list,color)
        ax.plot([], [], color=color, label=f"plant {plant_id}: {round(estimated_width*100,2)}cm FH {round(view_response['fractional_width']*100,2)}cm")
        
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),ncol=2)

    plt.savefig(str(out_path), bbox_inches="tight")
    plt.close(fig)
    '''

#plot blobs alone
def plot_blobs(image, out_path, blobs):
    graph_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    W,H = graph_rgb.shape[1], graph_rgb.shape[0]
    fig, ax = plt.subplots()
    ax.imshow(graph_rgb)
    ax.axis('on')

    color = color_pallet[1%len(color_pallet)]#choose color pallett

    add_green_blobs(ax,blobs,color)#draws green blobs around leaves
        
    #ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),ncol=2)

    plt.savefig(str(out_path), bbox_inches="tight")
    plt.close(fig)

    '''
    
def plot_calculated_displacements_graph_info(image, out_path, reference_tag):
    graph_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    W,H = graph_rgb.shape[1], graph_rgb.shape[0]
    fig, ax = plt.subplots()
    ax.imshow(graph_rgb)
    ax.axis('on')

    add_tag_displacement_relative_to_camera(ax, W, H, reference_tag)
    
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),ncol=2)
    plt.savefig(str(out_path), bbox_inches="tight")
    plt.close(fig)
    '''

    '''

def add_tag_displacement_relative_to_camera(ax, W, H, reference_tag, color=None):
    if color is None:
        color = ['red', 'green', 'blue']
    else:
        color = [color]*3
        
    displacement_d = reference_tag['displacements']['d']
    displacement_z = reference_tag['displacements']['z']
    displacement_x = reference_tag['displacements']['x']
    displacement_y = reference_tag['displacements']['y']
    center_x_image = W / 2
    center_y_image = H / 2
    center_x_ref = reference_tag['center'][0]
    center_y_ref = reference_tag['center'][1]

    ax.plot([center_x_image, center_x_ref], [center_y_image, center_y_ref], color=color[0], linewidth=2)
    ax.plot([center_x_image, center_x_ref], [center_y_ref, center_y_ref], color=color[1], linewidth=2)
    ax.plot([center_x_ref, center_x_ref], [center_y_image, center_y_ref], color=color[2], linewidth=2)

    ax.plot([], [], color=color[0], label='Depth Displacement(z)\n %.6f cm' % (displacement_z*100))
    ax.plot([], [], color=color[1], label='Horizontal Displacement (x)\n %.6f cm' % (displacement_x*100))
    ax.plot([], [], color=color[2], label='Vertical Displacement (y)\n %.6f cm' % (displacement_y*100))
    ax.plot([], [], color='black', label='Distance to QR Code (d)\n %.6f cm' % (displacement_d*100))
    '''
    '''

def add_camera_view_frustum(ax, heighest_green_pixel,camera_parameters, reference_tag, color='yellow'):
    width = camera_parameters["width"]
    height = camera_parameters["height"]
    fx = (camera_parameters["focal_length_mm"] / camera_parameters["sensor_width_mm"]) * width
    fy = (camera_parameters["focal_length_mm"] / camera_parameters["sensor_height_mm"]) * height
    displacement_z = reference_tag['displacements']['z']
    displacement_x = reference_tag['displacements']['x']
    displacement_y = reference_tag['displacements']['y']
    # find the difference between the x cordinate of the tallest pixel and the x cordinate of the center of the reference tag
    difference_x = reference_tag['center'][0] - heighest_green_pixel[0]
    point_x_camera = (displacement_z * (fx / displacement_z)) + (width / 2) - difference_x
    point_y_camera = height/2

    m, b = get_equation_of_line(heighest_green_pixel, (point_x_camera, point_y_camera))
    y_point = m * width + b 
    x_point = width - 1

    ax.plot([heighest_green_pixel[0], x_point], [heighest_green_pixel[1], y_point], color=color, linewidth=1)
    ax.plot([], [], color=color, label='Camera View to Heighest Plant Pixel \n (May indicate occlusion or steep angle)')
    '''
    '''

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
    ax.plot([], [], color=color, label=f'Tag ID: {tag["data"]}')
    '''

def add_green_blobs(ax, plant_blob_list, color='lime'):
    for blob in plant_blob_list:
        contours, _ = cv2.findContours(blob["mask"], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            contour = contour.squeeze()
            ax.plot(contour[:, 0], contour[:, 1], color=color, linewidth=2)

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
   
    


    