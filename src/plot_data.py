import math
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.text as text
import matplotlib.patheffects as patheffects

#import numpy as np
#from PIL import Image

def fig2rgb_array(fig):
    fig.canvas.draw()
    buf = fig.canvas.tostring_rgb()
    ncols, nrows = fig.canvas.get_width_height()
    return np.frombuffer(buf, dtype=np.uint8).reshape(nrows, ncols, 3)


"""

Function to see targets on an image
Inputs :
    - image
    - targets : Dict {"boxes" : list of boxes, size (N, 4) with [x1, y1, x2, y2]; "labels": list of labels, size (N)

"""

def make_image_labels(image, targets):

    image = image.cpu().permute(1, 2, 0)
    fig, ax = plt.subplots()
    ax.imshow(image)

    boxes = targets["boxes"].cpu()
    labels= targets["labels"]


    # Draw each box and label on the image
    for i in range(len(labels)):

        x1, y1, x2, y2 = boxes[i]
        ax.add_patch(patches.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor='red', lw=2))
        ax.text(x1, y2, str(labels[i]), color='white', fontsize=10, weight='bold', path_effects=[patheffects.withStroke(linewidth=5, foreground='black')])
        ax.set_axis_off()

    fig.tight_layout()

    return fig2rgb_array(fig)

def make_image_labels_without_cpu(image, targets):

    image = image.permute(1, 2, 0)
    fig, ax = plt.subplots()
    ax.imshow(image)

    boxes = targets["boxes"]
    labels= targets["labels"]


    # Draw each box and label on the image
    for i in range(len(labels)):

        x1, y1, x2, y2 = boxes[i]
        ax.add_patch(patches.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor='red', lw=2))
        ax.text(x1, y2, str(labels[i]), color='white', fontsize=10, weight='bold', path_effects=[patheffects.withStroke(linewidth=5, foreground='black')])
        ax.set_axis_off()

    fig.tight_layout()

    return fig2rgb_array(fig)


def concatenate_images(images):

    size = math.ceil(len(images) / 2)
    fig = plt.figure()

    x, y = 0, 0
    for i in range(len(images)): 
        
        if x >= size:
            x = 0
            y += 1

        fig.add_subplot(size, size, i + 1)
        plt.imshow(images[i][0])
        plt.axis('off')

        x += 1

    fig.subplots_adjust(wspace=0, hspace=0)

    return fig2rgb_array(fig)

