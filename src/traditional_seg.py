import numpy as np
import torch
import random
import skimage
import cv2 as cv
import copy
from glob import glob
from PIL import Image
from matplotlib import pyplot as plt
from skimage.segmentation import felzenszwalb, slic, morphological_chan_vese 
from skimage import filters
from skimage.color.adapt_rgb import adapt_rgb, each_channel, hsv_value
from skimage.exposure import rescale_intensity

def calculate_luminance(targets, img):

    luminance_matrix = 0.299 * img[:, :, 2] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 0]

    average_luminance = np.mean(luminance_matrix)


    luminance_targets = []
    for j, boxe in enumerate(targets["boxes"]):

        x_min, y_min, x_max, y_max = int(boxe[0].item()), int(boxe[1].item()), int(boxe[2].item()), int(boxe[3].item())
        roi = img[y_min:y_max, x_min:x_max]

        luminance_roi = 0.299 * roi[:, :, 2] + 0.587 * roi[:, :, 1] + 0.114 * roi[:, :, 0]

        if roi.shape[0] == 0 or roi.shape[1] == 0:
            luminance_targets.append(-1)
            continue

        luminance_targets.append(np.mean(luminance_roi)/average_luminance)

    return luminance_targets


def calculate_standard_deviation(targets, img, gray):

    std_targets = []
    for j, boxe in enumerate(targets["boxes"]):

        x_min, y_min, x_max, y_max = int(boxe[0].item()), int(boxe[1].item()), int(boxe[2].item()), int(boxe[3].item())
        roi_image = gray[y_min:y_max, x_min:x_max]

        if roi_image.shape[0] == 0 or roi_image.shape[1] == 0:
            std_targets.append(-1)
            continue

        std_targets.append(np.std(roi_image))

    return std_targets

def calculate_edge_density(targets, img, edge_img, inner_factor, display=False):

    edge_density_targets = []
    for j, boxe in enumerate(targets["boxes"]):
 
        x_min, y_min, x_max, y_max = int(boxe[0].item()), int(boxe[1].item()), int(boxe[2].item()), int(boxe[3].item())
        roi_image = img[y_min:y_max, x_min:x_max]

        if roi_image.shape[0] == 0 or roi_image.shape[1] == 0 or roi_image.shape[2] == 0 :
            print("[ERROR] WE HAVE an empty boxe .... ", img.shape, roi_image.shape, boxe)
            edge_density_targets.append(-1)
            continue

        # Calculate edge
        edge_roi = cv.Canny(roi_image, 100, 200)

        if display:
            fig = plt.figure()
            a = fig.add_subplot(3, 2, 1)
            plt.imshow(img)
            a = fig.add_subplot(3, 2, 2)
            plt.imshow(edge_img)
            a = fig.add_subplot(3, 2, 3)
            plt.imshow(roi_image)
            a = fig.add_subplot(3, 2, 4)
            plt.imshow(edge_roi)

        # Remove center edge
        x_distance = x_max - x_min
        y_distance = y_max - y_min
        x_padding = max(1, int(x_distance * inner_factor))
        y_padding = max(1, int(y_distance * inner_factor))
        edge_roi[y_padding:-y_padding, x_padding:-x_padding] = 0


        edge_density = edge_roi.sum() / (x_distance * 2) + (y_distance * 2)
        edge_density_targets.append(edge_density)


        if display:
            print(edge_density)
            a = fig.add_subplot(3, 2, 5)
            plt.imshow(edge_roi)
            
            plt.draw()
            plt.waitforbuttonpress(0)
            plt.close(fig)


    return edge_density_targets 
           

def calculate_color_contrast(targets, img, hist_img, ring_factor, display=False):

    color_contrast_targets = []
    for j, boxe in enumerate(targets["boxes"]):
 
        x_min, y_min, x_max, y_max = int(boxe[0].item()), int(boxe[1].item()), int(boxe[2].item()), int(boxe[3].item())
        roi_image = img[y_min:y_max, x_min:x_max]

        if roi_image.shape[0] == 0 or roi_image.shape[1] == 0 or roi_image.shape[2] == 0 :
            print("[ERROR] WE HAVE an empty boxe .... ", img.shape, roi_image.shape, boxe)
            color_contrast_targets.append(-1)
            continue

        hist_roi = compute_lab_hist(roi_image)


        x_distance = x_max - x_min
        y_distance = y_max - y_min
        x_padding = max(1, int(x_distance * ring_factor))
        y_padding = max(1, int(y_distance * ring_factor))

        y_min_expend = max(0 , y_min - y_padding)
        x_min_expend = max(0 , x_min - x_padding)
        x_max_expend = min(img.shape[1], x_max + x_padding)
        y_max_expend = min(img.shape[0], y_max + y_padding)

        expend_roi_image = img[y_min_expend:y_max_expend, x_min_expend:x_max_expend]
        hist_expend = compute_lab_hist(expend_roi_image)

        ring_image = copy.deepcopy(expend_roi_image)
        ring_image[y_padding:-y_padding, x_padding:-x_padding] = 0

        hist_ring = [hist_expend[0] - hist_roi[0], hist_expend[1] - hist_roi[1], hist_expend[2] - hist_roi[2]]

        hist_roi[0] /= hist_roi[0].sum()
        hist_roi[1] /= hist_roi[1].sum()
        hist_roi[2] /= hist_roi[2].sum()
        hist_ring[0] /= hist_ring[0].sum()
        hist_ring[1] /= hist_ring[1].sum()
        hist_ring[2] /= hist_ring[2].sum()
        hist_expend[0] /= hist_expend[0].sum()
        hist_expend[1] /= hist_expend[1].sum()
        hist_expend[2] /= hist_expend[2].sum()

        ##cv.HISTCMP_INTERSECT))#cv.HISTCMP_CORREL))#cv.HISTCMP_CHISQR))
        color_contrast = cv.compareHist(hist_roi[0], hist_ring[0], cv.HISTCMP_CHISQR)
        color_contrast += cv.compareHist(hist_roi[1], hist_ring[1], cv.HISTCMP_CHISQR)
        color_contrast += cv.compareHist(hist_roi[2], hist_ring[2], cv.HISTCMP_CHISQR)
        color_contrast_targets.append(color_contrast)

        if display:
            fig = plt.figure()
            a = fig.add_subplot(4, 2, 1)
            plt.imshow(img)
            a = fig.add_subplot(4, 2, 2)
            plt.plot(hist_img[0])
            a = fig.add_subplot(4, 2, 3)
            plt.imshow(roi_image)
            a = fig.add_subplot(4, 2, 4)
            plt.plot(hist_roi[0])
            a = fig.add_subplot(4, 2, 5)
            plt.imshow(expend_roi_image)
            a = fig.add_subplot(4, 2, 6)
            plt.plot(hist_expend[0])
            a = fig.add_subplot(4, 2, 7)
            plt.imshow(ring_image)
            a = fig.add_subplot(4, 2, 8)
            plt.plot(hist_ring[0])

            plt.draw()
            plt.waitforbuttonpress(0)
            plt.close(fig)

    return color_contrast_targets 
            


@adapt_rgb(each_channel)
def sobel_each(image):
    return filters.sobel(image)


@adapt_rgb(hsv_value)
def sobel_hsv(image):
    return filters.sobel(image)

def compute_lab_hist(image, display=False):

    lab_image = cv.cvtColor(image, cv.COLOR_BGR2LAB)

    if display:
        plt.imshow(image)
        plt.show()
        plt.imshow(lab_image)
        plt.show()

    #print(lab_image.shape)

    hist_l = cv.calcHist([lab_image], [0], None, [100], [0, 100])
    hist_a = cv.calcHist([lab_image], [1], None, [256], [-127, 127])
    hist_b = cv.calcHist([lab_image], [2], None, [256], [-127, 127])

    if display:
        plt.plot(hist_l, label="hist_l")
        plt.legend()
        plt.show()
        plt.plot(hist_a, label="hist_a")
        plt.legend()
        plt.show()
        plt.plot(hist_b, label="hist_b")
        plt.legend()
        plt.show()

    return [hist_l, hist_a, hist_b]


def compute_gray_hist(image, display=False):

    gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    hist = cv.calcHist([gray_image], [0], None, [256], [0, 256])

    if display:
        plt.figure()
        plt.title("Grayscale Histogram")
        plt.xlabel("Bins")
        plt.ylabel("# of Pixels")
        plt.plot(hist)
        plt.xlim([0, 256])
        plt.show()

    return hist



if __name__ == "__main__":

    from skimage.feature import hog
    from skimage import exposure
    from skimage import io
    from skimage.segmentation import slic
    from skimage.color import label2rgb
    import random
    import pywt

    image_files = glob('../data/bdd100k/train/images/*jpg')

    for i in range(10):


        image = cv.imread(image_files[i+25])

        segments1 = slic(image, n_segments=100, compactness=1)
        segmentation1 = label2rgb(segments1, image, kind='avg')
        segments2 = slic(image, n_segments=100, compactness=100)
        segmentation2 = label2rgb(segments2, image, kind='avg')

        # Définissez les dimensions de la région de l'intérêt (ROI)
        roi_width = 100  # Largeur de la ROI
        roi_height = 100  # Hauteur de la ROI

        # Sélectionnez une position aléatoire pour la ROI
        max_x = image.shape[1] - roi_width
        max_y = image.shape[0] - roi_height

        plt.subplot(5, 2, 1)
        plt.imshow(image)
        std_deviation = np.std(image)

        for i in range(4):

            x = random.randint(0, max_x)
            y = random.randint(0, max_y)

            roi = image[y:y+roi_height, x:x+roi_width]

            plt.subplot(3, 2, 2 + i)
            
            segments = slic(roi, n_segments=10, compactness=10)
            segmentation = label2rgb(segments, roi, kind='avg')
            plt.imshow(segmentation)

            plt.subplot(3, 2, 2 + i)

        plt.show()
        continue


        plt.figure(figsize=(10, 5))
        plt.subplot(221), plt.imshow(cv.cvtColor(image, cv.COLOR_BGR2RGB))
        plt.title('Image d\'origine')
        plt.subplot(222), plt.imshow(segmentation)
        plt.title('Superpixels')
        plt.subplot(223), plt.imshow(segmentation1)
        plt.subplot(224), plt.imshow(segmentation2)
        plt.show()
        

        continue

        # Read in the image
        image = cv.imread(image_files[i+25], cv.IMREAD_GRAYSCALE)

        plt.subplot(3, 2, 1)
        plt.imshow(image)
        std_deviation = np.std(image)
        plt.title("écart type : " + str(std_deviation))

        # Définissez les dimensions de la région de l'intérêt (ROI)
        roi_width = 100  # Largeur de la ROI
        roi_height = 100  # Hauteur de la ROI

        # Sélectionnez une position aléatoire pour la ROI
        max_x = image.shape[1] - roi_width
        max_y = image.shape[0] - roi_height


        for i in range(4):

            x = random.randint(0, max_x)
            y = random.randint(0, max_y)

            roi = image[y:y+roi_height, x:x+roi_width]

            plt.subplot(3, 2, 2 + i)
            plt.imshow(roi)
            # Calculez l'écart-type
            std_deviation = np.std(roi)

            # Appliquez une transformation en ondelettes (par exemple, la transformée de Haar)
            coeffs = pywt.wavedec2(roi, 'haar')

            # La première valeur dans coeffs est la matrice d'approximation (cA), tandis que les autres sont les matrices de détails (cD)
            cA, cD = coeffs[0], coeffs[1:]

            # Calculez l'écart-type des coefficients de détail
            std_deviation_more = [np.std(d) for d in cD]

            # Vous pouvez également calculer la moyenne des écarts-types des coefficients de détail
            mean_std_deviation = np.mean(std_deviation)

            print(mean_std_deviation)
            plt.title("écart type : " + str(std_deviation) + " mean : " + str(mean_std_deviation))

            print("L'écart-type du bruit dans la ROI aléatoire est : ", std_deviation, mean_std_deviation)

        plt.show()
        continue

        # Read in the image
        image = cv.imread(image_files[i+15])

        # Make a copy of the image
        image_copy = np.copy(image)

        # Change color to RGB (from BGR)
        image_copy = cv.cvtColor(image_copy, cv.COLOR_BGR2RGB)

        # Convert to grayscale
        gray = cv.cvtColor(image_copy, cv.COLOR_RGB2GRAY)
        gray = np.float32(gray)

        # Detect corners
        dst = cv.cornerHarris(gray, 2, 3, 0.04)

        # Dilate corner image to enhance corner points
        dst = cv.dilate(dst,None)

        plt.imshow(dst, cmap='gray')
        # This value vary depending on the image and how many corners you want to detect
        # Try changing this free parameter, 0.1, to be larger or smaller and see what happens
        thresh = 0.1*dst.max()

        # Create an image copy to draw corners on
        corner_image = np.copy(image_copy)

        # Iterate through all the corners and draw them on the image (if they pass the threshold)
        for j in range(0, dst.shape[0]):
            for i in range(0, dst.shape[1]):
                if(dst[j,i] > thresh):
                    # image, center pt, radius, color, thickness
                    cv.circle( corner_image, (i, j), 1, (0,255,0), 1)

        plt.imshow(corner_image)
        plt.show()


        continue

        # Load your image
        image = io.imread(image_files[i+15])

        # Specify the channel_axis for a color image (typically, the channel axis is 2 for RGB images)
        channel_axis = 2

        # Compute HOG features
        fd, hog_image = hog(image, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True, multichannel=True, channel_axis=channel_axis)


        # Rescale HOG features for better visualization
        hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))

        # Display the original image and the HOG representation
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
        ax1.axis('off')
        ax1.set_title('Original Image')
        ax1.imshow(image, cmap=plt.cm.gray)

        ax2.axis('off')
        ax2.set_title('HOG Features')
        ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)

        print(np.mean(hog_image_rescaled))

        plt.show()


        image = cv.imread(image_files[i+15])

        # Convertir l'image en niveaux de gris (HOG fonctionne généralement mieux avec des images en niveaux de gris)
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

        print("sobel")
        # Calculer les gradients de l'image
        gradient_x = cv.Sobel(gray, cv.CV_32F, 1, 0, ksize=1)
        gradient_y = cv.Sobel(gray, cv.CV_32F, 0, 1, ksize=1)

        # Calculer la magnitude et la direction des gradients
        magnitude, direction = cv.cartToPolar(gradient_x, gradient_y)

        # Display the magnitude of the gradients
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 4, 1)
        plt.imshow(image, cmap='gray')
        plt.title('Gradient Magnitude')


        plt.subplot(1, 4, 2)
        plt.imshow(magnitude, cmap='gray')
        plt.title('Gradient Magnitude')

        # Display the direction of the gradients
        plt.subplot(1, 4, 3)
        plt.imshow(direction, cmap='hsv')
        plt.title('Gradient Direction')


        # Create an HOG object
        cv_hog = cv.HOGDescriptor()

        # Calculate the HOG features
        features = cv_hog.compute(gray)

        # Reshape the HOG features into a format that can be displayed as an image
        print(image.shape, features.shape, gray.shape)
        #hog_image = features.reshape((gray.shape[0], gray.shape[1]))
        # Reshape the HOG features into a format that can be displayed as an image
        #hog_image = features.reshape((gray.shape[0] // 8, 8, gray.shape[1] // 8, 8, 9))
        #hog_image = hog_image.transpose((1, 0, 3, 2, 4)).reshape((gray.shape[0] // 8 * 8, gray.shape[1] // 8 * 8, 9))


        # Normalize the HOG image for better visualization
        hog_image = cv.normalize(features, None, 0, 255, cv.NORM_MINMAX)

        # Convert the HOG image to an 8-bit image
        hog_image = hog_image.astype('uint8')

        # Display the HOG descriptors as an image
        plt.subplot(1, 4, 4)
        plt.imshow(hog_image, cmap='gray')
        plt.title('HOG Descriptors')
        plt.show()
        continue


        print(len(histogram))
        print("plot")
        # Afficher l'histogramme
        plt.bar(np.arange(len(histogram[::1000])), histogram[::1000])
        plt.show()

        print("end plot")
        continue

        edges = cv.Canny(image,100,200)
        plt.subplot(121),plt.imshow(image,cmap = 'gray')
        plt.title('Original Image'), plt.xticks([]), plt.yticks([])
        plt.subplot(122),plt.imshow(edges,cmap = 'gray')
        plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
        plt.show()

        continue


        compute_lab_hist(image)

        continue
        image = np.array(Image.open(image_files[random.randint(0, len(image_files))]))

        segmented_image = felzenszwalb(image, scale=200, min_size=5)#, sigma=0.5, min_size=50)
        edge_image = sobel_each(image)
        segmented_edge_image = felzenszwalb(edge_image, sigma=0, scale=200)

        seg_img =  morphological_chan_vese(image, num_iter=15)#, n_segments=20000, compactness=0.001)
        seg_img_more = felzenszwalb(seg_img, sigma=0.8, scale=200)


        fig = plt.figure()
        a = fig.add_subplot(3, 2, 1)
        plt.imshow(image)
        a = fig.add_subplot(3, 2, 2)
        plt.imshow(segmented_image.astype(np.uint8))
        a = fig.add_subplot(3, 2, 3)
        plt.imshow(rescale_intensity(edge_image))
        a = fig.add_subplot(3, 2, 4)
        plt.imshow(segmented_edge_image)
        a = fig.add_subplot(3, 2, 5)
        plt.imshow(rescale_intensity(seg_img))
        a = fig.add_subplot(3, 2, 6)
        plt.imshow(seg_img_more)
        plt.draw()
        plt.waitforbuttonpress(0)
        plt.close(fig)
