import numpy as np
from skimage.transform import hough_line, hough_line_peaks
from skimage.transform import rotate
from skimage.feature import canny
from skimage.io import imread
from skimage.color import rgb2gray
import matplotlib.pyplot as plt
from scipy.stats import mode
import cv2
import matplotlib.pyplot as plt

def display_image(image, title="Image"):
    # Check if the image is grayscale or color
    if len(image.shape) == 3 and image.shape[2] == 3:
        # Convert BGR to RGB for color images
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        plt.imshow(image_rgb)
    else:
        # Grayscale image
        plt.imshow(image, cmap='gray')
    
    plt.title(title)
    plt.axis('off')  # Turn off axis numbers and ticks
    plt.show()

def read_image(path):
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    return image

def otsu_thresholding(image):
    """
    Extracts text from a black or white background image using Otsu's thresholding.

    Args:
        image: Grayscale image with text (any color) and black or white background.

    Returns:
        thresh: Binary image where text pixels are set to 255 and background pixels are set to 0.
    """

    # Apply Otsu's thresholding
    ret, thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Count foreground (ones) and background (zeros) pixels
    count_ones = np.count_nonzero(thresh == 255)
    count_zeros = np.count_nonzero(thresh == 0)

    if count_ones > count_zeros:
      thresh = 255 - thresh

    return thresh


def median_filter(image, kernel_size):
    # Apply median filter
    filtered_image = cv2.medianBlur(image, kernel_size)
    
    return filtered_image

def histogram_equalization(image):
  """
  Performs histogram equalization on a grayscale image.

  Args:
      image: Grayscale image.

  Returns:
      equalized_image: Image with equalized histogram.
  """

  # Equalize the histogram
  equalized_image = cv2.equalizeHist(image)

  return equalized_image

def skew_angle_hough_transform(image):
    # convert to edges
    edges = canny(image)
    # Classic straight-line Hough transform between 0.1 - 180 degrees.
    tested_angles = np.deg2rad(np.arange(0.1, 180.0))
    h, theta, d = hough_line(edges, theta=tested_angles)
    
    # find line peaks and angles
    accum, angles, dists = hough_line_peaks(h, theta, d)
    
    # round the angles to 2 decimal places and find the most common angle.
    most_common_angle = mode(np.around(angles, decimals=2))[0]
    
    # Check if most_common_angle is NaN
    if np.isnan(most_common_angle):
        most_common_angle = 0

    # convert the angle to degree for rotation.
    skew_angle = np.rad2deg(most_common_angle - np.pi/2)
    return skew_angle

def rotate_image(image, angle_degrees):
    height, width = image.shape[:2]

    # Define the rotation center (center of the image)
    center = (width // 2, height // 2)

    # Perform the rotation
    rotation_matrix = cv2.getRotationMatrix2D(center, angle_degrees, 1.0)
    rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height))

    return rotated_image

def sharpen_image(image, kernel_size=5):
    # Define the sharpening kernel
    kernel = np.ones((kernel_size, kernel_size)) * -1
    kernel[kernel_size // 2, kernel_size // 2] = kernel_size ** 2
    
    # Apply the kernel
    sharpened_image = cv2.filter2D(image, -1, kernel)

    return sharpened_image

def variance_of_laplacian(image):
    return cv2.Laplacian(image, cv2.CV_64F).var()

def preprocess(image, median_kernel_size=3, sharpen_kernel_size=3):
    """
    Preprocesses an image by applying median filtering, Otsu's thresholding, and histogram equalization.

    Args:
        image: Grayscale image with text (any color) and black or white background.
        median_kernel_size: Kernel size for the median filter.
        sharpen_kernel_size: Kernel size for the sharpening filter.

    Returns:
        preprocessed_image: Binary image where text pixels are set to 255 and background pixels are set to 0.
    """
    
    # Apply median filter

    # image = read_image(image_path)

    median_filtered_image = median_filter(image, median_kernel_size)

    # Detect salt and pepper noise
    # Check difference between original image and median filtered image
    noise = np.abs(image - median_filtered_image)
    noise = np.sum(noise) / (image.shape[0] * image.shape[1])

    if (noise > 60):
        image = median_filtered_image

    # Get variance of Laplacian
    laplacian_var = variance_of_laplacian(image)

    # Apply image sharpening
    sharpened_image = sharpen_image(image, sharpen_kernel_size)

    if (laplacian_var < 50):
        image = sharpened_image

    # Apply Otsu's thresholding
    image = otsu_thresholding(image)

    # Apply skew angle correction
    skew_angle = skew_angle_hough_transform(image)
    image = rotate_image(image, skew_angle)

    return image
# Not finished Yet

# Text rotation
# Text alignment
# Text size/weight
# Image blur


# Finished
# Text rotationsssssssss
# Text color
# Salt and pepper noise
# Brightness variation

