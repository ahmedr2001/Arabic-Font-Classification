
import numpy as np
import cv2
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans


# Feature Extraction Stage
def extract_features(image):
  """
  Extracts features from an image using dense sampling, SIFT descriptors, and PCA.

  Args:
      image: A grayscale or color image represented as a NumPy array.

  Returns:
      A NumPy array containing the reduced-dimensionality features.
  """

  # Convert to grayscale if necessary
  if len(image.shape) == 3:
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  else:
    gray_image = image

  # Dense sampling
  patches = []
  grid_sizes = [16, 24, 32, 40]
  stride = 8  # Step size between patches

  for grid_size in grid_sizes:
    for y in range(0, gray_image.shape[0], stride):
      for x in range(0, gray_image.shape[1], stride):
        # Ensure patch stays within image boundaries
        end_y = min(y + grid_size, gray_image.shape[0])
        end_x = min(x + grid_size, gray_image.shape[1])
        patch = gray_image[y:end_y, x:end_x]
        patches.append(patch)

  # SIFT descriptor calculation
  descriptors = []
  sift = cv2.xfeatures2d.SIFT_create()
  for patch in patches:
    keypoints, descriptor = sift.detectAndCompute(patch, None)
    # Check if patch has keypoints (avoid empty descriptors)
    if descriptor is not None:
      descriptors.append(descriptor)

  # PCA dimensionality reduction (assuming a pre-trained PCA object exists)
  if len(descriptors) > 0:  # Check if there are any descriptors before applying PCA
    pca = PCA(n_components=64)  # Assuming pre-trained PCA with 64 components
    reduced_descriptors = pca.transform(np.vstack(descriptors))
  else:
    reduced_descriptors = np.zeros((0, 64))  # Return empty array if no descriptors

  return reduced_descriptors


# Codebook Codebook
def generate_codebook(descriptors, codebook_size):
  """
  Generates a codebook using K-means clustering.

  Args:
      descriptors: A NumPy array of feature descriptors (e.g., SIFT descriptors).
      codebook_size: The desired number of clusters in the codebook.

  Returns:
      A NumPy array representing the codebook (cluster centroids).
  """

  # K-means clustering
  kmeans = KMeans(n_clusters=codebook_size)
  kmeans.fit(descriptors)
  codebook = kmeans.cluster_centers_

  return codebook


# Build Generation Stage
def contruct_codebook(training_images):
  # Collect feature descriptors from training data
  all_descriptors = []
  for image in training_images:
    features = extract_features(image)
    all_descriptors.extend(features)  # Combine features from all images

  # Define desired codebook size (number of visual words)
  codebook_size = 2048  # Example value, adjust based on your needs

  # Generate the codebook using K-means clustering
  codebook = generate_codebook(np.vstack(all_descriptors), codebook_size)

  return codebook

# BoF Vector Construction Stage
def construct_bof_vector(image, codebook):
  """
  Constructs a BoF (Bag-of-Features) vector for an image.

  Args:
      image: A grayscale or color image represented as a NumPy array.
      codebook: A NumPy array representing the codebook (cluster centroids).

  Returns:
      A NumPy array representing the BoF vector of the image.
  """

  # Feature quantization
  bof_vector = np.zeros(codebook.shape[0])
  for descriptor in extract_features(image):
    nearest_codeword = np.argmin(np.linalg.norm(codebook - descriptor, axis=1))
    bof_vector[nearest_codeword] += 1

  # Soft assignment (optional)
  # bof_vector = normalize(bof_vector, norm='l1' or norm='l2')  # Uncomment for soft assignment

  # Normalize (optional, but recommended)
  bof_vector = np.normalize(bof_vector)  # Normalize to unit length

  return bof_vector


# Pipeline
def arabic_font_recognition(image, codebook):
  bof_vector = construct_bof_vector(image, codebook)
  # Train a classifier (e.g., SVM, KNN) on a dataset of BoF vectors and corresponding font labels
  predicted_font = classifier.predict(bof_vector)
  return predicted_font