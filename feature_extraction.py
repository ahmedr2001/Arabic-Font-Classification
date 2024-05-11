
import numpy as np
import cv2
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans


def extract_features(image):
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(image, None)
    
    if descriptors is None:
        return None
    
    if descriptors.shape[0] == 0:
        return None
    
    
    # Pad or truncate descriptors to a fixed length (e.g., 128 dimensions)
    max_descriptors = 128
    if descriptors.shape[0] < max_descriptors:
        # Pad with zeros
        padded_descriptors = np.zeros((max_descriptors, descriptors.shape[1]), dtype=descriptors.dtype)
        padded_descriptors[:descriptors.shape[0], :] = descriptors
        descriptors = padded_descriptors
    
    # Perform PCA for dimensionality reduction
    pca = PCA(n_components=64)
    reduced_descriptors = pca.fit_transform(descriptors)
    
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
  i = 0
  for image in training_images:
    features = extract_features(image)
    if(features is not None):
      all_descriptors.extend(features)  # Combine features from all images
      i += 1
      print(i)
    else:
       print("None")
  # Define desired codebook size (number of visual words)
  codebook_size = 360  # Example value, adjust based on your needs

  # Generate the codebook using K-means clustering
  codebook = generate_codebook(all_descriptors, codebook_size)

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
