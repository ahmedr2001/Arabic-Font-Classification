
import numpy as np
import cv2
from joblib import load
from preprocessing import preprocess

def sift_features(image):
    sift = cv2.xfeatures2d.SIFT_create()
    kp, des = sift.detectAndCompute(image, None)
    return des if des is not None else []

def find_index(feature_vector, centers):
    # Calculate the Euclidean distance between the feature vector and each center
    distances = np.linalg.norm(centers - feature_vector, axis=1)
    # Find the index of the closest center (visual word)
    closest_index = np.argmin(distances)
    return closest_index

def histogram_from_sift(image, centers):
    descriptors = sift_features(image)
    histogram = np.zeros(len(centers))
    for feature_vector in descriptors:
        ind = find_index(feature_vector, centers)
        histogram[ind] += 1
    return histogram

def kmeans_centers():
    kmeans = load('../models/kmeans_model.pkl')
    return kmeans.cluster_centers_


def pipeline(image, centers):
    image = preprocess(image)
    histogram = histogram_from_sift(image, centers)
    return histogram