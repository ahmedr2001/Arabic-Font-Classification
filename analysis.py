import cv2
from joblib import load
import os
import numpy as np
import sys
import time
from preprocessing import preprocess
from feature_extraction import histogram_from_sift 

# Define paths & Params
classifier_path = "models\knn_model.pkl"
kmeans_path     = "models\kmeans_model.pkl"

kmeans = load(kmeans_path)
centers = kmeans.cluster_centers_
knn = load(classifier_path)

def classify_image(img):

    img = preprocess(img)
    histogram = histogram_from_sift(img, centers)
    pred = knn.predict([histogram])
    return pred[0] 
    



data_path = sys.argv[1]
print("Loading Images Paths.. ")
images_paths = []
for img_path in os.listdir(data_path):
    images_paths.append(int(img_path[:-5])) # remove the .jpeg
images_paths.sort()

print(f"Loaded: {len(images_paths)} images.")

print("Beginning test.")
timimg_file = open("time.txt" , 'w')
results_file = open("results.txt" , 'w')

for img_path in images_paths:
    # load the image
    img = cv2.imread(os.path.join(data_path , f"{str(img_path)}.jpeg"))
    start = time.time()
    prediction = classify_image(img)
    delta = time.time() - start
    print(f"Prediction for: \"{img_path}.jpeg\" = {prediction}  , in {delta:.3f} seconds")
    timimg_file.write(f"{delta:.3f}\n")
    results_file.write(f"{prediction}\n")

timimg_file.close()
results_file.close()
print("Test ended")