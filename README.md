# Arabic Font Classification
This is an arabic font classifier that based on a given sample of images containing arabic paragraphs classifies the paragraph as one of four fonts: IBM Plex Sans Arabic, Lemonada, Marhey, Scheherazade New. This was a winning model 🥇 in the competition that corresponded to the project in fulfillment of the classwork requirements of our pattern recognition course, achieiving 99.2% accuracy and making 1000 predictions in only 429 seconds.

## Datasets & Preprocessing 💾 
The dataset can be found <a href="https://www.kaggle.com/datasets/breathemath/fonts-dataset-cmp">here</a>. It is stored in the "fonts-dataset" folder. For preprocessing, we try our best to remove noise from the image without affecting the original image itself, since we rely on SIFT for feature extraction. We apply a median filter for salt and pepper noise followed by sharpening to remove blur (if needed). Finally, we binarize the image via otsu thresholding. The relevant file is "preprocessing.py". <br> <br>
This is a sample from the dataset: <br>
![0](https://github.com/ahmedr2001/Arabic-Font-Classification/assets/77215230/bc211430-5d90-4b9a-a195-262ed2d5d36a)


## Features Extracted 🤳
We first considered line, word, and character segmentation. We had made significant progress, but finally opted for SIFT due to the nature of our problem and the scale-invariant, rotation-invariant properties of SIFT. Our approach was as follows: first, we extract the SIFT descriptors; second, we perform segmentation of the extracted feature vectors via kmeans; third, we find the nearest cluster center to each feature vector and construct a histogram for each image based on the assigned clusters of its feature vectors; fourth, we return this histogram and use it for classification. <br> <br>

## Models Considered 🕹️
We have considered NN, KNN, SVM. Because both accuracy and performance mattered for the project (along with other constraints) only SVM made it to the final model. Additionally, SVM is suitable when the dataset isn't large but the features are complex, which was our case. 

## Running the Project 🚀
To test the final model, run "predict.ipynb". To train your own models using our features, run "train.ipynb". You can run <code>pip install -r requirements.txt</code> to install all dependencies. If you would like to use a browser, our model is deployed at <a href="https://ahmedr2001.pythonanywhere.com">ahmedr2001.pythonanywhere.com</a>.
