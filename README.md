# SCT_ML_03
Dog vs Cat Classification using HOG Features and SVM
This project involves classifying images of dogs and cats using Histogram of Oriented Gradients (HOG) features and a Support Vector Machine (SVM) classifier. The model is trained on grayscale images of dogs and cats, and it is evaluated based on accuracy and classification metrics.

Project Structure-

>Data Preprocessing: The dataset contains two folders, /dataset/dog and /dataset/cat, which store grayscale images of dogs and cats respectively. The images are loaded, resized, and labeled as 0 for dogs and 1 for cats.

>Feature Extraction: The Histogram of Oriented Gradients (HOG) method is used to extract features from the images. HOG captures edge directions and patterns, which helps in image classification.

>Model: The Support Vector Machine (SVM) classifier with a linear kernel is used to train the model on the extracted features.

>Evaluation: The model's performance is evaluated using accuracy, precision, recall, and F1-score. Additionally, a sample of images is displayed with their true and predicted labels.
