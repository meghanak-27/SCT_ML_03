import os
import numpy as np
from skimage.feature import hog
from skimage import io, color
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import cv2

dog_folder_path = '/dataset/dog'
cat_folder_path = '/dataset/cat'
image_data, labels = [], []

#Load data
def load_grayscale_images(folder, label, target_size=(64, 64)):
    data, lbls = [], []
    for file_name in os.listdir(folder):
        img_path = os.path.join(folder, file_name)
        try:
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue  # Skip invalid images
            img = cv2.resize(img, target_size)  
            data.append(img)
            lbls.append(label)
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
    return data, lbls

# Load dog images with label 0
print("Loading grayscale dog images...")
dog_images, dog_labels = load_grayscale_images(dog_folder_path, label=0)

# Load cat images with label 1
print("Loading grayscale cat images...")
cat_images, cat_labels = load_grayscale_images(cat_folder_path, label=1)

# Combine data and labels
image_data = dog_images + cat_images
labels = dog_labels + cat_labels

if not image_data:
    print("No images loaded. Check your directory paths.")
else:
    print(f"Loaded {len(image_data)} images.")

# Label and visualize distribution check
label_counts = dict(zip(*np.unique(labels, return_counts=True)))
print(f"Label distribution: Dog: {label_counts.get(0, 0)}, Cat: {label_counts.get(1, 0)}")


plt.bar(label_counts.keys(), label_counts.values(), tick_label=['Dog', 'Cat'], color=['blue', 'gray'])
plt.title("Label Distribution")
plt.xlabel("Labels")
plt.ylabel("Counts")
plt.show()

# Extracting HOG features
def extract_hog_features(image_list):
    hog_features = []
    for img in image_list:
        features = hog(img, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=False)
        hog_features.append(features)
    return np.array(hog_features)
features = extract_hog_features(image_data)

# Train-test split
X_train, X_test, y_train, y_test, img_train, img_test = train_test_split(
    features, labels, image_data, test_size=0.2, random_state=42
)

# SVM
svm = SVC(kernel='linear', random_state=42)
svm.fit(X_train, y_train)
print("SVM training completed.")

# Prediction and evaluation
y_pred = svm.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model accuracy: {accuracy * 100:.2f}%")
print(classification_report(y_test, y_pred, target_names=['Dog', 'Cat']))

# Displaying sample images with predictions
def display_images_with_predictions(images, true_labels, predicted_labels, num_images=10):
    plt.figure(figsize=(15, 7))
    indices = np.random.choice(len(images), size=min(num_images, len(images)), replace=False)
    for i, idx in enumerate(indices):
        plt.subplot(2, 5, i + 1)
        plt.imshow(images[idx], cmap='gray')  # Display as grayscale
        plt.title(f"True: {'Dog' if true_labels[idx] == 0 else 'Cat'}\n"
                  f"Pred: {'Dog' if predicted_labels[idx] == 0 else 'Cat'}",
                  color='green' if true_labels[idx] == predicted_labels[idx] else 'red')
        plt.axis('off')
    plt.tight_layout()
    plt.show()

# Visualize predictions
display_images_with_predictions(img_test, y_test, y_pred, num_images=10)
