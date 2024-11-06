# Synthetic Cats vs Dogs Classification
This project implements a basic image classification setup using Support Vector Machines (SVM) to classify synthetic images of cats and dogs. The dataset consists of randomly generated images that simulate a real-world dataset for practicing image classification pipelines.

# Project Overview
This project demonstrates how to:

Generate a synthetic dataset of images
Convert images to feature vectors
Use PCA to reduce dimensionality for visualization
Train an SVM classifier for binary classification
Note: The images are synthetic and randomly generated, so they lack meaningful features that represent cats or dogs. This project serves as a practice for building image pipelines rather than achieving high classification accuracy.

# Dataset
The synthetic dataset includes grayscale images of size 64x64 pixels:

100 images (50 labeled as 'cat' and 50 as 'dog').
Images are saved in two folders: synthetic_data/cats and synthetic_data/dogs.
A CSV file, synthetic_cat_dog_dataset.csv, lists each image’s path and label.
CSV Structure
The CSV file has two columns:

image_path: Path to the image file (e.g., synthetic_data/cats/cat_0.png)
label: Class label (cat or dog)

# Visualization
After applying PCA, a scatter plot shows the synthetic data’s structure. This helps visualize class separation (if any) and gives insight into the effectiveness of PCA for dimensionality reduction.

Sample Code for Visualization
plt.figure(figsize=(10, 6))
scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], c=labels, cmap='coolwarm', s=30, alpha=0.7)
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.title('2D Visualization of Synthetic Cats and Dogs Dataset using PCA')
plt.legend(handles=scatter.legend_elements()[0], labels=['Dog', 'Cat'])
plt.show()

# Results
Since the dataset is randomly generated, the results are not expected to have high classification accuracy. However, this project shows the mechanics of:

Image preprocessing
Dimensionality reduction
Data visualization
Setting up SVM for classification

# Further Improvements
To extend this project:

Use a real dataset of cats and dogs (e.g., from Kaggle) for meaningful classification.
Experiment with feature extraction techniques using pre-trained models (e.g., ResNet, VGG).
Apply other classifiers and compare results.
