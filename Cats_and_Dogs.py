import pandas as pd
import numpy as np
from PIL import Image
import os
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Load the dataset
synthetic_data = pd.read_csv('synthetic_cat_dog_dataset.csv')  # Ensure this is in your working directory

# Initialize lists for features and labels
features = []
labels = []

# Process each image
for _, row in synthetic_data.iterrows():
    try:
        # Load image as grayscale and resize to a consistent 64x64
        img = Image.open(row['image_path']).convert('L')
        img = img.resize((64, 64))
        
        # Flatten the 64x64 image into a 1D feature vector
        img_array = np.array(img).flatten()
        features.append(img_array)
        
        # Label encoding: 1 for 'cat', 0 for 'dog'
        labels.append(1 if row['label'] == 'cat' else 0)
    except FileNotFoundError:
        print(f"File {row['image_path']} not found.")

# Convert lists to numpy arrays
features = np.array(features)
labels = np.array(labels)

# Apply PCA for 2D visualization
pca = PCA(n_components=2)
features_2d = pca.fit_transform(features)

# Plot the 2D visualization
plt.figure(figsize=(10, 6))
scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], c=labels, cmap='coolwarm', s=30, alpha=0.7)
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.title('2D Visualization of Synthetic Cats and Dogs Dataset using PCA')
plt.legend(handles=scatter.legend_elements()[0], labels=['Dog', 'Cat'])
plt.show()
