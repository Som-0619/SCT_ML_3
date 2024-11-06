# Code to Generate a Synthetic Dataset for Cats and Dogs

import os
import numpy as np
import pandas as pd
from PIL import Image

# Parameters for the synthetic dataset
num_images = 100  # Total number of synthetic images (e.g., 50 cats, 50 dogs)
image_size = (64, 64)  # Image dimensions (64x64 pixels)

# Create folders to save synthetic images
os.makedirs('synthetic_data/cats', exist_ok=True)
os.makedirs('synthetic_data/dogs', exist_ok=True)

# Lists to store image paths and labels
image_paths = []
labels = []

# Function to generate a random grayscale image
def generate_random_image(size):
    return np.random.randint(0, 256, size, dtype=np.uint8)

# Generate synthetic cat images
for i in range(num_images // 2):
    img = generate_random_image(image_size)  # Random 64x64 image
    img = Image.fromarray(img, 'L')  # Convert to PIL Image in grayscale
    img_path = f'synthetic_data/cats/cat_{i}.png'
    img.save(img_path)
    image_paths.append(img_path)
    labels.append('cat')

# Generate synthetic dog images
for i in range(num_images // 2):
    img = generate_random_image(image_size)  # Random 64x64 image
    img = Image.fromarray(img, 'L')  # Convert to PIL Image in grayscale
    img_path = f'synthetic_data/dogs/dog_{i}.png'
    img.save(img_path)
    image_paths.append(img_path)
    labels.append('dog')

# Create a DataFrame and save to CSV
data = pd.DataFrame({'image_path': image_paths, 'label': labels})
data.to_csv('synthetic_cat_dog_dataset.csv', index=False)

print("Synthetic dataset generated and saved as 'synthetic_cat_dog_dataset.csv'")
