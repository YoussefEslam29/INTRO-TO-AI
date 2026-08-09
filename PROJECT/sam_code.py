import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
from google.colab import files

def custom_kmeans(image, k, metric='euclidean', max_iters=50, tol=1e-4):

    # Get image dimensions
    height, width, channels = image.shape
    
    # Reshape (H, W, 3) to a 2D array of pixels (N, 3)
    pixels = image.reshape((-1, channels)).astype(np.float32)
    num_pixels = pixels.shape[0]
    
    # Reproducible random initialization of K centroids
    np.random.seed(42)
    random_indices = np.random.choice(num_pixels, k, replace=False)
    centroids = pixels[random_indices].copy()
    
    for iteration in range(max_iters):
        # --- Distance Calculation ---
        # Expand dimensions to compute pairwise differences (N, K, 3)
        diff = pixels[:, np.newaxis, :] - centroids[np.newaxis, :, :]
        
        if metric.lower() == 'euclidean':
            # L2 Distance
            distances = np.linalg.norm(diff, axis=2)
        elif metric.lower() == 'manhattan':
            # L1 Distance
            distances = np.sum(np.abs(diff), axis=2)
        else:
            raise ValueError("Metric must be 'euclidean' or 'manhattan'")
            
        # --- Cluster Assignment ---
        labels = np.argmin(distances, axis=1)
        
        # --- Centroid Update ---
        new_centroids = np.zeros_like(centroids)
        for i in range(k):
            cluster_pixels = pixels[labels == i]
            if len(cluster_pixels) > 0:
                if metric.lower() == 'euclidean':
                    # Mean minimizes Euclidean distance
                    new_centroids[i] = np.mean(cluster_pixels, axis=0)
                else:
                    # Median minimizes Manhattan distance
                    new_centroids[i] = np.median(cluster_pixels, axis=0)
            else:
                new_centroids[i] = centroids[i]
                
        # --- Convergence Check ---
        shift = np.max(np.abs(new_centroids - centroids))
        centroids = new_centroids
        if shift < tol:
            break
            
    # Reshape 1D label array back to 2D image map (H, W)
    label_map = labels.reshape((height, width))
    return label_map

def process_satellite_dataset(k_values=[2, 3, 5]):
   
    valid_extensions = ['.jpg', '.jpeg', '.png']
    
    for img_num in range(1, 6):
        image_path = None
        
        # Find path for file named 1.jpg, 2.jpg, etc.
        for ext in valid_extensions:
            candidate = f"{img_num}{ext}"
            if os.path.exists(candidate):
                image_path = candidate
                break
                
        if image_path is None:
            print(f" Image '{img_num}' not found! Skipping... (Ensure files are named 1.jpg through 5.jpg)")
            continue
            
        # Read image
        bgr_img = cv2.imread(image_path)
        rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
        
   
        
        # Grid layout: Rows = K values, Columns = [Original, Euclidean, Manhattan]
        fig, axes = plt.subplots(len(k_values), 3, figsize=(15, 4 * len(k_values)))
        if len(k_values) == 1:
            axes = np.expand_dims(axes, axis=0)
            
        for row, k in enumerate(k_values):
            # Column 1: Original Image
            axes[row, 0].imshow(rgb_img)
            axes[row, 0].set_title(f"Image {img_num} (RGB) | k={k}")
            axes[row, 0].axis('off')
            
            # Column 2: Euclidean Segmentation
            labels_euc = custom_kmeans(rgb_img, k=k, metric='euclidean')
            axes[row, 1].imshow(labels_euc, cmap='viridis')
            axes[row, 1].set_title(f"k={k} | KMeans Euclidean")
            axes[row, 1].axis('off')
            
            # Column 3: Manhattan Segmentation
            labels_man = custom_kmeans(rgb_img, k=k, metric='manhattan')
            axes[row, 2].imshow(labels_man, cmap='viridis')
            axes[row, 2].set_title(f"k={k} | KMeans Manhattan")
            axes[row, 2].axis('off')
            
        plt.tight_layout()
        plt.show()

process_satellite_dataset(k_values=[2, 3, 5])