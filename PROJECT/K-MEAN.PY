import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

def kmeans_sec(data, K, metric='euclidean', max_iters=100):
    """
    K-Means algorithm structured specifically around the SEC material provided.
    """
    # 1. Initialization (Matching SEC slides exactly)
    data = np.array(data, dtype=float)
    centroids = data[:K].copy() # Initialize centroids using the first K points

    for iteration in range(1, max_iters+1):
        
        # --- Assign clusters ---
        # Vectorized distance calculation to handle image-scale data, 
        # mathematically identical to the SEC's nested loop approach.
        if metric == 'euclidean':
            # Equivalent to SEC's: np.sqrt(np.sum((point - centroid)**2))
            distances = np.linalg.norm(data[:, np.newaxis] - centroids, axis=2)
        elif metric == 'manhattan':
            # Sum of absolute differences
            distances = np.sum(np.abs(data[:, np.newaxis] - centroids), axis=2)
        else:
            raise ValueError("Unsupported metric")
            
        clusters = np.argmin(distances, axis=1)

        # --- Recompute centroids ---
        # Matching the SEC slide's exact list-appending structure
        new_centroids = []
        for k in range(K):
            if np.any(clusters == k):
                new_centroids.append(data[clusters == k].mean(axis=0))
            else:
                new_centroids.append(centroids[k])
        new_centroids = np.array(new_centroids)

        # --- Convergence check ---
        # Using the exact np.allclose check from the SEC slides
        if np.allclose(new_centroids, centroids):
            print(f"Converged at iteration {iteration} for K={K} ({metric})")
            break
            
        centroids = new_centroids

    return clusters, centroids

def segment_image(image_path, k_values=[3, 5]):
    """
    Reads an image, processes it with K-Means, and plots the results.
    """
    if not os.path.exists(image_path):
        print(f"File not found: {image_path}")
        return

    # Read image and convert to RGB
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Resize image to speed up processing
    h, w = img.shape[:2]
    new_w = 400
    new_h = int(h * (new_w / w))
    img_resized = cv2.resize(img, (new_w, new_h))
    
    # Flatten the image to an array of pixels: (N, 3)
    data = img_resized.reshape((-1, 3))

    for K in k_values:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle(f"Image: {image_path} | K = {K}", fontsize=16)

        # Plot Original
        axes[0].imshow(img_resized)
        axes[0].set_title("Original Image (RGB)")
        axes[0].axis('off')

        # --- Euclidean Distance Segmentation ---
        clusters_euc, centroids_euc = kmeans_sec(data, K, metric='euclidean')
        segmented_pixels_euc = centroids_euc[clusters_euc]
        segmented_image_euc = segmented_pixels_euc.reshape(img_resized.shape).astype(np.uint8)
        
        axes[1].imshow(segmented_image_euc)
        axes[1].set_title("k-Means\nEuclidean Distance")
        axes[1].axis('off')

        # --- Manhattan Distance Segmentation ---
        clusters_man, centroids_man = kmeans_sec(data, K, metric='manhattan')
        segmented_pixels_man = centroids_man[clusters_man]
        segmented_image_man = segmented_pixels_man.reshape(img_resized.shape).astype(np.uint8)
        
        axes[2].imshow(segmented_image_man)
        axes[2].set_title("k-Means\nManhattan Distance")
        axes[2].axis('off')

        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    dataset_images = [
        "image_23c660.jpg",
        "image_23c645.jpg",
        "image_23c640.jpg",
        "image_23c627.jpg",
        "image_23c623.jpg"
    ]
    
    k_test_values = [3, 5] 
    
    for img_file in dataset_images:
        print(f"\nProcessing {img_file}...")
        segment_image(img_file, k_values=k_test_values)