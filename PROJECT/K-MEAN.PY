import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

def compute_distances(X, centroids, metric='euclidean'):
    """
    Computes the distance between each pixel and each centroid.
    X: numpy array of shape (N, 3) representing image pixels.
    centroids: numpy array of shape (k, 3).
    """
    if metric == 'euclidean':
        # Euclidean distance: sqrt(sum((x - y)^2))
        return np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
    elif metric == 'manhattan':
        # Manhattan distance: sum(|x - y|)
        return np.sum(np.abs(X[:, np.newaxis] - centroids), axis=2)
    else:
        raise ValueError("Unsupported metric. Use 'euclidean' or 'manhattan'.")

def kmeans_clustering(pixels, k, metric='euclidean', max_iters=100, tol=1e-4):
    """
    Custom K-Means implementation supporting Euclidean and Manhattan distances.
    """
    # 1. Randomly initialize centroids from existing pixels
    np.random.seed(42) # For reproducible results
    random_indices = np.random.choice(pixels.shape[0], size=k, replace=False)
    centroids = pixels[random_indices]
    
    for i in range(max_iters):
        # 2. Assign each pixel to the nearest centroid
        distances = compute_distances(pixels, centroids, metric)
        labels = np.argmin(distances, axis=1)
        
        # 3. Update centroids (calculate the mean of assigned pixels)
        new_centroids = np.zeros_like(centroids)
        for cluster_idx in range(k):
            cluster_pixels = pixels[labels == cluster_idx]
            if len(cluster_pixels) > 0:
                new_centroids[cluster_idx] = cluster_pixels.mean(axis=0)
            else:
                # If a cluster is empty, reinitialize it randomly
                new_centroids[cluster_idx] = pixels[np.random.choice(pixels.shape[0])]
                
        # 4. Check for convergence
        shift = np.linalg.norm(new_centroids - centroids)
        if shift < tol:
            print(f"Converged at iteration {i+1} for k={k} ({metric})")
            break
            
        centroids = new_centroids
        
    return labels, centroids

def segment_image(image_path, k_values=[3, 5]):
    """
    Reads an image, processes it with K-Means for given k values and metrics,
    and plots the results.
    """
    if not os.path.exists(image_path):
        print(f"File not found: {image_path}")
        return

    # Read image and convert from BGR (OpenCV default) to RGB
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Resize image to speed up processing (optional but recommended for large satellite images)
    # Resize to a max width of 400px while maintaining aspect ratio
    h, w = img.shape[:2]
    new_w = 400
    new_h = int(h * (new_w / w))
    img_resized = cv2.resize(img, (new_w, new_h))
    
    # Flatten the image to an array of pixels: (N, 3)
    pixels = img_resized.reshape((-1, 3)).astype(np.float32)

    for k in k_values:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle(f"Image: {image_path} | Clusters (k): {k}", fontsize=16)

        # Plot Original
        axes[0].imshow(img_resized)
        axes[0].set_title("Original Image (RGB)")
        axes[0].axis('off')

        # --- Euclidean Distance Segmentation ---
        labels_euc, centers_euc = kmeans_clustering(pixels, k, metric='euclidean')
        # Map labels back to centroid colors
        segmented_pixels_euc = centers_euc[labels_euc]
        # Reshape back to image dimensions
        segmented_image_euc = segmented_pixels_euc.reshape(img_resized.shape).astype(np.uint8)
        
        axes[1].imshow(segmented_image_euc)
        axes[1].set_title("k-Means\nEuclidean Distance")
        axes[1].axis('off')

        # --- Manhattan Distance Segmentation ---
        labels_man, centers_man = kmeans_clustering(pixels, k, metric='manhattan')
        segmented_pixels_man = centers_man[labels_man]
        segmented_image_man = segmented_pixels_man.reshape(img_resized.shape).astype(np.uint8)
        
        axes[2].imshow(segmented_image_man)
        axes[2].set_title("k-Means\nManhattan Distance")
        axes[2].axis('off')

        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    # List of satellite image files provided in your dataset
    dataset_images = [
        "image_23c660.jpg",
        "image_23c645.jpg",
        "image_23c640.jpg",
        "image_23c627.jpg",
        "image_23c623.jpg"
    ]
    
    # Test different values of k as requested by the assignment
    k_test_values = [3, 5] 
    
    for img_file in dataset_images:
        print(f"Processing {img_file}...")
        segment_image(img_file, k_values=k_test_values)