"""
K-Means Clustering for Satellite Image Segmentation
=====================================================
This project applies K-Means clustering to segment satellite imagery (MOD09GA RGB)
into distinct regions (e.g., water, land, vegetation) using two distance measures:
    1. Euclidean Distance
    2. Manhattan Distance

The algorithm is implemented following the SEC section material structure:
    - Initialize centroids using the first K data points
    - Assign each pixel to the nearest centroid (cluster assignment)
    - Recompute centroids as the mean of assigned pixels
    - Repeat until convergence (np.allclose check)

Author: Student Project - Intro to AI
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2
import os


# =============================================================================
#  K-Means Algorithm (Based on SEC Slides)
# =============================================================================

def kmeans(data, K, metric='euclidean', max_iters=100):
    """
    K-Means clustering algorithm.
    
    This implementation follows the SEC section material exactly:
      - Step 1: Initialize centroids = first K data points
      - Step 2: Assign each point to the nearest centroid
      - Step 3: Recompute centroids as the mean of each cluster
      - Step 4: Repeat until convergence
    
    Parameters
    ----------
    data : array-like
        The dataset to cluster. Each row is a data point (e.g., a pixel's RGB values).
    K : int
        Number of clusters.
    metric : str
        Distance metric to use: 'euclidean' or 'manhattan'.
    max_iters : int
        Maximum number of iterations before stopping.
    
    Returns
    -------
    clusters : ndarray
        Cluster assignment for each data point (array of ints 0..K-1).
    centroids : ndarray
        Final centroid positions (K x D array).
    """
    # Convert to numpy array (as shown in SEC slides)
    data = np.array(data, dtype=float)
    
    # Initialize centroids using the first K points
    # (Matching SEC slide: centroids = data[:K].copy())
    centroids = data[:K].copy()

    for iteration in range(1, max_iters + 1):
        print(f"\nIteration {iteration}")

        # ----- Assign clusters -----
        # The SEC slides show a nested loop approach:
        #   for i, point in enumerate(data):
        #       min_dist = float('inf')
        #       for k, centroid in enumerate(centroids):
        #           dist = np.sqrt(np.sum((point - centroid)**2))  # Euclidean
        #           if dist < min_dist:
        #               min_dist = dist
        #               clusters[i] = k
        #
        # Below is the vectorized equivalent for efficiency on large images,
        # but the math is identical to the SEC loop.
        
        if metric == 'euclidean':
            # Euclidean distance: sqrt(sum((point - centroid)^2))
            # Equivalent to SEC's: np.sqrt(np.sum((point - centroid)**2))
            distances = np.linalg.norm(data[:, np.newaxis] - centroids, axis=2)
        elif metric == 'manhattan':
            # Manhattan distance: sum(|point - centroid|)
            distances = np.sum(np.abs(data[:, np.newaxis] - centroids), axis=2)
        else:
            raise ValueError(f"Unsupported metric: '{metric}'. Use 'euclidean' or 'manhattan'.")

        clusters = np.argmin(distances, axis=1)

        # Print points in each cluster (SEC slide style)
        for k in range(K):
            count = np.sum(clusters == k)
            print(f"Cluster {k+1}: {count} points")

        # ----- Recompute centroids -----
        # Matching the SEC slide's exact list-appending structure:
        #   new_centroids = []
        #   for k in range(K):
        #       if np.any(clusters == k):
        #           new_centroids.append(data[clusters == k].mean(axis=0))
        #       else:
        #           new_centroids.append(centroids[k])
        #   new_centroids = np.array(new_centroids)
        
        new_centroids = []
        for k in range(K):
            if np.any(clusters == k):
                new_centroids.append(data[clusters == k].mean(axis=0))
            else:
                # If a cluster has no points, keep the old centroid
                new_centroids.append(centroids[k])
        new_centroids = np.array(new_centroids)

        print("New centroids:", new_centroids)

        # ----- Convergence check -----
        # Using the exact np.allclose check from the SEC slides
        if np.allclose(new_centroids, centroids):
            print(f"\nConverged at iteration {iteration} for K={K} ({metric})")
            break

        centroids = new_centroids

    return clusters, centroids


# =============================================================================
#  Image Segmentation Function
# =============================================================================

def segment_image(image_path, k_values=[3, 5]):
    """
    Reads a satellite image, segments it using K-Means with both
    Euclidean and Manhattan distance, and displays the results.
    
    For each K value, it shows:
      - Original RGB image
      - Segmented image using Euclidean distance
      - Segmented image using Manhattan distance
    
    Parameters
    ----------
    image_path : str
        Path to the satellite image file.
    k_values : list of int
        List of K values (number of clusters) to test.
    """
    # Check if the file exists
    if not os.path.exists(image_path):
        print(f"Error: File not found: {image_path}")
        return

    # Read image using OpenCV and convert BGR -> RGB for display
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read image: {image_path}")
        return
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Resize image to speed up processing
    # (satellite images can be very large, resizing preserves the visual output)
    h, w = img.shape[:2]
    new_w = 300
    new_h = int(h * (new_w / w))
    img_resized = cv2.resize(img, (new_w, new_h))

    # Flatten the image from (H, W, 3) to (N, 3) - each row is one pixel's RGB
    data = img_resized.reshape((-1, 3))

    print(f"\n{'='*60}")
    print(f"Processing: {os.path.basename(image_path)}")
    print(f"Image size: {img_resized.shape[1]}x{img_resized.shape[0]} pixels")
    print(f"Total data points (pixels): {len(data)}")
    print(f"{'='*60}")

    for K in k_values:
        # Create a figure with 3 subplots: Original | Euclidean | Manhattan
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle(
            f"K-Means Image Segmentation  |  Image: {os.path.basename(image_path)}  |  K = {K}",
            fontsize=14, fontweight='bold'
        )

        # --- Plot 1: Original Image ---
        axes[0].imshow(img_resized)
        axes[0].set_title("MOD09GA RGB\n(Original Image)", fontsize=12)
        axes[0].axis('off')

        # --- Plot 2: K-Means with Euclidean Distance ---
        print(f"\n--- Running K-Means: K={K}, Metric=Euclidean ---")
        clusters_euc, centroids_euc = kmeans(data, K, metric='euclidean')
        
        # Reconstruct the segmented image:
        # Replace each pixel with its centroid's color
        segmented_pixels_euc = centroids_euc[clusters_euc]
        segmented_image_euc = segmented_pixels_euc.reshape(img_resized.shape).astype(np.uint8)

        axes[1].imshow(segmented_image_euc)
        axes[1].set_title("kMeans\nEuclidean Distance", fontsize=12)
        axes[1].axis('off')

        # --- Plot 3: K-Means with Manhattan Distance ---
        print(f"\n--- Running K-Means: K={K}, Metric=Manhattan ---")
        clusters_man, centroids_man = kmeans(data, K, metric='manhattan')
        
        # Reconstruct the segmented image
        segmented_pixels_man = centroids_man[clusters_man]
        segmented_image_man = segmented_pixels_man.reshape(img_resized.shape).astype(np.uint8)

        axes[2].imshow(segmented_image_man)
        axes[2].set_title("kMeans\nManhattan Distance", fontsize=12)
        axes[2].axis('off')

        plt.tight_layout()
        plt.show()


# =============================================================================
#  Plot Clusters Function (From SEC Slides - for 2D data visualization)
# =============================================================================

def plot_clusters(data, clusters, centroids, iteration):
    """
    Plots 2D clustered data with centroids.
    This function is taken directly from the SEC section slides.
    It is useful for visualizing K-Means on simple 2D datasets.
    
    Parameters
    ----------
    data : array-like
        The 2D dataset (N x 2).
    clusters : ndarray
        Cluster assignments.
    centroids : ndarray
        Current centroid positions.
    iteration : int
        Current iteration number (for the title).
    """
    data = np.array(data)
    K = len(centroids)
    colors = ['red', 'blue', 'green', 'purple', 'orange', 'brown']

    plt.figure(figsize=(6, 6))

    # Plot points by cluster
    for k in range(K):
        cluster_points = data[clusters == k]
        if len(cluster_points) > 0:
            plt.scatter(cluster_points[:, 0], cluster_points[:, 1],
                        s=80, c=colors[k % len(colors)], label=f"Cluster {k+1}")

    # Plot centroids
    for k, c in enumerate(centroids):
        plt.scatter(c[0], c[1], s=200, c='black', marker='X')
        plt.text(c[0] + 0.1, c[1] + 0.1, f"C{k+1}", fontsize=12)

    plt.title(f"K-Means Clustering (Iteration {iteration})")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.grid(True)
    plt.show()


# =============================================================================
#  2D Example (From SEC Slides - for understanding)
# =============================================================================

def run_2d_example():
    """
    Runs the K-Means algorithm on a simple 2D dataset, similar to the 
    example shown in the SEC section slides.
    This helps visualize how K-Means works step by step.
    """
    # Sample 2D data points (similar to SEC slide example)
    data_2d = np.array([
        [1, 2], [2, 5], [2, 10], [2, 9],
        [2, 5], [4, 9], [5, 8], [6, 4],
        [7, 5], [8, 4], [8, 4]
    ])
    
    K = 3
    print("\n" + "="*60)
    print("2D K-Means Example (from SEC slides)")
    print(f"Data points: {len(data_2d)}, K = {K}")
    print("="*60)
    
    clusters, centroids = kmeans(data_2d, K, metric='euclidean')
    plot_clusters(data_2d, clusters, centroids, iteration="Final")


# =============================================================================
#  Main Entry Point
# =============================================================================

if __name__ == "__main__":
    
    # -------------------------------------------------------------------------
    # Part 1: Run the 2D example from SEC slides (for understanding)
    # -------------------------------------------------------------------------
    run_2d_example()
    
    # -------------------------------------------------------------------------
    # Part 2: Satellite Image Segmentation
    # Apply K-Means on satellite imagery with different K values
    # and two distance metrics (Euclidean and Manhattan)
    # -------------------------------------------------------------------------
    
    # List of satellite images in the dataset
    dataset_images = [
        "k-means map1.jpeg",   # Sinai Peninsula / Red Sea region
        "k-means map2.jpeg",   # Aral Sea / Lake region
        "k-means map3.jpeg",   # Nile Delta / Mediterranean region
        "k-means map4.jpeg",   # Lake / desert region
        "k-means map5.jpeg",   # Large lake / sea region
    ]
    
    # Test with different numbers of clusters
    k_test_values = [3, 5]
    
    # Process each satellite image
    for img_file in dataset_images:
        # Build the full path (images are in the same directory)
        img_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), img_file)
        print(f"\n{'#'*60}")
        print(f"Processing: {img_file}")
        print(f"{'#'*60}")
        segment_image(img_path, k_values=k_test_values)
    
    print("\n" + "="*60)
    print("All images processed successfully!")
    print("="*60)
