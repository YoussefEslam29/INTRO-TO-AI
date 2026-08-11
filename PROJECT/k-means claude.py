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

Results are displayed in a GUI slideshow with left/right navigation.

Author: Student Project - Intro to AI
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for rendering to images
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
import cv2
import os
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import threading


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

    fig = Figure(figsize=(6, 6))
    ax = fig.add_subplot(111)

    # Plot points by cluster
    for k in range(K):
        cluster_points = data[clusters == k]
        if len(cluster_points) > 0:
            ax.scatter(cluster_points[:, 0], cluster_points[:, 1],
                        s=80, c=colors[k % len(colors)], label=f"Cluster {k+1}")

    # Plot centroids
    for k, c in enumerate(centroids):
        ax.scatter(c[0], c[1], s=200, c='black', marker='X')
        ax.text(c[0] + 0.1, c[1] + 0.1, f"C{k+1}", fontsize=12)

    ax.set_title(f"K-Means Clustering (Iteration {iteration})")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.legend()
    ax.grid(True)
    
    return fig


# =============================================================================
#  2D Example (From SEC Slides - for understanding)
# =============================================================================

def run_2d_example():
    """
    Runs the K-Means algorithm on a simple 2D dataset, similar to the 
    example shown in the SEC section slides.
    This helps visualize how K-Means works step by step.
    
    Returns the matplotlib Figure for display in the GUI.
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
    fig = plot_clusters(data_2d, clusters, centroids, iteration="Final")
    return fig


# =============================================================================
#  Generate slide figures for satellite images
# =============================================================================

def generate_slide_figure(image_path, K):
    """
    Generates a matplotlib figure with 3 subplots:
      Original (MOD09GA RGB) | kMeans Euclidean | kMeans Manhattan
    
    Returns a Figure object rendered for display in the GUI.
    """
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read image: {image_path}")
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Resize image to speed up processing
    h, w = img.shape[:2]
    new_w = 300
    new_h = int(h * (new_w / w))
    img_resized = cv2.resize(img, (new_w, new_h))

    # Flatten the image from (H, W, 3) to (N, 3)
    data = img_resized.reshape((-1, 3))

    print(f"\n{'='*60}")
    print(f"Processing: {os.path.basename(image_path)} | K={K}")
    print(f"Image size: {img_resized.shape[1]}x{img_resized.shape[0]} pixels")
    print(f"Total data points (pixels): {len(data)}")
    print(f"{'='*60}")

    # --- K-Means with Euclidean Distance ---
    print(f"\n--- Running K-Means: K={K}, Metric=Euclidean ---")
    clusters_euc, centroids_euc = kmeans(data, K, metric='euclidean')
    # Reshape cluster labels back to 2D image for colormap display
    labels_euc = clusters_euc.reshape(img_resized.shape[:2])

    # --- K-Means with Manhattan Distance ---
    print(f"\n--- Running K-Means: K={K}, Metric=Manhattan ---")
    clusters_man, centroids_man = kmeans(data, K, metric='manhattan')
    # Reshape cluster labels back to 2D image for colormap display
    labels_man = clusters_man.reshape(img_resized.shape[:2])

    # --- Build the figure ---
    fig = Figure(figsize=(14, 5), dpi=100)
    fig.patch.set_facecolor('#1a1a2e')

    ax1 = fig.add_subplot(1, 3, 1)
    ax2 = fig.add_subplot(1, 3, 2)
    ax3 = fig.add_subplot(1, 3, 3)

    ax1.imshow(img_resized)
    ax1.set_title("MOD09GA RGB", fontsize=12, fontweight='bold', color='white')
    ax1.axis('off')

    ax2.imshow(labels_euc, cmap='viridis')
    ax2.set_title(f"k={K} | KMeans Euclidean", fontsize=12, fontweight='bold', color='white')
    ax2.axis('off')

    ax3.imshow(labels_man, cmap='viridis')
    ax3.set_title(f"k={K} | KMeans Manhattan", fontsize=12, fontweight='bold', color='white')
    ax3.axis('off')

    fig.suptitle(
        f"Image: {os.path.basename(image_path)}   |   K = {K}",
        fontsize=14, fontweight='bold', color='#e94560', y=0.98
    )

    fig.tight_layout(rect=[0, 0, 1, 0.92])
    return fig


def fig_to_photoimage(fig, target_width=None):
    """Convert a matplotlib Figure to a PIL Image, then to a Tkinter PhotoImage."""
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    
    # Get the RGBA buffer
    buf = canvas.buffer_rgba()
    w, h = canvas.get_width_height()
    pil_image = Image.frombuffer('RGBA', (w, h), buf, 'raw', 'RGBA', 0, 1)
    
    # Resize if needed to fit the window
    if target_width and pil_image.width > target_width:
        ratio = target_width / pil_image.width
        new_h = int(pil_image.height * ratio)
        pil_image = pil_image.resize((target_width, new_h), Image.LANCZOS)
    
    plt.close(fig)
    return pil_image


# =============================================================================
#  GUI Slideshow Application
# =============================================================================

class SlideshowApp:
    """
    A tkinter GUI that displays K-Means segmentation results as a slideshow.
    Each slide shows: Original | Euclidean | Manhattan for one image+K combination.
    Navigation via left/right arrow buttons.
    """
    
    def __init__(self, root):
        self.root = root
        self.root.title("K-Means Satellite Image Segmentation")
        self.root.configure(bg='#0f0f23')
        self.root.state('zoomed')  # Start maximized on Windows
        
        self.slides = []       # List of PIL Images
        self.slide_titles = [] # Slide descriptions
        self.current_slide = 0
        self.photo_image = None  # Keep reference to prevent garbage collection
        
        self._build_ui()
        self._start_processing()
    
    def _build_ui(self):
        """Build the GUI layout."""
        # --- Title bar ---
        title_frame = tk.Frame(self.root, bg='#16213e', pady=12)
        title_frame.pack(fill=tk.X)
        
        title_label = tk.Label(
            title_frame,
            text="K-Means Clustering — Satellite Image Segmentation",
            font=("Segoe UI", 20, "bold"),
            fg='#e94560',
            bg='#16213e'
        )
        title_label.pack()
        
        subtitle_label = tk.Label(
            title_frame,
            text="Euclidean & Manhattan Distance  |  SEC Slides Implementation",
            font=("Segoe UI", 11),
            fg='#a3a3c2',
            bg='#16213e'
        )
        subtitle_label.pack()
        
        # --- Main image area ---
        self.image_frame = tk.Frame(self.root, bg='#0f0f23')
        self.image_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        self.image_label = tk.Label(
            self.image_frame,
            bg='#0f0f23',
            anchor='center'
        )
        self.image_label.pack(fill=tk.BOTH, expand=True)
        
        # --- Status / slide info ---
        self.status_label = tk.Label(
            self.root,
            text="Processing images... Please wait.",
            font=("Segoe UI", 12),
            fg='#e0e0e0',
            bg='#0f0f23',
            pady=5
        )
        self.status_label.pack()
        
        # --- Navigation bar ---
        nav_frame = tk.Frame(self.root, bg='#16213e', pady=12)
        nav_frame.pack(fill=tk.X, side=tk.BOTTOM)
        
        # Style for buttons
        btn_style = {
            'font': ("Segoe UI", 16, "bold"),
            'fg': '#ffffff',
            'bg': '#e94560',
            'activebackground': '#c81e45',
            'activeforeground': '#ffffff',
            'bd': 0,
            'padx': 30,
            'pady': 8,
            'cursor': 'hand2',
            'relief': tk.FLAT
        }
        
        self.prev_btn = tk.Button(
            nav_frame,
            text="◀  Previous",
            command=self._prev_slide,
            **btn_style
        )
        self.prev_btn.pack(side=tk.LEFT, padx=40)
        
        # Slide counter
        self.counter_label = tk.Label(
            nav_frame,
            text="— / —",
            font=("Segoe UI", 14, "bold"),
            fg='#a3a3c2',
            bg='#16213e'
        )
        self.counter_label.pack(side=tk.LEFT, expand=True)
        
        self.next_btn = tk.Button(
            nav_frame,
            text="Next  ▶",
            command=self._next_slide,
            **btn_style
        )
        self.next_btn.pack(side=tk.RIGHT, padx=40)
        
        # Keyboard bindings
        self.root.bind('<Left>', lambda e: self._prev_slide())
        self.root.bind('<Right>', lambda e: self._next_slide())
        self.root.bind('<Escape>', lambda e: self.root.destroy())
        
        # Disable buttons until processing is done
        self.prev_btn.config(state=tk.DISABLED)
        self.next_btn.config(state=tk.DISABLED)
    
    def _start_processing(self):
        """Start processing images in a background thread."""
        thread = threading.Thread(target=self._process_all_images, daemon=True)
        thread.start()
    
    def _process_all_images(self):
        """Process all satellite images and build slide images."""
        script_dir = os.path.dirname(os.path.abspath(__file__))
        
        dataset_images = [
            "k-means map1.jpeg",
            "k-means map2.jpeg",
            "k-means map3.jpeg",
            "k-means map4.jpeg",
            "k-means map5.jpeg",
        ]
        
        k_test_values = [3, 5]
        
        # --- First slide: 2D example ---
        self.root.after(0, lambda: self.status_label.config(
            text="Processing 2D example from SEC slides..."
        ))
        
        fig_2d = run_2d_example()
        pil_img_2d = fig_to_photoimage(fig_2d)
        self.slides.append(pil_img_2d)
        self.slide_titles.append("2D K-Means Example (SEC Slides)")
        
        # Show the first slide immediately
        self.root.after(0, self._show_first_slide)
        
        # --- Satellite image slides ---
        total = len(dataset_images) * len(k_test_values)
        done = 0
        
        for img_file in dataset_images:
            img_path = os.path.join(script_dir, img_file)
            
            if not os.path.exists(img_path):
                print(f"Warning: File not found: {img_path}")
                continue
            
            for K in k_test_values:
                done += 1
                self.root.after(0, lambda d=done, t=total, f=img_file, k=K:
                    self.status_label.config(
                        text=f"Processing {f} (K={k})... [{d}/{t}]"
                    )
                )
                
                fig = generate_slide_figure(img_path, K)
                if fig is not None:
                    pil_img = fig_to_photoimage(fig)
                    self.slides.append(pil_img)
                    self.slide_titles.append(f"{img_file}  |  K = {K}")
        
        # Enable navigation after processing
        self.root.after(0, self._processing_done)
    
    def _show_first_slide(self):
        """Display the first slide as soon as it's ready."""
        self.current_slide = 0
        self._display_slide()
        self.prev_btn.config(state=tk.NORMAL)
        self.next_btn.config(state=tk.NORMAL)
    
    def _processing_done(self):
        """Called when all images are processed."""
        self.status_label.config(
            text="All images processed! Use ◀ ▶ or arrow keys to navigate."
        )
        self._display_slide()
    
    def _display_slide(self):
        """Render the current slide onto the image label."""
        if not self.slides:
            return
        
        idx = self.current_slide
        pil_img = self.slides[idx]
        
        # Scale image to fit the available window width
        avail_w = self.image_frame.winfo_width() - 40
        avail_h = self.image_frame.winfo_height() - 20
        
        if avail_w < 100:
            avail_w = 1200
        if avail_h < 100:
            avail_h = 500
        
        # Scale maintaining aspect ratio
        img_w, img_h = pil_img.size
        scale = min(avail_w / img_w, avail_h / img_h, 1.0)
        new_w = int(img_w * scale)
        new_h = int(img_h * scale)
        
        if scale < 1.0:
            display_img = pil_img.resize((new_w, new_h), Image.LANCZOS)
        else:
            display_img = pil_img
        
        self.photo_image = ImageTk.PhotoImage(display_img)
        self.image_label.config(image=self.photo_image)
        
        # Update counter and title
        total = len(self.slides)
        self.counter_label.config(text=f"{idx + 1} / {total}")
        
        if idx < len(self.slide_titles):
            self.status_label.config(text=self.slide_titles[idx])
        
        # Update button states
        self.prev_btn.config(state=tk.NORMAL if idx > 0 else tk.DISABLED)
        self.next_btn.config(state=tk.NORMAL if idx < total - 1 else tk.DISABLED)
    
    def _prev_slide(self):
        """Go to the previous slide."""
        if self.current_slide > 0:
            self.current_slide -= 1
            self._display_slide()
    
    def _next_slide(self):
        """Go to the next slide."""
        if self.current_slide < len(self.slides) - 1:
            self.current_slide += 1
            self._display_slide()


# =============================================================================
#  Main Entry Point
# =============================================================================

if __name__ == "__main__":
    root = tk.Tk()
    app = SlideshowApp(root)
    root.mainloop()
    
    print("\n" + "="*60)
    print("All images processed successfully!")
    print("="*60)
