import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
import cv2
import os
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import threading
#definations 
def kmeans(data, K, metric='euclidean', max_iters=100):
    data = np.array(data, dtype=float) #conv to NP
    centroids = data[:K].copy()

    for iteration in range(1, max_iters + 1): #terminal
        print(f"\nIteration {iteration}")
        
        if metric == 'euclidean':
            distances = np.linalg.norm(data[:, np.newaxis] - centroids, axis=2)
        elif metric == 'manhattan':
            distances = np.sum(np.abs(data[:, np.newaxis] - centroids), axis=2)
        else:
            raise ValueError(f"Unsupported metric: '{metric}'. Use 'euclidean' or 'manhattan'.")

        clusters = np.argmin(distances, axis=1)

        for k in range(K):
            count = np.sum(clusters == k)
            print(f"Cluster {k+1}: {count} points")
        #update centroids convergence check
        new_centroids = []
        for k in range(K):
            if np.any(clusters == k):
                new_centroids.append(data[clusters == k].mean(axis=0))
            else:
                new_centroids.append(centroids[k])
        new_centroids = np.array(new_centroids)
    #last iteration convergence check
        print("New centroids:", new_centroids)

        if np.allclose(new_centroids, centroids):
            print(f"\nConverged at iteration {iteration} for K={K} ({metric})")
            break

        centroids = new_centroids

    return clusters, centroids

def generate_slide_figure(image_path, K):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read image: {image_path}")
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    h, w = img.shape[:2]
    new_w = 300
    new_h = int(h * (new_w / w))
    img_resized = cv2.resize(img, (new_w, new_h))
    #(h,w,3)to(n,3)
    data = img_resized.reshape((-1, 3))

    print(f"\n{'='*60}")
    print(f"Processing: {os.path.basename(image_path)} | K={K}")
    print(f"Image size: {img_resized.shape[1]}x{img_resized.shape[0]} pixels")
    print(f"Total data points (pixels): {len(data)}")
    print(f"{'='*60}")
    print(f"\n--- Running K-Means: K={K}, Metric=Euclidean ---")
    #n to h,w
    clusters_euc, centroids_euc = kmeans(data, K, metric='euclidean')
    labels_euc = clusters_euc.reshape(img_resized.shape[:2])

    print(f"\n--- Running K-Means: K={K}, Metric=Manhattan ---")
    clusters_man, centroids_man = kmeans(data, K, metric='manhattan')
    labels_man = clusters_man.reshape(img_resized.shape[:2])

    fig = Figure(figsize=(14, 5), dpi=100)
    fig.patch.set_facecolor('#1a1a2e')
    #create layout
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
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    
    buf = canvas.buffer_rgba()
    w, h = canvas.get_width_height()
    pil_image = Image.frombuffer('RGBA', (w, h), buf, 'raw', 'RGBA', 0, 1)
    
    if target_width and pil_image.width > target_width:
        ratio = target_width / pil_image.width
        new_h = int(pil_image.height * ratio)
        pil_image = pil_image.resize((target_width, new_h), Image.LANCZOS)
    
    plt.close(fig)
    return pil_image

class SlideshowApp:
    def __init__(self, root):
        self.root = root
        self.root.title("K-Means Satellite Image Segmentation")
        self.root.configure(bg='#0f0f23')
        self.root.state('zoomed')
        
        self.slides = []
        self.slide_titles = []
        self.current_slide = 0
        self.photo_image = None
        
        self._build_ui()
        self._start_processing()
    
    def _build_ui(self):
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
        
        self.image_frame = tk.Frame(self.root, bg='#0f0f23')
        self.image_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        self.image_label = tk.Label(
            self.image_frame,
            bg='#0f0f23',
            anchor='center'
        )
        self.image_label.pack(fill=tk.BOTH, expand=True)
        
        self.status_label = tk.Label(
            self.root,
            text="Processing images... Please wait.",
            font=("Segoe UI", 12),
            fg='#e0e0e0',
            bg='#0f0f23',
            pady=5
        )
        self.status_label.pack()
        
        nav_frame = tk.Frame(self.root, bg='#16213e', pady=12)
        nav_frame.pack(fill=tk.X, side=tk.BOTTOM)
        
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
        
        self.root.bind('<Left>', lambda e: self._prev_slide())
        self.root.bind('<Right>', lambda e: self._next_slide())
        self.root.bind('<Escape>', lambda e: self.root.destroy())
        
        self.prev_btn.config(state=tk.DISABLED)
        self.next_btn.config(state=tk.DISABLED)
    
    def _start_processing(self):
        thread = threading.Thread(target=self._process_all_images, daemon=True)
        thread.start()
    
    def _process_all_images(self):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        
        dataset_images = [
            "k-means map1.jpeg",
            "k-means map2.jpeg",
            "k-means map3.jpeg",
            "k-means map4.jpeg",
            "k-means map5.jpeg",
        ]
        
        k_test_values = [3, 5]
        
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
        
        self.root.after(0, self._processing_done)
    
    def _processing_done(self):
        self.status_label.config(
            text="All images processed! Use ◀ ▶ or arrow keys to navigate."
        )
        self._display_slide()
    
    def _display_slide(self):
        if not self.slides:
            return
        
        idx = self.current_slide
        pil_img = self.slides[idx]
        
        avail_w = self.image_frame.winfo_width() - 40
        avail_h = self.image_frame.winfo_height() - 20
        
        if avail_w < 100:
            avail_w = 1200
        if avail_h < 100:
            avail_h = 500
        
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
        
        total = len(self.slides)
        self.counter_label.config(text=f"{idx + 1} / {total}")
        
        if idx < len(self.slide_titles):
            self.status_label.config(text=self.slide_titles[idx])
        
        self.prev_btn.config(state=tk.NORMAL if idx > 0 else tk.DISABLED)
        self.next_btn.config(state=tk.NORMAL if idx < total - 1 else tk.DISABLED)
    
    def _prev_slide(self):
        if self.current_slide > 0:
            self.current_slide -= 1
            self._display_slide()
    
    def _next_slide(self):
        if self.current_slide < len(self.slides) - 1:
            self.current_slide += 1
            self._display_slide()

if __name__ == "__main__":
    root = tk.Tk()
    app = SlideshowApp(root)
    root.mainloop()
    
    print("\n" + "="*60)
    print("All images processed successfully!")
    print("="*60)
