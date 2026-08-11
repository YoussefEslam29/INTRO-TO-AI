# K-Means Satellite Image Segmentation — Code Walkthrough

## What Does This File Do?

This file implements **K-Means clustering** for **satellite image segmentation** using a full interactive GUI. It:
1. Processes 5 real satellite images using both **Euclidean** and **Manhattan** distances
2. Shows all results in a dark-themed **slideshow window** you can click through

---

## Code Structure at a Glance

| Section | Lines | Purpose |
|---------|-------|---------|
| Imports | 1–12 | Load all required libraries |
| `kmeans()` | 14–50 | Core K-Means algorithm |
| `generate_slide_figure()` | 52–105 | Run K-Means on a satellite image |
| `fig_to_photoimage()` | 107–121 | Convert Matplotlib figure → PIL image |
| `SlideshowApp` class | 123–322 | Tkinter GUI slideshow application |
| Entry point | 324–331 | Launch the app |

---

## Section 1: Imports (Lines 1–12)

```python
import numpy as np                                    # Line 1
import matplotlib                                     # Line 2
matplotlib.use('Agg')                                 # Line 3
import matplotlib.pyplot as plt                       # Line 4
from matplotlib.figure import Figure                  # Line 5
from matplotlib.backends.backend_agg import FigureCanvasAgg  # Line 6
import cv2                                            # Line 7
import os                                             # Line 8
import tkinter as tk                                  # Line 9
from tkinter import ttk                               # Line 10
from PIL import Image, ImageTk                        # Line 11
import threading                                      # Line 12
```

**Line 1** — `numpy` is used for all math: arrays, distance calculations, means.

**Lines 2–3** — `matplotlib.use('Agg')` must be called before importing pyplot. The `Agg` backend makes Matplotlib draw to an **in-memory buffer** instead of opening a pop-up window. This is critical because Tkinter handles our window — we don't want Matplotlib fighting over the display.

**Lines 4–6** — Three Matplotlib imports:
- `plt` — used only for `plt.close(fig)` to free memory
- `Figure` — creates an off-screen figure object (no window)
- `FigureCanvasAgg` — renders the figure into a pixel buffer we can extract

**Line 7** — `cv2` (OpenCV) reads and resizes the satellite image files from disk.

**Line 8** — `os` provides file path utilities like `os.path.join` and `os.path.exists`.

**Lines 9–10** — `tkinter` is Python's built-in GUI library. Used to build the window, buttons, and labels.

**Line 11** — `PIL` (Pillow) converts images between NumPy/Matplotlib format and Tkinter's format.

**Line 12** — `threading` lets the heavy image processing run in the **background** so the GUI stays responsive.

---

## Section 2: `kmeans()` — The Core Algorithm (Lines 14–50)

```python
def kmeans(data, K, metric='euclidean', max_iters=100):   # Line 14
```
**Line 14** — Defines the K-Means function. Parameters:
- `data` — the dataset (pixels as a 2D NumPy array)
- `K` — number of clusters
- `metric` — `'euclidean'` or `'manhattan'`
- `max_iters` — safety limit of iterations (default 100)

```python
    data = np.array(data, dtype=float)   # Line 15
    centroids = data[:K].copy()          # Line 16
```
**Line 15** — Converts the input to a float NumPy array for math operations.

**Line 16** — **Initialization**: takes the first K data points as the starting centroids. `.copy()` prevents accidentally modifying the original data. This matches the SEC slides exactly.

```python
    for iteration in range(1, max_iters + 1):    # Line 18
        print(f"\nIteration {iteration}")         # Line 19
```
**Lines 18–19** — Main loop, runs up to `max_iters` times. Prints the current iteration to the terminal.

```python
        if metric == 'euclidean':                                              # Line 21
            distances = np.linalg.norm(data[:, np.newaxis] - centroids, axis=2)  # Line 22
        elif metric == 'manhattan':                                            # Line 23
            distances = np.sum(np.abs(data[:, np.newaxis] - centroids), axis=2)  # Line 24
        else:                                                                  # Line 25
            raise ValueError(f"Unsupported metric: '{metric}'...")             # Line 26
```
**Lines 21–26** — Distance calculation using NumPy broadcasting:
- `data[:, np.newaxis]` reshapes from `(N, 3)` → `(N, 1, 3)`, then broadcasts against `centroids (K, 3)` to produce a `(N, K, 3)` difference array
- **Euclidean** (Line 22): `np.linalg.norm(..., axis=2)` gives straight-line distance → result shape `(N, K)`
- **Manhattan** (Line 24): `np.sum(np.abs(...), axis=2)` sums absolute differences → result shape `(N, K)`
- Line 25–26: raises a clear error if an invalid metric is given

```python
        clusters = np.argmin(distances, axis=1)   # Line 28
```
**Line 28** — **Assignment step**: for each pixel (row), finds the index of the nearest centroid. Each pixel gets a cluster label 0 to K-1.

```python
        for k in range(K):                              # Line 30
            count = np.sum(clusters == k)               # Line 31
            print(f"Cluster {k+1}: {count} points")    # Line 32
```
**Lines 30–32** — Prints how many pixels are in each cluster this iteration (useful for monitoring progress).

```python
        new_centroids = []                                        # Line 34
        for k in range(K):                                        # Line 35
            if np.any(clusters == k):                             # Line 36
                new_centroids.append(data[clusters == k].mean(axis=0))  # Line 37
            else:                                                 # Line 38
                new_centroids.append(centroids[k])                # Line 39
        new_centroids = np.array(new_centroids)                   # Line 40
```
**Lines 34–40** — **Update step**: recalculates each centroid as the **mean** of all pixels assigned to it.
- `data[clusters == k]` — selects only the pixels in cluster k
- `.mean(axis=0)` — averages the R, G, B values across all those pixels
- If a cluster is empty (Line 36 is False), the old centroid is kept to avoid NaN errors

```python
        print("New centroids:", new_centroids)   # Line 42
```
**Line 42** — Prints the updated centroid values to the terminal.

```python
        if np.allclose(new_centroids, centroids):                          # Line 44
            print(f"\nConverged at iteration {iteration} for K={K}...")    # Line 45
            break                                                           # Line 46
```
**Lines 44–46** — **Convergence check**: `np.allclose` returns True if centroids barely moved (within floating-point tolerance). When converged, the loop stops early.

```python
        centroids = new_centroids   # Line 48
```
**Line 48** — Updates centroids for the next iteration.

```python
    return clusters, centroids   # Line 50
```
**Line 50** — Returns the final cluster label for every pixel and the final centroid positions.

---

## Section 3: `generate_slide_figure()` — Satellite Image Processing (Lines 52–105)

```python
def generate_slide_figure(image_path, K):   # Line 52
```
**Line 52** — Takes a path to a satellite image file and the number of clusters K.

```python
    img = cv2.imread(image_path)            # Line 53
    if img is None:                         # Line 54
        print(f"Error: ...")                # Line 55
        return None                         # Line 56
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Line 57
```
**Lines 53–57** — Reads the image from disk. OpenCV loads images in **BGR** order by default, so Line 57 converts to **RGB** for correct display colors in Matplotlib.

```python
    h, w = img.shape[:2]                    # Line 59
    new_w = 300                             # Line 60
    new_h = int(h * (new_w / w))           # Line 61
    img_resized = cv2.resize(img, (new_w, new_h))  # Line 62
```
**Lines 59–62** — Resizes the image to **300 pixels wide** while keeping the aspect ratio. This is critical for performance — it reduces the number of pixels (data points) from hundreds of thousands down to ~80,000, making K-Means run in seconds.

```python
    data = img_resized.reshape((-1, 3))   # Line 64
```
**Line 64** — **Flattens** the image. Converts shape `(H, W, 3)` → `(N, 3)` where N = H × W. Each row is now one pixel with its `[R, G, B]` values — this is the dataset fed into `kmeans()`.

**Lines 66–70** — Print statements showing progress in the terminal (image name, size, pixel count).

```python
    clusters_euc, centroids_euc = kmeans(data, K, metric='euclidean')   # Line 73
    labels_euc = clusters_euc.reshape(img_resized.shape[:2])            # Line 74
```
**Lines 73–74** — Runs K-Means with **Euclidean** distance. Reshapes the flat `(N,)` label array back to `(H, W)` so it can be displayed as an image.

```python
    clusters_man, centroids_man = kmeans(data, K, metric='manhattan')   # Line 77
    labels_man = clusters_man.reshape(img_resized.shape[:2])            # Line 78
```
**Lines 77–78** — Same thing with **Manhattan** distance.

```python
    fig = Figure(figsize=(14, 5), dpi=100)      # Line 80
    fig.patch.set_facecolor('#1a1a2e')          # Line 81
```
**Lines 80–81** — Creates an off-screen Matplotlib figure (14×5 inches) with a dark navy background.

```python
    ax1 = fig.add_subplot(1, 3, 1)   # Line 83
    ax2 = fig.add_subplot(1, 3, 2)   # Line 84
    ax3 = fig.add_subplot(1, 3, 3)   # Line 85
```
**Lines 83–85** — Creates 3 side-by-side panels: original | Euclidean result | Manhattan result.

**Lines 87–97** — Each panel displays its image. `cmap='viridis'` maps cluster IDs (integers 0 to K-1) to a colorful gradient, making different regions visually distinct. `axis('off')` hides axis ticks.

**Lines 99–104** — Adds an overall title above all 3 panels and calls `tight_layout` to remove excess whitespace.

```python
    return fig   # Line 105
```
**Line 105** — Returns the finished figure to be converted and shown in the GUI.

---

## Section 4: `fig_to_photoimage()` — Figure Conversion (Lines 107–121)

```python
def fig_to_photoimage(fig, target_width=None):   # Line 107
    canvas = FigureCanvasAgg(fig)                # Line 108
    canvas.draw()                                # Line 109
```
**Lines 107–109** — Wraps the Matplotlib figure in an Agg canvas and renders it into memory (no window opens).

```python
    buf = canvas.buffer_rgba()                              # Line 111
    w, h = canvas.get_width_height()                       # Line 112
    pil_image = Image.frombuffer('RGBA', (w, h), buf, 'raw', 'RGBA', 0, 1)  # Line 113
```
**Lines 111–113** — Extracts the raw pixel data from the canvas buffer and wraps it into a PIL Image object.

```python
    if target_width and pil_image.width > target_width:   # Line 115
        ratio = target_width / pil_image.width            # Line 116
        new_h = int(pil_image.height * ratio)             # Line 117
        pil_image = pil_image.resize((target_width, new_h), Image.LANCZOS)  # Line 118
```
**Lines 115–118** — Optionally scales the image down to fit a target width using high-quality LANCZOS resampling.

```python
    plt.close(fig)     # Line 120
    return pil_image   # Line 121
```
**Line 120** — Closes the Matplotlib figure to **free memory** (important when processing 10 images).

**Line 121** — Returns the PIL image ready for Tkinter to display.

---

## Section 5: `SlideshowApp` — The GUI Application (Lines 123–322)

### `__init__()` — Initialization (Lines 124–136)

```python
class SlideshowApp:                        # Line 123
    def __init__(self, root):              # Line 124
        self.root = root                   # Line 125
        self.root.title("K-Means Satellite Image Segmentation")  # Line 126
        self.root.configure(bg='#0f0f23') # Line 127
        self.root.state('zoomed')         # Line 128
```
**Lines 123–128** — Defines the app class. Sets the window title, dark background color `#0f0f23`, and starts the window **maximized** (`'zoomed'`).

```python
        self.slides = []         # Line 130
        self.slide_titles = []   # Line 131
        self.current_slide = 0   # Line 132
        self.photo_image = None  # Line 133
```
**Lines 130–133** — Instance variables:
- `self.slides` — list of PIL images (one per processed image/K combo)
- `self.current_slide` — index tracking which slide is displayed
- `self.photo_image` — **must stay as an attribute** to prevent Python's garbage collector from deleting the image from the screen

```python
        self._build_ui()         # Line 135
        self._start_processing() # Line 136
```
**Lines 135–136** — Builds the GUI layout, then immediately kicks off background processing.

---

### `_build_ui()` — Layout (Lines 138–226)

**Lines 139–158** — Creates the **header bar**: a dark blue frame at the top with a bold red title label and a grey subtitle label.

**Lines 160–168** — Creates the **image display area**: a frame that expands to fill all available space, containing a label that will show the slide images.

**Lines 170–178** — Creates the **status bar**: a label below the image that shows the current processing status or slide title.

**Lines 180–194** — Creates the **bottom navigation bar** and defines the shared button style (red background, white text, flat relief, hand cursor).

**Lines 196–219** — Creates the **◀ Previous** and **Next ▶** buttons and the slide counter label in the center.

```python
        self.root.bind('<Left>',   lambda e: self._prev_slide())   # Line 221
        self.root.bind('<Right>',  lambda e: self._next_slide())   # Line 222
        self.root.bind('<Escape>', lambda e: self.root.destroy())   # Line 223
```
**Lines 221–223** — Binds **keyboard shortcuts**: left/right arrow keys to navigate, Escape to close the window.

```python
        self.prev_btn.config(state=tk.DISABLED)   # Line 225
        self.next_btn.config(state=tk.DISABLED)   # Line 226
```
**Lines 225–226** — Disables both buttons at start (no slides yet). They will be enabled after processing.

---

### `_start_processing()` — Background Thread (Lines 228–230)

```python
    def _start_processing(self):                                                    # Line 228
        thread = threading.Thread(target=self._process_all_images, daemon=True)    # Line 229
        thread.start()                                                              # Line 230
```
**Lines 228–230** — Spawns a background thread to run `_process_all_images`. `daemon=True` means the thread automatically stops when the main window is closed. This is why the GUI stays responsive while images are processing.

---

### `_process_all_images()` — Main Processing Loop (Lines 232–269)

```python
        dataset_images = [               # Line 235
            "k-means map1.jpeg",         # Line 236
            "k-means map2.jpeg",         # Line 237
            "k-means map3.jpeg",         # Line 238
            "k-means map4.jpeg",         # Line 239
            "k-means map5.jpeg",         # Line 240
        ]
        k_test_values = [3, 5]          # Line 243
```
**Lines 235–243** — Defines the 5 satellite image files and the two K values to test.

**Lines 245–246** — Calculates the total number of combinations (5 images × 2 K values = 10).

**Lines 248–267** — Main processing loop:
- Line 249: builds the full file path using the script's own directory
- Line 251–253: skips missing files gracefully with a warning
- Lines 257–261: uses `root.after(0, ...)` to safely update the status label from the background thread
- Line 263: calls `generate_slide_figure()` to run K-Means
- Lines 264–267: converts result to PIL image and appends to `self.slides`

```python
        self.root.after(0, self._processing_done)   # Line 269
```
**Line 269** — When all images are done, schedules `_processing_done` to run on the main GUI thread.

---

### `_processing_done()` — Finish (Lines 271–275)

```python
    def _processing_done(self):              # Line 271
        self.status_label.config(            # Line 272
            text="All images processed!..."  # Line 273
        )
        self._display_slide()               # Line 275
```
**Lines 271–275** — Updates the status label to the success message and triggers a display refresh to show the first completed slide.

---

### `_display_slide()` — Rendering (Lines 277–312)

```python
        avail_w = self.image_frame.winfo_width() - 40    # Line 284
        avail_h = self.image_frame.winfo_height() - 20   # Line 285
```
**Lines 284–285** — Gets the current pixel dimensions of the image display area.

**Lines 287–290** — Fallback values if the window hasn't been fully drawn yet.

```python
        scale = min(avail_w / img_w, avail_h / img_h, 1.0)   # Line 293
```
**Line 293** — Calculates the scaling factor to fit the image in the window without ever enlarging it (max scale = 1.0).

```python
        self.photo_image = ImageTk.PhotoImage(display_img)    # Line 302
        self.image_label.config(image=self.photo_image)       # Line 303
```
**Lines 302–303** — Converts the PIL image to Tkinter's `PhotoImage` format and sets it on the label. The result is stored in `self.photo_image` — if this were a local variable it would be garbage-collected immediately and the image would vanish.

**Lines 305–312** — Updates the slide counter label and enables/disables the navigation buttons based on position.

---

### `_prev_slide()` / `_next_slide()` — Navigation (Lines 314–322)

```python
    def _prev_slide(self):                 # Line 314
        if self.current_slide > 0:         # Line 315
            self.current_slide -= 1        # Line 316
            self._display_slide()          # Line 317
    
    def _next_slide(self):                         # Line 319
        if self.current_slide < len(self.slides) - 1:  # Line 320
            self.current_slide += 1                # Line 321
            self._display_slide()                  # Line 322
```
**Lines 314–322** — Move to the previous or next slide. Boundary checks prevent going out of range.

---

## Section 6: Entry Point (Lines 324–331)

```python
if __name__ == "__main__":      # Line 324
    root = tk.Tk()              # Line 325
    app = SlideshowApp(root)    # Line 326
    root.mainloop()             # Line 327
```
**Lines 324–327** — Standard Python entry point. Creates the Tkinter root window, instantiates the app (which immediately builds the UI and starts background processing), then enters the **event loop** — `mainloop()` blocks here until the window is closed.

```python
    print("\nAll images processed successfully!")   # Lines 329–331
```
**Lines 329–331** — Printed to the terminal after the window closes.

---

## How to Run

```bash
cd "d:\1) colloge\3.5) summer 3\AI\INTRO TO AI\PROJECT"
python "K-Means satellite.py"
```

The app will:
1. Open a **maximized dark-themed window**
2. Process all 5 satellite images in the background (K=3 and K=5, both distance metrics)
3. Show **10 slides** total — navigate with **◀ ▶ buttons** or **← → arrow keys**
4. Press **Escape** to close

---

## Files Required

| File | Description |
|------|-------------|
| `K-Means satellite.py` | Main implementation (332 lines) |
| `k-means map1.jpeg` — `map5.jpeg` | 5 satellite images (must be in same folder) |
