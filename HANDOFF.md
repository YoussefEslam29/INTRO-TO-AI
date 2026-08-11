# 📋 Project Handoff — K-Means Satellite Image Segmentation
### Intro to AI Course | Summer 3 | YoussefEslam29/INTRO-TO-AI

> This document covers **everything that happened across all previous AI assistant sessions** for this project so any new session (or the student) can pick up exactly where things left off.

---

## 🗂 Project Overview

| Item | Details |
|------|---------|
| **Course** | Intro to AI — Summer Semester 3 |
| **Project Topic** | K-Means Clustering for Satellite Image Segmentation |
| **Assignment Requirements** | Apply K-Means with Euclidean & Manhattan distance on 5 satellite images, using K = 3 and K = 5 |
| **Source of Algorithm** | Must follow SEC (course) slide structure exactly |
| **Project Folder** | `d:\1) colloge\3.5) summer 3\AI\INTRO TO AI\PROJECT\` |

---

## 📁 All Files in the Project Folder

```
PROJECT/
├── K-Means satellite.py          ← FINAL GUI version (Tkinter slideshow app)
├── K-MEAN.PY                     ← First working version (plain matplotlib, no GUI)
├── sam_code.py                   ← Google Colab version (uses random init + median for Manhattan)
├── k_means_colab.ipynb           ← Jupyter/Colab notebook version
├── K-cluster.docx                ← Project report document (Word)
├── K-cluster.pdf                 ← Project report document (PDF)
├── explaination walkthrough code.md  ← Line-by-line code explanation of K-Means satellite.py
├── k-means map1.jpeg             ← Satellite image 1 (dataset)
├── k-means map2.jpeg             ← Satellite image 2
├── k-means map3.jpeg             ← Satellite image 3
├── k-means map4.jpeg             ← Satellite image 4
├── k-means map5.jpeg             ← Satellite image 5
├── k-means project.jpeg          ← Additional reference image
├── slides_extract/               ← Extracted images from SEC lecture slides
│   ├── slide_13_Picture 3.png
│   ├── slide_14_Picture 4.png
│   └── ... (slides 13–23)
└── sheet3_extract/               ← Other extracted materials
```

---

## 🕐 Session History (Chronological)

---

### Session 1 — August 6, 2026
**Conversation ID:** `68a2524c-1145-4d47-b9f4-ed0293fa3427`

**What happened:**
- User uploaded 5 satellite images (`k-means map1.jpeg` → `map5.jpeg`) and a reference image
- Uploaded the original `K-MEAN.PY` — a basic skeleton of K-Means from the SEC slides
- Uploaded SEC lecture slides which were extracted to `slides_extract/`
- **Dependency issues were resolved** — several rounds of numpy/opencv version conflicts:
  - First installed `opencv-python==5.0` which required `numpy>=2`, but other packages (scipy, gensim, numba) needed `numpy<2`
  - Downgraded to `numpy==1.26.4` but that broke opencv
  - Final fix: `opencv-python<5` (installed `4.14.0.94`) with `numpy 2.5.1`
- **`K-MEAN.PY` was written** — a clean implementation following the SEC slide structure:
  - `kmeans_sec()` — core algorithm with euclidean/manhattan distance
  - `segment_image()` — loads image, runs K-Means, shows 3-panel plot (original | euclidean | manhattan)
  - First K values tested: `[3, 5]`
- **K-Means was successfully run** on all 5 images (task-86 log shows convergence data):
  - map1 K=3: Euclidean converged @ iter 18, Manhattan @ iter 12
  - map2 K=5: Euclidean converged @ iter 92, Manhattan @ iter 45
  - etc.
- A **walkthrough document** was created explaining the algorithm, project structure, and how to run it

**Artifacts created:**
- `walkthrough.md` in brain/68a2524c — K-Means concept explanation + code structure overview
- The final `K-MEAN.PY` file on disk

---

### Session 2 — August 9, 2026
**Conversation ID:** `95b0d367-0adb-4b7c-b5c4-1a016d96ce54`

**What happened:**
- User uploaded a photo (likely a screenshot or slide image — `media__1786303593384.jpg`)
- This session produced `sam_code.py` — a **Google Colab-compatible** version of K-Means
- Key differences in `sam_code.py` vs `K-MEAN.PY`:
  - Uses **random centroid initialization** (`np.random.seed(42)`) instead of `data[:K].copy()`
  - For Manhattan distance, uses **median** instead of mean for centroid update (mathematically correct — median minimizes L1 distance)
  - Uses tolerance-based convergence (`shift < tol`) instead of `np.allclose`
  - Designed for Google Colab: uses `from google.colab import files`, expects images named `1.jpg` through `5.jpg`
  - Grid layout: rows = K values, columns = [Original, Euclidean, Manhattan]
- `k_means_colab.ipynb` was also created/modified around this time

**Artifacts created:**
- `sam_code.py` (4235 bytes)
- `k_means_colab.ipynb` (12363 bytes)

---

### Session 3 — August 11, 2026 (5:24 AM)
**Conversation ID:** `a12203a3-ce9f-4a45-9e19-a116c40bbe9c`

**What happened:**
- Two screenshots/images uploaded (likely output screenshots or slides)
- **`K-Means satellite.py` was created** — the most advanced version with a full Tkinter GUI slideshow
- The script was run multiple times (task-18, task-40, task-65 all show identical output = same script run 3 times)
- Confirmed working output:
  - 2D example from SEC slides: K=3, converges at iteration 3
    - Cluster 1: 3 pts → centroid [1.67, 4.0]
    - Cluster 2: 4 pts → centroid [7.25, 4.25]
    - Cluster 3: 4 pts → centroid [3.25, 9.0]
  - All 5 satellite images processed with K=3 and K=5 (81,900 pixels each at 300px width)

**Artifacts created:**
- `K-Means satellite.py` (12,448 bytes) — the final main implementation

---

### Session 4 — August 11, 2026 (8:30 AM – Present)
**Conversation ID:** `4792f44d-3601-42db-b14b-1ac5e8b2c6a5` *(current session)*

**What happened:**
- **`K-Means satellite.py` was run** from the AI assistant — launched the Tkinter GUI in background.
- **Line-by-line walkthrough** of `K-Means satellite.py` was created, saved to:
  - `brain/4792f44d/walkthrough.md` (the AI artifact)
  - `PROJECT/explaination walkthrough code.md` (in the project folder itself)
- **Code Refactoring & Streamlining:**
  - Removed `plot_clusters()` and `run_2d_example()` to reduce code length by ~55 lines.
  - Streamlined `K-Means satellite.py` to focus exclusively on satellite image segmentation, making it easier to explain to the course professor.
  - Updated `explaination walkthrough code.md` to reflect the updated code structure and line numbers.
- **This HANDOFF.md** was created and updated.

---

## 🧠 The Three Versions — Comparison

| Feature | `K-MEAN.PY` | `sam_code.py` | `K-Means satellite.py` |
|---------|------------|---------------|------------------------|
| **Platform** | Local Python | Google Colab | Local Python |
| **GUI** | None (pyplot) | None (pyplot) | Tkinter slideshow |
| **Centroid init** | `data[:K].copy()` (SEC slides) | Random (`seed=42`) | `data[:K].copy()` (SEC slides) |
| **Manhattan centroid** | Mean | **Median** (correct) | Mean |
| **Convergence** | `np.allclose` | `shift < tol` | `np.allclose` |
| **Image resize** | 400px wide | No resize | 300px wide |
| **Threading** | No | No | Yes (background thread) |
| **K values** | [3, 5] | [2, 3, 5] | [3, 5] |
| **Images** | Hardcoded names | `1.jpg`–`5.jpg` | `k-means map1.jpeg`–`map5.jpeg` |

---

## 🔑 Key Algorithm Details (For All Versions)

### The K-Means Steps (matching SEC slides):

```
1. Initialize centroids = first K data points  →  centroids = data[:K].copy()
2. Loop until convergence:
   a. Compute distances from every point to every centroid
   b. Assign each point to nearest centroid  →  clusters = argmin(distances)
   c. Recompute centroids as mean of assigned points
   d. If np.allclose(new_centroids, old_centroids): STOP
```

### Vectorized Distance (key NumPy trick):
```python
# data shape: (N, D)  →  data[:, np.newaxis] shape: (N, 1, D)
# centroids shape: (K, D)
# After broadcast subtraction: (N, K, D)

# Euclidean:
distances = np.linalg.norm(data[:, np.newaxis] - centroids, axis=2)  # → (N, K)

# Manhattan:
distances = np.sum(np.abs(data[:, np.newaxis] - centroids), axis=2)  # → (N, K)
```

### Image → Pixel Data Conversion:
```python
img = cv2.imread(image_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # OpenCV is BGR by default!
img_resized = cv2.resize(img, (300, new_h))  # resize for speed
data = img_resized.reshape((-1, 3))          # (H, W, 3) → (N, 3) — each row = 1 pixel [R,G,B]
```

---

## ▶️ How to Run

### Main GUI Version (recommended):
```bash
cd "d:\1) colloge\3.5) summer 3\AI\INTRO TO AI"
python "PROJECT\K-Means satellite.py"
```
- Opens a **maximized dark-themed Tkinter window**
- Processes all 5 satellite images in background
- Navigate with **◀ ▶ buttons** or **← → arrow keys**, **Escape** to close
- Total slides: 5 images × 2 K values = **10 slides**

### Original Simple Version:
```bash
python "PROJECT\K-MEAN.PY"
```
- Opens matplotlib windows sequentially for each image

### Colab Version:
- Upload `sam_code.py` and images named `1.jpg` through `5.jpg` to Google Colab
- Remove or mock the `from google.colab import files` import if running locally

---

## 📊 Convergence Results (from actual runs)

| Image | K | Metric | Converges @ Iteration |
|-------|---|--------|-----------------------|
| map1 | 3 | Euclidean | 18 |
| map1 | 3 | Manhattan | 12 |
| map2 | 5 | Euclidean | 92 |
| map2 | 5 | Manhattan | 45 |
| map3 | 3 | Euclidean | 15 |
| map3 | 3 | Manhattan | 11 |
| 2D example | 3 | Euclidean | **3** |

---

## ⚠️ Known Issues & Gotchas

1. **numpy/opencv version conflict** — was resolved by using `opencv-python<5` (version 4.14.x). If you reinstall, use:
   ```bash
   pip install "opencv-python<5"
   ```

2. **`matplotlib.use('Agg')`** must be called BEFORE importing `matplotlib.pyplot` — it tells Matplotlib not to open its own window (since Tkinter handles the display in the GUI version).

3. **`self.photo_image` must be kept as an attribute** — if it goes out of scope (garbage collected), the image disappears from the Tkinter label.

4. **`root.after(0, callback)`** must be used to update GUI from background threads — never call Tkinter methods directly from a non-main thread.

5. **`sam_code.py` imports `google.colab`** — this will crash if run locally. Remove the `from google.colab import files` line for local use.

---

## 📄 Documentation Files

| File | Location | Description |
|------|----------|-------------|
| `walkthrough.md` | `brain/68a2524c/` | K-Means concept + code structure overview |
| `walkthrough.md` | `brain/4792f44d/` | Line-by-line walkthrough of `K-Means satellite.py` |
| `explaination walkthrough code.md` | `PROJECT/` | Same walkthrough saved in the project folder |
| `HANDOFF.md` | `brain/4792f44d/` | This document — full session history |

---

## 🚀 Next Steps / What's Left

- [ ] The **report** (`K-cluster.docx` / `K-cluster.pdf`) exists — check if it needs updating with final results
- [ ] Could add **K=2** to the GUI version as `sam_code.py` tests it
- [ ] Could improve `sam_code.py` to work locally (remove Colab dependency)
- [ ] Manhattan centroid update: `K-Means satellite.py` and `K-MEAN.PY` use **mean** — technically **median** is more correct for Manhattan (as `sam_code.py` does) — if the professor cares about this, update those files
