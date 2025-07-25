# 🛰️ OHRC DEM Generation & Image Conversion Toolkit

This repository contains essential Python scripts for working with **OHRC (Optical High-Resolution Camera)** satellite imagery:

1. **`test2.py`**: Reconstructs a Digital Elevation Map (DEM) from a windowed portion of an OHRC `.img` image using Shape-from-Shading (SfS).
2. **`convert2.py`**: Converts OHRC `.img` files to GeoTIFF format for easier visualization and compatibility with GIS software.

---

## 📁 Contents

| File                          | Description                                                 |
|-------------------------------|-------------------------------------------------------------|
| `test2.py`        | DEM reconstruction from a cropped OHRC image window         |
| `convert2.py` | Converts `.img` satellite image to `.tiff` using Rasterio   |
| `sfs_windowed_dem.npy`        | Example output DEM (saved as NumPy array)    (cache)               |
| `output.tiff`                 | Example output GeoTIFF image                                |

---

## 🧱 1. Shape-from-Shading DEM Generator (`test2.py`)

This script loads a windowed region of an OHRC `.img` file, estimates surface gradients using Shape-from-Shading (SfS), and reconstructs the relative surface elevation using Poisson integration.

### 📌 Features

- Raster window extraction using `rasterio`
- Image normalization and Gaussian smoothing
- Gradient estimation from sun angle metadata
- Poisson integration (via FFT) to recover height map
- DEM visualization using `matplotlib`
- Saves output as a NumPy `.npy` file

### 📥 Inputs

- `.img` file path (from OHRC imagery)
- Metadata from XML or known parameters:
  - Image size (width/height)
  - Sun azimuth and elevation
  - Pixel resolution (m/pixel)

### 📤 Outputs

- Plot showing input grayscale window and terrain elevation
- NumPy array file: `sfs_windowed_dem.npy`

# 🧠 Shape-from-Shading with Poisson Integration

This document explains the **core idea, mathematics, and pipeline** behind reconstructing surface height maps from a single image using **Shape-from-Shading (SfS)** and **Poisson's Equation**.

It is particularly useful for projects involving:
- Lunar/planetary terrain recovery (e.g., OHRC / Chandrayaan-2 imagery)
- Photometric refinement
- Single-image 3D shape recovery

---

## 🔧 Core Idea

Shape-from-Shading (SfS) aims to recover the **3D surface shape** $z(x, y)$ from image brightness $I(x, y)$, under the assumptions of:

- Known light source direction
- Lambertian reflectance
- Smooth surface
- Constant albedo $\rho$

We use **Poisson integration** to reconstruct height $z$ from estimated surface gradients.

---

## 🧠 Mathematical Formulation

### 1. Lambertian Reflectance Model

The observed image intensity:

```math
I(x, y) = \rho \left( \vec{n}(x, y) \cdot \vec{s} \right)
```

Where:
- $\rho$: surface albedo (assumed constant)
- $\vec{n}(x, y)$: surface normal
- $\vec{s}$: unit light source direction vector

### 2. Surface Gradients

From the reflectance model, estimate:

```math
(p, q) = \left( \frac{\partial z}{\partial x}, \frac{\partial z}{\partial y} \right)
```

### 3. Poisson Equation

Given the gradient field $\vec{g} = (p, q)$, recover height by solving:

```math
\nabla^2 z = \frac{\partial p}{\partial x} + \frac{\partial q}{\partial y} = \text{div}(\vec{g})
```

This is a **Poisson equation** in 2D.

---

## ✅ Why Use Poisson's Method?

- ✅ Robust to noise (unlike direct integration)
- ✅ Naturally smooths the solution
- ✅ Solves a global system instead of local path integration
- ✅ Can be solved efficiently using:
  - FFT (Fast Fourier Transform)
  - Multigrid methods
  - Conjugate Gradient

---

## 🧪 Practical Pipeline

1. **Preprocessing**
   - Normalize image intensity to [0, 1]
   - Apply Gaussian blur to reduce noise

2. **Lighting Geometry**
   - Compute light direction vector $\vec{s}$ from sun azimuth and elevation

3. **Estimate Gradients**
   - Use the reflectance equation to derive $p(x, y)$ and $q(x, y)$

4. **Compute Divergence** 
```math 
f(x, y) = \frac{\partial p}{\partial x} + \frac{\partial q}{\partial y}
``` 

5. **Solve Poisson Equation**
   - In frequency domain (FFT):
     ```python
     f_hat = fft2(f)
     z_hat = f_hat / laplacian_kernel
     z = ifft2(z_hat)
     ```
   - Or in spatial domain using iterative solvers

6. **Output**
   - Surface height map $z(x, y)$ in real-world units (e.g., meters)

---

## 🧩 Applications in this project

- 🪐 **Lunar & planetary terrain reconstruction** (e.g., Chandrayaan-2 OHRC imagery)


---

## 🛠 Tools & Libraries

| Language | Tools/Packages                                  |
|----------|--------------------------------------------------|
| Python   | `numpy`, `scipy.fft`, `rasterio`, `matplotlib`  |
| Python   | [`poisson_reconstruct`](https://pypi.org/project/poisson-reconstruct/) |
| C++      | OpenCV, Eigen, custom Poisson solvers           |
| MATLAB   | `poisson_solver_function`, `del2`, etc.         |

## Required Tools
Make sure system has `gdal` installed and configured properly for more information on requirements of gdal see [RasterIo Installation Guide](https://rasterio.readthedocs.io/en/stable/installation.html) for more information.

### ▶️ Run Script

```bash
### ▶️ How to Use
### add path to your image file and xml file in the script variables of test2.py before executing
python test2.py
```
### Sample Output

Output generated with data from ISRO data archive (`ch2_ohr_ncp_20250301t0406349947_d_img_d18.zip`):

![ch2_ohr_ncp_20250301t0406349947_d_img_d18 DEM Output](dem.png)