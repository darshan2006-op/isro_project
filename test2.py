import numpy as np
import rasterio
from rasterio.windows import Window
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from scipy.fft import fft2, ifft2
import xml.etree.ElementTree as ET

# ------------------------
# CONFIGURATION
# ------------------------

# Input image path (OHRC .img)
img_path = "input.img"
xml_path = "input.xml"

tree = ET.parse(xml_path)
root = tree.getroot()
ns = {'isda': 'https://isda.issdc.gov.in/pds4/isda/v1'}

# OHRC metadata from XML
sun_elevation_deg = float(root.find('.//isda:sun_elevation', ns).text)
sun_azimuth_deg = float(root.find('.//isda:sun_azimuth', ns).text)
pixel_resolution = float(root.find('.//isda:pixel_resolution', ns).text)

# Define a center crop window (you can change this)
win_size = 2048
col_off = 2048 * 2
row_off =2048 * 2
window = Window(col_off, row_off, win_size, win_size)

print(f"🪟 Windowing image: offset=({row_off},{col_off}), size={win_size}x{win_size}")


with rasterio.open(img_path, 'r') as src:
    I = src.read(1, window=window)

# ------------------------
# 2. PREPROCESS IMAGE
# ------------------------
I = I.astype(np.float32)
I = (I - I.min()) / (I.max() - I.min())  # Normalize to [0,1]
I = gaussian_filter(I, sigma=1)  # Optional: smooth to reduce noise

# ------------------------
# 3. COMPUTE LIGHT VECTOR FROM SUN AZIMUTH & ELEVATION
# ------------------------
az = np.deg2rad(sun_azimuth_deg)
el = np.deg2rad(sun_elevation_deg)
lx = np.cos(el) * np.sin(az)
ly = np.cos(el) * np.cos(az)
lz = np.sin(el)
s = np.array([lx, ly, lz])  # Light vector

# ------------------------
# 4. ESTIMATE SURFACE GRADIENTS (p, q)
# ------------------------
eps = 1e-4
nz = I * lz
nz = np.clip(nz, 0, 1)
denom = np.sqrt(1 - nz**2 + eps)

p = (I * lx) / denom
q = (I * ly) / denom

# ------------------------
# 5. POISSON INTEGRATION TO GET ELEVATION
# ------------------------
def poisson_solver(p, q):
    h, w = p.shape
    fy = np.zeros_like(p)
    fx = np.zeros_like(q)
    fy[1:, :] = p[1:, :] - p[:-1, :]
    fx[:, 1:] = q[:, 1:] - q[:, :-1]
    f = fx + fy
    f_hat = fft2(f)

    y, x = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
    denom = (2 * np.cos(2 * np.pi * x / w) - 2) + (2 * np.cos(2 * np.pi * y / h) - 2)
    denom[0, 0] = 1  # avoid divide by 0

    z_hat = f_hat / denom
    z_hat[0, 0] = 0  # remove DC offset
    Z = np.real(ifft2(z_hat))
    return Z

Z = poisson_solver(p, q)
Z_scaled = Z * pixel_resolution  # Elevation in meters

# ------------------------
# 6. DISPLAY & SAVE
# ------------------------
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.title("Windowed Input Image")
plt.imshow(I, cmap='gray')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title("DEM from SfS (meters)")
plt.imshow(Z_scaled, cmap='terrain')
plt.colorbar(label='Elevation (m)')
plt.axis('off')

plt.tight_layout()
plt.show()

# Save elevation data
np.save('sfs_windowed_dem.npy', Z_scaled)
print("✅ DEM saved as 'sfs_windowed_dem.npy'")
