import rasterio
import numpy as np
import rasterio as rio
import scipy
from rasterio.windows import Window, from_bounds
from rasterio.windows import transform
import matplotlib.pyplot as plt

IMG_PATH = 'output.tiff'
# IMG_PATH = 'C:/Users/darsh/Downloads/ch2_ohr_ncp_20250612T2031048828_d_img_d18/data/calibrated/20250612/ch2_ohr_ncp_20250612T2031048828_d_img_d18.img'

# CW = 1024 * 2
# CH = 1024 * 2
#
# print(f"Opening file: {IMG_PATH}")
#
# with rio.open(IMG_PATH) as src:
#     print(f"Shape: {src.width} x {src.height}")
#     print(f"No of Bands: {src.count}")
#     print(f"No pixels: {src.width * src.height}")
#     print(f"Window Size: {CW} x {CH}")
#
#     no_patch = (src.width//CW)*(src.height//CH)
#     print(f"No of Patches: {no_patch}")
#
#     window = Window(CW, CH, CW, CH)
#     win_transform = transform(window, src.transform)
#     x,y = win_transform * (0,0)
#     print(f"Window Transform: {win_transform}")
#     print(f"Top left corner coordinates: {x}, {y}")
#     data = src.read(window=window).reshape(CW, CH, 1)
#     plt.imshow(data)
#     plt.show()

class Processor:
    CW = 1024 * 2
    CH = 1024 * 2
    def __init__(self, path: str):
        self.path: str = path
        self.width: int | None = None
        self.height: int | None = None
        self.n_bands: int | None = None
        self.data = None

        with rasterio.open(self.path) as src:
            self.width = src.width
            self.height = src.height
            self.n_bands = src.count

    def print_data_info(self):
        print("-"*40)
        print(f"Filepath: {self.path}")
        print(f"Size: {self.width} x {self.height}")
        print(f"Number of Bands: {self.n_bands}")
        print(f"Number of pixels: {self.width * self.height}")
        print("-"*40)

    def print_processing_info(self):
        print("-"*40)
        print(f"Patch Size: {self.CW} x {self.CH}")
        print(f"Patch Space Size: {self.width // self.CW} x {self.height // self.CH}")
        print(f"Number of Patches: {(self.width // self.CW) * (self.height // self.CH)}")
        print("-"*40)

    def process_data(self):
        self.data = scipy.ndimage.gaussian_filter(self.data, sigma=0.5)

    def calculate_slope(self):
        dy, dx = np.gradient(self.data)
        slope = np.radians(np.arctan(np.sqrt(dx**2 + dy**2)))
        return slope

    def load_patch(self, col: int, row: int):
        print(f"Loading patch at ({col},{row})")
        with rasterio.open(self.path) as src:
            print(src.tags(ns="TIFF"))
            window = Window(self.CW*col, self.CH*row, self.CW, self.CH)
            self.data = src.read(window=window).reshape(self.CW, self.CH, self.n_bands)

            if hasattr(src, 'nodata') and src.nodata is not None:
                self.data = np.where(self.data == src.nodata, np.nan, self.data)

            print(f"Patch Memory Usage: {self.data.nbytes/1024**2} mb")

    def pyshape_shading(self):

    def shape_from_shading(self, iterations=100):
        """
        Implement shape-from-shading algorithm for height estimation
        """
        print("Applying shape-from-shading...")
        processed_image = self.data.astype(np.float32)
        # Initialize height map
        height = np.zeros_like(processed_image)

        # Get illumination parameters
        sun_elevation = -0.493025
        sun_azimuth = 153.986068

        # Convert to radians
        sun_elev_rad = np.radians(sun_elevation)
        sun_azim_rad = np.radians(sun_azimuth)

        # Light direction
        light_x = np.cos(sun_elev_rad) * np.sin(sun_azim_rad)
        light_y = np.cos(sun_elev_rad) * np.cos(sun_azim_rad)
        light_z = np.sin(sun_elev_rad)

        # Iterative shape-from-shading
        for i in range(iterations):
            # Calculate gradients
            grad_x = np.gradient(height, axis=1)
            grad_y = np.gradient(height, axis=0)

            # Calculate surface normals
            normal_x = -grad_x
            normal_y = -grad_y
            normal_z = np.ones_like(height)

            # Normalize
            norm = np.sqrt(normal_x ** 2 + normal_y ** 2 + normal_z ** 2)
            normal_x /= norm
            normal_y /= norm
            normal_z /= norm

            # Calculate reflectance
            reflectance = normal_x * light_x + normal_y * light_y + normal_z * light_z
            reflectance = np.clip(reflectance, 0, 1)

            # Update height based on intensity difference
            intensity_diff = processed_image - reflectance
            height += 0.01 * intensity_diff  # Learning rate

            if i % 20 == 0:
                print(f"  Iteration {i}/{iterations}")

        return height

    def visualize(self):
        plt.imshow(self.data)
        plt.colorbar()
        plt.show()

    def frankot_chellappa_method(self):
        """
        Frankot-Chellappa method using FFT integration
        Integrates gradient field to recover depth
        """
        # Estimate gradients from image
        x,y,_ = self.data.shape
        img = self.data
        print(img.shape)
        p, q = np.gradient(img, axis=1), np.gradient(img, axis=0)

        # FFT-based integration
        height, width, _ = p.shape

        # Create frequency domain coordinates
        u = np.fft.fftfreq(width, 1.0 / width)
        v = np.fft.fftfreq(height, 1.0 / height)
        U, V = np.meshgrid(u, v)

        # Avoid division by zero
        denom = U ** 2 + V ** 2
        denom[0, 0] = 1  # Set DC component to avoid division by zero

        # FFT of gradients
        P = np.fft.fft2(p)
        Q = np.fft.fft2(q)

        # Integrate in frequency domain
        Z_fft = -(1j * U * P + 1j * V * Q) / denom
        Z_fft[0, 0] = 0  # Set DC component to zero (remove arbitrary constant)

        # Inverse FFT to get depth
        z = np.real(np.fft.ifft2(Z_fft))

        return z

def main():
    processor = Processor(IMG_PATH)
    processor.print_data_info()
    processor.print_processing_info()

    processor.load_patch(4, 1)
    # processor.process_data()
    processor.visualize()
    # dem = processor.frankot_chellappa_method()
    # stat = dem - processor.data
    # print(stat.min())
    # print(stat.max())
    # print(np.std(stat))
    # print(np.mean(stat.flatten()))
    # print(dem.shape)
    # plt.imshow(dem)
    # plt.colorbar()
    # plt.show()


if __name__ == "__main__":
    main()
